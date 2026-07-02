"""Flow-matching denoiser for training and sampling with the JiT model.

This module provides the `Denoiser` class, which wraps the JiT transformer
backbone to implement:

1. **Flow Matching Training**: Implements flow-matching objectives with
   configurable prediction targets (x, v, e) and loss functions (x, v, e).

2. **Classifier-Free Guidance (CFG)**: Supports unconditional generation by
   randomly dropping class labels during training.

3. **ODE Sampling**: Implements Euler and Heun solvers for deterministic
   sampling from the learned flow.

4. **EMA (Exponential Moving Average)**: Maintains two EMA copies of model
   weights for stable sampling.

Flow Matching Background:
    The forward process defines a path from data x to noise e:
        z_t = t * x + (1 - t) * e

    The velocity field is:
        v = dz_t/dt = x - e = (x - z_t) / (1 - t)

    This module supports 9 combinations of prediction targets and loss functions:
    - Prediction targets: x (clean data), v (velocity), e (noise)
    - Loss functions: x-loss, v-loss, e-loss

Sampling:
    Starting from z_0 = e ~ N(0, σ²I), solve the ODE:
        dz_t/dt = v_pred(z_t, t)

    using either Euler or Heun integration up to t = 1 to get x.

References:
    - Flow Matching: "Flow Matching for Generative Modeling"
    - CFG: "Classifier-Free Diffusion Guidance"
    - JiT: "Back to Basics: Let Denoising Generative Models Denoise"
"""

from typing import Literal

import torch
import torch.nn as nn

from jit_tfg.models.jit.model import JiT_models

PredTarget = Literal["x", "v", "e"]
LossType = Literal["x", "v", "e"]


class Denoiser(nn.Module):
    """Flow-matching denoiser wrapper for JiT.

    Wraps the JiT model to provide training and sampling functionality
    for flow-based generative modeling with configurable prediction targets
    and loss functions.

    Supported Configurations (3x3 = 9 combinations):
        Prediction targets:
            - "x": Network predicts clean data x
            - "v": Network predicts velocity v = x - e
            - "e": Network predicts noise e

        Loss functions:
            - "x": L2 loss in x-space (clean data)
            - "v": L2 loss in v-space (velocity)
            - "e": L2 loss in e-space (noise)

    Conversion Formulas (from paper Table 2):
        Given z_t = t * x + (1 - t) * e:

        From x-prediction:
            v = (x - z) / (1 - t)
            e = (z - t * x) / (1 - t)

        From e-prediction:
            x = (z - (1 - t) * e) / t
            v = (z - e) / t

        From v-prediction:
            x = (1 - t) * v + z
            e = z - t * v

    Attributes:
        net: The JiT transformer backbone.
        pred_target: Prediction target ("x", "v", or "e").
        loss_type: Loss function type ("x", "v", or "e").
        img_size: Expected input image size.
        num_classes: Number of classes in the dataset.
        label_drop_prob: Probability of dropping labels during training for CFG.
        P_mean: Mean of the logit-normal timestep distribution.
        P_std: Std of the logit-normal timestep distribution.
        t_eps: Small epsilon to prevent division by zero.
        noise_scale: Scale of initial noise for sampling.
        ema_decay1: Decay rate for first EMA.
        ema_decay2: Decay rate for second EMA.
        method: Sampling method ("euler" or "heun").
        steps: Number of sampling steps.
        cfg_scale: Classifier-free guidance scale.
        cfg_interval: Tuple (min, max) for CFG application range.
    """

    def __init__(self, args) -> None:
        """Initialize the denoiser.

        Args:
            args: Configuration namespace containing:
                - model: Model variant name (e.g., "JiT-L/16")
                - pred_target: Prediction target ("x", "v", or "e")
                - loss_type: Loss function type ("x", "v", or "e")
                - img_size: Input image size
                - class_num: Number of classes
                - attn_dropout: Attention dropout rate
                - proj_dropout: Projection dropout rate
                - label_drop_prob: Label dropout probability for CFG
                - P_mean: Timestep distribution mean
                - P_std: Timestep distribution standard deviation
                - t_eps: Epsilon for numerical stability
                - noise_scale: Initial noise scale
                - ema_decay1: First EMA decay rate
                - ema_decay2: Second EMA decay rate
                - sampling_method: "euler" or "heun"
                - num_sampling_steps: Number of sampling steps
                - cfg: Classifier-free guidance scale
                - interval_min: CFG interval minimum
                - interval_max: CFG interval maximum
        """
        super().__init__()

        # Initialize the JiT backbone from registry
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=3,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )
        self.img_size = args.img_size
        self.num_classes = args.class_num

        # Prediction target and loss type configuration
        self.pred_target: PredTarget = getattr(args, "pred_target", "x")
        self.loss_type: LossType = getattr(args, "loss_type", "v")

        # Training hyperparameters
        self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # EMA (Exponential Moving Average) parameters
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        # Sampling hyperparameters
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def _convert_prediction(
        self,
        net_output: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert network output to x, v, e predictions.

        Based on the prediction target, converts the raw network output
        to all three representations (x, v, e) using the formulas from
        the paper (Table 2).

        Note:
            Numerical stability: Conversions involve division by t or (1-t),
            which can cause instability near t=0 or t=1. We use clamp_min(t_eps)
            to prevent division by zero. For production use, consider:
            - Loss weighting to down-weight samples near boundaries
            - Monitoring predictions for abnormally large values

        Args:
            net_output: Raw network output of shape (B, C, H, W).
            z: Noisy sample z_t of shape (B, C, H, W).
            t: Timestep of shape (B, 1, 1, 1).

        Returns:
            Tuple of (x_pred, v_pred, e_pred), each of shape (B, C, H, W).
        """
        if self.pred_target == "x":
            x_pred = net_output
            v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)
            e_pred = (z - t * x_pred) / (1 - t).clamp_min(self.t_eps)
        elif self.pred_target == "e":
            e_pred = net_output
            x_pred = (z - (1 - t) * e_pred) / t.clamp_min(self.t_eps)
            v_pred = (z - e_pred) / t.clamp_min(self.t_eps)
        elif self.pred_target == "v":
            v_pred = net_output
            x_pred = (1 - t) * v_pred + z
            e_pred = z - t * v_pred
        else:
            raise ValueError(f"Unknown pred_target: {self.pred_target}")

        return x_pred, v_pred, e_pred

    def _compute_loss(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        e: torch.Tensor,
        x_pred: torch.Tensor,
        v_pred: torch.Tensor,
        e_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss based on the configured loss type.

        Args:
            x: Ground truth clean data of shape (B, C, H, W).
            v: Ground truth velocity of shape (B, C, H, W).
            e: Ground truth noise of shape (B, C, H, W).
            x_pred: Predicted clean data of shape (B, C, H, W).
            v_pred: Predicted velocity of shape (B, C, H, W).
            e_pred: Predicted noise of shape (B, C, H, W).

        Returns:
            Scalar loss tensor.
        """
        if self.loss_type == "x":
            loss = (x - x_pred) ** 2
        elif self.loss_type == "v":
            loss = (v - v_pred) ** 2
        elif self.loss_type == "e":
            loss = (e - e_pred) ** 2
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss.mean(dim=(1, 2, 3)).mean()

    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Randomly drop class labels for classifier-free guidance training.

        With probability `label_drop_prob`, replaces class labels with the
        null class index (num_classes), enabling the model to learn both
        conditional and unconditional generation.

        Args:
            labels: Class labels of shape (B,) with values in [0, num_classes-1].

        Returns:
            Modified labels of shape (B,) where some entries are replaced
            with num_classes (the null class index).
        """
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None) -> torch.Tensor:
        """Sample timesteps from a logit-normal distribution.

        The logit-normal distribution provides better coverage of the full
        [0, 1] range compared to uniform sampling.

        Sampling procedure:
            z ~ Normal(P_mean, P_std)
            t = sigmoid(z)

        Args:
            n: Number of timesteps to sample (batch size).
            device: Device to create tensors on.

        Returns:
            Sampled timesteps of shape (n,) with values in (0, 1).
        """
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the flow matching training loss.

        Implements the flow matching training objective with configurable
        prediction target and loss function.

        Args:
            x: Clean images of shape (B, C, H, W), normalized to [-1, 1].
            labels: Class labels of shape (B,) with values in [0, num_classes-1].

        Returns:
            Scalar loss tensor.
        """
        labels_dropped = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        v = x - e

        net_output = self.net(z, t.flatten(), labels_dropped)
        x_pred, v_pred, e_pred = self._convert_prediction(net_output, z, t)

        loss = self._compute_loss(x, v, e, x_pred, v_pred, e_pred)
        return loss

    @torch.no_grad()
    def generate(self, labels: torch.Tensor) -> torch.Tensor:
        """Generate images by solving the probability flow ODE.

        Starting from pure noise, integrates the learned velocity field
        from t=0 to t=1 to generate images. Uses classifier-free guidance
        to improve sample quality and class fidelity.

        Args:
            labels: Target class labels of shape (B,) with values in
                [0, num_classes-1].

        Returns:
            Generated images of shape (B, C, H, W) in [-1, 1] range.
        """
        device = labels.device
        bsz = labels.size(0)

        z = self.noise_scale * torch.randn(bsz, 3, self.img_size, self.img_size, device=device)

        timesteps = (
            torch
            .linspace(0.0, 1.0, self.steps + 1, device=device)
            .view(-1, *([1] * z.ndim))
            .expand(-1, bsz, -1, -1, -1)
        )

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels)

        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)

        return z

    @torch.no_grad()
    def _forward_sample(self, z: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute velocity prediction with classifier-free guidance.

        Runs the model twice (conditional and unconditional) and combines
        the predictions using the CFG formula. Regardless of the prediction
        target, returns velocity for ODE integration.

        Args:
            z: Current noisy sample of shape (B, C, H, W).
            t: Current timestep of shape (B, 1, 1, 1).
            labels: Class labels of shape (B,).

        Returns:
            Guided velocity prediction of shape (B, C, H, W).
        """
        net_cond = self.net(z, t.flatten(), labels)
        _, v_cond, _ = self._convert_prediction(net_cond, z, t)

        net_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        _, v_uncond, _ = self._convert_prediction(net_uncond, z, t)

        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Perform one Euler integration step.

        Args:
            z: Current state of shape (B, C, H, W).
            t: Current timestep of shape (B, 1, 1, 1).
            t_next: Next timestep of shape (B, 1, 1, 1).
            labels: Class labels of shape (B,).

        Returns:
            Updated state of shape (B, C, H, W).
        """
        v_pred = self._forward_sample(z, t, labels)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Perform one Heun (improved Euler) integration step.

        Args:
            z: Current state of shape (B, C, H, W).
            t: Current timestep of shape (B, 1, 1, 1).
            t_next: Next timestep of shape (B, 1, 1, 1).
            labels: Class labels of shape (B,).

        Returns:
            Updated state of shape (B, C, H, W).
        """
        v_pred_t = self._forward_sample(z, t, labels)
        z_next_euler = z + (t_next - t) * v_pred_t

        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels)
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)

        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self) -> None:
        """Update EMA parameters from current model weights.

        Maintains two EMA copies with different decay rates:
            ema_param = decay * ema_param + (1 - decay) * current_param
        """
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params, strict=True):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params, strict=True):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)
