# Quick Start

This guide walks you through training a JiT model and generating images.

## Training

### Single GPU Training

```bash
python experiments/jit_train.py \
    --data_path /path/to/imagenet \
    --model JiT-B/16 \
    --img_size 256 \
    --batch_size 128 \
    --epochs 200 \
    --output_dir ./outputs/jit_b_16
```

### Multi-GPU Training

For distributed training across multiple GPUs:

```bash
torchrun --nproc_per_node=8 experiments/jit_train.py \
    --data_path /path/to/imagenet \
    --model JiT-L/16 \
    --img_size 256 \
    --batch_size 64 \
    --epochs 400 \
    --output_dir ./outputs/jit_l_16
```

### Training Configuration

Key training arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `JiT-B/16` | Model variant |
| `--img_size` | `256` | Image resolution |
| `--batch_size` | `128` | Per-GPU batch size |
| `--epochs` | `200` | Total training epochs |
| `--blr` | `5e-5` | Base learning rate |
| `--warmup_epochs` | `5` | LR warmup epochs |
| `--lr_schedule` | `constant` | LR schedule (`constant` or `cosine`) |

### Flow Matching Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--P_mean` | `-0.8` | Timestep distribution mean |
| `--P_std` | `0.8` | Timestep distribution std |
| `--noise_scale` | `1.0` | Initial noise scale |
| `--t_eps` | `5e-2` | Epsilon for numerical stability |
| `--label_drop_prob` | `0.1` | CFG label dropout probability |

## Image Generation

### Generate from Checkpoint

```bash
python experiments/jit_train.py \
    --resume ./outputs/jit_l_16 \
    --evaluate_gen \
    --num_images 50000 \
    --cfg 4.0 \
    --sampling_method heun \
    --num_sampling_steps 50
```

### Programmatic Generation

```python
import torch
from types import SimpleNamespace
from jit_tfg.models.jit.denoiser import Denoiser

# Model configuration
args = SimpleNamespace(
    model="JiT-L/16",
    img_size=256,
    class_num=1000,
    attn_dropout=0.0,
    proj_dropout=0.0,
    label_drop_prob=0.1,
    P_mean=-0.8,
    P_std=0.8,
    t_eps=5e-2,
    noise_scale=1.0,
    ema_decay1=0.9999,
    ema_decay2=0.9996,
    sampling_method="heun",
    num_sampling_steps=50,
    cfg=4.0,
    interval_min=0.0,
    interval_max=1.0,
)

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Denoiser(args).to(device)
model.eval()

# Load checkpoint (optional)
# checkpoint = torch.load("checkpoint.pth", map_location=device)
# model.load_state_dict(checkpoint["model"])

# Generate images
batch_size = 4
labels = torch.randint(0, 1000, (batch_size,), device=device)

with torch.no_grad():
    images = model.generate(labels)  # (B, 3, 256, 256) in [-1, 1]

# Convert to [0, 255] for display
images = ((images + 1) / 2 * 255).clamp(0, 255).byte()
```

### Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling_method` | `heun` | ODE solver (`euler` or `heun`) |
| `num_sampling_steps` | `50` | Integration steps |
| `cfg` | `4.0` | Classifier-free guidance scale |
| `interval_min` | `0.0` | CFG interval start |
| `interval_max` | `1.0` | CFG interval end |

## Evaluation

### Compute FID and IS

During training with online evaluation:

```bash
python experiments/jit_train.py \
    --data_path /path/to/imagenet \
    --online_eval \
    --eval_freq 40 \
    --num_images 50000
```

### Standalone Evaluation

```bash
python experiments/jit_train.py \
    --resume ./outputs/jit_l_16 \
    --evaluate_gen \
    --num_images 50000 \
    --cfg 4.0
```

## Understanding the Training Loop

The training implements flow matching with the following steps:

```mermaid
graph TD
    A[Clean Image x] --> B[Sample t ~ LogitNormal]
    A --> C[Sample noise e ~ N(0, I)]
    B --> D[Create z_t = t*x + (1-t)*e]
    C --> D
    D --> E[Model predicts x̂]
    E --> F[Compute velocity v̂ = (x̂-z)/(1-t)]
    A --> G[True velocity v = (x-z)/(1-t)]
    F --> H[Loss = MSE(v, v̂)]
    G --> H
```

### Loss Functions

The codebase currently implements **x-prediction with v-loss**:

- **Prediction target**: Model outputs \\(\hat{x}_0\\) (clean image estimate)
- **Loss computation**: Computed in velocity space:

\\[
\mathcal{L} = \mathbb{E}_{t, x, \epsilon} \left[ \left\| v - \hat{v} \right\|^2 \right]
\\]

where \\(v = (x - z_t) / (1-t)\\) and \\(\hat{v} = (\hat{x} - z_t) / (1-t)\\).

## Monitoring Training

### TensorBoard

Training logs are written to TensorBoard:

```bash
tensorboard --logdir ./outputs/jit_l_16
```

Tracked metrics:

- `train_loss`: Flow matching loss
- `lr`: Learning rate
- `fid_cfg{X}_res{Y}`: FID score (if online_eval enabled)
- `is_cfg{X}_res{Y}`: Inception Score (if online_eval enabled)

### Checkpoints

Checkpoints are saved with the following structure:

```
outputs/jit_l_16/
├── checkpoint-last.pth  # Latest checkpoint
├── checkpoint-0.pth     # Epoch 0
├── checkpoint-100.pth   # Epoch 100
├── events.out.*         # TensorBoard logs
└── ...
```

Each checkpoint contains:

- `model`: Model state dict
- `model_ema1`: First EMA state dict
- `model_ema2`: Second EMA state dict
- `optimizer`: Optimizer state
- `epoch`: Current epoch

## Next Steps

- [Architecture Overview](../concepts/architecture.md): Deep dive into model design
- [Research Context](../concepts/research-context.md): Understand the research goals
- [API Reference](../api/index.md): Detailed code documentation
