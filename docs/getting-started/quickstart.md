# Quick Start

This guide walks you through generating images and reproducing the paper's
headline metrics. The release is **evaluation / inference only** — the four main
models (JiT, DiT, SiT, PixelFlow) are used from their pretrained checkpoints; no
training code is shipped.

Before you start, make sure you have installed the environment and downloaded
the checkpoints and FID statistics — see [Installation](installation.md) and
[`CHECKPOINTS.md`](../../CHECKPOINTS.md).

All experiment entry points live in `experiments/` (plus the self-contained
`spiral_test/` toy ablation). Each writes images, a `config.json`, and a
`metrics.json` under `outputs/`. Runs are resumable — re-invoking a command
skips already-generated images and cached metrics.

## CFG-only baselines (FID / IS)

Generate the zero-guidance baseline for any of the four models. The default is
1000 classes × 50 images = 50K samples:

```bash
uv run python experiments/imagenet_no_guidance.py --model jit-h-16
```

Swap `--model` for `dit`, `sit`, or `pixelflow`. The reported CFG-only FID is
≈ 2 for all four models (JiT-H 1.86 · PixelFlow 1.98 · SiT 2.06 · DiT 2.27).

## Headline guidance study (bird Child-FID / Validity)

The headline experiment is a **DPS ρ-sweep**: sweep `--rho_override` to trace
the Pareto frontier of guided-class FID (Child-FID) vs. classifier validity.

```bash
# x-prediction model (JiT-H/16, Heun, x-space guidance)
uv run python experiments/finegrained_bird_tfg.py \
    --model jit-h-16 --guidance_mode dps --rho_override 0.5

# epsilon-prediction model (DiT-XL/2, 100-step DDPM)
uv run python experiments/finegrained_bird_tfg.py \
    --model dit --guidance_mode dps --rho_override 0.5 --nfe 100
```

Sweep `--rho_override` over several values (e.g. `0.1 0.3 0.5 ...`) to produce
the frontier. `--model sit` and `--model pixelflow` are also supported. The
zero-guidance reference point comes from
`experiments/finegrained_bird_no_guidance.py`. Note that guided runs for `dit`
and `sit` need `--nfe 100` to match the paper's guided protocol (NFE ≈ 100),
because their defaults in `experiments/model_configs.json` are the CFG-only
published-optimum configuration (NFE 250).

## Inverse problems (deblur / super-resolution)

```bash
# Gaussian deblur
uv run python experiments/deblur_sr.py --task deblur \
    --model jit-h-16 --guidance_mode dps --imagenet_dir <path/to/imagenet/val>

# 4x super-resolution
uv run python experiments/deblur_sr.py --task super_resolution \
    --model jit-h-16 --guidance_mode dps --imagenet_dir <path/to/imagenet/val>
```

These report LPIPS / PSNR / SSIM against degraded ImageNet validation images and
need a local ImageNet validation set passed via `--imagenet_dir`.

## Toy manifold ablation (crossed lines)

`spiral_test/` is a self-contained 2D→512D toy ablation that trains tiny MLPs
from scratch (it does **not** use the pretrained Diffusion Transformers):

```bash
uv run python spiral_test/generate_data.py --name crossed_lines
uv run python spiral_test/train.py --data crossed_lines
uv run python spiral_test/inference.py --exp crossed_lines_<YYYYMMDD_HHMMSS> -s 0.0 2.0 5.0 -n 50
```

See [`spiral_test/README.md`](../../spiral_test/README.md) for details.

## Useful shared flags

The bird / inverse / CFG scripts share several flags:

| Flag | Description |
|------|-------------|
| `--nfe`, `--sampling_method` | Override the per-model sampler defaults |
| `--images_per_class` | Samples generated per class |
| `--batch_size` | Per-batch sample count (lower it if you hit OOM) |
| `--device` | `cuda` (default) or `cpu` |
| `--seed` | Random seed |
| `--force_rerun` | Ignore cached images/metrics and regenerate |

Run any script with `--help` for the full list.

## Programmatic generation

You can drive the JiT model directly through the supported loading API,
`load_checkpoint_for_inference`. It reads the checkpoint, infers the model
variant from the filename (`jit-h-16.pth` → `JiT-H/16`), builds a `Denoiser`
with the matching official configuration (Heun, 50 steps, CFG 2.2 for JiT-H/16),
and loads the EMA weights:

```python
import torch
from jit_tfg.models.jit.utils.checkpoint import load_checkpoint_for_inference

device = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpoint path from CHECKPOINTS.md
model, config = load_checkpoint_for_inference(
    checkpoint_path="checkpoints/jit/jit-h-16.pth",
    device=device,
)

labels = torch.randint(0, 1000, (4,), device=device)
images = model.generate(labels)  # (B, 3, 256, 256) in [-1, 1]

images = ((images + 1) / 2 * 255).clamp(0, 255).byte()
```

> Do not construct `Denoiser` by hand from an ad-hoc args object — its
> `__init__` requires many more fields (dropout rates, EMA decays, `t_eps`,
> `noise_scale`, CFG interval, ...) than the sampling-related ones. Use
> `load_checkpoint_for_inference`, or the experiment scripts in `experiments/`,
> which are the canonical, tested entry points.

## Next Steps

- [Architecture Overview](../concepts/architecture.md): Deep dive into model design.
- [Research Context](../concepts/research-context.md): Understand the research goals.
- [Timestep Conventions](../concepts/timestep-conventions.md): How DiT/SiT/JiT time conventions differ.
