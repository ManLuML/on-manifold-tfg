# Installation

This guide covers setting up `on-manifold-tfg` for **evaluation / inference**.
The release ships the four pretrained Diffusion Transformers' wrappers plus the
TFG sampler and evaluation code — **no model training is included**, and the
four main models are used from their authors' pretrained checkpoints.

## Prerequisites

- **Python**: 3.11 or higher
- **CUDA**: 11.8+ recommended for GPU inference (CPU works but is slow)
- **Git**: For cloning the repository
- **uv**: [`uv`](https://docs.astral.sh/uv/) for dependency management

## Quick Installation

This release is **self-contained** — no Git submodules and no upstream clones
are required. Install the locked environment with `uv`:

```bash
# Clone the repository
git clone https://github.com/ManLuML/on-manifold-tfg.git
cd on-manifold-tfg

# Create the virtual environment and install all dependencies from uv.lock
uv sync
```

All commands below are run with `uv run python ...` from the repository root.

## Download checkpoints and FID statistics

The repository never commits model weights or reference statistics. Fetch them
with the two helper scripts (see [`CHECKPOINTS.md`](../../CHECKPOINTS.md) for the
full per-model details and expected local paths):

```bash
# Diffusion model checkpoints -> checkpoints/
uv run python scripts/download_checkpoints.py --all

# FID reference statistics -> src/jit_tfg/evaluation/generation/fid_stats/
uv run python scripts/download_fid_stats.py
```

The guidance/evaluation classifiers used by the bird benchmark are pulled from
the HuggingFace Hub automatically on first run (cached under
`~/.cache/huggingface`), so they need no manual step. The Stable Diffusion VAE
used by the latent models (DiT, SiT) is likewise fetched at load time.

## Verify Installation

After installation, verify everything works:

```bash
# Run the unit tests
uv run pytest tests/

# Check that the model wrappers import and the JiT variants register
uv run python -c "from jit_tfg.models.jit.model import JiT_models; print(list(JiT_models.keys()))"
```

Expected output:

```
['JiT-B/16', 'JiT-B/32', 'JiT-L/16', 'JiT-L/32', 'JiT-H/16', 'JiT-H/32']
```

## GPU Setup

Inference runs on a single GPU. Confirm CUDA is visible to PyTorch:

```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
uv run python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

Pass `--device cpu` to the experiment scripts to run without a GPU (slow).

## Troubleshooting

### ImportError: No module named 'jit_tfg'

Run commands through the project environment with `uv run ...`, or activate the
environment created by `uv sync` (`source .venv/bin/activate`).

### CUDA out of memory

Reduce the per-batch sample count with `--batch_size` on the experiment scripts.

### Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/ManLuML/on-manifold-tfg/issues)
- **Discussions**: For questions and feature requests

## Next Steps

- [Quick Start Guide](quickstart.md): Generate images and reproduce the paper metrics.
- [Architecture Overview](../concepts/architecture.md): Understand the model design.
