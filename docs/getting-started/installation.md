# Installation

This guide covers setting up the JiT-TFG development environment.

## Prerequisites

- **Python**: 3.11 or higher
- **CUDA**: 11.8+ (for GPU training)
- **Git**: For cloning the repository
- **uv**: Recommended for fast dependency management

## Quick Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager that we recommend for development.

```bash
# Clone the repository
git clone https://github.com/ManLuML/on-manifold-tfg.git
cd on-manifold-tfg

# Install with make (sets up environment and pre-commit hooks)
make install
```

This will:

1. Create a virtual environment
2. Install all dependencies
3. Set up pre-commit hooks for code quality
4. Generate the `uv.lock` file

### Using pip

```bash
# Clone the repository
git clone https://github.com/ManLuML/on-manifold-tfg.git
cd on-manifold-tfg

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Dependencies

### Core Dependencies

The project requires these main packages:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.9.1 | Deep learning framework |
| `torchvision` | latest | Image utilities |
| `numpy` | latest | Numerical computing |
| `einops` | latest | Tensor operations |
| `timm` | latest | Vision model utilities |
| `tensorboard` | latest | Training visualization |
| `scipy` | latest | Scientific computing |
| `opencv-python` | latest | Image I/O |

### Development Dependencies

For development work, additional packages are required:

| Package | Purpose |
|---------|---------|
| `pytest` | Testing framework |
| `ruff` | Linting and formatting |
| `mkdocs` | Documentation |
| `mkdocs-material` | Documentation theme |
| `mkdocstrings[python]` | API documentation |

## Verify Installation

After installation, verify everything works:

```bash
# Run tests
make test

# Or using pytest directly
uv run pytest tests/

# Check imports
python -c "from jit_tfg.models.jit.model import JiT_models; print(list(JiT_models.keys()))"
```

Expected output:

```
['JiT-B/16', 'JiT-B/32', 'JiT-L/16', 'JiT-L/32', 'JiT-H/16', 'JiT-H/32']
```

## GPU Setup

### CUDA Configuration

For GPU training, ensure CUDA is properly configured:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'cuDNN version: {torch.backends.cudnn.version()}')"
```

### Multi-GPU Setup

For distributed training, the codebase uses PyTorch's DistributedDataParallel (DDP):

```bash
# Example: 4 GPU training
torchrun --nproc_per_node=4 experiments/jit_train.py \
    --data_path /path/to/imagenet \
    --model JiT-L/16 \
    --batch_size 64
```

## Data Preparation

### ImageNet Dataset

The training script expects ImageNet in the standard torchvision format:

```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   └── ...
└── val/
    ├── n01440764/
    │   └── ...
    └── ...
```

### Preparing FID Reference Statistics

For evaluation, prepare reference statistics:

```bash
python -m jit_tfg.models.jit.prepare_ref \
    --data_path /path/to/imagenet \
    --output_path imagenet-train-256 \
    --img_size 256
```

This creates preprocessed images for FID computation.

## Troubleshooting

### Common Issues

#### ImportError: No module named 'jit_tfg'

Ensure the package is installed in editable mode:

```bash
pip install -e .
```

#### CUDA out of memory

Reduce batch size or use gradient accumulation:

```bash
python experiments/jit_train.py --batch_size 32
```

#### torch.compile errors

If you encounter compilation errors, disable torch.compile:

```python
# In your training script, set:
torch._dynamo.config.suppress_errors = True
```

### Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/ManLuML/on-manifold-tfg/issues)
- **Discussions**: For questions and feature requests

## Next Steps

- [Quick Start Guide](quickstart.md): Run your first training and generation
- [Architecture Overview](../concepts/architecture.md): Understand the model design
