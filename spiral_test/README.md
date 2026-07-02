# Spiral Test — JiT + Train-Free Guidance (TFG) Toy Experiment

A 2D toy-dataset experiment that compares the Train-Free Guidance behavior of diffusion models across **prediction targets (x, e, v)**.

## Overview

This experiment is designed to answer:

> **Which prediction target is most favorable for Train-Free Guidance?**

### Experimental design

1. Generate 2D toy data (e.g., a double spiral).
2. Random orthogonal projection from 2D → D dimensions (D = 2, 8, 32, 128, 512).
3. For each D, train a **classifier** + **three diffusion models** (x-pred, e-pred, v-pred).
4. Steer samples toward a target class with DPS (classifier) guidance.
5. Re-project back to 2D and visualize to compare guidance behavior.

### Pipeline

```
generate_data.py → train.py → inference.py
   (generate data)   (train)    (sample + visualize)
```

---

## 1. Data generation (`generate_data.py`)

### Supported datasets

| Name | Description | #Classes | Default noise |
|------|-------------|----------|---------------|
| `doublespiral` | Two interleaved spirals | 2 | 0.5 |
| `concentric_rings` | Concentric circles (radius ratio 1:2) | 2 | 0.1 |
| `circular_gaussians` | 8 Gaussians arranged in a circle (4 classes) | 4 | 0.3 |
| `grid_gaussians` | 3×3 grid of Gaussians | 9 | 0.2 |
| `crossed_lines` | Two crossing (X-shaped) lines | 2 | 0.1 |
| `half_arcs` | Two half-circle arcs (top/bottom) | 2 | 0.1 |

### Usage

```bash
# Default (doublespiral)
uv run python spiral_test/generate_data.py --name doublespiral

# A different dataset
uv run python spiral_test/generate_data.py --name circular_gaussians

# Custom parameters
uv run python spiral_test/generate_data.py \
    --name doublespiral \
    --total_points 20000 \
    --noise 0.3 \
    --data_range 3.0 \
    --seed 42
```

### Output files

```
spiral_test/data/<dataset_name>/
├── data.npz            # points_2d (N,2), labels (N,)
├── class_info.json     # metadata: #classes, colors, etc.
├── config.json         # parameters used for generation
└── visualization.png   # scatter plot of the data
```

---

## 2. Training (`train.py`)

For each value of D, this trains:
- **Classifier** — MLP-based (binary: sigmoid, multi-class: softmax)
- **Diffusion Models** — one each of x-pred, e-pred, v-pred (3 total)

### Flow matching definition

```
Forward process:  z_t = t · x + (1 - t) · ε
Prediction target:
  x-pred → network predicts clean data x
  e-pred → network predicts noise ε
  v-pred → network predicts velocity v = x - ε
```

### Usage

```bash
# Default (D=2,8,32,128,512, 500 epochs)
uv run python spiral_test/train.py --data doublespiral

# Select D values
uv run python spiral_test/train.py --data doublespiral --d_values 2 8 16

# Adjust epoch counts
uv run python spiral_test/train.py \
    --data doublespiral \
    --diffusion_epochs 1000 \
    --classifier_epochs 200

# Use CPU
uv run python spiral_test/train.py --data doublespiral --device cpu
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | `doublespiral` | Dataset name |
| `--d_values` | `2 8 32 128 512` | List of dimensions to train |
| `--diffusion_epochs` | `500` | Diffusion training epochs |
| `--classifier_epochs` | `100` | Classifier training epochs |
| `--seed` | `42` | Random seed |
| `--device` | `cuda` | Device (falls back to CPU automatically if no CUDA) |

### Output structure

```
spiral_test/output/<dataset>_<YYYYMMDD_HHMMSS>/
├── train_config.json           # training config
├── D2/
│   ├── proj_matrix.npy         # projection matrix (identity for D=2)
│   ├── classifier.pt           # trained classifier
│   ├── diffusion_x.pt          # x-prediction diffusion model
│   ├── diffusion_e.pt          # e-prediction diffusion model
│   └── diffusion_v.pt          # v-prediction diffusion model
├── D8/
│   └── ...
├── D32/
│   └── ...
├── D128/
│   └── ...
└── D512/
    └── ...
```

### Model architecture

- **Classifier**: MLP (128→128→128→output), ReLU activation
- **Diffusion Model**: ResMLP with sinusoidal time embedding
  - Hidden dim: 256, Residual blocks: 5
  - Each block: LayerNorm → Linear → ReLU → time projection → Linear → ReLU + skip connection

---

## 3. Inference (`inference.py`)

Loads the trained models, performs DPS-guidance sampling, and visualizes the results.

### DPS (classifier) guidance

```
v_guided = v_pred + s · ∇_z log p(y=target | x̂₀(z_t))
```

- `s` = guidance scale (0 means unconditional)
- `x̂₀` = clean data predicted by the diffusion model
- `p(y | x̂₀)` = classifier's predicted probability

### Usage

```bash
# Default (scale=0,2,5 / steps=50,100)
uv run python spiral_test/inference.py --exp doublespiral_20260113_123456

# Specify guidance scale and number of steps
uv run python spiral_test/inference.py \
    --exp doublespiral_20260113_123456 \
    -s 2.0 \
    -n 50

# Multiple scale × step combinations
uv run python spiral_test/inference.py \
    --exp doublespiral_20260113_123456 \
    -s 0.0 2.0 5.0 10.0 \
    -n 50 100 200

# Change target class
uv run python spiral_test/inference.py \
    --exp doublespiral_20260113_123456 \
    --target_class 0

# Adjust number of samples
uv run python spiral_test/inference.py \
    --exp doublespiral_20260113_123456 \
    --num_samples 5000
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--exp` | **(required)** | Experiment folder name under `spiral_test/output/` |
| `-s`, `--guidance_scale` | `0.0 2.0 5.0` | Guidance scale (multiple values allowed) |
| `-n`, `--num_steps` | `50 100` | Number of sampling steps (multiple values allowed) |
| `--num_samples` | `10000` | Number of samples to generate |
| `--target_class` | `1` | Target class for guidance |
| `--seed` | `42` | Random seed |
| `--device` | `cuda` | Device |

### Pipeline stages

1. **Load data & models** — 2D data + per-D projection matrices, classifier, and diffusion models (x/e/v).
2. **DPS-guidance sampling** — Euler ODE sampling for each (guidance_scale, num_steps) combination.
3. **Compute metrics** — quantitative quality evaluation of the generated samples.
4. **Visualize** — re-project to 2D over the classifier decision-boundary background.

### Metrics

For each (D, prediction target) combination, five metrics are computed:

| Metric | Description |
|--------|-------------|
| `on_manifold_rate` | Fraction of samples on the manifold (`crossed_lines`, `half_arcs` only; `null` otherwise) |
| `source_mmd` | MMD between generated samples ↔ all GT (Gaussian kernel, median heuristic) |
| `target_mmd` | MMD between generated samples ↔ target-class GT |
| `kl_div` | KL(p_gen ‖ p_target), via dual KDE (`null` when the covariance is singular) |
| `class_accuracy` | Fraction classified as the target class by the classifier |

### Output

```
<exp_dir>/results/
├── metrics_s0.0_steps50.json   # per-scale, per-step quantitative metrics
├── metrics_s2.0_steps100.json
├── s0.0_steps50.png            # per-scale, per-step visualization image
├── s2.0_steps100.png
├── ...
└── inference_config.json       # config used for inference
```

Each image is a **(D values × 4)** grid:

|  | Ground Truth | x-pred | e-pred | v-pred |
|--|--------------|--------|--------|--------|
| **D=2** | original data | generated | generated | generated |
| **D=8** | original data | generated | generated | generated |
| **D=32** | original data | generated | generated | generated |
| **D=128** | original data | generated | generated | generated |
| **D=512** | original data | generated | generated | generated |

The background shows the classifier's decision boundary as a color map.

---

## Quick Start

```bash
# 1. Generate data
uv run python spiral_test/generate_data.py --name doublespiral

# 2. Train (quick test: small D, few epochs)
uv run python spiral_test/train.py \
    --data doublespiral \
    --d_values 2 8 \
    --diffusion_epochs 100

# 3. Inference (after checking the output folder name)
uv run python spiral_test/inference.py \
    --exp doublespiral_<YYYYMMDD_HHMMSS> \
    -s 0.0 2.0 5.0 \
    -n 50
```

---

## Notes

- **Self-contained**: the code in this folder is **fully independent** of the main codebase (`src/jit_tfg/`). It imports no external modules, so it is unaffected by changes to the main code.
- **Dependencies**: `torch`, `numpy`, `matplotlib`, `scipy`, `tqdm` (all included in the project's base dependencies).
- **GPU**: falls back to CPU automatically when CUDA is unavailable.
- **Reproducibility**: the `--seed` option makes data generation, training, and inference all reproducible.
