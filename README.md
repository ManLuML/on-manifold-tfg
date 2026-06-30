# Not All Prediction Targets Keep Training-Free Diffusion Guidance on the Manifold

**The diffusion prediction target (x, ε, or v) decides whether training-free guidance fails *gracefully* or *catastrophically*. x-prediction keeps guided samples on the data manifold; ε-prediction does not.**

[![ECCV 2026 — Accepted](https://img.shields.io/badge/ECCV%202026-Accepted-brightgreen)](#citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[**Paper**](#) (coming soon) &nbsp;·&nbsp; [**arXiv**](#) (coming soon) &nbsp;·&nbsp; [**Project Page**](https://ManLuML.github.io/on-manifold-tfg) &nbsp;·&nbsp; [**Code**](https://github.com/ManLuML/on-manifold-tfg)

> Authors: **Yunsung Lee**, **Hyeongmin Lee** · maum.ai

---

## Abstract

Training-free guidance (TFG) steers diffusion models without retraining, but strong guidance risks driving samples off-manifold. The resulting catastrophic failures—collapsed, artifact-ridden images—differ from graceful failures, where guidance misses the target but images remain realistic. We trace this distinction to the prediction target: all TFG methods compute guidance from a clean-data estimate x̂, so its fidelity governs manifold preservation. We prove a strict error amplification hierarchy: ε-prediction's recovery formula divides by *t*, amplifying errors unboundedly at high noise; v-prediction incurs bounded amplification; x-prediction incurs none. On ImageNet 256×256, we evaluate four pretrained Diffusion Transformers spanning all three targets. At matched classifier accuracy, guided-class FID (Child FID) reveals a 5.2-point gap between x- and ε-prediction (**32.9** vs. **38.1**)—manifold damage invisible to standard evaluation. This extends to style transfer, establishing x-prediction as the target that keeps guidance failures graceful rather than catastrophic.

---

## Installation

This release is **self-contained** — no Git submodules. We use [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/ManLuML/on-manifold-tfg.git
cd on-manifold-tfg
uv sync
```

All commands below are run with `uv run python ...` from the repository root. The code is **evaluation/inference only** — no model training is shipped, and the four main diffusion models are used pretrained.

---

## Pretrained Models

The paper evaluates four pretrained Diffusion Transformers that span all three prediction targets at comparable unconditional quality (CFG-only FID ≈ 2 on ImageNet 256×256):

| Model | Prediction target | Space | Sampler (default) | CFG-only FID ↓ |
|-------|:-----------------:|:-----:|:-----------------:|:--------------:|
| **JiT-H/16** | x | Pixel | Heun | **1.86** |
| **PixelFlow-XL** | v | Pixel | Euler (per-stage) | 1.98 |
| **SiT-XL/2** | v | Latent | Heun | 2.06 |
| **DiT-XL/2** | ε | Latent | DDPM (250 steps) | 2.27 |

> **Checkpoints** point at each model's official source. DiT / SiT / PixelFlow download automatically via the helper scripts; **JiT-H/16 is a manual download** from the original authors' repo (see below). Expected local paths:
>
> | Model | Local checkpoint path | Source |
> |-------|-----------------------|--------|
> | JiT-H/16 | `checkpoints/jit/jit-h-16.pth` | [`LTH14/JiT`](https://github.com/LTH14/JiT) — official weights (Dropbox), manual download |
> | DiT-XL/2 | `checkpoints/dit/DiT-XL-2-256x256.pt` | `facebook/DiT-XL-2-256` |
> | SiT-XL/2 | `checkpoints/sit/SiT-XL-2-256.pt` | official SiT release |
> | PixelFlow-XL | `checkpoints/pixelflow/` | `ShoufaChen/PixelFlow-Class2Image` |
>
> See [`CHECKPOINTS.md`](CHECKPOINTS.md) for the per-model download instructions.

The two evaluation classifiers used by the bird benchmark (a guidance classifier and a separate validity classifier) are pulled from the HuggingFace Hub on first run.

---

## Reproduce the Paper

All experiment entry points live in `experiments/` and `spiral_test/`. Each writes images, a `config.json`, and a `metrics.json` under `outputs/`. Runs are resumable — re-invoking a command skips already-generated images and cached metrics.

The headline guidance study is a **DPS ρ-sweep** (rho-only guidance): sweep `--rho_override` to trace the Pareto frontier. (The scripts retain other TFG presets and guidance-space flags from development; the paper reports only the DPS ρ-sweep.)

| Paper result | Command | Notes |
|--------------|---------|-------|
| **ImageNet-256 guided — Child-FID / Validity ρ-sweep** (abstract headline; Child-FID **32.9** (x) vs. **38.1** (ε) at matched classifier accuracy) | `uv run python experiments/imagenet_tfg.py --model jit-h-16 --guidance_mode dps --rho_override <ρ>`<br>`uv run python experiments/imagenet_tfg.py --model dit --guidance_mode dps --rho_override <ρ>` | Sweep `<ρ>` (e.g. `0.1 0.3 0.5 ...`) to trace the Child-FID / Validity Pareto frontier on ImageNet 256×256 — the numbers behind the abstract's 5.2-point x-vs-ε gap. Also `--model sit` and `--model pixelflow`. |
| **Fine-grained bird — Child-FID / Validity ρ-sweep** (headline; `rho_fid_vs_child_fid.png`, `rho_fid_vs_validity.png`, `vis_{on,off}_{jit,dit}`) | `uv run python experiments/finegrained_bird_tfg.py --model jit-h-16 --guidance_mode dps --rho_override <ρ>`<br>`uv run python experiments/finegrained_bird_tfg.py --model dit --guidance_mode dps --rho_override <ρ>` | Sweep `<ρ>` (e.g. `0.1 0.3 0.5 ...`). JiT-H uses Heun + `--guidance_space x`; DiT uses DDPM (250 steps, x-space only). Also `--model sit` (Heun) and `--model pixelflow`. |
| **Fine-grained bird — CFG-only baseline** (zero-guidance reference point) | `uv run python experiments/finegrained_bird_no_guidance.py --model jit-h-16` | One run per model (`jit-h-16`, `dit`, `sit`, `pixelflow`). |
| **Inverse problems — Gaussian deblur** | `uv run python experiments/deblur_sr.py --task deblur --model jit-h-16 --guidance_mode dps --imagenet_dir <path/to/imagenet/val>` | Reports LPIPS / PSNR / SSIM. Swap `--model dit` etc. |
| **Inverse problems — 4× super-resolution** | `uv run python experiments/deblur_sr.py --task super_resolution --model jit-h-16 --guidance_mode dps --imagenet_dir <path/to/imagenet/val>` | Same metrics as deblur. |
| **CFG-only baseline — 4 models, FID/IS** (Table; JiT-H 1.86 · PixelFlow 1.98 · SiT 2.06 · DiT 2.27) | `uv run python experiments/imagenet_no_guidance.py --model jit-h-16` | One run per model. Defaults to 1000 classes × 50 images = 50K samples. |
| **Crossed-lines manifold ablation** (2D → 512D; `onmanifold_vs_dim`) | `uv run python spiral_test/generate_data.py --name crossed_lines`<br>`uv run python spiral_test/train.py --data crossed_lines`<br>`uv run python spiral_test/inference.py --exp crossed_lines_<YYYYMMDD_HHMMSS> -s 0.0 2.0 5.0 -n 50` | Toy x/ε/v ablation across dimensions D ∈ {2, 8, 32, 128, 512}; self-contained, trains tiny MLPs from scratch. See [`spiral_test/README.md`](spiral_test/README.md). |

Useful shared flags for the bird / inverse / CFG scripts: `--nfe` and `--sampling_method` (override the per-model defaults), `--images_per_class`, `--batch_size`, `--device`, `--seed`, `--skip_fid` (validity only), `--force_rerun`. Run any script with `--help` for the full list.

> **Style transfer (CLIP-Gram)** is reported in the paper (`style_gram_vs_validity.png`) but the style-transfer code is **not** part of this release.

### FID reference statistics

FID/Child-FID computation needs precomputed Inception reference statistics, which are **not committed** to the repository. Fetch them with:

```bash
uv run python scripts/download_fid_stats.py
```

This populates `src/jit_tfg/evaluation/generation/fid_stats/` (`imagenet/in256_stats.npz`, `finegrained/{parent,child}_fid_stats.npz`). The stats are hosted on the HuggingFace Hub dataset repo [`ManLuML/onmanifold-tfg-fid-stats`](https://huggingface.co/datasets/ManLuML/onmanifold-tfg-fid-stats) and fetched automatically by the script. Until you download the stats, runs without these files print a warning and report FID as `0.0`; validity and IS are unaffected.

---

## Repository Structure

```
on-manifold-tfg/
├── src/jit_tfg/                  # Python package (import name: jit_tfg)
│   ├── tfg/                      # UnifiedSampler + TFG guiders + config/calibration
│   │   ├── unified_sampler.py    # Unified x/ε/v sampling + guidance injection
│   │   └── guiders/              # base, finegrained, imagenet, inverse guiders
│   ├── models/                   # 4 model wrappers
│   │   ├── jit/                  # JiT  (x-prediction, pixel space)
│   │   ├── dit/                  # DiT  (ε-prediction, latent space + VAE)
│   │   ├── sit/                  # SiT  (v-prediction, latent space)
│   │   └── pixelflow/            # PixelFlow (v-prediction, pixel space)
│   └── evaluation/               # FID/IS + classification validity
│       ├── generation/           # Inception features, FID, IS, stats paths
│       └── guidance/             # ImageNet + fine-grained bird validity evaluators
├── experiments/                  # Paper experiment entry points
│   ├── finegrained_bird_tfg.py         # Bird Child-FID / Validity ρ-sweep (headline)
│   ├── finegrained_bird_no_guidance.py # Bird CFG-only baseline
│   ├── deblur_sr.py                    # Inverse problems: deblur + 4× SR
│   ├── imagenet_tfg.py                 # ImageNet-256 guided Child-FID / Validity ρ-sweep (headline)
│   ├── imagenet_no_guidance.py         # CFG-only FID/IS for 4 models
│   └── utils.py
├── spiral_test/                  # Crossed-lines 2D→512D toy manifold ablation
├── config/                       # Model YAML configs
│   ├── dit/dit_xl2_256.yaml
│   ├── sit/sit_xl2_256.yaml
│   └── pixelflow/pixelflow_xl_256.yaml
├── scripts/                      # download_fid_stats.py (FID reference stats)
├── tests/                        # Unit tests (sampler, models, schedules, …)
├── docs/                         # concepts/ · getting-started/ · evaluation/
├── pyproject.toml · uv.lock · Makefile · .pre-commit-config.yaml
└── LICENSE · CONTRIBUTING.md · THIRD_PARTY_LICENSES.md
```

> The experiment scripts also expect two small data files in `experiments/` — `model_configs.json` (per-model checkpoint paths, samplers, NFE, CFG scale) and `finegrained_bird_mapping.json` (fine-grained-species → ImageNet-parent class mapping). The deblur/SR script additionally needs a local ImageNet validation set, passed via `--imagenet_dir`.

---

## Citation

```bibtex
@inproceedings{lee2026onmanifold,
  title     = {Not All Prediction Targets Keep Training-Free Diffusion Guidance on the Manifold},
  author    = {Lee, Yunsung and Lee, Hyeongmin},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE) © Yunsung Lee.

It builds on several upstream projects with their own terms (JiT, DiT, SiT, PixelFlow, TFG, InverseBench, EDM). See [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for full attribution. **Note:** the DiT components derive from Meta's DiT, which is licensed **CC-BY-NC 4.0 (non-commercial)** — that restriction applies to the DiT-derived code (`src/jit_tfg/models/dit/`) regardless of the top-level MIT license.
