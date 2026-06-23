# Checkpoints & Reference Statistics

This release ships **evaluation/inference code only** — no model weights are
committed to the repository. This document explains how to obtain every
pretrained checkpoint and auxiliary classifier used by the paper
*Not All Prediction Targets Keep Training-Free Diffusion Guidance on the
Manifold* (ECCV 2026).

> **We never commit weights.** All paths below are gitignored. The repository
> only tracks `.gitkeep` placeholders inside the target directories.

Two helper scripts automate most of this:

```bash
# Diffusion model checkpoints -> checkpoints/
uv run python scripts/download_checkpoints.py --all

# FID reference statistics -> src/jit_tfg/evaluation/generation/fid_stats/
uv run python scripts/download_fid_stats.py
```

The guidance/evaluation **classifiers** are pulled automatically from the
HuggingFace Hub the first time you run an experiment (they are cached under
`~/.cache/huggingface`), so they usually need no manual step — see
[Classifiers](#3-guidanceevaluation-classifiers) for the exact model IDs and
how to pre-fetch them.

---

## 1. Diffusion model checkpoints (4 pretrained models)

Target directory: `checkpoints/` (gitignored). Approximate disk sizes are for
the FP32 weights as published; expect some variation.

| Model | Target | Space | Params | Source | Target path | Approx. size |
|-------|--------|-------|--------|--------|-------------|--------------|
| **JiT-H/16** | x | Pixel | 953M | TODO — see note below | `checkpoints/jit/jit-h-16.pth` | ~3.7 GB |
| **DiT-XL/2** | ε | Latent | 675M (+49M VAE) | `facebook/DiT-XL-2-256` (HF) / Meta URL | `checkpoints/dit/DiT-XL-2-256x256.pt` | ~2.7 GB |
| **SiT-XL/2** | v | Latent | 675M (+49M VAE) | `nyu-visionx/SiT-collections` (HF) | `checkpoints/sit/` | ~2.7 GB |
| **PixelFlow** | v | Pixel | 677M | `ShoufaChen/PixelFlow-Class2Image` (HF) | `checkpoints/pixelflow/` | ~2.7 GB |

The latent models (DiT, SiT) additionally need the Stable Diffusion VAE
decoder. It is fetched automatically from HuggingFace (`stabilityai/sd-vae-ft-ema`,
~335 MB) by the model wrappers at load time; no manual download is required.

### JiT-H/16 (x-prediction) — manual step required

JiT-H/16 is the primary x-prediction model (from *Back to Basics: Let Denoising
Generative Models Denoise*, Li & He 2025). At release time we do not have a
verified public download URL encoded in the code, so the download script uses a
**clearly-marked placeholder**:

```python
# scripts/download_checkpoints.py
JIT_H16_URL = "TODO-set-jit-h16-url"  # FIXME: official JiT-H/16 checkpoint URL
```

Until that URL is set, obtain the checkpoint from the official *JiT* release and
place it at `checkpoints/jit/jit-h-16.pth`. The loader infers the model variant
from the filename (`jit-h-16.pth` -> `JiT-H/16`), so keep that exact name. The
loader reads weights with `torch.load(..., weights_only=True)`.

### DiT-XL/2 (ε-prediction)

Two equivalent sources; the download script prefers HuggingFace and falls back
to the Meta URL:

- HuggingFace: `facebook/DiT-XL-2-256`
- Meta direct URL: `https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt`

`config/dit/dit_xl2_256.yaml` points at `facebook/DiT-XL-2-256` via
`from_pretrained`; set a local `checkpoint:` key there to use the file instead.

### SiT-XL/2 (v-prediction)

```bash
huggingface-cli download nyu-visionx/SiT-collections --local-dir checkpoints/sit
```

(The download script wraps this.) SiT checkpoints share DiT's format
(`ema` / `model` key) and JiT's time convention, so no conversion is needed.

### PixelFlow (v-prediction, pixel-space)

```bash
huggingface-cli download ShoufaChen/PixelFlow-Class2Image --local-dir checkpoints/pixelflow
```

PixelFlow operates directly in pixel space (no VAE). Recall that the `nfe` in
the config is **per-stage** (total NFE = nfe × 4 stages).

---

## 2. FID reference statistics

Target directory: `src/jit_tfg/evaluation/generation/fid_stats/` (the `.npz`
files there are gitignored). The evaluation code resolves these exact paths:

| Statistic | Used for | Target path | Approx. size |
|-----------|----------|-------------|--------------|
| ImageNet 256 | Parent-FID (P-FID), CFG-only FID | `fid_stats/imagenet/in256_stats.npz` | ~25 MB |
| Bird parent | P-FID against ImageNet bird parents | `fid_stats/finegrained/parent_fid_stats.npz` | ~10 MB |
| Bird child | **Child-FID (C-FID)** — the headline metric | `fid_stats/finegrained/child_fid_stats.npz` | ~10 MB |

Download with:

```bash
uv run python scripts/download_fid_stats.py
```

> **Hosting is deferred.** The HuggingFace dataset that will host these `.npz`
> files is not yet published. `scripts/download_fid_stats.py` defines
> `HF_FID_STATS_REPO = "TODO-ManLuML/<dataset>"` and prints a clear message
> instead of downloading until that constant is set to the real repo ID.

These statistics are Inception-V3 (pool3, 2048-dim) mean/covariance pairs
(`mu`, `sigma`) — the standard FID reference format. You can regenerate them
from the corresponding image sets if needed; see `docs/evaluation/`.

---

## 3. Guidance/evaluation classifiers

These HuggingFace classifiers are loaded on demand by the experiment scripts and
cached under `~/.cache/huggingface`. They are small (tens to hundreds of MB) and
need no manual placement, but you can pre-fetch them to run offline.

| Role | Model ID | Used by |
|------|----------|---------|
| ImageNet **validity** eval | `facebook/deit-small-patch16-224` (DeiT-Small) | `imagenet_no_guidance.py`, `finegrained_bird_no_guidance.py` |
| ImageNet guidance (DPS) | `google/vit-base-patch16-224` (ViT-B/16) | TFG/DPS on ImageNet classes |
| Fine-grained bird **guidance** (DPS) | `dennisjooo/Birds-Classifier-EfficientNetB2` | `finegrained_bird_tfg.py` |
| Fine-grained bird **validity** eval | `chriamue/bird-species-classifier` | `finegrained_bird_tfg.py` |

Separate models are used for guidance vs. evaluation to avoid circular
evaluation. Pre-fetch all four:

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
for repo in [
    "facebook/deit-small-patch16-224",
    "google/vit-base-patch16-224",
    "dennisjooo/Birds-Classifier-EfficientNetB2",
    "chriamue/bird-species-classifier",
]:
    print("fetching", repo)
    snapshot_download(repo)
PY
```

---

## 4. Directory layout after download

```
checkpoints/                       # gitignored (weights never committed)
├── jit/jit-h-16.pth               # JiT-H/16  (x)   — manual until URL is set
├── dit/DiT-XL-2-256x256.pt        # DiT-XL/2  (ε)
├── sit/...                        # SiT-XL/2  (v)
└── pixelflow/...                  # PixelFlow (v)

src/jit_tfg/evaluation/generation/fid_stats/   # .npz gitignored
├── imagenet/in256_stats.npz
└── finegrained/
    ├── parent_fid_stats.npz
    └── child_fid_stats.npz
```

---

## 5. Licensing of the upstream weights

The weights are governed by their **original** licenses, not this repo's MIT
license. In particular **DiT (Meta) is released under CC-BY-NC-4.0
(non-commercial)**. JiT, SiT, and PixelFlow are MIT-licensed. Review and comply
with each upstream license before use. See the repository `LICENSE` and the
attribution notes in the source tree.
