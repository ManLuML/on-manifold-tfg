# Third-Party Licenses and Attribution

`on-manifold-tfg` (import name `jit_tfg`) is released under the MIT License,
Copyright (c) 2025 Yunsung Lee (see [`LICENSE`](LICENSE)).

This project incorporates, adapts, or derives from a number of third-party
research codebases. Each retains its own license and copyright. The table below
lists every upstream component the released code is based on, how it is used
here, and the governing license.

> **NON-COMMERCIAL NOTICE.** Components derived from **facebookresearch/DiT**
> are licensed under **Creative Commons Attribution-NonCommercial 4.0
> International (CC-BY-NC 4.0)**. **The DiT-derived parts of this repository may
> be used for research and other non-commercial purposes only.** The MIT license
> on the rest of this project does **not** relax that restriction. If you need a
> fully permissive subset, avoid the `jit_tfg.models.dit` package and the DiT
> model path.

The full upstream clones are **not** shipped with this release (they are
git-ignored). Only our wrappers and the minimal vendored subsets noted below are
included. Where the original license requires it, the upstream copyright and
permission notices are preserved alongside the derived code.

---

## Vendored / derived components

| Component | Upstream | License | Copyright holder | How it is used here |
|-----------|----------|---------|------------------|---------------------|
| **JiT** | [LTH14/JiT](https://github.com/LTH14/JiT) | MIT | (c) 2025 Tianhong Li | JiT (pixel-space, x-prediction Transformer) model wrapper and denoiser in `src/jit_tfg/models/jit/` derived from LTH14/JiT. Pretrained checkpoints loaded at inference. |
| **DiT** | [facebookresearch/DiT](https://github.com/facebookresearch/DiT) | **CC-BY-NC 4.0 (NON-COMMERCIAL)** | (c) Meta Platforms, Inc. and affiliates | DiT (latent-space, ε-prediction / DDPM Transformer) model wrapper in `src/jit_tfg/models/dit/`, and the vendored DDPM diffusion utilities in `src/jit_tfg/models/dit/diffusion/` (schedules + DDIM/DDPM sampling), derived from facebookresearch/DiT. Pretrained checkpoints loaded at inference. **Non-commercial use only.** |
| **SiT** | [willisma/SiT](https://github.com/willisma/SiT) | MIT | (c) Meta Platforms, Inc. and affiliates | SiT (latent-space, v-prediction flow-matching Transformer) model wrapper in `src/jit_tfg/models/sit/`, and the vendored `transport` module (linear interpolation path + ODE/SDE integrators) in `src/jit_tfg/models/sit/transport/`, derived from willisma/SiT. Pretrained checkpoints loaded at inference. |
| **PixelFlow** | [ShoufaChen/PixelFlow](https://github.com/ShoufaChen/PixelFlow) | MIT | (c) 2025 Shoufa Chen | PixelFlow (pixel-space, multi-stage flow Transformer) model wrapper in `src/jit_tfg/models/pixelflow/` derived from ShoufaChen/PixelFlow. Pretrained checkpoints loaded at inference. |
| **TFG (Training-Free Guidance)** | [YWolfeee/Training-Free-Guidance](https://github.com/YWolfeee/Training-Free-Guidance) | MIT (upstream `LICENSE.md`) | upstream authors | Our TFG guidance implementation (`src/jit_tfg/tfg/`) follows the algorithm and several conventions of the original TFG codebase, including the gradient-rescaling behavior referenced from `edm/tasks/utils.py` in the upstream tree (see code comments in `src/jit_tfg/tfg/utils.py` and `config.py`). No upstream source is vendored verbatim. |
| **InverseBench** | [devzhk/InverseBench](https://github.com/devzhk/InverseBench) | MIT | (c) 2025 Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu, Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki, Zachary E. Ross, Katherine L. Bouman, Yisong Yue | Inverse-problem (Gaussian deblur, 4× super-resolution) evaluation protocol, degradation operators, and DPS-style measurement guidance in `experiments/deblur_sr.py` informed by the InverseBench benchmark conventions. No upstream source is vendored verbatim. |
| **EDM** | [NVlabs/edm](https://github.com/NVlabs/edm) | **CC-BY-NC-SA 4.0 (NON-COMMERCIAL)** (upstream `LICENSE.txt`) | (c) 2022 NVIDIA Corporation | Not vendored. Referenced only as the provenance of the gradient-rescaling logic inherited via the TFG codebase (`edm/tasks/utils.py`); see code comments in `src/jit_tfg/tfg/utils.py` and `config.py`. No EDM source code is included in this release. |

## Additional upstreams consulted (not vendored verbatim)

| Component | Upstream | License | Copyright holder | How it is used here |
|-----------|----------|---------|------------------|---------------------|
| **Flow Guidance** | "On the Guidance of Flow Matching" (Feng et al., ICML 2025) | MIT (upstream `LICENSE` names no copyright holder) | upstream authors | The v-space / "flow guidance" guidance-space heritage in `src/jit_tfg/tfg/` (velocity-modification scaling `λ_t = (1-t)/t`) follows the Flow Guidance formulation. No upstream source is vendored verbatim. |
| **Diffusion Conditional Sampling** | [Diffusion Conditional Sampling](https://diffusion-conditional-sampling.github.io) (Patsenker, Li et al., AISTATS 2026) | MIT (stated in upstream README) | Jonathan Patsenker, Henry Li, Myeongseob Ko, Ruoxi Jia, Yuval Kluger | Inverse-problem conditional-sampling concepts consulted for the deblur / super-resolution experiments (`experiments/deblur_sr.py`). No upstream source is vendored verbatim. |

## Evaluation dependencies

| Component | Upstream | License | How it is used here |
|-----------|----------|---------|---------------------|
| **torch-fidelity** | [toshas/torch-fidelity](https://github.com/toshas/torch-fidelity) | Apache-2.0 (upstream `LICENSE.md`) | Inception-v3 feature extraction in `src/jit_tfg/evaluation/generation/inception.py` uses torch-fidelity's Inception-v3 weights/implementation for exact compatibility with the pre-computed FID/IS reference statistics. Pulled in as a normal Python dependency (see `pyproject.toml`); not vendored. Note: `pyproject.toml` pins the LTH14 fork of torch-fidelity. |

## Notes

- **Pretrained models.** The four main models (JiT, DiT, SiT, PixelFlow) are
  evaluated using their authors' **pretrained checkpoints**. Those checkpoints
  are distributed by the respective upstream projects under the upstream
  licenses, not by this repository. Downloading and using a checkpoint subjects
  you to that model's license (notably, **DiT checkpoints are non-commercial**).
- **FID/IS reference statistics** are not committed to this repository; the
  helper `scripts/download_fid_stats.py` fetches them from the HuggingFace
  dataset `ManLuML/onmanifold-tfg-fid-stats`.
- The upstream licenses above were confirmed against each project's published
  `LICENSE` file at release time. As always, re-confirm the current upstream
  license before redistributing any derived component.

If you believe an attribution here is incomplete or incorrect, please open an
issue so it can be fixed.
