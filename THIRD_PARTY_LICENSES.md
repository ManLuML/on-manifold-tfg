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
included. Where the original license requires it (e.g. the MIT license's
condition that "the above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software"), the upstream
copyright and permission notices are reproduced verbatim in the
[Full license texts and copyright notices](#full-license-texts-and-copyright-notices)
section at the end of this document.

---

## Vendored / derived components

| Component | Upstream | License | Copyright holder | How it is used here |
|-----------|----------|---------|------------------|---------------------|
| **JiT** | [LTH14/JiT](https://github.com/LTH14/JiT) | MIT | (c) 2025 Tianhong Li | JiT (pixel-space, x-prediction Transformer) model wrapper and denoiser in `src/jit_tfg/models/jit/` derived from LTH14/JiT. Pretrained checkpoints loaded at inference. |
| **DiT** | [facebookresearch/DiT](https://github.com/facebookresearch/DiT) | **CC-BY-NC 4.0 (NON-COMMERCIAL)** | (c) Meta Platforms, Inc. and affiliates | DiT (latent-space, ε-prediction / DDPM Transformer) model wrapper in `src/jit_tfg/models/dit/`, and the vendored DDPM diffusion utilities in `src/jit_tfg/models/dit/diffusion/` (schedules + DDIM/DDPM sampling), derived from facebookresearch/DiT. Pretrained checkpoints loaded at inference. **Non-commercial use only.** |
| **SiT** | [willisma/SiT](https://github.com/willisma/SiT) | MIT | (c) Meta Platforms, Inc. and affiliates | SiT (latent-space, v-prediction flow-matching Transformer) model wrapper in `src/jit_tfg/models/sit/`, and the vendored `transport` module (linear interpolation path + ODE/SDE integrators) in `src/jit_tfg/models/sit/transport/`, derived from willisma/SiT. Pretrained checkpoints loaded at inference. |
| **PixelFlow** | [ShoufaChen/PixelFlow](https://github.com/ShoufaChen/PixelFlow) | MIT | (c) 2025 Shoufa Chen | PixelFlow (pixel-space, multi-stage flow Transformer) model wrapper in `src/jit_tfg/models/pixelflow/` derived from ShoufaChen/PixelFlow. Pretrained checkpoints loaded at inference. |
| **TFG (Training-Free Guidance)** | [YWolfeee/Training-Free-Guidance](https://github.com/YWolfeee/Training-Free-Guidance) | MIT (declared in upstream README; **no license file is published upstream** — see the [full-texts section](#tfg--ywolfeeetraining-free-guidance-mit-declared-no-license-file-published-upstream) below) | upstream authors (Ye et al., NeurIPS 2024) | Our TFG guidance implementation (`src/jit_tfg/tfg/`) follows the algorithm and several conventions of the original TFG codebase, including the inverse-problem guiders and degradation operators in `src/jit_tfg/tfg/guiders/inverse.py` (which follow the upstream `gaussian_deblur.py` / `super_resolution.py` tasks and `image_inverse_operator.py`) and the gradient-rescaling behavior referenced from `edm/tasks/utils.py` in the upstream tree (see code comments in `src/jit_tfg/tfg/utils.py` and `config.py`). No upstream source is vendored verbatim. |
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
  `LICENSE` file (except TFG, whose repository declares MIT in its README but
  publishes no license file — see below). As always, re-confirm the current
  upstream license before redistributing any derived component.

---

## Full license texts and copyright notices

This section preserves, verbatim, the upstream copyright notices and license
texts for the projects this repository derives code from, as required by the
MIT license condition that "the above copyright notice and this permission
notice shall be included in all copies or substantial portions of the
Software." Texts were copied from the upstream repositories' license files on
2026-07-03; source links are given per component.

### JiT — LTH14/JiT (MIT)

Applies to code derived from JiT in `src/jit_tfg/models/jit/`.
Source: <https://github.com/LTH14/JiT/blob/main/LICENSE>

```text
MIT License

Copyright (c) 2025 Tianhong Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### SiT — willisma/SiT (MIT)

Applies to code derived from SiT in `src/jit_tfg/models/sit/`, including the
vendored `transport` module in `src/jit_tfg/models/sit/transport/`.
Source: <https://github.com/willisma/SiT/blob/main/LICENSE.txt>

```text
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### PixelFlow — ShoufaChen/PixelFlow (MIT)

Applies to code derived from PixelFlow in `src/jit_tfg/models/pixelflow/`.
Source: <https://github.com/ShoufaChen/PixelFlow/blob/main/LICENSE>

```text
MIT License

Copyright (c) 2025 Shoufa Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### InverseBench — devzhk/InverseBench (MIT)

Applies to the inverse-problem evaluation protocol and conventions followed in
`experiments/deblur_sr.py`. (The square brackets in the copyright line below
appear verbatim in the upstream license file.)
Source: <https://github.com/devzhk/InverseBench/blob/main/LICENSE>

```text
MIT License

Copyright (c) 2025 [Hongkai Zheng, Wenda Chu, Bingliang Zhang, Zihui Wu, Austin Wang, Berthy T. Feng, Caifeng Zou, Yu Sun, Nikola Kovachki, Zachary E. Ross, Katherine L. Bouman, Yisong Yue]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### DiT — facebookresearch/DiT (CC-BY-NC 4.0, NON-COMMERCIAL)

Applies to code derived from DiT in `src/jit_tfg/models/dit/`, including the
vendored DDPM diffusion utilities in `src/jit_tfg/models/dit/diffusion/`.

The upstream `LICENSE.txt`
(<https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt>) is the full
Creative Commons Attribution-NonCommercial 4.0 International legalcode and
contains no copyright line of its own. Meta's copyright notice appears in the
upstream source file headers and is reproduced verbatim here:

```text
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
```

License: **Creative Commons Attribution-NonCommercial 4.0 International
(CC-BY-NC 4.0)**. Under this license you may share and adapt the material for
**non-commercial purposes only**, provided you give appropriate credit,
indicate if changes were made, and do not apply additional legal or
technological restrictions. The license text is long and is therefore
incorporated here by reference rather than pasted in full; the authoritative
legalcode is available at
<https://creativecommons.org/licenses/by-nc/4.0/legalcode> and is mirrored
verbatim in the upstream repository at
<https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt>.

Note on provenance: DiT's own `diffusion/` utilities are, per the upstream
headers, "Modified from OpenAI's diffusion repos" (GLIDE, ADM/guided-diffusion,
IDDPM), which are MIT-licensed, `Copyright (c) 2021 OpenAI`
(<https://github.com/openai/guided-diffusion/blob/main/LICENSE>).

### TFG — YWolfeee/Training-Free-Guidance (MIT declared; no license file published upstream)

Applies to the TFG guidance implementation in `src/jit_tfg/tfg/` (algorithm and
conventions followed; includes `src/jit_tfg/tfg/guiders/inverse.py`). No
upstream source is vendored verbatim.

As of 2026-07-03, the upstream repository's README states "MIT. Check
`LICENSE.md`." and carries an MIT license badge, **but no `LICENSE.md` (or any
other license file) is actually present in the repository**, GitHub's license
detection reports no license, and the upstream source files carry no copyright
headers. There is therefore no upstream copyright notice or license text that
can be reproduced verbatim here. We rely on the README's express MIT
declaration (<https://github.com/YWolfeee/Training-Free-Guidance#-license>) and
attribute the work to its authors: Copyright (c) the TFG authors (Haotian Ye et
al., "TFG: Unified Training-Free Guidance for Diffusion Models," NeurIPS 2024).
If upstream publishes its license file, its exact text should be added here.

### EDM — NVlabs/edm (CC-BY-NC-SA 4.0, NON-COMMERCIAL) — not vendored

No EDM source code is included in this release; EDM is listed as the provenance
of the gradient-rescaling logic inherited by way of the TFG codebase. For
attribution completeness, the upstream `LICENSE.txt`
(<https://github.com/NVlabs/edm/blob/main/LICENSE.txt>) begins with the
following copyright notice, reproduced verbatim:

```text
Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Attribution-NonCommercial-ShareAlike 4.0 International
```

License: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International (CC-BY-NC-SA 4.0)**; the authoritative legalcode is available at
<https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>.

### Other components

The "Additional upstreams consulted" components (Flow Guidance, Diffusion
Conditional Sampling) were consulted at the level of algorithms and
formulations; no upstream source is vendored verbatim, and their licensing is
recorded in the table above. `torch-fidelity` (Apache-2.0) is consumed as an
ordinary Python package dependency, not vendored; its license text ships inside
the installed package.

If you believe an attribution here is incomplete or incorrect, please open an
issue so it can be fixed.
