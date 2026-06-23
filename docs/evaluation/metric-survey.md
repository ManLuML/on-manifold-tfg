# Evaluation Metric Literature Survey

A comprehensive survey of generative model evaluation metrics, with focus on manifold-aware metrics for guided diffusion. This document supports metric choices in our ECCV 2026 paper on prediction target selection for Train-Free Guidance.

For the practical overview and recommendations, see [Manifold-Aware Evaluation Metrics](manifold-metrics.md).

---

## 1. Foundational Metrics

### 1.1 FID (Heusel et al., NeurIPS 2017)

Fréchet Inception Distance measures the distance between real and generated distributions in Inception-v3 feature space, assuming both are Gaussian:

\\[
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})
\\]

- **Standard**: The most widely reported metric for generative models
- **Limitations**: Conflates fidelity and diversity into one number; assumes Gaussian features; Inception-v3 bias (see Section 4)
- **Sample requirement**: \\(N \geq 50{,}000\\) for stable estimates; unstable below 10,000

### 1.2 IS (Salimans et al., NeurIPS 2016)

Inception Score measures quality (sharp class predictions) and diversity (uniform marginal):

\\[
\text{IS} = \exp\left(\mathbb{E}_x \left[ D_{\text{KL}}(p(y|x) \| p(y)) \right]\right)
\\]

- Higher IS = more confident per-image predictions with uniform class coverage
- **Limitation**: Only measures ImageNet class consistency; insensitive to intra-class variation

### 1.3 Original Precision & Recall (Sajjadi et al., NeurIPS 2018)

First formalization of separate fidelity and diversity axes for generative models:

- **Precision** = fraction of generated samples within the support of real data distribution
- **Recall** = fraction of real samples covered by the generated distribution
- Constructs PRD (Precision-Recall Distribution) curves by clustering embeddings with mini-batch k-means (k=20) in Inception Pool3 feature space, then comparing cluster assignment histograms

**Limitations**: Relies on explicit density estimation in high-dimensional space; computationally expensive; superseded by k-NN approaches.

---

## 2. Improved Precision & Recall (IPR)

**Kynkäänniemi, Karras, Laine, Lehtinen, Aila. NeurIPS 2019.** ([arXiv:1904.06991](https://arxiv.org/abs/1904.06991))

800+ citations. The de facto standard for disentangled fidelity/diversity evaluation.

### Algorithm

1. Embed real samples \\(\{x_i\}\\) and generated samples \\(\{y_j\}\\) via a pretrained feature extractor
2. For each real sample \\(x_i\\): form hypersphere \\(B(x_i, r_i)\\) with radius = distance to \\(k\\)-th nearest real neighbor
3. For each generated sample \\(y_j\\): form hypersphere \\(B(y_j, r_j)\\) with radius = distance to \\(k\\)-th nearest generated neighbor
4. **Precision** = fraction of \\(y_j\\) inside any real hypersphere
5. **Recall** = fraction of \\(x_i\\) inside any generated hypersphere

### Parameter Sensitivity

| k value | Effect | Recommendation |
|---------|--------|----------------|
| k=1 | Most sensitive to outliers | Avoid |
| k=3 | **Recommended** (original paper) | Standard |
| k=5 | Slightly more robust | Also acceptable |
| k>10 | Over-smoothing, saturates at 1.0 | Avoid |

Results are monotonically increasing in k; the choice of k is far less impactful than the choice of feature extractor.

### Known Limitations

1. **Outlier sensitivity**: A single real-data outlier creates a large hypersphere, potentially encompassing many generated samples and inflating Precision. Naeem et al. (2020) showed Precision can jump by ~0.9 with one outlier.
2. **Binary membership**: No gradation between "barely inside" and "deep inside" the manifold.
3. **Constant-density assumption**: Treats the interior of each hypersphere as uniform.
4. **Hubness** (Radovanovic et al., JMLR 2010): In high-dimensional spaces, some points become "hubs" --- appearing disproportionately as nearest neighbors of many query points, inflating effective coverage of certain real-data hyperspheres and biasing Precision upward in hub-adjacent regions.

**Code**: [github.com/kynkaat/improved-precision-and-recall-metric](https://github.com/kynkaat/improved-precision-and-recall-metric)

---

## 3. Density & Coverage (PRDC)

**Naeem, Oh, Uh, Choi, Yoo. ICML 2020.** ([arXiv:2002.09797](https://arxiv.org/abs/2002.09797))

Addresses the key weaknesses of IPR with two improved metrics.

### How PRDC Improves on IPR

| IPR Problem | PRDC Fix |
|-------------|----------|
| Precision inflated by outlier with large hypersphere | **Density** counts how many real-neighborhood spheres contain each fake sample (normalized by k), not just binary in/out |
| Recall inflated by unrealistic fake samples overestimating the generated manifold | **Coverage** builds neighborhoods around real samples only, measuring what fraction has at least one generated neighbor |
| Recall expensive (k-NN over fake varies per model) | Coverage uses real-sample manifold only (computed once, reused) |

### Formal Definitions

**Density** (continuous fidelity):

\\[
\text{Density} = \frac{1}{kM} \sum_{j=1}^{M} \sum_{i=1}^{N} \mathbb{1}\left[ y_j \in B(x_i, r_i) \right]
\\]

where \\(r_i\\) is the distance from \\(x_i\\) to its \\(k\\)-th nearest real neighbor (same radius used in Precision).

**Coverage** (robust diversity):

\\[
\text{Coverage} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[ \exists\, j : y_j \in B(x_i, r_i) \right]
\\]

- Default k=5
- Density is unbounded (can exceed 1); values around 1 indicate matching the real density
- Both are computed from the same k-NN structure, adding negligible overhead to Precision/Recall computation

**Code**: [github.com/clovaai/generative-evaluation-prdc](https://github.com/clovaai/generative-evaluation-prdc)

---

## 4. Feature Space Research

### 4.1 Exposing Flaws of Inception-Based Metrics

**Stein et al. NeurIPS 2023.** ([arXiv:2306.04675](https://arxiv.org/abs/2306.04675))

The most comprehensive evaluation study to date: 9 feature extractors, 17 metrics, 41 generative models, 4 datasets, plus human evaluation.

**Key findings**:

1. **No existing metric strongly correlates with human judgment** when using Inception-v3 features
2. **Inception-v3 systematically favors GANs** over diffusion models due to architectural/training similarities
3. **DINOv2 ViT-L/14 is the recommended replacement**: Self-supervised features align significantly better with human perceptual quality ratings
4. **Practical recommendations**: DINOv2-ViT-L/14 for final metrics; DINOv2-ViT-B/14 for training-time monitoring (4x more efficient)

**Library**: `dgm-eval` ([github.com/layer6ai-labs/dgm-eval](https://github.com/layer6ai-labs/dgm-eval)) --- 17 metrics, 9 encoders

### 4.2 Current Consensus (2024--2025)

DINOv2 is rapidly becoming the standard for evaluation:

- **GFCG (Shenoy, 2024)**: Reports FD-DINOv2 alongside FID, demonstrates DINOv2's superior stability to JPEG artifacts
- **Calibration + Regularization (2025)**: Reports Precision and Recall in DINOv2 space
- **torchmetrics**: Feature request for native FD-DINOv2 support (May 2025), indicating widespread adoption
- **However**: Many 2024 papers still report FID with Inception for backward comparability

**Best practice**: Report both FID (Inception, for comparability) and FD-DINOv2 (for reliability). Compute PRDC metrics in DINOv2 space.

---

## 5. Distribution Distance Alternatives

### 5.1 FD-DINOv2

Fréchet Distance with DINOv2 features. Same mathematical formula as FID, different feature extractor.

- Better human judgment alignment (Stein et al., NeurIPS 2023)
- More robust to compression artifacts
- **Implementation**: [github.com/justin4ai/FD-DINOv2](https://github.com/justin4ai/FD-DINOv2) or via `dgm-eval`

### 5.2 CMMD (Jayasumana et al., CVPR 2024)

**CLIP Maximum Mean Discrepancy.** ([arXiv:2401.09603](https://arxiv.org/abs/2401.09603))

Squared MMD with Gaussian RBF kernel between CLIP-ViT-L/14 embeddings:

\\[
\text{CMMD}^2 = \mathbb{E}[k(x, x')] + \mathbb{E}[k(y, y')] - 2\mathbb{E}[k(x, y)]
\\]

where \\(k\\) is a Gaussian kernel.

**Advantages over FID**:

| Property | FID | CMMD |
|----------|-----|------|
| Distribution assumption | Gaussian | None |
| Sample efficiency | ~50K needed | ~300 sufficient |
| Bias | Biased estimator | Unbiased |
| Monotonicity | Can decrease misleadingly | Monotonic with degradation |

**Implementation**: [github.com/sayakpaul/cmmd-pytorch](https://github.com/sayakpaul/cmmd-pytorch)

### 5.3 FWD --- Fréchet Wavelet Distance (ICLR 2025)

**Veeramacheneni et al.** ([arXiv:2312.15289](https://arxiv.org/abs/2312.15289))

Projects images into wavelet packet coefficients (Haar wavelet, level 4) and computes Fréchet distance per packet.

- **Domain-agnostic**: No pretrained neural network required
- Captures spatial and frequency information
- Per-packet interpretability
- **Implementation**: [github.com/BonnBytes/PyTorch-FWD](https://github.com/BonnBytes/PyTorch-FWD)

### 5.4 FLD+ (2024)

**Jeevan, Nixon, Sethi. arXiv 2024.** ([arXiv:2411.15584](https://arxiv.org/abs/2411.15584))

Normalizing flow-based metric achieving stable results with **fewer than 300 images**.

- Particularly useful for fine-grained evaluation where per-species reference sets are small

---

## 6. Advanced Precision & Recall Variants

### 6.1 PP&PR --- Probabilistic Precision and Recall

**Park & Kim. ICCV 2023.** ([arXiv:2309.01590](https://arxiv.org/abs/2309.01590))

Replaces binary in/out-of-hypersphere with probabilistic scores via kernel density estimation. Assigns different scores to different samples, addressing the constant-density assumption.

**Code**: [github.com/kdst-team/Probablistic_precision_recall](https://github.com/kdst-team/Probablistic_precision_recall) (repository name sic)

### 6.2 Clipped Density & Coverage

**Salvy, Talbot, Thirion. ICLR 2026.** ([arXiv:2507.01761](https://arxiv.org/abs/2507.01761))

Clips individual sample contributions and k-NN ball radii to prevent outlier-driven bias.

- **Key property**: **Linear score degradation** as proportion of bad samples increases --- scores are directly interpretable as equivalent proportion of good samples
- Uses DINOv2 ViT-L/14, following Stein et al. (2023)
- Addresses the ICML 2025 position paper's criticism that no metric provides absolute (non-relative) evaluation

### 6.3 GICDM (Salvy et al., 2026)

**Generative Iterative Contextual Dissimilarity Measure.** ([arXiv:2602.16449](https://arxiv.org/abs/2602.16449))

Applies Iterative Contextual Dissimilarity Measure (ICDM) to correct hubness-distorted k-NN neighborhoods before computing any distance-based metric. Resolves hubness-induced failures in high-dimensional spaces.

!!! note
    Very recent (February 2026). Not yet widely adopted but addresses a known theoretical weakness of all k-NN-based metrics.

### 6.4 Unified P&R Framework

**Sykes, Simon, Rabin. arXiv 2024.** ([arXiv:2405.01611](https://arxiv.org/abs/2405.01611))

Unifies Sajjadi (2018), Kynkäänniemi (2019), Naeem (2020), and Park & Kim (2023) under a common framework related to divergence frontiers. Shows all existing P&R variants are special cases of a construction based on \\(\alpha\\)-divergences.

---

## 7. Evaluation in Guided Diffusion Generation

### 7.1 TFG Benchmark Protocol

**Ye et al. NeurIPS 2024 Spotlight.** ([arXiv:2409.15761](https://arxiv.org/abs/2409.15761))

The TFG benchmark evaluates on two axes:

- **Guidance Validity**: Top-1 classifier accuracy on generated samples
- **Generation Fidelity**: FID score

Notably, TFG does **not** report Precision, Recall, Density, or Coverage. This leaves a significant gap: FID conflates fidelity and diversity, and accuracy does not measure manifold proximity.

### 7.2 The Adversarial Gradient Problem

**Shen et al. NeurIPS 2024.** ([NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/c4edc5113b4ffd4632718558fb66b9ef-Paper-Conference.pdf))

Training-free guidance is susceptible to **adversarial gradients**: minimal input perturbations causing disproportionate gradient direction changes. Key findings:

1. Training-free guidance does **not** approximate the exact conditional score
2. Resilience characterized by the Lipschitz constant of the guidance network
3. Off-the-shelf (time-independent) classifiers have worse Lipschitz properties than noise-conditional classifiers
4. Proposed mitigations: Gaussian perturbations for gradient smoothing, Polyak step size scheduling

**Implication for evaluation**: Adversarial gradients can produce samples that fool classifiers (high validity) while moving off-manifold (low Precision). This makes Precision/Density especially important for TFG evaluation.

### 7.3 Recent Guided Generation Papers --- Metric Choices

| Paper | Venue | Year | Metrics Used | Notable |
|-------|-------|------|-------------|---------|
| GFCG ([Shenoy et al.](https://arxiv.org/abs/2411.15393)) | arXiv | 2024 | FD-DINOv2, Precision, Recall | Among the first guided generation papers to report FD-DINOv2 and Precision |
| Calibration + Regularization ([Javid et al.](https://arxiv.org/abs/2511.05844)) | NeurIPS Workshop | 2025 | Precision, Recall, FID | Shows Hellinger guidance: P=0.80, R=0.58 |
| CFG++ | ICLR | 2025 | FID, CLIP-similarity | Identifies off-manifold phenomenon in CFG |
| APG ([Sadat et al.](https://arxiv.org/abs/2410.02416)) | ICLR | 2025 | FID, Precision, Recall, **Saturation score** | Novel saturation metric for color artifacts |
| TAG ([Cho et al.](https://arxiv.org/abs/2510.04533)) | arXiv | 2025 | FID, IS, CLIP | Tangential guidance to suppress off-manifold drift |
| MPGD ([He et al.](https://arxiv.org/abs/2311.16424)) | ICLR | 2024 | LPIPS, SSIM, PSNR, Style Score | Manifold-preserving approach |
| Flow Guidance | ICML | 2025 | FID, task-specific | First theoretically principled guidance for flow matching |

### 7.4 Open Gap

No widely adopted, unified metric suite has been specifically designed for evaluating guided generation. While individual papers (TAG, CFG++, MPGD) address manifold-proximity concerns, the field still largely applies unconditional metrics (PRDC) to conditional samples plus task-specific accuracy. A standardized evaluation protocol jointly measuring "on-manifold quality" and "guidance adherence" remains an open research direction.

---

## 8. Per-Sample Quality Metrics

### 8.1 Realism Score

Per-sample variant of Precision from Kynkäänniemi et al. (2019). For each generated sample, computes the maximum ratio of real-sample k-NN radius to distance:

\\[
\text{Realism}(y_j) = \max_{i \in \text{NN}_k(y_j)} \frac{r_i}{\|y_j - x_i\|_2}
\\]

- Score \\(\geq\\) 1.0: on-manifold; Score \\(<\\) 1.0: off-manifold
- Enables per-image analysis and failure mode diagnosis
- Not available in the `prdc` package, but available in the original Kynkäänniemi et al. [repository](https://github.com/kynkaat/improved-precision-and-recall-metric); alternatively, a straightforward custom implementation using k-NN distances

### 8.2 VAE Reconstruction Error / AEROBLADE (CVPR 2024)

**Ricker et al.** ([arXiv:2401.17879](https://arxiv.org/abs/2401.17879))

Training-free detection of AI-generated images using autoencoder reconstruction error:

- Generated images are more accurately reconstructed by the AE than real images
- Uses **LPIPS** as reconstruction distance metric
- Achieves mean AP of 0.992 across Stable Diffusion, Kandinsky, Midjourney
- Provides per-pixel reconstruction error maps

**Application to TFG evaluation**: For latent-space models (DiT, SiT), compute VAE reconstruction error (LPIPS) on guided samples. Lower error = closer to VAE's learned manifold. Compare across prediction targets under same guidance strength.

**Code**: [github.com/jonasricker/aeroblade](https://github.com/jonasricker/aeroblade)

### 8.3 FIRE (CVPR 2025)

Frequency-guided reconstruction error analysis. Mid-frequency components are hardest for diffusion models to reconstruct, providing diagnostic information beyond global LPIPS.

### 8.4 Geometric Perspective on Reconstruction-Based Detection (2025)

([arXiv:2510.25141](https://arxiv.org/html/2510.25141))

Theoretical grounding: *"When an image lies off the learned data manifold, its projection induces a residual in the normal space, leading to a non-trivial lower bound governed by the decoder's singular values."*

This provides mathematical justification for using reconstruction error as a manifold distance proxy.

---

## 9. Fine-Grained Domain Considerations

### Small Reference Set Challenge

Fine-grained evaluation (bird species, car models) typically has few reference images per category:

| Dataset | Classes | Images/class | Total |
|---------|---------|-------------|-------|
| CUB-200-2011 | 200 | ~60 | 11,788 |
| Our bird mapping | 143 | variable | ~9,000--14,000 |

FID requires ~20,000+ images per distribution for stability. Per-class FID with ~60 images is unreliable.

### Solutions for Small Reference Sets

| Metric | Min Samples | Why It Works |
|--------|-------------|--------------|
| **CMMD** | ~300 | Unbiased MMD estimator |
| **FLD+** | ~300 | Normalizing flow density estimation |
| **PRDC** (global) | ~5,000 | Pool all species' images together |
| **Per-class FID** | ~2,000+ | \\(\rightarrow\\) unstable below this |

**Practical recommendation**: Compute PRDC metrics globally (all species pooled) rather than per-species. Report global Precision/Density alongside per-species Validity.

---

## 10. Practical Guide

### Minimum Sample Sizes

| Source | Finding | Recommended N |
|--------|---------|---------------|
| Kynkäänniemi et al. (2019) | Standard configuration | 50,000 |
| Naeem et al. (2020) | PRDC tested | 10,000 |
| General k-NN stability | Variance \\(\propto O(1/\sqrt{N})\\) | \\(\geq\\) 5,000 |

### k Values

| Metric Framework | Default k | Notes |
|------------------|-----------|-------|
| IPR (Kynkäänniemi 2019) | k=3 | Robust across datasets |
| PRDC (Naeem 2020) | k=5 | More robust to outliers |
| General recommendation | k=3 to k=5 | Fix one value and report consistently |

### Computational Cost

| Operation | Complexity | Practical Time |
|-----------|------------|----------------|
| DINOv2 feature extraction | O(N) forward passes | ~5 min for 50K images (GPU) |
| k-NN computation | O(N log N) construction + O(M·k·log N) query | ~30--60s for 50K in 1024-dim |
| PRDC (all 4 metrics) | O((N+M)·k·log N) with ball tree | ~2 min for 50K with scikit-learn |

### Known Failure Modes Summary

| Issue | Affected Metrics | Severity | Mitigation |
|-------|------------------|----------|------------|
| Outlier sensitivity | Precision | Medium | Use Density instead or alongside |
| Hubness (high-dim) | All k-NN metrics | Medium | GICDM (2026), or dimensionality reduction |
| Inception bias | FID, Inception-space P&R | High | Use DINOv2 features |
| Small sample instability | All distributional metrics | High | Pool classes, use CMMD/FLD+ |
| Binary membership | Precision | Low | Use Density for continuous scores |

!!! warning
    **Räisä et al. (ICML 2025 Position Paper)**: *"All current fidelity and diversity metrics fail at least 40% of sanity checks."* No single metric is fully reliable; always report a suite of complementary metrics.

---

## Full Reference Table

| Metric | Paper | Venue | Year | What It Measures | Key Strength | Key Weakness |
|--------|-------|-------|------|-----------------|--------------|--------------|
| FID | Heusel et al. | NeurIPS | 2017 | Distribution distance | Standard, comparable | Gaussian assumption, Inception bias |
| IS | Salimans et al. | NeurIPS | 2016 | Quality + diversity | Simple, standard | Only ImageNet classes |
| P&R Curves | Sajjadi et al. | NeurIPS | 2018 | Fidelity + diversity | First decomposition | Fragile density estimation |
| IPR (k-NN) | Kynkäänniemi et al. | NeurIPS | 2019 | Fidelity + diversity | Simple, non-parametric | Outlier sensitive, hubness |
| PRDC | Naeem et al. | ICML | 2020 | Fidelity + diversity | Outlier robust D&C | Density unbounded |
| PP&PR | Park & Kim | ICCV | 2023 | Fidelity + diversity | Probabilistic, smooth | Less adopted |
| FD-DINOv2 | Stein et al. | NeurIPS | 2023 | Distribution distance | Human-aligned features | Still Gaussian assumption |
| CMMD | Jayasumana et al. | CVPR | 2024 | Distribution distance | No Gaussian assumption, sample-efficient | Single number |
| FWD | Veeramacheneni et al. | ICLR | 2025 | Distribution distance | Domain-agnostic, no NN | Single number, newer |
| FLD+ | Jeevan et al. | arXiv | 2024 | Distribution distance | Works with ~300 samples | Requires retraining per domain |
| Clipped D&C | Salvy et al. | ICLR | 2026 | Fidelity + diversity | Interpretable, linear degradation | Recently accepted |
| GICDM | Salvy et al. | arXiv | 2026 | Corrected D&C | Hubness correction | Very recent |
| Unified P&R | Sykes et al. | arXiv | 2024 | Meta-framework | Theoretical unification | No new practical tool |
| AEROBLADE | Ricker et al. | CVPR | 2024 | Manifold membership | Per-pixel, training-free | Designed for detection, not evaluation |

---

## Citation Guide

For an ICML-level paper justifying Precision/Density as evaluation metrics, cite this chain:

1. **Sajjadi et al. (NeurIPS 2018)** --- Original P&R formulation
2. **Kynkäänniemi et al. (NeurIPS 2019)** --- Improved P&R with k-NN manifold estimation
3. **Naeem et al. (ICML 2020)** --- Density and Coverage as more reliable alternatives
4. **Stein et al. (NeurIPS 2023)** --- DINOv2 as the recommended feature space
5. **Shen et al. (NeurIPS 2024)** --- Adversarial gradient problem motivating why Precision is critical for TFG
6. Optionally: **Räisä et al. (ICML 2025)** --- Acknowledge known limitations of all metrics
