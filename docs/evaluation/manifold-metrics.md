# Manifold-Aware Evaluation Metrics

This document proposes evaluation metrics that measure whether TFG-guided samples remain on the **true data manifold** of the target class, rather than merely fooling a classifier. These metrics are critical for capturing x-prediction's core advantage: preserving source distribution fidelity under guidance.

For the detailed literature survey behind these recommendations, see [Metric Literature Survey](metric-survey.md).

---

## Why Validity Is Insufficient

The current **validity** metric (top-1 classification accuracy from a fine-grained bird classifier) answers only one question: *"Is this image classified as the target species?"*

This conflates two qualitatively different outcomes:

| Outcome | Validity | Realistic? | Example (Toy) |
|---------|----------|------------|----------------|
| On-manifold, correct class | High | Yes | Point on the red bar |
| Off-manifold, correct class | High | **No** | Point in the red region but off the bar |
| On-manifold, wrong class | Low | Yes | Point on the blue bar (graceful failure) |
| Off-manifold, wrong class | Low | No | Point in empty space |

The second row is the critical failure mode: **adversarial-like samples** that exploit the classifier's decision boundary without being realistic images of the target species. Under TFG, this happens because off-the-shelf classifiers produce adversarial-like gradients at high noise levels.

### The Circular Evaluation Problem

When the guidance classifier and evaluation classifier share similar feature-space biases, artifacts introduced by the guidance gradients may go undetected by evaluation:

1. Guidance maximizes \\(\nabla_{z_t} \log p_\phi(y \mid \hat{x}_0)\\) using classifier \\(\phi\\)
2. Validity measures \\(p_\psi(y \mid x_{\text{generated}})\\) using classifier \\(\psi\\)
3. If \\(\phi\\) and \\(\psi\\) share similar feature-space blind spots, artifacts in the guidance gradient may be rewarded by evaluation

!!! warning
    Shen et al. (NeurIPS 2024, "Understanding and Improving Training-free Loss-based Diffusion Guidance") demonstrate that training-free guidance is more susceptible to misaligned gradients than classifier guidance trained on noisy data, due to worse Lipschitz properties of off-the-shelf classifiers. This suggests classifier-based validity alone may be insufficient for comparing TFG across prediction targets.

**Solution**: Evaluate in a feature space that is **independent of the guidance network** (though not necessarily of visual semantics in general).

---

## Proposed Metrics

### Precision (Kynkäänniemi et al., NeurIPS 2019)

**The single most important metric for our research question.**

Precision measures the fraction of generated samples that fall within the **support** of the real data distribution, estimated via k-nearest-neighbor hyperspheres in a learned feature space.

#### Algorithm

1. Embed real samples \\(\{x_i\}_{i=1}^N\\) and generated samples \\(\{y_j\}_{j=1}^M\\) into feature space (DINOv2)
2. For each real sample \\(x_i\\), form a hypersphere \\(B(x_i, r_i)\\) where \\(r_i\\) = distance to its \\(k\\)-th nearest real neighbor
3. Precision = fraction of generated samples falling within *any* real hypersphere:

\\[
\text{Precision} = \frac{1}{M} \sum_{j=1}^{M} \mathbb{1}\left[ y_j \in \bigcup_{i=1}^{N} B(x_i, r_i) \right]
\\]

#### Interpretation for TFG

| Scenario | Validity | Precision | Meaning |
|----------|----------|-----------|---------|
| x-pred (ideal) | High | **High** | Guided samples are realistic AND correct |
| e-pred (adversarial) | High | **Low** | Samples fool classifier but are off-manifold |
| x-pred (graceful fail) | Low | **High** | Failed guidance, but samples are still realistic |

**Note**: "Graceful failure" means the output remains a realistic image (useful for some applications), but it is also a *silent* failure mode --- the user receives a plausible but incorrect result without obvious visual cues of failure.

This 2D decomposition directly supports the paper's core claim:

> *x-prediction maintains manifold fidelity even when guidance partially fails, while e/v-prediction can produce high-validity but off-manifold samples.*

#### Practical Details

- **Default k**: 5 (PRDC standard; IPR original uses k=3, results robust across k=3--5)
- **Sample size**: \\(N \geq 5{,}000\\) for stable estimates, \\(N = 50{,}000\\) ideal
- **Package**: `prdc` ([github.com/clovaai/generative-evaluation-prdc](https://github.com/clovaai/generative-evaluation-prdc))
- **Citations**: Widely adopted, de facto standard in GAN/diffusion evaluation

---

### Density (Naeem et al., ICML 2020)

Density is a continuous extension of Precision that measures **how deeply** each generated sample sits within the real distribution's support, rather than just binary in/out.

\\[
\text{Density} = \frac{1}{kM} \sum_{j=1}^{M} \sum_{i=1}^{N} \mathbb{1}\left[ y_j \in B(x_i, r_i) \right]
\\]

where \\(r_i\\) is the distance from \\(x_i\\) to its \\(k\\)-th nearest real neighbor (same radius used in Precision).

- **Density > 1**: Generated samples on average fall within multiple real-sample hyperspheres (well-represented region)
- **Density \\(\approx\\) 0**: Samples are at the manifold edge or off-manifold
- **Advantage over Precision**: Distinguishes "barely on-manifold" from "squarely on-manifold"
- **Caveat**: High Density can also indicate mode collapse (concentration in a narrow region) --- interpret alongside Coverage
- **Outlier robust**: Unlike Precision, a single real-data outlier with a large hypersphere does not inflate Density because the count is normalized

Computed simultaneously with Precision via the `prdc` package at zero additional cost.

---

### Coverage (Naeem et al., ICML 2020)

Coverage measures what fraction of the real data manifold is "covered" by generated samples --- a robust replacement for Recall:

\\[
\text{Coverage} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[ \exists\, j : y_j \in B(x_i, r_i) \right]
\\]

- Detects **mode collapse**: if guidance causes all generated birds to look the same, Coverage drops
- TFG guidance inherently reduces diversity; Coverage reveals whether prediction targets differ in **how much** diversity they sacrifice
- **Why Coverage over Recall**: Recall uses generated-sample hyperspheres, so unrealistic fake samples can inflate their radii and overestimate the generated manifold (Naeem et al., 2020). Coverage uses **real-sample hyperspheres only** --- more robust and reusable across models

Computed simultaneously with Precision and Density via the `prdc` package at zero additional cost. Recall is also computed by `prdc` and should be reported for reference, though Coverage is preferred for interpretation.

---

### Realism Score (Per-Sample)

A continuous, per-image score measuring manifold proximity. The original definition (Kynkäänniemi et al., 2019) takes the max over **all** real samples. The computationally efficient approximation restricts to the \\(k\\) nearest real neighbors of each generated sample:

\\[
\text{Realism}(y_j) = \max_{i \in \text{NN}_k(y_j)} \frac{r_i}{\|y_j - x_i\|_2}
\\]

where \\(\text{NN}_k(y_j)\\) denotes the indices of the \\(k\\) nearest **real** samples to \\(y_j\\), and \\(r_i\\) is the distance from real sample \\(x_i\\) to its \\(k\\)-th nearest real neighbor. (This approximation is standard in practice. The k-NN methodology is similar to that used in the `prdc` package, but the Realism Score itself requires a separate implementation.)

- **Score \\(\geq\\) 1.0**: The sample is inside at least one real hypersphere (on-manifold)
- **Score \\(<\\) 1.0**: The sample is outside all real hyperspheres (off-manifold)

#### Visualization Potential

Plot the distribution of Realism Scores across prediction targets:

- **x-pred**: Expected unimodal distribution (most scores > 1, on-manifold)
- **e-pred**: Expected bimodal distribution (some on-manifold, some off-manifold)
- **v-pred**: Expected intermediate pattern

This makes a compelling paper figure showing qualitatively different failure modes.

---

## Feature Space: DINOv2

All manifold metrics must be computed in a feature space **independent of the guidance network** to avoid shared evaluation vulnerabilities.

### Why DINOv2 ViT-L/14

| Property | DINOv2 | Inception-v3 | Guidance Classifier |
|----------|--------|--------------|---------------------|
| Training | Self-supervised (DINO) | Supervised (ImageNet) | Supervised (species) |
| Architecture | ViT-L/14 | Inception-v3 (CNN) | EfficientNetB2/ViT |
| Feature dim | 1024 | 2048 | varies |
| Human alignment | **Best** (Stein 2023) | Biased against diffusion | N/A |

- **Stein et al. (NeurIPS 2023)** conducted the largest evaluation study (9 encoders, 17 metrics, 41 models, 4 datasets) and found DINOv2 ViT-L/14 most aligned with human perceptual judgment
- Self-supervised training means fewer explicit class biases --- captures holistic image structure (plumage, posture, habitat) rather than discriminative features (though semantic structure correlated with classes is still learned)
- Features are not directly optimized by guidance classifiers (EfficientNetB2, ViT-B/16) or evaluation classifiers (bird-species-classifier, DeiT-Small), though they share training data overlap (LVD-142M includes ImageNet-like images)

### Library Support

- **`dgm-eval`** ([github.com/layer6ai-labs/dgm-eval](https://github.com/layer6ai-labs/dgm-eval)): Supports 17 metrics (including PRDC) with 9 feature extractors (including DINOv2)
- **`torch.hub`**: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')` for direct use

### Practical Recommendation

- **Primary**: DINOv2 ViT-L/14 for all manifold metrics (Precision, Density, Coverage, Realism)
- **Secondary**: Inception-v3 for backward compatibility (report alongside DINOv2 results)

---

## Complementary Metrics

### FD-DINOv2 (Fréchet Distance with DINOv2)

Same formula as FID but using DINOv2 features instead of Inception-v3:

\\[
\text{FD} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})
\\]

- Mitigates known Inception bias against diffusion models (Stein et al., NeurIPS 2023)
- Rapidly becoming the standard complementary metric (2024-2025 adoption growing)
- Can replace Source FID / Target FID with DINOv2 versions for cross-validation

### CMMD (CLIP Maximum Mean Discrepancy)

- **Distribution-free**: No Gaussian assumption (unlike FID/FD)
- **Sample-efficient**: Stable with as few as 300 samples (Jayasumana et al., CVPR 2024)
- Useful for fine-grained evaluation where per-species reference sets are small (~60 images)

### Multi-Classifier Agreement

Evaluate with 2--3 architecturally diverse classifiers. Report:

- **Per-classifier validity**: Individual accuracy for each architecture
- **Agreement validity**: Fraction where **all** classifiers agree on target class
- **Rationale**: Adversarial-like samples tend to exploit one architecture's decision boundary but fail on others

Current infrastructure already uses two classifiers (guide: EfficientNetB2, eval: bird-species-classifier). Adding a third architecturally different model (e.g., ConvNeXt or ResNet) enables 3-way agreement.

### VAE Reconstruction Error (AEROBLADE, CVPR 2024)

Pass generated images through a pretrained autoencoder and measure reconstruction error (LPIPS):

- **Images matching the autoencoder's training distribution**: Low reconstruction error (autoencoder can faithfully reconstruct)
- **Out-of-distribution images**: High reconstruction error (outside the autoencoder's learned manifold)
- Directly applicable to latent-space models (DiT, SiT) since their VAE is already available

!!! note
    **Caveat for TFG**: AEROBLADE was designed to distinguish real vs. AI-generated images, not to evaluate guided generation quality. The VAE's learned manifold does not directly correspond to the data manifold in the evaluation sense. The relationship between VAE reconstruction error and data manifold proximity requires careful interpretation in the TFG context.

---

## Toy Example (2D Spiral) Metrics

The k-NN based metrics (Precision, Density, Realism) are **dimension-agnostic** and work directly on 2D coordinates without any feature extractor:

```python
# No feature extraction needed for 2D data
real_red_bar = ground_truth_red_class_points   # (N, 2)
generated = tfg_generated_points               # (M, 2)

from prdc import compute_prdc
metrics = compute_prdc(real_red_bar, generated, nearest_k=5)
# metrics['precision']: fraction of generated points on the red bar
# metrics['density']:   how deep in the red bar distribution
```

Additionally, 2D data enables exact distributional comparisons:

| Metric | What It Measures | Implementation |
|--------|------------------|----------------|
| k-NN Precision | On-bar fraction | `prdc` package |
| k-NN Density | Bar-depth | `prdc` package |
| KDE + KL Divergence | Distributional match | `scipy.stats.gaussian_kde` |
| Wasserstein Distance | Optimal transport distance | `scipy.stats.wasserstein_distance` |

**Key experiment**: Plot Precision vs. dimension (D=2, 8, 32, 128, 512) across prediction targets. One plausible outcome: x-pred Precision remains more stable while e/v-pred Precision degrades faster with increasing dimension, though the reverse (all targets degrade similarly) or confounding effects (k-NN metric breakdown in high-D) are also possible.

---

## Application in the Paper (Appendix J)

The paper's camera-ready version carries out this analysis in **Appendix J
("Precision--Recall Analysis")**, which complements the FID-based metrics
(P-FID, C-FID) and classifier Validity of the main text with manifold-aware
Precision and Recall curves.

**Setup.** Appendix J uses the \\(k\\)-nearest-neighbour Precision and Recall of
Naeem et al. (2020), computed in **DINOv2 ViT-B/14** feature space
(768-dimensional CLS tokens, \\(k=5\\)), traced over the same DPS
\\(\rho\\)-sweep as the main bird-guidance experiments for all six models
(JiT-B/L/H, DiT, SiT, PixelFlow). Precision measures the fraction of generated
samples inside the real data manifold (fidelity); Recall measures the fraction
of real samples covered by the generated distribution (diversity).

**What Appendix J reports:**

1. **Capacity expands coverage, not sharpness.** Across the JiT family, Recall
   improves steadily with scale (JiT-B 0.37 → JiT-L 0.49 → JiT-H 0.59) while
   Precision stays in a narrow band (~0.17--0.19).
2. **ε-prediction shows the mode-collapse signature.** DiT attains the highest
   Precision of any model (0.24 at \\(\rho=0.05\\)), but its Recall never
   exceeds ~0.49, and under stronger guidance its Precision erodes (0.19 at
   \\(\rho=0.5\\)) while Recall stays low (~0.45) --- the joint high-Precision,
   low-Recall pattern the paper identifies as ε-prediction's mode collapse.
3. **v-prediction gains little diversity from guidance.** PixelFlow and SiT
   remain in the Recall range 0.40--0.53, below JiT-H at every operating point;
   PixelFlow's Precision drifts down (0.21 → 0.19) with Recall essentially
   unchanged, so guidance degrades fidelity without adding diversity.

These sample-level observations agree with the paper's distribution-level
C-FID analysis and with the prediction that x-prediction preserves manifold
proximity under guidance. See the paper appendix
(`section_latex/appendix/J_prdc_analysis.tex`, section "Precision--Recall
Analysis") for the full figure and discussion.

> Note: the paper's Appendix J uses DINOv2 ViT-B/14; the ViT-L/14
> recommendation earlier in this document reflects the general guidance from
> Stein et al. (2023) for future extensions (Density, Coverage, Realism), which
> are surveyed here but not reported in the paper.

---

## Reference Data Requirements

Computing Precision/Density requires **raw feature vectors** from real images (not just mean/covariance as for FID).

### Options

| Approach | What's Needed | Storage |
|----------|---------------|---------|
| Pre-compute DINOv2 features | Run feature extraction on real images, save as `.npz` | ~56 MB for 14K bird images |
| Transfer feature files | Compute on device with data, transfer `.npz` only | Network transfer |
| Use `dgm-eval` library | Handles feature extraction + metric computation | Requires image access |

Reference feature files follow the existing pattern in `evaluation/generation/fid_stats/`:
```
fid_stats/
  finegrained/
    child_fid_stats.npz          # Existing: Inception mu/sigma
    child_dinov2_features.npz    # NEW: DINOv2 raw features (N, 1024)
    child_dinov2_stats.npz       # NEW: DINOv2 mu/sigma for FD-DINOv2
```

---

## References

- Kynkäänniemi et al., "Improved Precision and Recall Metric for Assessing Generative Models", *NeurIPS 2019*
- Naeem et al., "Reliable Fidelity and Diversity Metrics for Generative Models", *ICML 2020*
- Stein et al., "Exposing Flaws of Generative Model Evaluation Metrics and Their Unfair Treatment of Diffusion Models", *NeurIPS 2023*
- Jayasumana et al., "Rethinking FID: Towards a Better Evaluation Metric for Image Generation", *CVPR 2024*
- Shen et al., "Understanding and Improving Training-free Loss-based Diffusion Guidance", *NeurIPS 2024*
- Ricker et al., "AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error", *CVPR 2024*
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", *TMLR 2024*

See [Metric Literature Survey](metric-survey.md) for the complete reference list.
