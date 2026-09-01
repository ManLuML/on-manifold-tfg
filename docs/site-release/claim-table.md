# Project-page claim table

Verified on 2026-09-01 against camera-ready paper commit `22954ee7f9f7d0e58d61db34474f3d6a15ec8a07`. Presentation materials and the legacy page were not used as claim authorities.

| Public claim | Camera-ready authority | Qualification used on the page |
| --- | --- | --- |
| Title, author order, affiliations, equal contribution, and corresponding author | `root.tex`, with final user display override | Yunsung Lee¹*; Hyeongmin Lee²*†. `*` means equal contribution; `†` means corresponding author. Public affiliation labels are simplified to “Maum AI” and “Seoul National University of Science and Technology,” with no country labels or logos. |
| High-noise convention | `section_latex/3_method.tex` | `z_t = t x + (1-t) ε`; `t → 0` is pure noise. |
| x recovery has no multiplier | `section_latex/3_method.tex`; `section_latex/appendix/A_proofs.tex` | `‖x̂^(x)-x‖ = δ_x`. |
| v recovery is bounded | Same | `‖x̂^(v)-x‖ = (1-t)δ_v`. |
| ε recovery diverges at high noise | Same | `‖x̂^(ε)-x‖ = ((1-t)/t)δ_ε`; the coefficient diverges as `t → 0`. |
| Overall TFG robustness ordering | Abstract, method, and results | Public copy states “x first, v second, ε third” and labels the direction from most to least robust, avoiding an ambiguous bare inequality. It does not claim that the three recovery coefficients alone have one strict order at every `t`. |
| Crossed-lines D=512 on-manifold rates | `section_latex/appendix/C_experimental_protocols.tex`, Table `tab:crossed_lines_full` | x 93.3%, v 21.5%, ε 0.5%. Fully controlled, identical network design and training protocol. |
| Matched-validity Child FID | `section_latex/5_results.tex`; Appendix C full sweep | JiT-H/x 32.85 → 32.9, SiT/v 34.66 → 34.7, DiT/ε 38.11 → 38.1, all near 26.6% validity. |
| x-versus-ε Child FID gap | Abstract and results | Exact table difference is 5.26; camera-ready prose reports 5.2 points. |
| Bird benchmark scale | `section_latex/4_experiments.tex`; Appendix C | 143 species under 30 ImageNet parents; 64 samples per species; 9,152 generated images per plotted point. |
| Capacity-reversed evidence | `section_latex/appendix/F_confound_discussion.tex` | JiT-B 131M reaches C-FID 31.3; DiT and SiT at 675M reach 36.7 and 34.4. Capacity is not a sufficient explanation. |
| LGD and FreeDoM | Appendix C, `fig:tfg_family` | JiT-H has the lowest C-FID frontier under both. No invented numeric table is shown. |
| Butterfly domain | Appendix C, `fig:butterfly` | 34 species under six ImageNet parents; JiT-H has the lowest C-FID frontier. |
| Style transfer | Results and Appendix C | At `ρ=10`, JiT-H retains 80% content accuracy while DiT reaches 1.5%; separation is explicitly weaker than on birds. |
| Precision and recall | `section_latex/appendix/J_prdc_analysis.tex` | JiT-H recall reaches 0.59. DiT precision peaks at 0.24, recall remains around 0.49, and precision falls under stronger guidance. |
| Inverse problems | Appendix C tables | Best x-prediction LPIPS: 0.2140 deblur and 0.1886 4× super-resolution. No model beats the degraded-input PSNR baselines. |
| Cross-model limitations | Conclusion and `section_latex/appendix/F_confound_discussion.tex` | The pretrained models also vary in architecture, operating space, capacity, and sampler. The page never says the full comparison differs only by target. |
| Theory and method scope | Conclusion and Appendix F | Lipschitz energy, well-trained predictors, and comparable Jacobian assumptions; gradient-based guidance only; 256×256 experiments; attention-only guidance outside scope. |

## Citation

The paper tree has no self-citation entry. The public BibTeX is constructed from camera-ready title, authors, venue, and year and matches the public repository citation.
