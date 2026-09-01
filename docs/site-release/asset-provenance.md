# Site asset provenance ledger

Verification date: 2026-09-01. The website build does not fetch runtime assets from a CDN.

## Affiliation marks

The final user direction supersedes the earlier logo-placement requirement. The redesigned page and all newly generated assets contain no MAUM.AI or SeoulTech logo. The supplied originals remain unchanged only in the user’s pre-existing local source checkout. Public page affiliations are text-only: “Maum AI” and “Seoul National University of Science and Technology.” Automated source, distribution, and browser checks require zero affiliation-logo paths in the new page.

During release preparation, the existing legacy deployment independently published `/slides/`. Its full 111-file subtree remains recoverable at tag `legacy-pages-pre-astro-2026-09-01` (commit `c3f04b42c4857ac21b62515879bf6e9dd6e6f2a5`, tree `50045b8e083c4b481a9497afd4b72c076b6aa3d4`). The stable route is retained as one minimal English recovery page that links the immutable source and snapshot download. It republishes none of the deck’s third-party educational quotations, fonts, or institution artwork.

## Author-owned paper figures

The authors are Yunsung Lee and Hyeongmin Lee. These figures are used on their paper project page and are not covered by the website-code license. All source files come from the camera-ready paper at `22954ee7f9f7d0e58d61db34474f3d6a15ec8a07`, except the combined teaser copied from public repository `e842dff11f255961e188d1271e6aaa8d4642fabc` and verified against the final teaser composition.

| Public use | Source and SHA-256 | Web derivatives | Transformation and accessible description |
| --- | --- | --- | --- |
| Combined thesis teaser | `assets/teaser.png`, `1a6cbc8ce077953387ec5eefe0c9667ecd8a3bf32a44a92158843dfe5ce7e777` | `teaser-{720,1200,1800}.{avif,webp,png}` and legacy aliases | Aspect-ratio-preserving resize. Alt and nearby prose describe on-manifold success, graceful failure, and catastrophic failure. |
| Camera-ready Figure 1(a) | `figs/teaser_panel_a.pdf`, `81ea5625cc4934f2aa809efa472881e44256b37ffb8dd3565b02664b0f7d0d73`; Poppler render source `cdd3db8cccc9c29ab90e7ff21ffeee96f7b7f9045f068bd737a62dabb85e46fc` | `figure-1a-{720,960}.{avif,webp,png}` | Rendered from the vector PDF at 600 DPI, then resized. Replaces the lower-quality hand-built manifold diagram; caption and alt explain the source/target manifold paths. |
| Crossed-lines full grid | `figs/crossed_lines_grid.pdf`, `246bd5e24b05448d8dc5aeea1ccc92d84ecf60c92e6726d0b271828a71c0c89a`; deterministic Poppler render source `06e863…771666` | `crossed-lines-grid-{720,1400}.{avif,webp,png}` | Rendered with `pdftoppm -png -singlefile -r 180`, then resized. Alt and D=512 HTML table state the trend and exact rates. |
| Dimension curve | `figs/onmanifold_vs_dim.pdf`, `dbb5fdefdea5ab30210b5014f036c847e3a7df2dac2b9ca78ad13912bd7b474c`; render source `6c3c9a…693f6` | `onmanifold-vs-dim-720.{avif,webp,png}` | Poppler render at 220 DPI and resize. Retained for provenance/template use. |
| Validity and Child-FID plots | Source hashes `5272a4…d37` and `f77734…437` | `rho-fid-vs-{validity,child-fid}-{720,1200}.{avif,webp,png}` plus legacy aliases | Aspect-ratio-preserving resize. Captions and HTML table explain matched validity, direction, values, and 9,152 samples per point. |
| Strong-guidance model grids | JiT `6c3c9c…aca`, SiT `8c35f7…086d9`, DiT `869985…eb90` | `{jit,sit,dit}-failure-grid-{720,1400,2200}.{avif,webp,jpg}` | Aspect-ratio-preserving resize. Each tab panel has target-specific alt text and prose. |
| Legacy JiT/DiT grid paths | `vis_{on,off}_{jit,dit}.jpg` source hashes recorded in the build tree | Exact `static/images/*.jpg` aliases and WebP derivatives | Compatibility only; existing public URLs remain fetchable. |
| LGD and FreeDoM | `bird_lgd_a.png` `0e9414…f40`; `bird_lgd_b.png` `13bbdd…5b1` | `{lgd,freedom}-{640,960}.{avif,webp,png}` | Resize only. Captions state the qualitative lowest-frontier result without inventing numbers. |
| Butterfly | `butterfly_pareto.png` `7ed670…19f` | `butterfly-{640,960}.{avif,webp,png}` | Resize only. Alt and prose name 34 species, six parents, and the lowest frontier. |
| Style quantitative plot | `style_gram_vs_validity.png` `bcf22a…4e0` | `style-{640,960}.{avif,webp,png}` plus legacy aliases | Resize only. Avoids uncleared qualitative WikiArt thumbnails. Prose records 400 images and weaker separation. |
| Precision/recall | `rho_precision_vs_recall.png` `eaf3d6…e28` | `precision-recall-{640,960}.{avif,webp,png}` | Resize only. Alt/prose give the headline precision and recall trends. |
| Project social card | Camera-ready Figure 1(a) source above | `site/public/og/project-card.png`, 1200×630, SHA-256 `0ba6afd2921b9d4e2b26dc449ffb9f83ce491617cc2f7670ead490e604acbfa2` | Reviewed Sharp composition with exact paper title, venue, authors, explicit “x first · v second · ε third” TFG robustness ranking, and the author-owned manifold visual. No affiliation marks. |

Sharp-generated AVIF/WebP derivatives use committed source files, fixed widths, quality 57/84, and fixed encoding options in `site/scripts/process-assets.mjs`. Fallbacks are PNG for plots and JPEG for photographic grids. `site/assets/generated-assets.json` locks the generator, dependency lockfile, every source, and every reviewed output by SHA-256; CI verifies this manifest instead of regenerating social-card typography with platform-dependent system fonts. Intrinsic dimensions are present in HTML; below-fold figures use lazy loading.

## Excluded material

- Stale `figs/teaser.pdf`.
- `rebuttal_delta_precision.*`, old root P-FID plots, `pareto_cfid_validity.pdf`, and rebuttal figures.
- Third-party DDPM and JiT presentation crops.
- `style_vis_dit` and `style_vis_jit`, because their WikiArt thumbnail reuse rights are not individually recorded.
- Slides and all slide-bundle fonts.
- The ECCV navbar SVG. Its official origin was verified, but no current written grant for republication on an independent project site was found; the release uses the text venue label only.
- New MAUM.AI and SeoulTech derivatives for the redesigned page, following the final text-only affiliation decision. The legacy deck is preserved in immutable history rather than republished.

## 2026-09-01 slide-publication addendum

The ledger above records the first Astro project-page release and remains the provenance record for that release. The authors subsequently and explicitly superseded the slide-exclusion policy while retaining the text-only affiliation policy for the root Astro project page.

The publication boundary is now path-specific:

- `/on-manifold-tfg/` and its normal Astro assets remain text-only for affiliations. MAUM.AI, SeoulTech, and ECCV logos remain excluded from the root page, shared shell, metadata, social card, favicon, and reusable project-page assets.
- `/on-manifold-tfg/slides/**` publishes the complete 17-slide author-prepared ECCV deck. The slide bundle is intentionally retained as a unit, including the MAUM.AI wordmark, SeoulTech affiliation artwork, official ECCV logo SVG, quoted educational figures, presentation fonts, Reveal support files, and all media required by the deck.
- The earlier excluded-material entries for slides, slide-bundle fonts, the ECCV logo, and affiliation artwork are therefore superseded only within `/slides/**`. They continue to apply to the root Astro page.
- Slide-specific logos, quoted figures, artwork, and fonts are presentation material rather than website-code assets. Inclusion in the deployed bundle does not place them under the repository’s MIT license or transfer any copyright or trademark rights.

The immutable legacy tag and snapshot branch remain valid rollback evidence. They are no longer the only public recovery mechanism for the presentation; the current Actions artifact is intended to serve the complete deck directly from the stable `/slides/` URL.
