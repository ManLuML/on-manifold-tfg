# Legacy project-page baseline

Captured on 2026-09-01 before the Astro deployment.

| Item | Frozen value |
| --- | --- |
| Research default branch baseline | `e842dff11f255961e188d1271e6aaa8d4642fabc` |
| Final live legacy Pages commit | `c3f04b42c4857ac21b62515879bf6e9dd6e6f2a5` |
| Final durable rollback tag | `legacy-pages-pre-astro-2026-09-01` |
| Directly deployable snapshot branch | `legacy-pages-pre-astro` at `c3f04b42c4857ac21b62515879bf6e9dd6e6f2a5` |
| Earlier rollback tag | `legacy-pages-2026-09-01` at `702070226b69410cae8d97902d348db734b05064` |
| Earlier successful legacy commit | `772dba2bc1fb4ce9ad3c8f4ab9bc6d2cb36d0205` |
| Pages mode | Legacy branch deployment from `gh-pages:/` |
| Canonical URL | `https://manluml.github.io/on-manifold-tfg/` |
| Final live `index.html` SHA-256 | `921b64006173a083c12ca9c2a3c12ec77bfd535da540680d037deec7bc2a9980` |
| Final live slide subtree | Tree `50045b8e083c4b481a9497afd4b72c076b6aa3d4`; 111 files; `slides/index.html` SHA-256 `3d169a2591e9ee2bd73e72d3b075e44df75773dce08270147f1d138857fd272a` |
| Camera-ready authority | `22954ee7f9f7d0e58d61db34474f3d6a15ec8a07` |

The legacy page exposed `#abstract`, `#method`, `#results`, and `#bibtex`. All four IDs remain in the rebuilt page.

The compatibility set contains the following 18 URLs below `/on-manifold-tfg/static/images/`: `favicon.svg`, `og_card.png`, PNG/WebP pairs for teaser, both rho plots, and the style plot, plus JPG/WebP pairs for on/off JiT and DiT grids. Automated browser tests require each URL to return an image response.

The legacy branch advanced during release preparation to publish `/on-manifold-tfg/slides/`. Its complete deck remains immutable and recoverable through the final tag and snapshot branch. The Actions artifact keeps the stable `/slides/` route as a one-file recovery page but does not republish the deck’s excluded third-party quotations, fonts, or institution artwork.

The legacy page used Bulma, Font Awesome, Academicons, and MathJax at runtime. The rebuild preserves useful metadata, alternative text, intrinsic image dimensions, WebP coverage, the canonical URL, and copyable BibTeX while removing the runtime CDN dependencies and placeholder paper link.
