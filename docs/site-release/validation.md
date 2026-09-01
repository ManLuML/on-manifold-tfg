# Coordinated release validation record

Release candidate verified on 2026-09-01 in Asia/Seoul. Production fields are appended only after GitHub Pages reports successful deployments.

## Frozen authorities and preservation

- Research baseline: `origin/main` at `e842dff11f255961e188d1271e6aaa8d4642fabc` in a fresh worktree; the user’s five-commit-behind original checkout and its untracked plan/logo inputs remain untouched.
- Camera-ready paper: `22954ee7f9f7d0e58d61db34474f3d6a15ec8a07`.
- Final legacy live page: `c3f04b42c4857ac21b62515879bf6e9dd6e6f2a5`, frozen as tag `legacy-pages-pre-astro-2026-09-01` and directly deployable branch `legacy-pages-pre-astro`. The earlier tag `legacy-pages-2026-09-01` at `702070226b69410cae8d97902d348db734b05064` also remains intact; `gh-pages` history is not rewritten or deleted.
- Out-of-scope dirty worktrees `on-manifold-tfg-page` and `on-manifold-tfg-page-slides` were not used.
- Final affiliation override: text-only “Maum AI” and “Seoul National University of Science and Technology”; no country labels and no affiliation-logo original or derivative in either release.

## Local artifact evidence

| Artifact | Files | Size | `index.html` SHA-256 | 404 SHA-256 | Social-card SHA-256 |
| --- | ---: | ---: | --- | --- | --- |
| ManLuML root | 37 | 2.3 MB | `fe6458c24c38a5786e061ca32fc65cfebd4d2cd55b07d79787b75f0fe32b71bf` | `d900ca92004ea8f52bbcb02c28b1453f05a9a80c16f219cfcb363d21e2440833` | `8408297a1d46fae70b9413d2c8c660f91bf38a002f3ce460d10320c0fa3084f6` |
| On-Manifold TFG | 132 | 29.0 MB | `c01aa2cbe34e4b4895d135997b0f9f53e96c5817d897f1ccf1a936042eadf5e8` | `56fd249dec86def5cdd7b2e4ea11125b229b07592c3586c3be12aac880d011cd` | `0ba6afd2921b9d4e2b26dc449ffb9f83ce491617cc2f7670ead490e604acbfa2` |

Both `npm test` commands pass from locked dependencies. Root: seven manifest/unit assertions and six browser tests. Project: five asset/slide assertions and thirteen browser tests. Astro reports zero type/schema errors, warnings, or hints. Dependency audit reports zero vulnerabilities. Deterministic-integrity checks cover 25 root inputs/outputs and 136 project inputs/outputs, including every generated figure variant, social card, license, recovery page, and generator source.

## Contract checks

- Schema: explicit manifest accepts one reviewed published feature, rejects duplicate slugs, invalid/missing links, and unreviewed publication; draft/private and future-project fixtures remain invisible.
- Build and paths: root and `/on-manifold-tfg/` build statically with trailing slashes; project assets use the configured base. No server runtime, database, analytics, cookies, or external runtime CDN requests.
- Content: claim table covers formulas, D=512 values, matched-validity values, 5.2-point gap, 143 species, 9,152 samples, broader controls, and limitations. Search/assertion checks find no blanket “only target differs” cross-model claim.
- Interactions: tabs support pointer, Arrow Left/Right, Home, and End; the native slider supports keyboard boundaries and updates coefficient values; initial HTML keeps all panels, balanced accessible formulas, the camera-ready Figure 1(a), static table, resources, and citation with JavaScript disabled; copy feedback is visible and announced.
- Accessibility: automated axe reports zero violations; semantic browser tree contains ordered headings, landmarks, named controls, tables/captions, MathML, figure alternatives/captions, status feedback, and a working skip link. Keyboard, reduced-motion, no-JavaScript, and 320px reflow paths pass.
- Affiliation override: two simplified text affiliations render; no `Republic of Korea` string and zero affiliation-logo image paths or files.
- Compatibility: `#abstract`, `#method`, `#results`, and `#bibtex` exist; all 18 legacy `static/images/*` URLs return an image response in the production artifact server. `/slides/` returns a tested, noindex, English recovery page (SHA-256 `ab17f833b436836fccad257267aaf1310a2be8b707476a1d682199e6425d99a1`) linking the immutable legacy tag and archive without republishing the excluded 111-file deck.
- Metadata: unique titles/descriptions, absolute canonicals, Open Graph/X fields, 1200×630 cards, root Organization/WebSite JSON-LD, project ScholarlyArticle/citation metadata, sitemap, robots files, favicons, and noindexed 404 pages pass assertions.
- Links: every independently hosted public resource returned HTTP 200 in the prelaunch check, including arXiv, GitHub profiles/repository, the immutable legacy snapshot, and all three Hugging Face resources. Dedicated post-deployment workflow checks verify both site origins and bidirectional root/project navigation.
- Lab performance across the final worktree and clean-index runs: root LCP 72–140 ms and CLS 0.0005; project LCP 68–104 ms, CLS 0.0125, and slider interaction proxy 0.1–12.5 ms. These are local Chromium lab values, not field p75 data.
- Visual evidence: committed 320px screenshot baselines cover both first viewports and the text-only affiliation block. Desktop review captures cover the root hero, project hero, explicit x-first/v-second/ε-third robustness statement, balanced formulas, camera-ready Figure 1(a), and the reviewed 1200×630 social card.

## Rollback restoration evidence

The complete immutable tag artifact was materialized and served independently before launch. Tag and snapshot branch both resolve to `c3f04b42c4857ac21b62515879bf6e9dd6e6f2a5`; the artifact has 151 files in total, including 111 slide files and 18 `static/images/*` compatibility assets. Its root `index.html` SHA-256 is `921b64006173a083c12ca9c2a3c12ec77bfd535da540680d037deec7bc2a9980`; its slide index SHA-256 is `3d169a2591e9ee2bd73e72d3b075e44df75773dce08270147f1d138857fd272a`; and the slide tree is `50045b8e083c4b481a9497afd4b72c076b6aa3d4`. All four compatibility fragments passed.

At 2026-09-01 17:29 KST, Pages was switched from `gh-pages` to the frozen `legacy-pages-pre-astro:/` source and a fresh legacy Pages build was explicitly triggered. Build commit `c3f04b42c4857ac21b62515879bf6e9dd6e6f2a5` completed successfully. The live root and slide hashes matched the immutable artifact exactly, all four fragments remained present, and all 18 legacy asset URLs returned HTTP 200. GitHub added `legacy-pages-pre-astro` to the `github-pages` environment branch policy alongside `gh-pages` and `main`. This exercises the documented recovery path without rewriting `gh-pages` or either tag.

## Verified resource limitation

The publicly linked PDF is arXiv v1 and differs from the accepted camera-ready source used to validate the page. The site therefore labels it “arXiv preprint,” pins scientific claims separately to camera-ready commit `22954ee7f9f7d0e58d61db34474f3d6a15ec8a07`, and does not imply that the public PDF is the accepted manuscript. No verified public camera-ready or proceedings PDF was available at release time.

## Production evidence

To be appended after launch: repository commit SHAs, workflow run IDs, Pages deployment IDs, timestamped HTTP/metadata/asset/fragment checks, production screenshots, production lab metrics, cross-navigation, and rollback restoration evidence.
