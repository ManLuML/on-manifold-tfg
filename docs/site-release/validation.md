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

Production verification completed on 2026-09-01 at 17:56 KST.

| Site | Release commit | Pages workflow | Pages deployment | Public URL |
| --- | --- | --- | ---: | --- |
| ManLuML root | `9b16a9dc2522a2a1df7c2678771c67b5a85d3883` | [run 33488364896](https://github.com/ManLuML/manluml.github.io/actions/runs/33488364896), success | `6197663416` | `https://manluml.github.io/` |
| On-Manifold TFG | `c355ecf4278285b53606351f71eba834c5d337f9` | [run 33488814933](https://github.com/ManLuML/on-manifold-tfg/actions/runs/33488814933), success | `6197743540` | `https://manluml.github.io/on-manifold-tfg/` |

The existing Python [CI run 33488814927](https://github.com/ManLuML/on-manifold-tfg/actions/runs/33488814927) also passed lint and tests for the project release commit. Both Pages sites report workflow mode with HTTPS enforced. The root workflow’s post-deploy check verified the root canonical and root-to-project link; the project workflow’s post-deploy check verified both canonicals and bidirectional navigation.

Fresh no-cache HTTP checks matched the reviewed artifacts byte-for-byte: root index `fe6458c24c38a5786e061ca32fc65cfebd4d2cd55b07d79787b75f0fe32b71bf`, root card `8408297a1d46fae70b9413d2c8c660f91bf38a002f3ce460d10320c0fa3084f6`, project index `c01aa2cbe34e4b4895d135997b0f9f53e96c5817d897f1ccf1a936042eadf5e8`, project card `0ba6afd2921b9d4e2b26dc449ffb9f83ce491617cc2f7670ead490e604acbfa2`, and slides recovery page `ab17f833b436836fccad257267aaf1310a2be8b707476a1d682199e6425d99a1`. Both custom 404 routes returned HTTP 404 with `noindex,follow`.

Production smoke checks confirmed all four legacy fragments, all 18 legacy image URLs, the one-file noindex `/slides/` recovery route, exact simplified affiliations, explicit x-first/v-second/ε-third ranking, citation-version disclosure, Open Graph/X cards, and zero affiliation-logo or ECCV-logo references. Every public resource and both cross-site directions returned HTTP 200.

Production Chromium lab checks observed root LCP 544 ms and CLS 0.00038; project LCP 308 ms, CLS 0.01759, and slider interaction proxy 0.1 ms. Axe reported zero violations on both desktop pages, console errors and failed requests were empty, 320 CSS-pixel layouts had no document overflow, and both normal and 200% text-size accessibility checks passed. These remain lab measurements; no field p75/CrUX data exists for the newly launched pages.

Production screenshots are committed as `production-root-desktop.png`, `production-root-320.png`, `production-project-desktop.png`, `production-project-320.png`, `production-project-formulas.png`, and `production-project-manifold-figure.png` under `docs/site-release/screenshots/`. They visually confirm the desktop/mobile hierarchy, balanced formulas, and camera-ready Figure 1(a) on the public origin.

## 2026-09-01 slide-publication addendum

All evidence above remains the immutable validation record for the first Astro release. A later author-approved change supersedes only the slide-route exclusion: the Actions artifact is now expected to serve the complete 17-slide ECCV deck at `/on-manifold-tfg/slides/` instead of the one-file recovery placeholder.

The release contract is deliberately path-scoped:

- The root project page must continue to show affiliations as text and must contain no MAUM.AI, SeoulTech, or ECCV logo references.
- The `/slides/**` subtree must preserve the authored deck intact, including its MAUM.AI, SeoulTech, and ECCV logos, quoted educational figures, presentation fonts, Reveal support files, and required media.
- Automated logo exclusions must therefore inspect the root Astro site while exempting the isolated slide bundle; they must not remove or rewrite presentation assets.
- Slide validation must confirm the full 17-slide structure, slide numbering, required assets, and absence of private speaker notes before deployment.

Production deployment completed on 2026-09-01 at 21:16 KST from squash commit `e874846b08761e1dae76c5acdf17f8c71eb2402d`. Pull request [#4](https://github.com/ManLuML/on-manifold-tfg/pull/4) passed both Pages quality checks and the repository Python CI before merge. Pages [run 33506672393](https://github.com/ManLuML/on-manifold-tfg/actions/runs/33506672393) successfully built, uploaded, deployed, and completed the production cross-site verifier.

The live slide index is 233,339 bytes with SHA-256 `846aa170e3d7678f496e00b7c04f22a271aabd80384f97cb5416876041147892`, matching the reviewed artifact. A fresh public-origin browser check confirmed 17 direct slides, numbering from `1 / 17`, the MAUM.AI, SeoulTech, and ECCV logos, restored source section 19, all four outcome-animation groups, zero public speaker notes, zero broken images, no document-level horizontal overflow, and no legacy-placeholder copy. The ECCV, MAUM.AI, SeoulTech, outcome-photo, and vendored Reveal-runtime URLs all returned HTTP 200.

The editable slide source branch was also fast-forwarded from `c3f04b4` to `6756b6a`; Pages remains workflow-driven from `main`, so this preserves the QMD/CSS source without changing the production deployment source.
