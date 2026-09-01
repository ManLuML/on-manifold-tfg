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
| On-Manifold TFG | 258 | 44.7 MB | `7e45ab2430d18ab70e0538f291e255598e6d1b6fbb56744826d83d4aff7b5f1f` | `56fd249dec86def5cdd7b2e4ea11125b229b07592c3586c3be12aac880d011cd` | `0ba6afd2921b9d4e2b26dc449ffb9f83ce491617cc2f7670ead490e604acbfa2` |

Both `npm test` commands pass from locked dependencies. Root: seven manifest/unit assertions and six browser tests. Project: eight asset/slide/citation assertions and fifteen browser tests. Astro reports zero type/schema errors, warnings, or hints. Dependency audit reports zero vulnerabilities. Deterministic-integrity checks cover 25 root inputs/outputs and 262 project inputs/outputs, including every generated figure variant, mobile comparison crop, social card, license, complete presentation asset, and generator source.

## Contract checks

- Schema: explicit manifest accepts one reviewed published feature, rejects duplicate slugs, invalid/missing links, and unreviewed publication; draft/private and future-project fixtures remain invisible.
- Build and paths: root and `/on-manifold-tfg/` build statically with trailing slashes; project assets use the configured base. No server runtime, database, analytics, cookies, or external runtime CDN requests.
- Content: claim table covers the `t=1` endpoint interpretation, high-noise formulas, D=512 values, realistic TFG task, pooled Child FID, matched-validity values, 5.2-point gap, 143 species, 9,152 samples, and broader controls. Search/assertion checks find no blanket “only target differs” cross-model claim.
- Scroll story: all x/v/ε failure bands, balanced accessible formulas, high-noise explanation, Figure 1(a), dimension plot, benchmark definition, resources, and citation exist in initial HTML. Tabs and the timestep slider are absent; copy feedback remains visible and announced.
- Accessibility: automated axe reports zero violations; semantic browser tree contains ordered headings, landmarks, named controls, tables/captions, MathML, figure alternatives/captions, status feedback, and a working skip link. Keyboard, reduced-motion, no-JavaScript, and 320px reflow paths pass.
- Affiliation override: two simplified text affiliations render; no `Republic of Korea` string and zero affiliation-logo image paths or files.
- Compatibility: `#abstract`, `#method`, `#results`, and `#bibtex` exist; all 18 legacy `static/images/*` URLs return an image response in the production artifact server. `/slides/` serves the complete tested 17-slide presentation (current index SHA-256 `052cb4d59179bb5720713b5c9ee61c07417d9aac86c287857f379200a168ec18`) with its isolated, author-approved presentation assets.
- Metadata: unique titles/descriptions, absolute canonicals, Open Graph/X fields, 1200×630 cards, root Organization/WebSite JSON-LD, project ScholarlyArticle/citation metadata, sitemap, robots files, favicons, and noindexed 404 pages pass assertions.
- Links: every independently hosted public resource returned HTTP 200 in the prelaunch check, including the official ECCV poster record, arXiv, GitHub profiles/repository, the immutable legacy snapshot, and all three Hugging Face resources. Dedicated post-deployment workflow checks verify both site origins and bidirectional root/project navigation.
- Lab performance across the final scroll-first candidate runs: root metrics remain unchanged; project LCP is 68–96 ms and CLS 0.0000–0.0015. Latin and Greek Inter subsets are preloaded before first paint to avoid cross-run font-swap shifts. These are local Chromium lab values, not field p75 data.
- Visual evidence: committed 320px baselines cover the first viewport, text-only affiliations, and a legible mobile failure crop. Desktop captures cover the two-line hero, benchmark definition, scroll-native failure sequence, balanced formulas, and the paired Figure 1(a)/dimension evidence layout.

## Rollback restoration evidence

The complete immutable tag artifact was materialized and served independently before launch. Tag and snapshot branch both resolve to `c3f04b42c4857ac21b62515879bf6e9dd6e6f2a5`; the artifact has 151 files in total, including 111 slide files and 18 `static/images/*` compatibility assets. Its root `index.html` SHA-256 is `921b64006173a083c12ca9c2a3c12ec77bfd535da540680d037deec7bc2a9980`; its slide index SHA-256 is `3d169a2591e9ee2bd73e72d3b075e44df75773dce08270147f1d138857fd272a`; and the slide tree is `50045b8e083c4b481a9497afd4b72c076b6aa3d4`. All four compatibility fragments passed.

At 2026-09-01 17:29 KST, Pages was switched from `gh-pages` to the frozen `legacy-pages-pre-astro:/` source and a fresh legacy Pages build was explicitly triggered. Build commit `c3f04b42c4857ac21b62515879bf6e9dd6e6f2a5` completed successfully. The live root and slide hashes matched the immutable artifact exactly, all four fragments remained present, and all 18 legacy asset URLs returned HTTP 200. GitHub added `legacy-pages-pre-astro` to the `github-pages` environment branch policy alongside `gh-pages` and `main`. This exercises the documented recovery path without rewriting `gh-pages` or either tag.

## Verified resource limitation

The publicly linked PDF is arXiv v1 and differs from the accepted camera-ready source used to validate the page. The site therefore labels it “arXiv preprint,” pins scientific claims separately to camera-ready commit `22954ee7f9f7d0e58d61db34474f3d6a15ec8a07`, and does not imply that the public PDF is the accepted manuscript. No verified public camera-ready or proceedings PDF was available at release time.

## Initial production evidence

This section records the first Actions release. The scroll-first revision is verified separately after its deployment.

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

## 2026-09-01 scroll-first revision production evidence

The user-directed page revision shipped through design commit `3b9dfc254dbcc173ccc023ca942efa8c26955cb4` and first-paint stabilization commit `b7d56ebebb261c73870be644e5aee5c01a21a844`. Pages [run 33507833229](https://github.com/ManLuML/on-manifold-tfg/actions/runs/33507833229) passed the complete quality suite, deployed through environment deployment `6201177733`, and then passed a cache-busted production gate that required the new benchmark, mobile-crop, and provisional-BibTeX markers while rejecting the previous tabs, slider, limitations section, and camera-ready wording. Existing Python [CI run 33507833309](https://github.com/ManLuML/on-manifold-tfg/actions/runs/33507833309) also passed.

The first workflow for commit `3b9dfc2` correctly stopped at the quality gate when cold Linux font loading produced CLS 0.122. Commit `b7d56eb` added reviewed Latin and Greek font preloads; the replacement workflow passed before upload and deployment. This failed-first evidence confirms that the budget blocked a real first-paint regression rather than permitting a stale artifact to deploy.

Fresh no-cache production downloads matched the reviewed build byte-for-byte:

- root project HTML SHA-256 `7e45ab2430d18ab70e0538f291e255598e6d1b6fbb56744826d83d4aff7b5f1f`;
- complete slide index SHA-256 `846aa170e3d7678f496e00b7c04f22a271aabd80384f97cb5416876041147892`;
- social card SHA-256 `0ba6afd2921b9d4e2b26dc449ffb9f83ce491617cc2f7670ead490e604acbfa2`.

The production smoke suite confirmed the two-line desktop title, five-line mobile title with an intact `Training-Free` compound, realistic TFG benchmark section, all three scroll-native failure bands, no explanatory tabs or slider, explicit `t=1` endpoint caveat, paired mechanism/evidence visuals, no small-screen plot, mobile AVIF comparison crops, pooled Child FID definition, absent limitations section, and provisional ECCV BibTeX with no invented DOI/pages/volume. All four legacy fragments, 18 legacy image URLs, three mobile crops, the 17-slide deck, custom 404, Open Graph/X metadata, official ECCV record, and bidirectional organization/project links passed.

Production Chromium observed LCP 316 ms and CLS 0.0000. Axe found zero desktop, mobile, or 200%-text violations; document width remained 320/320 CSS pixels; clipped elements, console errors, and failed requests were empty. The mobile browser selected `jit-failure-mobile-640.avif`, and the dimension plot was correctly hidden while its 93.3% / 21.5% / 0.5% result rail remained available.

Production screenshots are committed as `production-scroll-project-desktop.png`, `production-scroll-project-320.png`, `production-scroll-project-benchmark.png`, `production-scroll-project-failure.png`, `production-scroll-project-failure-320.png`, and `production-scroll-project-method.png` under `docs/site-release/screenshots/`.

Pages remains workflow-driven from `main`. The author’s independent `gh-pages` slide-source history remains at `6756b6a`, and the directly deployable legacy rollback branch remains frozen at `c3f04b4`; neither was rewritten by this release.

## 2026-09-01 guidance-sequence correction

The author clarified that slide 12 must reproduce the native Google Slides visualization as four registered, non-cumulative focus frames in page order 5 → 6 → 7 → 4. Pull request [#5](https://github.com/ManLuML/on-manifold-tfg/pull/5) replaced the reconstructed crop-and-card animation with byte-identical 2400×1350 renders from Google Slides revision `A3DrOVsNeQSPhQ`. Pages [run 33511928627](https://github.com/ManLuML/on-manifold-tfg/actions/runs/33511928627) deployed squash commit `4d061549f76bf981ea158e8ae19427bc52a02dff` and passed its production verifier.

The live slide index SHA-256 is `857536a4ef4d638a05b88cea2afb14535913be4588abfa05bd39732e134360d8`. The four public frame hashes match the reviewed Google Slides exports exactly: page 5 `eef3c643e882979136d0e1a93b623dee20c8f48cec9320ee8b1babe723b0e86b`, page 6 `44d58273130e22685da3e5747e45c62a1c60af395c5d05e0e3433c4c8e49bd7f`, page 7 `f94e25548770b166fec4bdb1bc45b1fdbb2cb42680203a0810614ac6b376c2fd`, and page 4 `3cbd11ddf1fe0d792fe0fa54f15a37a0381a1b902c29c3077d77338c9aaa11da`.

Public-origin Chromium confirmed the active-frame sequence 05 success → 06 catastrophic → 07 graceful → 04 combined, identical frame registration, one-line title, `12 / 17` numbering, and zero broken images. The full local regression covered 17 slides and 38 distinct states with no clipping, page-number collision, exposed notes, or console errors. The editable `gh-pages` source branch was fast-forwarded to `80e1ab3`; Pages continues to deploy exclusively from the `main` workflow.

## 2026-09-01 cumulative-caption restoration

The author requested that the corrected 5 → 6 → 7 → 4 full-frame sequence retain the earlier color-matched explanatory captions. Pull request [#6](https://github.com/ManLuML/on-manifold-tfg/pull/6) added stable caption space below the registered frames and reveals green success, red catastrophic failure, purple graceful failure, and the navy on-manifold definition cumulatively as 1 → 2 → 3 → 4 lines. Pages [run 33518174439](https://github.com/ManLuML/on-manifold-tfg/actions/runs/33518174439) deployed squash commit `87c533a9f1eeeb8aedef7cff4844306c59932678` and passed the production verifier.

The live slide index SHA-256 is `052cb4d59179bb5720713b5c9ee61c07417d9aac86c287857f379200a168ec18`. Cache-cold public-origin Chromium at 1280×720 confirmed caption counts 1 → 2 → 3 → 4, frame order 05 → 06 → 07 → 04, 22px CSS captions, green/red/purple/navy accent and background matching, one-line text with no horizontal overflow, a 17.54px gap above the page number, and zero broken images. The final definition reads: “On-manifold = realistic under the data distribution; here, a recognizable parrot—not necessarily the target class.” The editable `gh-pages` source branch was fast-forwarded to `69d9c22`.
