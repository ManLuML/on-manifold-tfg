# On-Manifold TFG project site

The Astro source for `https://manluml.github.io/on-manifold-tfg/` lives in this isolated directory. It has its own Node dependency graph and does not change the Python package runtime.

## Commands

```bash
npm ci
npm run check
npm run build
npm run test:e2e
npm run check:production
```

`npm test` runs the production build and browser suite together. The build uses `site: "https://manluml.github.io"`, `base: "/on-manifold-tfg"`, and trailing slashes. Every internal asset URL is derived from the Astro base path.

`npm run check:production` is a post-deployment check. It requires both public sites to be live, then verifies the project canonical and bidirectional organization/project navigation.

## Content authority

Scientific content is frozen in `../docs/site-release/claim-table.md` against camera-ready paper commit `22954ee7f9f7d0e58d61db34474f3d6a15ec8a07`. Asset rights, transformations, and exclusions are recorded in `../docs/site-release/asset-provenance.md`.

## Progressive enhancement

The result comparison and recovery-error slider are the two explanatory interactions. All target panels, formulas, coefficient values, tables, prose, resources, and BibTeX are present in initial HTML. The copy button is a utility action and leaves selectable citation text available when clipboard access fails.

`public/slides/index.html` keeps a slide URL published on the legacy deployment recoverable while linking its immutable rollback tag. It does not republish the deck’s excluded assets and is not linked as a primary project resource.

### 2026-09-01 slide-publication addendum

The paragraph above records the initial Astro launch policy and is superseded for the slide route only. `public/slides/` now contains the complete 17-slide ECCV presentation and is published at `https://manluml.github.io/on-manifold-tfg/slides/`. The bundle is kept intact, including its MAUM.AI, SeoulTech, and ECCV logo artwork, quoted educational figures, presentation fonts, Reveal runtime, and supporting media.

This change does not alter the root project-page policy. The Astro page at `/on-manifold-tfg/` continues to render affiliations as text and must not import affiliation or venue logos. Logo-bearing material is permitted only within the independently authored `/slides/**` presentation subtree.

## Deployment and rollback

Pull requests run the same schema, type, build, accessibility, interaction, screenshot, link, and performance checks used by deployment. The default branch deploys through `.github/workflows/pages.yml`. The final legacy production is preserved at tag `legacy-pages-pre-astro-2026-09-01` and snapshot branch `legacy-pages-pre-astro`; the earlier rollback tag also remains intact. See `../docs/site-release/rollback.md`.
