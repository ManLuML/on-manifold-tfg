# GitHub Pages rollback procedure

The legacy branch and its history remain untouched. Tag `legacy-pages-pre-astro-2026-09-01` permanently identifies the final legacy production commit, `c3f04b42c4857ac21b62515879bf6e9dd6e6f2a5`. Branch `legacy-pages-pre-astro` points directly to the same commit so GitHub Pages can deploy the snapshot without moving `gh-pages`. The earlier tag `legacy-pages-2026-09-01` at `702070226b69410cae8d97902d348db734b05064` also remains intact.

## Verified artifact test

Before switching production to Actions, materialize the tagged tree in a disposable directory, serve it locally, and verify:

- its `index.html` SHA-256 is `921b64006173a083c12ca9c2a3c12ec77bfd535da540680d037deec7bc2a9980`;
- the four compatibility fragments exist;
- all 18 `static/images/*` files are present; and
- the legacy `slides` tree is `50045b8e083c4b481a9497afd4b72c076b6aa3d4` with 111 files and slide-index SHA-256 `3d169a2591e9ee2bd73e72d3b075e44df75773dce08270147f1d138857fd272a`; and
- the artifact renders without relying on either dirty local page worktree.

The validation record captures the command output and timestamp.

## Production rollback

First require the deployable snapshot branch to remain frozen:

```bash
test "$(git ls-remote origin refs/heads/legacy-pages-pre-astro | cut -f1)" = "c3f04b42c4857ac21b62515879bf6e9dd6e6f2a5"
```

If this assertion fails, recreate a new snapshot branch from the immutable tag rather than force-updating any existing branch. An administrator can then restore the frozen legacy deployment:

```bash
gh api --method PUT repos/ManLuML/on-manifold-tfg/pages \
  -f build_type=legacy \
  -f 'source[branch]=legacy-pages-pre-astro' \
  -f 'source[path]=/'
```

After the Pages deployment succeeds, verify the canonical URL, the recorded legacy `index.html` hash, the four fragments, all 18 assets, and the legacy slide hash.

## Restore the Actions release

```bash
gh api --method PUT repos/ManLuML/on-manifold-tfg/pages -f build_type=workflow
gh workflow run pages.yml --repo ManLuML/on-manifold-tfg --ref main
```

Wait for the workflow and Pages deployment to succeed, then rerun the production smoke suite. Do not delete or force-update `gh-pages`, either rollback tag, the snapshot branch, or the Actions deployment history during the rollback window.
