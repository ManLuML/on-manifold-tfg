# on-manifold-tfg project page

Static academic project page for *Not All Prediction Targets Keep Training-Free Diffusion Guidance on the Manifold* (ECCV 2026).

The landing page is static: `index.html` loads Bulma and MathJax from CDNs,
figures live under `static/images/`, and the published presentation bundle
lives under `slides/`.

## Deploy

Published as the `gh-pages` branch of [ManLuML/on-manifold-tfg](https://github.com/ManLuML/on-manifold-tfg). Push the contents of this directory to that branch and enable GitHub Pages (source: `gh-pages`, root). The site is then served at `https://manluml.github.io/on-manifold-tfg/`.

The ECCV 2026 presentation is served at `https://manluml.github.io/on-manifold-tfg/slides/`. This repository owns both the editable source under `slides-src/` and the note-free public bundle under `slides/`.

## Build the presentation

The validated toolchain is Quarto 1.8.27 with Python 3, ripgrep, and rsync.
Run:

```bash
bash scripts/build-slides.sh
```

The script renders `slides-src/on-manifold-tfg.qmd`, strips speaker notes,
updates the public bundle, and verifies every local asset reference. Private
speaker-note backups belong in the gitignored `.speaker-notes/` directory.

## TODO

- Add the ECCV proceedings Paper-PDF link once available (currently a `#` placeholder; the arXiv link is already live).
