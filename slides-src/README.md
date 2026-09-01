# Presentation source

This directory is the canonical editable source for the ECCV 2026 presentation
*Not All Prediction Targets Keep Training-Free Diffusion Guidance on the
Manifold*.

- `on-manifold-tfg.qmd` contains the 17-slide deck.
- `on-manifold-tfg.css` contains the deck-local design and full-screen
  occupancy settings.
- `clean-academic.scss` is the pinned Reveal theme dependency.
- `on-manifold-tfg.deck.yml` records the audience, duration, language, and
  source premises.
- `quality_reports/` preserves the deck's plans, verification, and migration
  history.
- Figure files and their provenance manifest live in `../slides/assets/`.

Run `bash scripts/build-slides.sh` from the repository root to regenerate the
public bundle at `slides/`. The build was validated with Quarto 1.8.27 and also
uses Python 3, ripgrep, and rsync. It refuses source speaker notes and strips
any rendered note elements as a second safety check.

Speaker notes are private and must not be committed. The migrated local backup
is stored at `.speaker-notes/on-manifold-tfg.json`, which is gitignored.

The source was fully transferred from `alohays/paper2pr` after Paper2PR commit
`7c8bb70`. Paper2PR is no longer the source of truth for this presentation.
