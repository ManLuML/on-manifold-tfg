# On-Manifold TFG ECCV Polish Plan

## Goal

Polish the project-owned ECCV 2026 video deck by using the official conference
identity, replacing the redundant paper-title transition with a visual
hypothesis slide, adding subtle page numbers, and restoring the original proof
slide.

## Requirements

- Use the official ECCV 2026 logo supplied from the ECCV website on slide 1.
- Preserve the title, author, affiliation, and project-link content.
- Replace the `ECCV 2026 · Our paper` transition with the paper's representative
  figure and a concise hypothesis/test framing grounded in the paper README.
- Replace generic `Our paper` eyebrow labels with experiment-specific labels.
- Add bottom-right current/total numbering to the full deck.
- Restore source section 19 from Hyeongmin Lee's authoritative exported deck,
  preserving every statement, proof, formula, and attribution.
- Keep all content inside the 1280 x 720 canvas after the deck grows to 17
  slides.

## Verification

1. Rebuild the public bundle from `slides-src/on-manifold-tfg.qmd`.
2. Compare the restored proof formulas and text against source section 19.
3. Inspect all 17 slides and every fragment state at the supplied Retina
   full-screen aspect ratio and at 1280 x 720.
4. Verify numbering, canonical URL, local asset closure, broken images, and
   zero public speaker notes.
5. Confirm a clean Git archive reproduces the committed public bundle.
