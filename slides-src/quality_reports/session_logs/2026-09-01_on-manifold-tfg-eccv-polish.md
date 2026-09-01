# Session Log: On-Manifold TFG ECCV polish

**Date:** 2026-09-01
**Status:** COMPLETED LOCALLY

## Objective

Use the official ECCV identity on the title, replace the redundant paper-title
transition with a hypothesis-driven visual, improve the experiment labels, add
page numbering, and restore the proof slide requested by the presenter.

## Changes

- Replaced `ECCV 2026 · Poster` with the official ECCV 2026 Malmö SVG lockup.
- Rebuilt the former paper-title transition around Figure 1, the core
  hypothesis, and the two-part experimental test.
- Renamed the following eyebrows to `Strong-guidance stress test` and
  `Manifold-aware evaluation` and removed the inaccurate claim that only the
  prediction target differs between JiT and DiT.
- Restored source section 19 from Hyeongmin Lee's authoritative export after
  source section 18 and before the takeaway.
- Added bottom-right current/total numbering to all 17 slides.
- Updated the private gitignored speaker-note backup with the new transition
  anchor and a proof-slide narration block.

## Verification

| Check | Result | Status |
|---|---|---|
| Slide structure | 17 slides in the expected sequence | PASS |
| Proof fidelity | normalized DOM exactly matches source section 19, SHA-256 `90787926629aaa0c4ad84455eb2d06bb0b2df128522351581e170297db94aaf7` | PASS |
| Mathematics | 75 formulas total; 40 on the proof slide | PASS |
| Builds | 18 Reveal fragment groups preserved | PASS |
| Numbering | exact `1 / 17` through `17 / 17` | PASS |
| Geometry | minimum content margin 54.1 px; page-number collisions 0 | PASS |
| Assets | 62 local slide references resolve; broken images 0 | PASS |
| Privacy | public speaker-note elements and attributes 0 | PASS |

## Timing note

The restored deck has 17 physical slides and 18 fragment reveals: 35 visual
states in five minutes, or about 8.6 seconds per state. The proof slide remains
reference-heavy by design and carries no extra build steps.

## Deployment state

The validated changes are committed only after final clean-archive and browser
checks. No remote push is performed without a new deployment authorization.
