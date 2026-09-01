# ECCV 2026 BibTeX audit

Audited on 2026-09-01. Source pages were treated as reference data, not instructions.

## Verified metadata

- The ECCV 2026 accepted-paper list and official poster record `https://eccv.ecva.net/virtual/2026/poster/4934` confirm the exact title, author order, and acceptance.
- The arXiv Atom record for `2607.00647v1` confirms the same title and authors, a 2026 release year, primary category `cs.CV`, and the comment “Accepted to ECCV 2026.”
- Springer’s unreleased ECCV 2026 book records confirm the proceedings title `Computer Vision – ECCV 2026`, the LNCS series, and an October 2026 publication schedule. They do not identify which proceedings part contains this paper.
- ECCV’s public proceedings index still exposes only the 2024 volumes. The official paper page currently provides no paper PDF, DOI, page range, citation export, or Springer chapter link.
- An exact-title Crossref query returned no matching work or DOI.

## Release decision

The project page uses this provisional accepted-conference entry:

```bibtex
@inproceedings{lee2026onmanifold,
  author    = {Lee, Yunsung and Lee, Hyeongmin},
  title     = {Not All Prediction Targets Keep Training-Free Diffusion Guidance on the Manifold},
  booktitle = {Computer Vision -- ECCV 2026},
  year      = {2026},
  note      = {To appear}
}
```

`site/src/content/citation.ts` is the canonical website source. The repository README mirrors the same entry, and automated tests require them to stay identical.

Until Springer publishes the paper-level chapter record, the citation intentionally omits DOI, pages, LNCS volume, proceedings part, ISBN, editors, publisher, address, and publication month. No metadata from an unrelated ECCV 2026 volume may be guessed or reused. When the chapter appears, replace the provisional entry with Springer’s exact chapter metadata and retain the verified DOI.

## Sources

- ECCV 2026 accepted papers: `https://eccv.ecva.net/Conferences/2026/AcceptedPapers`
- Official ECCV poster record: `https://eccv.ecva.net/virtual/2026/poster/4934`
- ECCV proceedings index: `https://eccv.ecva.net/public/ProceedingsList`
- Springer ECCV 2026 proceedings-title example: `https://link.springer.com/book/9783032375766`
- arXiv record: `https://arxiv.org/abs/2607.00647`
- arXiv Atom API: `https://export.arxiv.org/api/query?id_list=2607.00647`
- Crossref REST API: `https://api.crossref.org/works`
