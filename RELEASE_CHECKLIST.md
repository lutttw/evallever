# Release checklist (execute at acceptance — NOT before)

Plan A: GitHub-only release, main entry = repo; Zenodo DOI optional insurance
via release integration (decide at acceptance; TACL norm is GitHub-only —
185/380 papers in our 2026-11 survey).

1. [ ] After revision settles: refresh `statistics/` with revision-round files
       (r64 series + any recomputed registries), bump this snapshot to v1.1;
       keep v1.0 tag pointing at the as-submitted state.
2. [ ] Re-run sha256 check; regenerate `MANIFEST.txt` for the new file set
       (keep the original 197-entry block intact, append new section).
3. [ ] Scrub pass: no `internal_review/`, no reviewer-identifying text, no
       credentials/tokens in scripts.
4. [ ] Create GitHub repo `<org>/evallever`, push, tag `v1.0-submission`
       (and `v1.1-accepted` later).
5. [ ] GitHub Release for the tag; attach heavy artifacts exceeding repo
       limits as release assets (2 GB/file cap). **`ttmre/logits_evaluation.npy`
       (140 MB) is already untracked via `.gitignore` — GitHub hard-blocks
       >100 MB files — and MUST be attached to the Release.** Optional, if
       trimming repo weight: `docunet_large/logits_evaluation.npy` (70 MB,
       over the 50 MB advisory), `ud_mech_pathb/align/cs_dev.jsonl` (20 MB),
       `docunet_large/keys_evaluation.jsonl` (20 MB), `ttmre/keys_evaluation.jsonl`
       (18 MB) — these may stay tracked.
6. [ ] Optional: connect repo to Zenodo → mint concept DOI → add to README
       citation block; reference DOI in camera-ready FN1 replacement.
7. [ ] Camera-ready FN1: replace the "Upon acceptance we release..." promise
       with the fulfilled sentence (repo URL + optional DOI) — see
       REVISION_PREP_20261031.md release-plan notes.
8. [ ] arXiv v2 Comments field may add `code: github.com/<org>/evallever`.
