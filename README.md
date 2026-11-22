# EvalLever — v1.0.2 (Neurocomputing submission; canonical git copy)

> This repository is the canonical public copy of the evidence bundle.
> Archived snapshot DOI: 10.5281/zenodo.24442304
> v1.0.2 changes vs v1.0.1 zip: 124 files LF-normalized to match MANIFEST
> digests (CRLF packaging artifact; no content change), 2 post-manifest doc
> rehashes (数字对照表.md, amendment_summary_en.csv), 5 entries moved to an
> explicit excluded block, `.gitattributes` disables eol conversion so the
> sha256 verification stays byte-stable.

# EvalLever — v1.0.1 ([venue] submission snapshot, rev211 + r48–r64 additions)

Artifacts backing **"When Model Rankings Move Without Model Change:
A Preregistered Audit of Evaluation Contracts in Neural Prediction Pipelines"**
(Hao Lu, Weixing Zhu, Ruiyang Li; submitted to *[venue]*
(NLP track), 2026-11). A preprint of the earlier version is available as
arXiv:2611.04099; the revised version accompanies this submission (arXiv v2).

The v1.0 baseline (197 files, frozen 2026-10-30, [venue] submission [id])
is **byte-identical to tag `v1.0-submission`**. v1.0.1 **adds** the
paper-cited evidence series that postdate that freeze — the r48–r64
statistics, their paired analysis scripts, and two UD phase-4 reports —
without modifying any v1.0 file. `MANIFEST.txt` covers all 321 files in
three labeled blocks.

## Layout

| Directory | Contents |
|---|---|
| `evallever/` | The EvalLever Python package (leverage, audit, seedRx, CLI, selftest) |
| `code/` | Per-experiment scripts (r-series arms, inference+scoring, audits, incl. r64 E-series) |
| `statistics/` | Registry/statistics JSONs backing §4–§6 tables and figures (r1–r64) |
| `prereg/` | Preregistration: `预注册_v1_20261003.md` (frozen 2026-10-03, sha256-pinned), `预注册_登记.json` (**the amendment trail: 147 amendments, each timestamped with post-amendment sha256**), plus per-revision snapshots |
| `docunet/`, `docunet_large/`, `dreeam/`, `ttmre/`, `kddocre/` | Per-model frozen artifacts: per-document predictions, train/infer reports, configs |
| `ud_mech_pathb/` | UD-evolution mechanism probe (path-B frozen audits + phase-4 reports) |
| `chinese_analysis/` | Chinese-arm truncation audit |
| `adapter_acceptance/` | Cross-task adapter acceptance (BC5CDR, RAMS) artifacts |
| `human_verification/` | Blind human-verification samples (violation sampling, fresh-20 parse-failure diagnostics) |
| `theory/` | E5 report + Kitagawa decomposition backing §3 |
| `provenance/` | Selection-bias analysis |
| `paper/` | Reference list used at submission |
| `notes/` | Benchmarking notes |
| `数字对照表.md` | Crosswalk: paper numbers → artifact files |
| `MANIFEST.txt` | sha256 manifest: 197 v1.0 entries + 74 v1.0.1 additions + 1 per-seed verdict file (272 total) |
| `verify_gauge_synth.py` + output | Gauge-synthesis verification for §3 |
| `PRACTITIONER_GUIDE.md` | Two-page plain-language practitioner guide (protocol, adoption profile, vocabulary; no bundle identifiers) |

## Claim → artifact quick map

| Paper element | Backing artifacts |
|---|---|
| §4 57-check registry, 13 flips | `statistics/r1_rank_flip.json`, `statistics/r26_holm_flips.json` |
| §4.2 published-claims re-check (v2 + per-seed) | `statistics/r56_published_claims_audit_v2.json`, `r56b_published_claims_perseed.json` (+ `_v1` and the V1BUG marker), `docunet*/per_doc_*.json` |
| §4.3 violation surface | `statistics/p15_violation_cis.json`, `human_verification/*` |
| §4.3 variance–level ordering | `statistics/r57_error_decomposition.json`, `r48_nstar_rho_sensitivity.json` |
| §5 AER canonical grids (R=2000 re-freeze) | `statistics/r60_canonical_aer.json`, `r60_VERDICT.json`, `r59_e2_aer_advantage.json` |
| §5 TOST equivalence + ε anchoring | `statistics/r58_tost_null_bounds.json`, `r61_e14_anchor_sens.json`, `r64_ea6_tost_matrix.json`, `r64_ea6_epsilon_anchor.json`, `r64_ea6b_*.json` |
| §5 held-out coverage (newstest2021) | `statistics/r64_ea3_holdout_coverage.json` |
| §5 learned-metric sensitivity (MetricX/COMET) | `statistics/r64_ea4_metricx_ensemble.json`, `r64_ea4_comet_ensemble.json` (+ seg/src inputs) |
| §5 tie machinery on DocRE flips | `statistics/r64_ec1_docre_aer_coverage.json` |
| §5.1 judge grid | `code/p15_hacred_icl.py` (judge arm), `statistics/r54_multitask_judge.json` |
| §5.1 format/language tie layers | `statistics/r64_ea5a_format_tie_layer.json`, `r64_ea5b_language_tie_layer.json` |
| §6 cross-task families (r49–r52) | `statistics/r49_multitask_retrieval.json` … `r52_multitask_code.json` |
| §6 released-gold audits (TACRED/MCQA/UD/MT) | `statistics/r59_n1_tacred_gauges.json` (45-system run), `r59_n1b_*.json`, `r59_n2_mmluredux.json`, `r59_n3*.json`, `r59_e3_ud_released.json`, `r59_e1_para_refs.json` |
| §6 symmetric-flip audit (TACREV, 88/990) | `statistics/r64_ea1_symmetric_flips.json` |
| §6 AER transfer (retrieval/QA/code) | `statistics/r59_n4_multitask_aer.json`, `r53_aer_mt_validation.json`, `r55_aer_value.json` |
| §6 degenerate-CI + seed-variance (Paper-3 seeds) | `statistics/r64_ec2_judge_degenerate_ci.json`, `r64_ec3_seed_variance.json` |
| §6 boundary (UD, gauge) | `ud_mech_pathb/` (incl. `ud_mech_phase4_report.json`), `theory/`, `verify_gauge_synth.py` |
| Tie budget / mask (exploratory) | `statistics/r62_e21_tie_budget.json`, `r63_e33_tie_mask.json` |
| Preregistered paradigm, 147 amendments | `prereg/预注册_登记.json` |
| Cross-task transfer | `adapter_acceptance/*` |

## Reproduction environment

- Python 3.13 (conda env), `sacrebleu 2.6.0`, `bert_score`, PyTorch 2.x (CUDA
  for judge/inference arms; CPU sufficient for registry/statistics layers)
- Entry points: `evallever/cli.py` (register → aer → audit → seed), self-test
  via `evallever/selftest.py`. Note: the self-test reconciles against
  frozen statistics at their project-tree paths (`out/revision_v5_experiments/
  statistics/…`, mirroring the authors’ workspace); run it from a directory
  laid out accordingly, or use it as a read-along check of the receipts.
- Bootstrap conventions as preregistered: doc-paired resampling, seed 20260926,
  R=2000 (registered) / R=1000 (exploratory, disclosed per artifact)

## Verification

`MANIFEST.txt` carries sha256 for all shipped files in two blocks: the 197
v1.0 frozen files (verified 2026-11-02) and the 74 v1.0.1 additions
(generated 2026-11-11). Four v1.0 manifest entries are **deliberately not
shipped** in this repository, by design:

| Excluded | Reason |
|---|---|
| `internal_review/R1_methodology.md` | authors' private review-simulation |
| `internal_review/R2_taskarea.md` | authors' private review-simulation |
| `internal_review/修订清单.md` | authors' private review-simulation |
| `notes/[venue]_benchmarking.md` | internal formatting memo, not an artifact |

Everything else matches the manifest byte-for-byte. Conflict note: for
`statistics/r59_n1_tacred_gauges.json` two historical runs existed (35-system
2026-10-27 01:00 vs 45-system 19:26); the bundle ships the **45-system run**,
which is the one the submitted manuscript cites. Third-party code included
for reproducibility: `dreeam/official_code_YoumiMa-dreeam-476a700.tar.gz`
(DREEAM by Yixiao Ma et al., MIT license, pinned to upstream commit `476a700`).

## License

- Code: MIT (see `LICENSE`)
- Data/artifacts: CC BY 4.0
- Redistribution scope: this bundle contains re-scoring results, per-document counts, logs, and analysis scripts only; no benchmark text or gold-standard data is redistributed (all golds are referenced by document hash).

## Citation

Archived snapshot DOI: 10.5281/zenodo.24442304 (see
`RELEASE_CHECKLIST.md`). 
## Contact

Corresponding author: Hao Lu (luhao_0210@163.com) ([email redacted]). First author: Hao Lu
([email redacted]).

## Artifact field semantics (2026-11-22)

- `statistics/r64_ea22_standin_battery.json`: per-cell booleans are
  side-neutral — `ci_excludes_zero` = either tail excludes zero;
  `kd_side_positive_significant` = CI above zero (the registered CL2
  verdict is stated KD-side); `sign_consistent_with_baseline` = the
  perturbed delta keeps the unperturbed cell's sign;
  `cl2_verdict_summary` aggregates the verdict per side. An earlier
  circulation of this file used `gD_ci_excludes_zero`/`verdict_survives`
  with KD-perspective-only semantics; this version supersedes it.
- `verify_counts.py`: single source of truth for every "N/N files"
  claim quoted from MANIFEST.txt in the manuscript or letters.

## Language and redaction notes (public release)

- The frozen evidence files (statistics JSONs, preregistration ledger and
  snapshots, analysis reports, code comments) were **authored in Chinese**
  during the study and are shipped as generated; numbers, identifiers, and
  file paths are language-neutral, and every number in the paper resolves to
  a JSON field via `number_crosswalk.md`. The preregistration ledger ships
  with an English one-line summary sheet (`prereg/amendment_summary_en.csv`),
  as disclosed in the paper.
- Author contact details are redacted (`[email redacted]`) and submission
  venue references are anonymized (`[venue]`/`[venue-2]`/`[id]`) in the
  public copies of the ledger and documents; redaction sites are marked
  in place.
- Human-facing documents in this repository (README, crosswalk, guides,
  verifier output) are in English.
