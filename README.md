# EvalLever — v1.0 (submission snapshot, rev211)

Artifacts backing **"Evaluation Contracts and Evaluation Leverage: Sliced-Gap
Readings in Structured Prediction Can Move Without Model Change"**
(Transactions of the ACL, submission 11952, submitted 2026-10-30, rev211).

This snapshot is **frozen as submitted**; post-review additions (r64 series,
revision-round statistics) will appear in v1.1 after the revision settles.

## Layout

| Directory | Contents |
|---|---|
| `evallever/` | The EvalLever Python package (leverage, audit, seedRx, CLI, selftest) |
| `code/` | Per-experiment scripts (r-series arms, inference+scoring, audits) |
| `statistics/` | Registry/statistics JSONs backing §4–§5 tables and figures (r-series) |
| `prereg/` | Preregistration: `预注册_v1_20261003.md` (frozen 2026-10-03, sha256-pinned), `预注册_登记.json` (**the amendment trail: 147 amendments, each timestamped with post-amendment sha256**), plus per-revision snapshots |
| `docunet/`, `docunet_large/`, `dreeam/`, `ttmre/`, `kddocre/` | Per-model frozen artifacts: per-document predictions, train/infer reports, configs |
| `ud_mech_pathb/` | UD-evolution mechanism probe (path-B frozen audits) |
| `chinese_analysis/` | Chinese-arm truncation audit |
| `adapter_acceptance/` | Cross-task adapter acceptance (BC5CDR, RAMS) artifacts |
| `human_verification/` | Blind human-verification samples (violation sampling, fresh-20 parse-failure diagnostics) |
| `theory/` | E5 report + Kitagawa decomposition backing §3 |
| `provenance/` | Selection-bias analysis |
| `paper/` | Reference list used at submission |
| `notes/` | Benchmarking notes |
| `数字对照表.md` | Crosswalk: paper numbers → artifact files |
| `MANIFEST.txt` | sha256 manifest of the 197 frozen files |
| `verify_gauge_synth.py` + output | Gauge-synthesis verification for §3 |

## Claim → artifact quick map

| Paper element | Backing artifacts |
|---|---|
| §4 57-check registry, 13 flips | `statistics/r1_rank_flip.json`, `statistics/r26_holm_flips.json` |
| §4.2 published-claims re-check | `statistics/r56_published_claims_audit*.json`, `docunet*/per_doc_*.json` |
| §4.3 violation surface | `statistics/p15_violation_cis.json`, `human_verification/*` |
| §4.4 symmetric TACRED place | `statistics/r59_n1_tacred_gauges.json`, `r59_n1b_*.json` |
| §5 AER canonical grids | `statistics/r60_canonical_aer.json`, `r59_e2_aer_advantage.json` |
| §5.1 judge grid | `code/p15_hacred_icl.py` (judge arm), `statistics/r54_multitask_judge.json` |
| §5 gate / ε anchoring | `statistics/r58_tost_null_bounds.json`, `r61_e14_anchor_sens.json` |
| §6 boundary (UD, gauge) | `ud_mech_pathb/`, `theory/`, `verify_gauge_synth.py` |
| Preregistered paradigm, 147 amendments | `prereg/预注册_登记.json` |
| Cross-task transfer | `adapter_acceptance/*` |

## Reproduction environment

- Python 3.13 (conda env), `sacrebleu 2.6.0`, `bert_score`, PyTorch 2.x (CUDA
  for judge/inference arms; CPU sufficient for registry/statistics layers)
- Entry points: `evallever/cli.py` (register → aer → audit → seed), self-test
  via `evallever/selftest.py`
- Bootstrap conventions as preregistered: doc-paired resampling, seed 20260926,
  R=2000 (registered) / R=1000 (exploratory, disclosed per artifact)

## Verification

`MANIFEST.txt` carries sha256 for the 197 frozen submission files (verified
2026-11-02). Four manifest entries are **deliberately not shipped** in this
repository, by design:

| Excluded | Reason |
|---|---|
| `internal_review/R1_methodology.md` | authors' private review-simulation |
| `internal_review/R2_taskarea.md` | authors' private review-simulation |
| `internal_review/修订清单.md` | authors' private review-simulation |
| `notes/TACL_benchmarking.md` | internal formatting memo, not an artifact |

Everything else matches the manifest byte-for-byte. Third-party code included
for reproducibility: `dreeam/official_code_YoumiMa-dreeam-476a700.tar.gz`
(DREEAM by Yixiao Ma et al., MIT license, pinned to upstream commit `476a700`).

## License

- Code: MIT (see `LICENSE`)
- Data/artifacts: CC BY 4.0

## Citation

Zenodo concept DOI: *to be minted at public release* (see
`RELEASE_CHECKLIST.md`). Until then, cite the paper.

## Contact

Corresponding author: Weixing Zhu (zwx@aeu.edu.cn). First author: Hao Lu
(luhao_0210@163.com).
