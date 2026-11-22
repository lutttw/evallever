# Paper-number crosswalk (rev162, review material T1)

Purpose: respond to the review concern that "key released resources are not
verifiable at review time". Every headline number in the paper is mapped to
its frozen statistics file and field; each frozen file carries a sha256 (see
MANIFEST). Spot-check: pick any row, read the "file -> field" path in the
JSON, and the value must match the paper.

| Paper number | Frozen file (statistics/) | Field / location | Value |
|---|---|---|---|
| Headline flip +0.0363 [+0.0204,+0.0529] | r1_rank_flip.json | top5_robust_flips | 0.03628 |
| Intro gD diff +0.0309 / gR -0.0053 | r1_rank_flip.json | leverage_gold_swap components | 0.0309 / -0.0053 |
| Confirmatory C1 +0.036, p=.0006 | confirmatory_comparisons.json | C1 | 0.0355 / adj p .0006 |
| Confirmatory C2 +0.039, p=.0016 (F1) | confirmatory_comparisons.json | C2 | 0.0386 / adj p .0016 |
| C2 power 0.44 / MDE 0.031 | E4_report.json | power@0.02 / MDE | 0.4378 / 0.03108 |
| C3 +0.015, p=.011 (5/15 reversals) | confirmatory_comparisons.json | C3 | 0.0148 / adj p .011 |
| 57 checks / 13 flips / 11 CI-excl. | r1_rank_flip.json + r26_holm_flips.json | flip_matrix / est | 57/13/11 |
| Holm survivors 34/57, 10/11 | r26_holm_flips.json | survivors | 34 / 10 |
| Kendall tau in [-0.333,+1.000] | r1_rank_flip.json | kendall_tau | -0.3333 / 1.0000 |
| Density-arm rho +0.64 / within-family +0.68 | r1 + r28 recompute | spearman | 0.6364 / 0.6786 |
| Variance ordering rho 0.656 (p=.0045) | r21 + r28 | h3_pooled | 0.6559 / .0045 |
| Dedup cascade 0.588/0.493/0.359 | r28 | dedup cascade | 0.5876/0.4925/0.3586 |
| Convergence 0.991@n=8; exact track n*=5 | r2_seed_convergence.json | curve/trackB | 0.99130 / 8 / 5 |
| rho-hat=0.34, range 0.14-0.56 (5 seeds) | r2 | rho_cal_exact_pool | 0.33854 / 0.1359 / 0.5623 |
| Imputation validated <3% | r2 | implied vs exact | rel 2.6% |
| H1 0.867 (13/15); P(X>=13)=0.40 | r28 | h1_binomial | 0.398 |
| LOO 0.860 vs 0.772, p<0.002 | r37 | acc/majority/permutation | 0.8596 / 0.7719 / p=0 (500 draws) |
| Grouped arms 0.018-0.030 (8 generators) | r29 + r31/r32/r35 | per-generator arms | .0300/.0280/.0210/.0290/.0200/.0207/.0178/.0223 |
| Violation gate, 17-cell CIs (Table 4) | p15_violation_cis.json | 17 rows | positionally identical |
| Violation regression OR 0.43 / 1.02 | r41 | fixed/mixed | 0.4268 / 1.0205 |
| Positive control -0.004 [-0.006,-0.002] | r38 | oracle | -0.0039 |
| Fault injection 98.19% / Wilson 1.15% / 26/26 | stage45 acceptance record | detection/localization/integrity | 98.19% / 1.151% / 26 |
| X2 scoring decomposition +0.005/+0.022/+0.024 | r40_decomp | recall/F1/exact | +0.0054/+0.0217/+0.0240 |
| NER rescuable 6.4-7.8% | r36 | rescuable | 6.44% |
| Chinese inventory channel 14.4% -> 22.6% | r47 | relation-name component | 14.4 -> 22.56 |

Note: the rev79_test_population directory (no population replication) and
the UD-probe raw outputs live in the corresponding `out/` frozen-tree
subdirectories; the release MANIFEST covers all files named above.

## v1.0.1 additions (r48-r64, 2026-11-11)

Paper-cited evidence series added after the v1.0 freeze are mapped at
file level in the README "Claim -> artifact quick map" (Sec. 4.3 variance
cascade / Sec. 5 AER canonical+TOST+held-out+learned-metric / Sec. 6
r49-r52 + r59-series + r64 E-series). Per-number rows in the same format
as the table above ship with the accepted-version v1.1 freeze; v1.0.1
relies on the file-level mapping. Added-file sha256 values: MANIFEST.txt
second block.
