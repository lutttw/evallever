# r54 broadcast-layer sweep statement (E7.2)

Bug (self-discovered, disclosed in paper S5.1/r64/E-C2): the judge-grid
ensemble reading z-scores each judge's SYSTEM MEAN and broadcasts that value
to every segment, so segment-level bootstrap draws return identical system
means and the published interval [0.70, 0.70] was degenerate by construction.

Sweep rule: a statistics file is AFFECTED iff it reports a judge-grid
ensemble interval/coverage computed from per-segment z of broadcast system
means. Files reporting (a) segment metrics resampled from raw per-segment
scores, (b) DocRE document-level counts, (c) system-level point readings
without segment resampling, are UNAFFECTED (immune by construction).

## Sweep results (r54-r61 files in statistics/)

- r54_multitask_judge.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r55_aer_value.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r56_published_claims_audit.V1BUG.gold-axis-not-exercised.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r56_published_claims_audit.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r56_published_claims_audit_v2.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r56b_published_claims_perseed.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r56c_gamma_docre.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r56d_subset_covariates.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r57_error_decomposition.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r58_tost_null_bounds.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r59_aer_experiments.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r59_e1_para_refs.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r59_e2_aer_advantage.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r59_e3_ud_released.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r59_e5_gauge_break.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r59_n1_tacred_gauges.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r59_n1b_ci_excluding.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r59_n1b_retacred_subset.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r59_n2_mmluredux.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r59_n3_wmt_scorer_axis.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r59_n3b_full_matrix.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r59_n4_multitask_aer.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r60_VERDICT.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r60_VERDICT.txt: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r60_canonical_aer.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r61_e13_microf1_scope.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r61_e14_anchor_sens.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)
- r61_e16_refsets.json: unaffected (raw per-segment/document values; no judge-grid ensemble interval, no broadcast path)

## Affected statistics and their published replacements

- r54 judge-grid ensemble interval [0.70,0.70] (degenerate): superseded by
  the per-segment recount tau_b = .700, CI95 [.683,.750] (BCa [.667,.717]);
  the repaired values are the ones printed in the paper (S5.1) and in
  r64_ec2_judge_degenerate_ci.py/json, bit-exact revalidated.
- Sign-unstable pair recount: published 0 was an artifact of the same
  broadcast layer; repaired count = 9 (r64/E-A12), printed in the paper.

## Immunity argument for the remaining confirmatory-tier numbers

C1-C3 (confirmatory comparisons), the 57-check flip matrix (r1/r26), and
the AER validation (r59-r61) operate on document/segment-level raw metric
values (DocRE relation counts, WMT segment scores) with document-paired or
segment resampling - no system-mean broadcast anywhere in their paths
(verified: the broadcast pattern appears only in r64_ec2 and the E-C2
repair script; grep 'broadcast' over code/ returns only these files plus
an unrelated numpy broadcast_to in r64_ea20, which broadcasts a gold
matrix against per-token logits and is not a resample path).

Verdict: the bug's blast radius is exactly the judge-grid ensemble
interval and its derived sign-unstable recount, both already repaired in
the submitted manuscript with bit-exact revalidation; no other published
number shares the affected code path.
