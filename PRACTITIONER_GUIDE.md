# EvalLever Practitioner Guide

A two-page, plain-language guide to the contract-audit protocol.
For the formal account, propositions, and full evidence, see the paper
("When Model Rankings Move Without Model Change", this bundle's
companion manuscript) and `MANIFEST.txt` for the artifact map.

---

## 1. The problem, in one line

A reported number ("our intra–inter gap is X", "we beat them by Y")
is not a property of the model alone: it is a joint property of the
model and the **evaluation contract** — the scoring function, gold
version, decision threshold, output format, and prompt language that
produced it. Switch one contract coordinate and the ranking of two
systems can reverse, with neither model changed.

## 2. The protocol (freeze → audit → report)

1. **Freeze.** Before comparing, write down the contract: scorer,
   gold version, threshold, format, language. Register the legal
   alternatives you could equally have chosen.
2. **Audit.** With predictions frozen, re-score under the declared
   alternatives. The `evallever` stack:
   - `register` — the contract set, before running;
   - `aer` — the marginal reading, with ties declared;
   - `audit` — violation scan and survivor-selection bound;
   - `seed` — the minimum seed count for this contrast.
3. **Report.** Edges, ties, and reversals as contract-indexed
   readings, with ties first-class — not a single number.

The audit step runs on CPU in seconds over frozen predictions.
(Prediction generation and contract-alternate construction are extra;
see §5 below.)

## 3. Will it work for my setting? (adoption profile)

| Your data situation | Expected coverage | Recommended action |
|---|---|---|
| Released gold alternates; flip incidence low (≤ ~9%) | Near-full tie coverage (88/88 TACRED; 14/14 WMT19 declared) | Marginalize with AER; promote on tier + gate pass |
| Released gold alternates; dense flips (~57% of pairs) | Coverage degrades (3/13) | Contract selection or normative-gold adjudication first; audit as diagnostic |
| Scorer / format alternates only | Format and language arms; near-zero expected for strong models | Report with ties |
| Single gold, no alternates | Leverage unmeasurable (B1) | Freeze-and-state; the protocol still disciplines reporting |
| LLM outputs | Violation scan flags most settings (16/17) | Triage to human review; the scan is a flag, not an automated block |

Flip incidence = the fraction of system pairs whose ordering changes
under some declared contract. It is computable from any frozen grid
before adopting AER, and it orders the coverage gradient.

## 4. Reading the outputs (vocabulary)

- **Reading** ρ(M, c): any real-valued number reported as capability
  evidence, taken under contract c.
- **Contract** c = (scorer, gold version, threshold, format, language).
- **Arm** |Δρ|: how far a reading moves under a contract switch.
- **Flip**: a rank reversal between two systems across contracts.
- **Tie**: a difference whose CI contains zero — declared, not hidden.
- **Violation**: an output failing structural grounding; the gate
  flags settings above 5% as unverifiable (survivor-selection bias is
  bounded, not ignored).
- **Evidence tiers**: T1 preregistered and audit-passed; T2 multi-seed
  unaudited; T3 single-seed. A T1 + gate-pass reading supports
  promotion; T2/T3 or flagged readings inform but do not ground a
  go/no-go.

## 5. Limits (read before relying on it)

- **Legal alternates required.** No released gold/scorer alternates →
  no leverage measurement. The freeze-and-state discipline still
  applies.
- **Coverage is grid-specific.** Full tie coverage on low-incidence
  grids; it degrades as flips get dense. Check your flip incidence
  first (§3 above).
- **The gate is a triage flag**, not an automated block. Flagged
  settings route to human review.
- **Seed prescriptions are contrast-specific** (derived from one
  contrast curve; imputation-conditional).
- **Benchmarks, not operational logs.** The evidence base is public
  benchmarks; a worked case on live operational logs is future work.

## 6. Where to look in the paper

- Protocol formalization: §3 and the propositions.
- The DocRE reversal example: §1 and Figure 1.
- AER construction and validation: §5.
- Boundary conditions (where it fails): §6, B1–B5.
- The audit protocol and toolkit: §7–§8.
- Adoption and cost: §9.
