# Selection — §4 applied, and a verdict on every registered prediction

**Job J-16. Written 2026-08-10. Sole writer of this file.**

This document does exactly two things: it applies `report/prereg-improvement.md` §4's selection
rule, and it returns a verdict on every registered prediction in that file's §3. It amends nothing.
No threshold, prediction or selection rule in the pre-registration is altered, extended or
reinterpreted to make an answer come out convenient.

---

## 0. Conventions used in this document, stated once

**Split labels.** Every measured quantity below carries a bracketed tag naming where it came from:

| Tag | Meaning |
|---|---|
| `[validation]` | Validation rows `[8544, 11520)`. §4.1's endpoint. |
| `[test]` | Test rows `[11424, 14400)`. §4.3's reporting split. Never a selection input. |
| `[training]` | Training rows `[0, 8640)`. Estimator and criterion statistics only. |
| `[config]` | Not a split quantity — parameter counts, hashes, flags read from a resolved config. |

`val_MSE ≈ 1.8 × test_MSE` at every horizon on this split (0.672 `[validation]` vs 0.373
`[test]` at H = 96, both means over seeds 2024/2025/2026). Any sentence mixing the two without
labels reads as an error, so none does.

**Note on the `[training]` and `[config]` tags.** J-16's acceptance criterion 2 asks for
`[validation]` or `[test]` with zero exceptions. Three registered predictions — Arm B's estimator
result, Arm D's criterion, Arm D's parameter count — are not split-split quantities at all, and
labelling a training-split ACF peak `[test]` would be false. They are tagged `[training]` and
`[config]` instead. Flagged rather than fudged.

**Mean convention.** All three-seed means are `statistics.mean` over the three seed values —
exact-rational accumulation, correctly rounded once at the end. This matters at the last bit:
`sum(x)/3` on the reconstruction's three validation values returns `0.6724990175677815`
`[validation]` and `math.fsum(x)/3` returns `0.6724990175677813` `[validation]`, while
`statistics.mean` returns `0.6724990175677814` `[validation]`, which is the value
`report/w_curve.md` prints and the value J-16's tripwire names. The same convention reproduces
Arm D's `0.6805545682661754` `[validation]` where `sum(x)/3` gives `…753` `[validation]`. Stated because a reader recomputing these means with a different accumulator will differ in
the sixteenth digit and should know why.

**Standard deviation convention.** Sample sd, n−1 in the denominator, per
`report/horizon_sigma.md`'s stated convention.

**Traceability.** Every file cited anywhere below appears in §5 with its sha256. Numbers not
traceable to a named result file are not printed — see §5.2 for the one place that bites.

---

## 0.1 ⚠ The seed-2024 sidecar on disk holds a superseded value — read before any number below

**In words, because a reader who misses this will compute the wrong selection.**
The file `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json`
(sha256 `ed715742bd1602e447e99fe1af2c9138f6c87a96e59911ab18bd9834124ea6e3`) currently holds
`val_MSE = 0.6869550701723053` `[validation]`. **That number is superseded. It is an epoch-3
artefact of a checkpoint overwritten during the §7j incident, not a measurement of the
reconstruction.** The sidecar is regenerated from whatever checkpoint sits on disk, and the
checkpoint sitting there is an early-stopping snapshot from an interrupted, `head`-truncated
training run — so the file is a faithful measurement of a broken model.

**The reconstruction's validation MSE at H = 96, seed 2024, is `0.6712632722155959`
`[validation]`.** Its sources:

| Source | Path | Evidence |
|---|---|---|
| Run log, both states | `results/validation/validation_metrics.log` | Logs `val_MSE=0.6712632722155959` for this setting before the incident and `0.6869550701723053` after — one file carrying both states of the same directory over time |
| Corrected curve row | `report/w_curve.md`, W = 24 row | Corrected by J-15d; the superseded value retained struck through |
| Independent retrain | `results/validation/ETTh1_96_96_reconstruction_v2_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` | `0.6712632718563285` `[validation]`, −3.6e-10 |
| Independent retrain | `results/validation/ETTh1_96_96_armB_auto_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` | `0.6712632724477633` `[validation]`, +2.3e-10 |

Corroborating the artefact reading: `results/validation/ETTh1_96_96_armB_smoke_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json`
holds `0.6869550703100449` `[validation]` — an independently `head`-truncated run of the same
setting, agreeing with the poisoned sidecar to 1.4e-10. Two runs truncated the same way landing on
the same value to ten significant figures is deterministic training stopped at the same epoch.
Three runs trained to completion agree with each other to under 4e-10; the truncated pair agree with
each other and disagree with all three by ~0.0157.

**Tripwire cleared.** The reconstruction three-seed validation mean at H = 96 computed in this
document is **`0.6724990175677814`** `[validation]`. Reading the poisoned sidecar instead would
give `0.6777296168866845` `[validation]`; that value does not appear anywhere in this document's
reasoning.

The epoch-3 checkpoint
(`TQNet/checkpoints/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024/checkpoint.pth`,
sha256 `c5d0f7bbc057d48608c15e60a4872712b363a5fa4c12238ba77fb16860773ca2`) is retained as evidence
for §7j / F6. It was hashed read-only for this document and not touched otherwise.

---

# Part 1 — §4's selection rule, applied

## 1.1 The primary endpoint, as §4.1 states it

> **Primary endpoint: validation MSE at H = 96, averaged over seeds 2024/2025/2026.**
> Validation, not test. Mean of three seeds, not a best-of.

Two constraints follow directly and are applied without exception below: the comparison is
`[validation]`, and it is a three-seed mean — an arm with fewer than three seeds at H = 96 has not
produced the endpoint quantity and cannot be ranked on it.

## 1.2 Every arm that reached measurement on that endpoint

| Arm | Reached the §4.1 endpoint? | Why |
|---|---|---|
| **A** — damped-trend instance norm | **No** | Abandoned at §3's registered gate. Only seed 2024 was run, at φ = 0.8 and φ = 1.0; φ = 0.9 and φ = 0.95 were never run and no φ was ever frozen. No three-seed mean exists at any φ. |
| **B** — estimate *W* from the training split | **Yes** | Estimator returned W = 24 `[training]`; three seeds evaluated at W = 24. |
| **C** — quantile head with pinball loss | **No** | Never implemented. Gate lapsed 2026-08-09 17:00; PLAN §6 forbids the unregistered late experiment. No checkpoint, no sidecar, no run record exists. |
| **D** — channel-count-conditional attention | **Yes** | Three seeds evaluated at H = 96. |

**Two arms reached measurement: B and D.**

### Arm B — mean validation MSE at H = 96

| Seed | val_MSE `[validation]` | Sidecar |
|---|---|---|
| 2024 | `0.6712632722155959` | **Not the on-disk sidecar** — see §0.1. Sourced from `results/validation/validation_metrics.log`, corroborated by `results/validation/ETTh1_96_96_reconstruction_v2_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` (`0.6712632718563285`) and `results/validation/ETTh1_96_96_armB_auto_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` (`0.6712632724477633`) |
| 2025 | `0.6735143635280876` | `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2025.json` |
| 2026 | `0.6727194169596606` | `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2026.json` |

**Mean = `0.6724990175677814` `[validation]`.** Three-seed sd (n−1) = `0.0011416150591374683`
`[validation]`.

**Why this row is also the reconstruction's row.** Arm B's deliverable at its estimated *W* is
`--cycle auto`, which resolved to 24 `[training]` and therefore produces the byte-identical setting
string `ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed<seed>` and an architecturally identical
model (`n_params = 661640` `[config]`, identical to the reconstruction). The W = 24 row was
therefore not retrained — it re-evaluates the reconstruction's three checkpoints. **Arm B's
three-seed mean at its estimated W and the reconstruction's three-seed mean are the same number, by
construction, not by coincidence.** The one run that exercised `--cycle auto` end-to-end
(`ETTh1_96_96_armB_auto…`, seed 2024) returns `0.6712632724477633` `[validation]`, +2.3e-10 from
the pre-incident anchor — an independent confirmation that the auto path builds the same model.

### Arm D — mean validation MSE at H = 96

| Seed | val_MSE `[validation]` | Sidecar |
|---|---|---|
| 2024 | `0.6795092456048932` | `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024_tq0ca0.json` |
| 2025 | `0.6812630824349228` | `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2025_tq0ca0.json` |
| 2026 | `0.6808913767587101` | `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2026_tq0ca0.json` |

**Mean = `0.6805545682661754` `[validation]`.** Three-seed sd (n−1) = `0.0009241568465767698`
`[validation]`.

All six sidecars above assert `split_hash = b66ee6b47e2b2eb8` with `split_hash_ok: true` `[config]`
(standing order 12).

### Arms that did not reach the endpoint — what exists instead

Recorded for completeness; neither enters the ranking.

**Arm A**, seed 2024 only, `[validation]`:

| Setting | val_MSE `[validation]` | Sidecar |
|---|---|---|
| φ = 0.8 | `0.6952801971213933` | `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024_dphi0.8.json` |
| φ = 1.0 | `0.9491608951697321` | `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024_dphi1.json` |

**Arm C**: no measurement of any kind exists.

## 1.3 §4.2 applied

> **The arm with the lowest mean validation MSE becomes the "improved" column of the F5 table.**
> Ties inside 0.0005 are broken by the pre-stated preference order **A > B > D > C**.

| Arm | Mean val_MSE at H = 96 `[validation]` | Rank |
|---|---|---|
| **B** | **`0.6724990175677814`** | **1 (lowest)** |
| D | `0.6805545682661754` | 2 |

**Arm B takes the "improved" column of F5.**

Margin, B under D: `0.008055550698394032` `[validation]`.

---

## 1.4 Ruling (a) — the selected arm changes nothing, and §4 requires that outcome

**The question.** Arm B at its estimated *W* is numerically the reconstruction. §4.2's "improved"
column therefore lands on an arm that alters no weight, no parameter count and no output. Does §4,
as frozen, require that? And does §4.2's tiebreak (`A > B > D > C`, ties inside 0.0005) apply, given
that A is abandoned and C is dead?

**Ruling, in three parts.**

**(a-i) §4.2 selects Arm B, and the fact that Arm B is numerically the reconstruction is not a
ground for setting the selection aside.** §4.2's operative clause is a total order on a single
measured quantity: *"the arm with the lowest mean validation MSE becomes the 'improved' column."* It
conditions on nothing else — not on the arm having changed the model, not on the margin being
material, not on the improvement being interesting. Arm B's mean, `0.6724990175677814`
`[validation]`, is the lowest among arms that reached the endpoint. The rule is applied as written.
§4's own preamble — *"fixed now, applied without discretion later"* — forecloses reading an
unstated "must differ from the baseline" precondition into §4.2 after seeing the result. Doing so
would be exactly the post-hoc rule-editing §4 exists to prevent.

**(a-ii) §4.2's tiebreak does not apply, because there is no tie.** The tiebreak clause is
conditional on its own face: *"Ties inside 0.0005 are broken by…"*. The margin between the two arms
that reached the endpoint is `0.008055550698394032` `[validation]`, which is **16.1×** the 0.0005
tie band. The condition is not met, so the clause is inert and the preference order
`A > B > D > C` is never consulted. **The status of A (abandoned) and C (dead) is therefore
irrelevant to this selection and does not need to be resolved.** This is worth stating plainly
because a tie *would* have forced the awkward question — whether an abandoned arm can win a
tiebreak it is ranked first in — and the measured margin means that question never arises. It is
not resolved here, and it does not need to be.

**(a-iii) Selection is not a win. §4.4 denies Arm B the "win" label, and the two findings are
consistent.** §4.4: *"A 'win' requires the improvement to beat the reconstruction by more than 1σ
on the metric and horizon its own §3 prediction named. Anything smaller is reported as 'no
measurable effect'."* Arm B's §3 prediction named **equality**, not improvement — *"MSE at estimated
W equals the reconstruction to within ±0.0005 (it is the same model)."* Arm B's measured difference
from the reconstruction is `0` by construction on the shared W = 24 checkpoints, and
`2.321674e-10` `[validation]` via the independent `--cycle auto` run — roughly seven orders of
magnitude below 1σ (`σ` = `0.0011416150591374683` `[validation]`; see ruling (c)). **Arm B is
selected under §4.2 and is simultaneously "no measurable effect" under §4.4.** Those are not in
tension: §4.2 ranks, §4.4 classifies, and the frozen text applies them independently.

**The awkward answer is the finding, and F6 carries it.** Stated without softening: *this study's
four-arm screen selected, as its improved column, an arm whose contribution is a procedure for
deriving a hyperparameter that was already correct — producing a model numerically identical to the
baseline it was screened against.* No workaround was invented, no arm was promoted on a secondary
criterion, and no threshold was relaxed. This is the pre-registration operating exactly as designed:
it registered in §4 that *"all four arms null … is a coherent, publishable-shaped result … it is not
a failed project"*, and the outcome it produced is that result's concrete form. **F6 carries this
paragraph.**

---

## 1.5 Ruling (b) — Arm D's test-split advantage did not enter the decision

**The facts, both labelled.** At H = 96, Arm D is **better** than the reconstruction on
`[test]` and **worse** on `[validation]`:

| Split | Reconstruction, 3-seed mean | Arm D, 3-seed mean | Δ (D − reconstruction) | Δ/σ |
|---|---|---|---|---|
| `[test]` | `0.3726664257448507` | `0.3719532075084556` | `−0.0007132182363950856` | **−0.331** (σ = `0.0021541981747125473` `[test]`) |
| `[validation]` | `0.6724990175677814` | `0.6805545682661754` | `+0.008055550698394032` | **+7.056** (σ = `0.0011416150591374683` `[validation]`) |

The two splits point in opposite directions. On `[test]` Arm D looks marginally preferable; on
`[validation]` it is decisively worse.

**Ruling. Selection is on validation, and Arm D loses.** §4.1 fixes the primary endpoint as
*"validation MSE at H = 96, averaged over seeds 2024/2025/2026. Validation, not test."* The
parenthetical is not decoration — it is the operative instruction, and §4.5 makes the prohibition
explicit and absolute: *"Selecting by test MSE is forbidden. If any post-hoc reasoning of the form
'arm X did better on test so let's feature it' appears, the study is invalid and this document is
the evidence of that."* Arm D's `[test]` result at H = 96 is precisely the form of reasoning §4.5
names. It is therefore excluded from the selection by the frozen text, not by judgment.

**Stated explicitly, as required: the test-split direction did not enter the decision.** Arm D was
ranked second on the `[validation]` endpoint alone, and would have been ranked second on that
endpoint if its `[test]` result had never been computed. The `[test]` figures appear in this
document only under §4.3 — *"test is then read once per arm per horizon, and every arm's test result
is reported"* — as disclosure after the fact, and in Part 2, where §3's per-horizon predictions are
adjudicated on the split those predictions were written against.

**A note on which split §3's Arm D prediction is read on, since this ruling could be misread as
settling it.** Arm D's registered prediction — *"MSE within ±1σ of the reconstruction at every
horizon"* — does not name a split. It is adjudicated in Part 2 on `[test]`, because §1's σ (the
quantity the prediction is denominated in) was measured on the test-split reconstruction runs, and
§4.3 makes `[test]` the split on which each arm's per-horizon result is reported. §4.1's
"validation, not test" is scoped by its own sentence to the *selection endpoint* and does not
retroactively redefine §3's predictions. **This is the one place in §4 where a careful reader could
land differently, and it is flagged in §5.3 rather than buried.**

---

## 1.6 Ruling (c) — which σ §4.4's "more than 1σ" means

**The question.** §4.4 parenthesises `σ = 0.00215`. That figure is `report/horizon_sigma.md`'s
**test-split** H = 96 MSE sample sd, `0.0021541981747125473` `[test]`. The reconstruction's
**validation** three-seed sd at H = 96 is `0.0011416150591374683` `[validation]`. §4.1 defines the
primary endpoint on validation. Which σ applies to a validation comparison?

**Ruling: a comparison on the validation split is judged against the validation σ,
`0.0011416150591374683` `[validation]`. §4.4's parenthesised `0.00215` `[test]` governs comparisons
made on the test split, and only those.**

**Justification, from §4's own text and §1.1's registered substitution rule.**

1. **§4.4's σ is a noise floor, and a noise floor is only meaningful in the units of the quantity it
   bounds.** §4.4's function is stated in its own final sentence: *"At σ = 0.00215, a 0.001 gain is
   not a gain."* It asks whether an observed difference exceeds run-to-run variation. Run-to-run
   variation is a property of a specific measurement on a specific split — here, `0.0021541981747125473`
   `[test]` and `0.0011416150591374683` `[validation]`, which differ by a factor of 1.89. Dividing a
   difference measured on validation by a spread measured on test compares two quantities that were
   never measured on the same rows. That is not a conservative choice or a liberal one; it is a
   category error, and §4.4's stated purpose cannot survive it.

2. **§4.1 fixes the endpoint quantity as validation MSE, and §4.4 is applied to that quantity.**
   §4.4 speaks of *"the improvement"* beating *"the reconstruction"* — the same comparison §4.1
   defines. Reading §4.4's σ as split-agnostic would make §4.1's explicit "Validation, not test"
   apply to the numerator and not the denominator of the same ratio.

3. **§1.1 registered in advance that measured quantities replace assumed ones.** Its words:
   *"the per-horizon σ measured in P3 replaces 0.00215 in every threshold below."* §1.1's registered
   principle is that `0.00215` is a placeholder standing in for a properly measured seed spread, to
   be superseded wherever a better-matched measurement exists. §1.1 wrote that rule for the horizon
   axis because the split axis was not anticipated — §4.1's validation endpoint only became
   executable after J-10b built the validation evaluator. Applying the same registered principle
   along the split axis is the reading consistent with §1.1's intent, and it substitutes a measured
   quantity for a mismatched one rather than inventing a threshold.

**Arm D's validation delta under both σ, re-derived here and not copied.**

Δ (Arm D − reconstruction), three-seed means, H = 96 `[validation]`
= `0.6805545682661754` − `0.6724990175677814` = **`0.008055550698394032`** `[validation]`.

| σ applied | Value | Source | Δ/σ |
|---|---|---|---|
| **σ_validation** (the ruling) | `0.0011416150591374683` `[validation]` | Reconstruction's own three-seed validation sd at H = 96, from the three sidecars in §1.2 | **+7.056276** |
| σ_test (§4.4's parenthesised figure) | `0.0021541981747125473` `[test]` | `report/horizon_sigma.md`, H = 96 MSE sd (n−1) | **+3.739466** |

Both re-derived from the per-seed values in §1.2 and `report/horizon_sigma.md`; both agree with the
dispatch's arithmetic to the precision it quoted (`+7.06σ` / `+3.74σ`).

**Does the choice change any ruling? No — and this is stated as a checked fact, not an assumption.**

- **The selection (§4.2) does not use σ at all.** §4.2 ranks raw means. Ruling (c) cannot move it.
- **Arm D's position under §4.4 is unchanged.** Under either σ, Arm D's validation delta is a
  *degradation* of far more than 1σ (+7.06 or +3.74), so Arm D cannot claim a win under §4.4 on
  either reading. The sign, not just the magnitude, is what settles it.
- **Arm B's position under §4.4 is unchanged.** Arm B's difference from the reconstruction is
  `2.321674e-10` `[validation]` at most. Against `0.0011416150591374683` that is 2.03e-7 σ; against
  `0.0021541981747125473` it is 1.08e-7 σ. Both are so far below 1σ that no threshold in the
  plausible range separates them.
- **The W-curve flatness verdict is unchanged in outcome, though not in margin.** Against 2σ_validation
  (`0.0022832301182749365`) the spread is 0.669 × 2σ — flat. Against 2σ_test (`0.004308396349425095`)
  it is 0.354 × 2σ — flat by a wider margin. Confirmed either way. The validation σ is the
  stricter test and is the one this document applies, which is the conservative direction for a
  prediction of a null.
- **No verdict in Part 2 flips.** Checked row by row; the two σ never straddle a threshold for any
  registered prediction.

**Recorded honestly:** the ruling above is the reading this document applies, and it is a reading —
§4.4 as frozen names one number and §4.1 names a different split, and the frozen text does not
reconcile them itself. What removes the risk is that the choice is **outcome-neutral across every
ruling and every verdict in this document**, as itemised above. Had it not been, this job would
have returned to the PM rather than choosing.

---

## 1.7 Recorded, not acted on — §7l item 4

`report/w_curve.md` and `STAGE2_WORKPLAN_2026-08-09.md` §7l item 4 leave open whether
`ETTh1_96_96_reconstruction_v2` formally supersedes the protected checkpoint as *the*
reconstruction; it differs from the anchor used throughout this document by `3.6e-10`
`[validation]` — immaterial at every precision printed here, and roughly six orders of magnitude
below σ — so this document proceeds on the pre-incident anchor and records that the decision is
Amitay's and is not Gate 4's.

---

# Part 2 — a verdict on every registered prediction in §3

**Permitted verdicts: `confirmed` · `falsified` · `unevaluable`. Nothing else.** No prediction below
is recorded as partial, mixed, or directionally-something. Where a registered prediction names four
horizons, it is adjudicated as four rows, because each horizon carries its own registered threshold
and its own measured σ.

## 2.1 The verdict table

| # | Prediction as written (§3) | What was measured — value + source path | Verdict | Justification |
|---|---|---|---|---|
| **A1** | Arm A, H = 96: ΔMSE vs reconstruction `0 ± 0.002` (null, explicitly); within 1σ | φ=0.8: `0.6952801971213933` `[validation]` (`results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024_dphi0.8.json`); φ=1.0: `0.9491608951697321` `[validation]` (`…_dphi1.json`); reconstruction seed 2024 `0.6712632722155959` `[validation]`. Δ = `+0.0240169249057974` and `+0.2778976229541362` `[validation]` | **falsified** | Δ at φ=0.8 is 12.0× the registered ±0.002 band and +21.04 σ_validation; at φ=1.0, 139× the band. Degradation, not null, at both φ actually run. |
| **A2** | Arm A, H = 192: `−0.002 or better` (~1σ) | No run exists at H = 192 for Arm A — no checkpoint, no sidecar, no run record | **unevaluable** | The arm was abandoned at §3's registered H=96 gate before any longer horizon ran. Cause: **arm not run**. Distinct from A3/A4 below. |
| **A3** | Arm A, H = 336: `−0.004 or better` (measured vs P3's σ) | Measured σ at H = 336 = `0.004789584272220202` `[test]` (`report/horizon_sigma.md`). No Arm A run at H = 336 | **unevaluable** | §1.1 registered that P3's **measured** σ replaces the assumed 0.00215 at every horizon ≠ 96. The measured σ (`0.004790`) **exceeds the predicted effect** (`0.004`), so the prediction is smaller than the noise floor it must clear and is untestable as written — §1.1's own registered consequence ("declared untestable"). Cause: **effect below measured σ**. |
| **A4** | Arm A, H = 720: `−0.008 or better` (measured vs P3's σ) | Measured σ at H = 720 = `0.022769078789010144` `[test]` (`report/horizon_sigma.md`). No Arm A run at H = 720 | **unevaluable** | Same registered cause as A3: measured σ (`0.022769`) is 2.85× the predicted effect (`0.008`). Untestable as written under §1.1. Cause: **effect below measured σ**. |
| **A5** | Arm A, shape: "no effect at short horizon, growing effect at long horizon — the *shape* is the prediction"; a uniform improvement would disconfirm the mechanism | One horizon of four measured (H = 96, `[validation]`, seed 2024 only); H = 192/336/720 never run | **unevaluable** | A shape across four horizons cannot be evaluated from one horizon. Not falsified-by-A1: A1 falsifies the H=96 point, while the shape claim is about the profile across horizons, which has no measurement. |
| **B1** | Arm B: estimator returns **W = 24** from ACF **and** periodogram independently | ACF largest local maximum = **24** (`ac[24] = 0.8851540915365546`); periodogram argmax = **24** (power `19641471.641798608`); `agree = True` `[training]`, rows `[0, 8640)`. `report/cycle_estimate.md`; independently reproduced in `w_curve.log` and recorded as `cycle_source: "estimated"` `[config]` in `results/validation/ETTh1_96_96_armB_auto_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` | **confirmed** | Both methods returned 24 independently on training rows only, under a strict integer-equality agreement rule that counts harmonics as disagreement. Exactly the registered claim. |
| **B2** | Arm B: MSE at estimated W equals the reconstruction to within **±0.0005** | `--cycle auto` run: `0.6712632724477633` `[validation]` (`results/validation/ETTh1_96_96_armB_auto_…seed2024.json`) vs reconstruction seed 2024 `0.6712632722155959` `[validation]`. Difference = `2.321674e-10` `[validation]` | **confirmed** | `2.32e-10` is inside the registered ±0.0005 band by ~6.5 orders of magnitude. Independently, the W = 24 three-seed row *is* the reconstruction's checkpoints re-evaluated, so the difference there is exactly 0 `[validation]`. |
| **B3** | Arm B: **the W-curve is flat — all eight values within ±2σ of each other** | Eight per-W three-seed means `[validation]`, max `0.672930702161326` (W = 12) − min `0.6714039036970284` (W = 6) = **`0.0015267984642975962`** `[validation]`; 2σ_validation = `0.0022832301182749365` → **0.6687 × 2σ**. All 24 sidecars listed in §5.1 | **confirmed** | The largest gap between any two of the eight means is 0.669 × 2σ, so every pair is inside ±2σ. Confirmed under σ_validation (the stricter test) and under σ_test (0.354 × 2σ) alike — see ruling (c). |
| **C1** | Arm C: MSE at H = 96 **degrades** by 0.002–0.008 (1–4σ) | No run, no checkpoint, no sidecar, no run record exists for Arm C | **unevaluable** | Arm C was never implemented; its go/no-go gate lapsed 2026-08-09 17:00 and PLAN §6 forbids the unregistered late experiment. Nothing was measured. |
| **C2** | Arm C: MAE at H = 96 **improves** by 0.001–0.004 | As C1 — no measurement exists | **unevaluable** | As C1. |
| **C3** | Arm C: MdAE improves, by more than MAE does | As C1 — no measurement exists | **unevaluable** | As C1. |
| **C4** | Arm C: 10–90 coverage 80% nominal; accept **72–88%** | As C1 — no measurement exists | **unevaluable** | As C1. |
| **D1** | Arm D: criterion fires **"drop"** on ETTh1 (C = 7) | `mean_abs_offdiag_pearson_correlation = 0.3110131176996824` vs threshold `0.3` `[training]`, rows `[0, 8640)` → `decision = drop (use_tq=0, channel_aggre=0)`. `armD_runs.log`; `channel_criterion_check.log` | **confirmed** | The criterion evaluated on training rows only and returned "drop", exactly as registered. The margin is 0.011 — narrow, and F6 should say so — but the registered prediction was that it fires, and it fired. |
| **D2** | Arm D: parameters **661,640 → 624,224 (−5.7%)** | `n_params = 661640` `[config]` (`results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json`) → `n_params = 624224` `[config]` (`…_seed2024_tq0ca0.json`), both from instantiated models. Difference = 37,416 = **5.6550%** | **confirmed** | Both absolute counts match the registered figures exactly. 5.6550% rounds to −5.7% at the registered precision. |
| **D3** | Arm D, H = 96: MSE within **±1σ** of the reconstruction | Reconstruction `0.3726664257448507` `[test]`; Arm D `0.3719532075084556` `[test]`; Δ = `−0.0007132182363950856` `[test]`; σ = `0.0021541981747125473` `[test]` → **Δ/σ = −0.331** | **confirmed** | \|Δ/σ\| = 0.331 < 1. Inside the registered band. (Direction is nominally favourable; §4.4 classifies anything under 1σ as "no measurable effect", which is the correct reading here.) |
| **D4** | Arm D, H = 192: MSE within **±1σ** of the reconstruction | Reconstruction `0.4305209998692137` `[test]`; Arm D `0.4278058235574245` `[test]`; Δ = `−0.0027151763117891914` `[test]`; σ = `0.0008376947093460351` `[test]` → **Δ/σ = −3.241** | **falsified** | See §2.2 — this row is written out in full because it must not be rounded. \|Δ/σ\| = 3.241 > 1, so the registered band is broken; the sign is negative, so it is broken by the smaller model being **better**. |
| **D5** | Arm D, H = 336: MSE within **±1σ** of the reconstruction | Reconstruction `0.47695683245890524` `[test]`; Arm D `0.47992154301875284` `[test]`; Δ = `+0.0029647105598475942` `[test]`; σ = `0.004789584272220202` `[test]` → **Δ/σ = +0.619** | **confirmed** | \|Δ/σ\| = 0.619 < 1. Inside the registered band. |
| **D6** | Arm D, H = 720: MSE within **±1σ** of the reconstruction | Reconstruction `0.5039997349691424` `[test]`; Arm D `0.5044445613048515` `[test]`; Δ = `+0.0004448263357090809` `[test]`; σ = `0.022769078789010144` `[test]` → **Δ/σ = +0.020** | **confirmed** | \|Δ/σ\| = 0.020 < 1. Inside the registered band, by the widest relative margin of the four horizons — though against the largest σ. |
| **D7** | Arm D: training wall-clock **≥ 10% faster** on the same CPU | Arm D wall-clock exists (`armD_wallclock.log`, 12 values, H=96 seed 2024 `258s` … H=720 seed 2026 `166s`). **No reconstruction wall-clock exists at any horizon, in any file.** No run record in `results/runs/` or `TQNet/results_armD/*/metrics.json` carries any duration field — the only time-like key is `timestamp` | **unevaluable** | A relative claim needs a baseline and no baseline was ever recorded. See §2.3 — this is a result about the study's instrumentation, and §1.1 registered in advance that measured quantities replace assumed ones. |

**Row count: 19 predictions — A×5, B×3, C×4, D×7.** Verdicts: **8 confirmed** (B1, B2, B3, D1, D2,
D3, D5, D6), **2 falsified** (A1, D4), **9 unevaluable** (A2, A3, A4, A5, C1, C2, C3, C4, D7).
8 + 2 + 9 = 19, matching the row count.

### Registered abandon conditions — not predictions, so no verdict vocabulary applies

§3 also registers an abandon condition per arm. These are gates, not falsifiable claims, so they are
recorded separately rather than forced into the confirmed/falsified/unevaluable vocabulary.

| Arm | Abandon condition as written | Fired? |
|---|---|---|
| A | "H = 96 degrades by more than 1σ" | **Fired.** `+0.0240169249057974` `[validation]` at φ = 0.8 = +21.04 σ_validation. Arm abandoned; J-12b/J-12c/J-13 cancelled. |
| B | "ACF and periodogram disagree on the dominant period" | Did not fire. Both returned 24 `[training]`. |
| C | "coverage falls outside 60–95%" | Never reached — arm not implemented. |
| D | "MSE degrades by more than 1σ at any horizon" | Did not fire on `[test]`, the split D3–D6 are adjudicated on (max degradation +0.619σ at H = 336). Note for the record: read on `[validation]` at H = 96 it would fire (+7.056 σ_validation), and this document does not re-adjudicate an in-flight gate after the fact. Flagged in §5.3. |

## 2.2 Arm D at H = 192, written as what it is

Arm D registered, for every horizon, that its MSE would sit **within ±1σ of the reconstruction —
"statistically indistinguishable."** At H = 192 it does not. The measured difference is
`−0.0027151763117891914` `[test]`, against a measured σ of `0.0008376947093460351` `[test]` — the
**smallest** of the four per-horizon σ — giving **Δ/σ = −3.241** `[test]`.

**The registered prediction is falsified, and the direction of the falsification is favourable: the
smaller model is better than the reconstruction by 3.24σ.** The prediction that broke was a
prediction of *no difference*, and what broke it was an *improvement*.

This is neither of the two summaries it will be tempting to compress it into. It is not the arm
succeeding — the arm predicted indistinguishability and did not get it, so a registered claim was
wrong. It is not the arm failing — the abandon condition is a *degradation* of more than 1σ, and the
measured effect runs the other way, so nothing was abandoned and no mechanism was shown to be doing
harm. **A pre-registered null broken in the favourable direction is a distinct and more interesting
outcome than either, precisely because it was registered in advance and could not have been claimed
after the fact.** F6 reports it in those terms.

Two qualifications, both stated rather than left for a reader to raise. The falsification is against
the smallest σ of the four horizons (`0.0008376947093460351` `[test]`, against `0.0021541981747125473`
at H = 96, `0.004789584272220202` at H = 336 and `0.022769078789010144` at H = 720 — σ is not
monotonic in the horizon on this split), so the same absolute Δ would clear 1σ at H = 192 and not at
any other horizon. And the effect is isolated: the other three horizons are inside ±1σ, so this is
one horizon of four, not a trend.

## 2.3 Arm D's efficiency prediction, and why "unevaluable" is the honest verdict

Arm D registered **"training wall-clock ≥ 10% faster on the same CPU."** That is a *relative* claim,
and it needs two measurements. Only one exists.

`armD_wallclock.log` records twelve Arm D wall-clock values as J-10 was instructed to measure them.
**No reconstruction wall-clock was ever recorded, at any horizon, in any file** — the twelve
reconstruction run records in `results/runs/`, the twelve Arm D run records, and the twelve
`TQNet/results_armD/*/metrics.json` files carry no duration field of any kind; the only time-like
key in the schema is `timestamp`. There is nothing to compute a percentage against.

The Arm D numbers that do exist are additionally incoherent as a compute measure: H = 96 averages
242s while H = 336 averages 98s `[config]`. A shorter horizon taking 2.5× longer reflects
early-stopping epoch counts and machine load, not the cost of the architecture.

**The verdict is `unevaluable`, and the cause is instrumentation, not data.** The prediction was
registered, the instrument to test it was never built, and §1.1 registered in advance that measured
quantities replace assumed ones — here there is no measured quantity to substitute in. **This is a
result about how the study was instrumented and it is reported as one**, not quietly dropped, not
estimated from epoch counts, and not rescued by a late reconstruction re-run whose wall-clock would
be measured hours later on a differently-loaded laptop and would not mean anything.

**Distinguished from Arm A's unevaluability, since both carry the same verdict for different
reasons.** A3 and A4 are unevaluable because the **effect the prediction named is smaller than the
σ measured for that horizon** — the quantity exists, the threshold is registered, and §1.1's
substitution of measured σ for assumed σ makes the prediction untestable as written. D7 is
unevaluable because **the measurement was never taken at all** — no threshold problem, no σ
problem, simply no baseline. A2 is a third cause again: the arm was abandoned before the horizon
ran. Same verdict, three different causes, and F6 should not collapse them.

## 2.4 Arm A overall — the abandonment is a measured outcome

Arm A was abandoned at §3's registered gate, and it matters that the abandonment came **after**
J-12d found and fixed a real origin bug (the level was carried from the window centre while the
damped trend projected from the window end). The arm was not abandoned on a broken implementation;
it was abandoned on a corrected one. Measured at H = 96, seed 2024, against the reconstruction's
seed-2024 value `0.6712632722155959` `[validation]`: φ = 0.8 gives `0.6952801971213933`
`[validation]`, a degradation of `+0.0240169249057974` = **+21.04 σ_validation**; φ = 1.0 gives
`0.9491608951697321` `[validation]`, `+0.2778976229541362` = **+243.43 σ_validation**. The registered
grid {0.8, 0.9, 0.95, 1.0} is monotone in this direction across the two values run, and never
contained a viable φ.

**Arm A's conclusion does not depend on which reconstruction anchor is used** (§7l item 4).
Recomputed against the epoch-3 value `0.6869550701723053` `[validation]` instead, φ = 0.8 is
**+7.29 σ_validation** and φ = 1.0 is **+229.68 σ_validation**; recomputed against
`reconstruction_v2`'s `0.6712632718563285` `[validation]`, they are **+21.04** and **+243.43** —
identical to the pre-incident anchor at the precision that matters. All three anchors clear the
1σ abandon threshold by between one and two orders of magnitude, so **the abandonment is robust to
the anchor question and nothing in §7l item 4 can reopen it.**

**This is a measured outcome, not a non-result.** The mechanism was implemented, verified against a
deliberate off-by-one that a weaker gate would have passed (§7h), corrected once a real bug was
found, measured, and found to degrade the model decisively. The pre-registration named the
condition under which it would stop, and it stopped on that condition. That is the
pre-registration working.

**One number withheld, per standing order T15′.** §-1 of `STAGE2_WORKPLAN_2026-08-09.md` and
`PM_HANDOFF_2026-08-10_1730.md` both state that the in-window linear fit explains a median **4.0%**
of variance with **λ̂ = −0.34** on the training split — the mechanism-level explanation for why the
slope anti-predicts. **Neither figure could be traced to any result file in this repository.** They
appear only in narrative prose; `damped_trend_measure.py` computes numerical-identity checks, not
this quantity, and no log, sidecar or run record contains either value. T15′ — *"a number that
cannot be traced to a named result file, by path and sha256, does not get printed"* — so they are
not printed as findings here. The qualitative claim they support (the in-window slope carries little
of the within-window variance and is mean-reverting, so projecting it is worse than projecting
nothing) is consistent with the measured degradation above, but the two figures need a result file
before F4/F6 may cite them. Raised to the PM in §5.3.

## 2.5 The flat W-curve and the ablation corroborate each other

§3's Arm B registered the mutual-reinforcement claim **in advance**, before either result was in:
*"It follows from the ablation: if the Temporal Query contributes nothing measurable at C = 7, then
nothing indexed by W can matter either, so W-sensitivity must vanish. If the curve is flat, two
independent results agree and each becomes much stronger."* Both halves now hold, and they hold on
different evidence. The ablation fires: Arm D's criterion drops both mechanisms `[training]` and the
resulting model is inside ±1σ of the reconstruction at three of four horizons `[test]` (D3, D5, D6),
its one departure being an *improvement* (D4) — removing the Temporal Query and the channel-attention
layer costs nothing measurable, and at H = 192 it helps. The curve is flat: eight values of W
spanning 6 to 168 — a 28-fold range straddling the true diurnal period — produce three-seed
validation means whose entire spread is `0.0015267984642975962` `[validation]`, or
**0.669 × 2σ** (2σ_validation = `0.0022832301182749365`), so every pair of the eight sits inside the
registered ±2σ band. **A flat curve plus a firing ablation is strictly stronger than either alone,
because each closes the other's most natural escape route.** A flat W-curve on its own invites the
reading that the sweep was insensitive — that W was never reaching the mechanism, or that the
mechanism was saturated, so of course nothing moved; the ablation refutes that by removing the
mechanism outright and measuring the same null, which is only possible if the mechanism genuinely
contributes nothing at C = 7. Conversely, a firing ablation on its own invites the reading that the
comparison was underpowered or that the single ablated configuration was unlucky; the curve refutes
that by showing the model is indifferent across a 28-fold range of the very hyperparameter that
indexes the ablated mechanism, which is a far denser sampling of the null than one on/off contrast
can provide. The two results are produced by different interventions — one varies the mechanism's
parameter, the other deletes the mechanism — and they cannot both be artefacts of the same
insensitivity. That the agreement was **predicted before either was measured**, and that the
alternative outcome was pre-declared as *"a more interesting finding still"* rather than a failure,
is what makes the conjunction evidence rather than a post-hoc pattern. **F6 uses this paragraph.**

---

# 5. Traceability

## 5.1 sha256 of every file cited in this document

**Documents read**

| File | sha256 |
|---|---|
| `report/prereg-improvement.md` | `d5ee1bd6130e21d8694f290e106d8f2c6a3b3b03983332de64e6d1bc6f70611f` |
| `report/horizon_sigma.md` | `b28c8f2376091cabb2cf5934bc6869f73cd8bd6d97730479cfcb8af30b706538` |
| `report/w_curve.md` | `0e320a97388ac59997b5389a349236bf42f29c4811417a1733b64c5ee1e56e89` |
| `report/cycle_estimate.md` | `34ed05b46a4775f37a3a9b67df1574602a7f74c31ebe636ae0bef89d1b0b235c` |

**Validation sidecars and logs — H = 96 selection endpoint**

| File | sha256 |
|---|---|
| `results/validation/validation_metrics.log` | `4af37f4fe4be4fcc01fb17e30a15db0b06c3b390fc764d22d205e0b4bb9b573d` |
| `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` **(superseded — see §0.1)** | `ed715742bd1602e447e99fe1af2c9138f6c87a96e59911ab18bd9834124ea6e3` |
| `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2025.json` | `9907d633e153b6d667b8f9f4007091e86b1160313256d64b21e2e738be650fbb` |
| `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2026.json` | `193f4d1976563ce99ca1d467596df85908e1a64c6309da24b2f41daa42c60094` |
| `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024_tq0ca0.json` | `5f1e3f69eb8628e31e6fc3a684c85839e7e4bd8be6aed942b52d9234cd7bd13d` |
| `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2025_tq0ca0.json` | `896415fcf2113915aa8a5a4662df2188ec0c2e300447c7ae07c51ca9b1e0e862` |
| `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2026_tq0ca0.json` | `8933efc4958712bc66041ce36a539bb35e6697c61172a806d1acaf8beb76b5b7` |
| `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024_dphi0.8.json` | `6d79c1e20bd7f3bf7fdf673c80ffef153280e785ffea40ac9d7367d1c24be75e` |
| `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024_dphi1.json` | `86b90be10d9287f75767fcd656f385448d13d5aa62aa1476df7e9183893b930f` |
| `results/validation/ETTh1_96_96_reconstruction_v2_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` | `577938c9d080f64327e18504ac3e46f45a4495cfb3b404c07f407b13e1ae430a` |
| `results/validation/ETTh1_96_96_armB_auto_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` | `23ec5927826b1e7c56cb183d59caa718b5a743bc4299253e646af92dec02c2ce` |
| `results/validation/ETTh1_96_96_armB_smoke_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024.json` | `7e8bd2d55eb35c520339e72e8437faa8331be43b97ace30f133fd9736fa8e980` |

**W-curve validation sidecars (B3)** — all `results/validation/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle<W>_seed<S>.json`

| W | seed 2024 | seed 2025 | seed 2026 |
|---|---|---|---|
| 6 | `2c60d00c88f4944a427c96427f2ad4c2874b981642624283c5f850f3fe748bf9` | `23a9f46536b9455c9e2a3a0818b1b2211b074dde238ae1d75d4fb478f1537dff` | `2e8ab6951c0bdce7867718644710228fab2ec410953b61a33ca9a254924f1979` |
| 8 | `402c99d0adef28914287ebea41c554e2dcae3b83f9c8aabe353e6f656d27150f` | `26c6f9084c63df42f480027f0fc961921fa91e677684e05c3ecccff1eaa8d05b` | `702f03b48d2768421c1b350abd1ef8381b25e6243bac23219ef90d8f59a91678` |
| 12 | `ad1eabf0fa04fcd5ec11d31a4ea0c7b920caea5abc948e38e352424f05694a0f` | `9e86323697c191360abe68b437681636d2f29086c70f5751801af60da0c67f1e` | `779d312c4d885d534a540ea3c0728bfe6cc1d0c053176a25ff199d6fb0ce0ec6` |
| 23 | `7548cf197169bff72c810177033311309a138f1f7e3276c55b41a4cd15415291` | `c0154d36baa9c5cf37390588e4563082f7be89d77824e2034867bce3accb6385` | `0673b759dd64840dea104228731f728d8f2d121eb02efbd2f39928598f19aa85` |
| 24 | *(the three `cycle24` sidecars above; seed 2024 sourced per §0.1)* | | |
| 25 | `420c90e89731f2a8c659c02b5a8ed36962e422c68abbdf431f7fa03eb03677b1` | `0e62e2e07948adc559878c5217f25d5c105f690c7e17db6df58125a60f19a0bc` | `805c18c8840cd97f8f5762bf753b580a8c245c2bc6e808ff6247753eb7b9966f` |
| 48 | `32a8870a3ec57c5649b0cb7bbab1ebb126bfc9050e32e341701c47750a161717` | `3b84e9023f37cb6d23bcb9a34c9e2352e549414a89964c339081c7b138c921fb` | `d1de62d4278ea706b4eafc3ad9d2e3e3f4f7a84894cf80fad5cc071990a216b9` |
| 168 | `ad4b47a8856ea746810f1a14ab0ebc8694c99bd5bcc344d1bc6acedac714b55c` | `32dc50bb03f8076acc20dc74f9990ee7a9000fec15148dc213753764c706dca7` | `5c40d68ca64c2751484c515f5add9446a1b5cd2a0fbb85f96e45521a2a63ae4d` |

**Test-split run records (D3–D6, §1.5)** — all under `results/runs/`

| File | sha256 |
|---|---|
| `reconstruction-TQNet-s2024-h96-1786272620501519500.json` | `ee3aadfbcd36c71fb4d6484b8d4f7fd24a115cc076dcd0a657bfba9491910ed2` |
| `reconstruction-TQNet-s2025-h96-1786272620971503500.json` | `cbe8e00571441c003962f479a3a259373e398338804969690f81f7524f2da33a` |
| `reconstruction-TQNet-s2026-h96-1786272621439125400.json` | `9b6c0f9a8016dcc284c660deec7c0306309ecb232d992421930ff7a2de723331` |
| `reconstruction-TQNet-s2024-h192-1786272608407577500.json` | `7be5729c11e440dc956420374a6fff4fdf8c9423079b6c98bc9257d6e26b5763` |
| `reconstruction-TQNet-s2025-h192-1786272609122571600.json` | `0dbe75b6d28a8620e7aa02a6cab75bfed70ef73d7810c20d3c6f269f4262eb61` |
| `reconstruction-TQNet-s2026-h192-1786272609983184400.json` | `8b203b106bef508dc5fe602b8953ed0a27af0e73b36ec6bb82f8fd2770a870a2` |
| `reconstruction-TQNet-s2024-h336-1786272610921132100.json` | `188087a7592fbc8c304ea75bb655c8392384cb212c68ae2a7755d3a63bb1e190` |
| `reconstruction-TQNet-s2025-h336-1786272612028778200.json` | `354dcda05065a76f6d56b500538ae8fa93415f4cd4c0d4c8956ccea97ac64b39` |
| `reconstruction-TQNet-s2026-h336-1786272613140900800.json` | `a9ddf148a4872504e7c611c131c520ad84107dfdcca6e372420218130d6efc3f` |
| `reconstruction-TQNet-s2024-h720-1786272614741150700.json` | `91657408166f067ac17981cda475b096661b62b4a4136a57a1c6085adf330656` |
| `reconstruction-TQNet-s2025-h720-1786272617013299400.json` | `7962bc6a564fe4eecae5036d242ae166008ae320f54843c6ca30b20bb845244b` |
| `reconstruction-TQNet-s2026-h720-1786272619212584300.json` | `78c935ec5c68a6af1c135ac32e190110c7aab2150886cd4a9ee3ec32aebff507` |
| `improved-TQNet-s2024-h96-1786297731786703500.json` | `a4d58b13c7128c4ea125cce5599cfaecf583edea7f10c37c42ab285bc54beb0a` |
| `improved-TQNet-s2025-h96-1786297732554574700.json` | `d4175c438439f667e402ee6cfae003abeb1f32f7ea9f7ec2ddb4cadf52b38c72` |
| `improved-TQNet-s2026-h96-1786297733835667500.json` | `916c65a688cde09e49c2ddb2567a00e8731077d39ea031410c32ff9e7d43301b` |
| `improved-TQNet-s2024-h192-1786297718824254300.json` | `a2d44d32e9411e1bbde9c8571bc42b74d9ccbaf95464c49af5458d26d601dd1d` |
| `improved-TQNet-s2025-h192-1786297719581725300.json` | `1600d177f47c568df2c284e91f68e9df99b9c56fa9ed116707ea9ccb3fe4f055` |
| `improved-TQNet-s2026-h192-1786297720360776900.json` | `f1949a204882108a59412b3ae8e96c04da94a88bc9ce603993457b83161b2892` |
| `improved-TQNet-s2024-h336-1786297721348461400.json` | `bcd1377b5f7ecbadfdcd9b3161243cae862b68216136b7c4a94a4a3a298ebaf8` |
| `improved-TQNet-s2025-h336-1786297722570589100.json` | `8109ee0c335a300de84517766e0307008490964b4edb55c6911d39d7f84cac79` |
| `improved-TQNet-s2026-h336-1786297723838791500.json` | `12647d65b980bd261fd5deb3fd3a9cf2702a55ee36d0ff8f8539cd0c163a71e6` |
| `improved-TQNet-s2024-h720-1786297725895635200.json` | `4d098c16a2cf3544f9dc04e16db55e9e7573b3c873e53f9193d41cd99fe5ae2a` |
| `improved-TQNet-s2025-h720-1786297728236759100.json` | `a94583a1c904b099c3cc737376ab0ad08de576ffbe1120d6c9a26a85772feecf` |
| `improved-TQNet-s2026-h720-1786297730459978500.json` | `d65f38c9874a853fe77d9b7d1c165261fc737bd10922399662998069a60af3d5` |

**Logs and the protected checkpoint**

| File | sha256 |
|---|---|
| `armD_runs.log` | `32a0ff132caba3705e0c19096e10917a06111814660b32c4cc0af8dda0241bc7` |
| `armD_wallclock.log` | `7ef94b9e64dcff638d8baffa9770c42b9e0cb5fd3ede0175fed439ffff8ec0d9` |
| `armA_phi.log` | `1915f800b34d973abf7ff7496724841afb161d3131844ab8b04cead6b5068227` |
| `w_curve.log` | `2909b511665bfa1e3865af807bfb8ef62a84cc2ed7a1bdc2ed9e1fabd0bd45d4` |
| `TQNet/checkpoints/ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024/checkpoint.pth` (epoch-3 artefact, retained as §7j evidence) | `c5d0f7bbc057d48608c15e60a4872712b363a5fa4c12238ba77fb16860773ca2` |

## 5.2 Numbers deliberately not printed

Per T15′, the following were encountered but are not printed as findings because no result file
carries them: the Arm A mechanism figures "median **4.0%** of within-window variance" and
"λ̂ = **−0.34**" (see §2.4). They appear only in `STAGE2_WORKPLAN_2026-08-09.md` §-1 and
`PM_HANDOFF_2026-08-10_1730.md`, both narrative documents.

## 5.3 Raised to the PM, not decided here

1. **`report/prereg-improvement.md` contains one `## Amendments` block, not two** (the dated
   `2026-08-09 — Arm D parameter-count correction` entry). J-16's dispatch instructed reading
   "both `## Amendments` blocks."
2. **Arm A's "4.0% / λ̂ = −0.34" have no result file.** F4/F6 may not cite them under T15′ until one
   exists. See §5.2.
3. **D7's stated cause needs one word of precision.** The dispatch says "no run record in this
   project ever carried a duration field" — true of `results/runs/*.json` and
   `TQNet/results_armD/*/metrics.json`, but `armD_wallclock.log` does hold twelve Arm D wall-clock
   values. The verdict is unchanged (`unevaluable`); the operative reason is the **absent
   reconstruction baseline**, not the absence of all timing data. See §2.3.
4. **`report/w_curve.md` prints 2σ one ulp high.** It states
   `2σ = 0.0022832301182749366`; `2 × 0.0011416150591374683` = **`0.0022832301182749365`** by every
   accumulation route tested (`sd*2`, `sd+sd`, `numpy`). Standing order 5 — the run wins — so this
   document uses `…365`. The difference is ~1e-19 relative and changes nothing; flagged only so the
   two files do not appear to disagree unexplained.
5. **The split on which §3's Arm D per-horizon MSE prediction is read.** Adjudicated here on
   `[test]`, justified in §1.5. On `[validation]` at H = 96 the same prediction would be falsified
   in the *unfavourable* direction and Arm D's abandon condition would read as fired. This document
   does not re-adjudicate an in-flight gate; if the PM disagrees with the split reading, D3 and the
   Arm D abandon row are the two entries that change.

---

*End of J-16. This document selected an arm and ruled on nineteen predictions. It wrote no other
file, ran no training, ran no writing git command, and executed no evaluator.*
