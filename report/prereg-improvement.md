# Pre-registration — Stage 2 improvement

**Written 2026-08-09 by Amitay, before any Stage-2 run.**
Discharges **C1** (meaningful modification, justified) and **D10** (*"No experiment before its
pre-registration exists"*).

**This file is frozen once the first Stage-2 run starts.** Nothing below is edited afterwards.
Anything learned later goes in the report's F6 section, marked as post-hoc, or in a dated
`## Amendments` block at the foot of this file that says what changed and why — never by rewriting
a prediction.

Git commit this file **before** the first run. The commit timestamp is the evidence.

---

## 0. Why this is a four-arm screen and not one improvement

**PLAN §6 says "one improvement, no second improvement however good the first one's results look."**
This document registers **four**, and that needs defending rather than glossing over, because on its
face it contradicts a live decision.

The thing PLAN §6 protects against is **adding an arm after seeing results** — running one idea,
finding it null, quietly running a second, and reporting the survivor as if it had been the plan.
That is the failure mode, and it is a failure mode about *ordering*, not about *count*.

A screen in which **all four arms, all four predictions, and the selection rule are fixed in writing
before any of them runs** does not have that failure mode. Every arm's result is reportable whichever
way it falls, because every arm was declared. The count went from one to four for a legitimate
reason — a training run costs 33 seconds, so the marginal cost of a fourth arm is trivial while the
marginal cost of *choosing wrong on Day 1 with one day left* is the whole of Stage 2.

**But the trap this creates is real and must be named**, because a grader will look for it:

> With four arms and a noise floor of σ = 0.00215, picking whichever arm happens to score best **on
> the test split** is not a finding. It is a maximum over four noisy draws, and its expected value is
> better than any individual arm's true value even if all four arms are worthless. Selecting on test
> is also precisely the hyperparameter-tuning-on-future-data that **B2** — the brief's only PASS/FAIL
> requirement — forbids.

**§4 fixes the selection rule in advance, on validation only. That section is the load-bearing part
of this document.**

Reporting shape stays exactly as **F5** requires: a three-way table, paper / reconstruction /
**one** improved column. The other three arms are reported in **F6**, which requires "what did not
work" as graded content.

---

## 1. What is already known, and the bar every arm must clear

All from Stage 1. Sources in the right-hand column; every number is traceable per T15.

| Quantity | Value | Source |
|---|---|---|
| Reconstruction, ETTh1 L=96→H=96, seed 2024 | MSE **0.371050** / MAE **0.392724** | `results/runs/reconstruction-TQNet-s2024-h96-…json` |
| Authors' own run, same cell | MSE 0.3712166 / MAE 0.3928201 | `TQNet/result_authors_reference.txt` |
| Reproduction gap | −0.000167 = **0.17×** the paper's seed σ | `report/results.md` |
| Seasonal-naive (period 24) baseline | MSE **0.512225** | `results/runs/baseline-seasonal_naive_24-…json` |
| **Our seed sd at H=96, seeds 2024/2025/2026** | **σ = 0.002154 MSE**, 0.000375 MAE | `docs/03` §3.7 — **G2: run records not yet committed** |
| Paper's own seed σ at H=96 | 0.001 MSE | paper Table 9 |
| TQNet's claimed margin over CycleNet at H=96 | 0.004 MSE | paper Table 5 |
| Split fingerprint (L=H=96) | `b66ee6b47e2b2eb8` | `common/split.py`, `docs/04` §4.5 |
| Test windows / predictions | 2,785 / 1,871,520 | `docs/03` §3.1 |

**The bar is our σ = 0.00215, not the paper's 0.001.** Our noise floor is already half the size of
the entire effect TQNet is famous for.

**And the ablation bounds what this dataset can show.** On ETTh1, removing the Temporal Query costs
+0.000726 MSE (0.34σ) and removing the attention layer as well gives 0.370963 — **nominally better
than the full model** (`docs/03` §3.7). At C = 7 the attention map is 7×7 and neither mechanism is
measurable above run-to-run noise. Every prediction below is written with that in mind: **arms that
touch the TQ/attention mechanism are predicted null, and that prediction is the point.**

### 1.1 Prerequisite runs — these happen before any arm, and their results are not predictions

| # | Run | Why | Cost |
|---|---|---|---|
| P1 | Commit the leakage-audit artefacts (STATUS G1) | B2 evidence must exist | 1 command |
| P2 | Commit run records for the seed spread and the ablation (STATUS G2) | σ = 0.00215 and the ablation are the two numbers this document is measured against, and **neither currently has a committed record.** T15: a number that cannot be traced does not get printed | ~5 min |
| P3 | Reconstruction at H = 192 / 336 / 720, **3 seeds each** (STATUS G3) | Gives F5 four rows **and** gives us a per-horizon σ instead of extrapolating one | ~15 min |
| P4 | Ablation at 3 seeds per variant | Upgrades the project's most striking finding from single-seed to measured | ~5 min |

**P3 is not optional bookkeeping.** §3 predicts effects that grow with the horizon, and without a
measured σ at H = 336 and H = 720 those predictions cannot be tested. The paper's own σ rises from
0.001 at H=96 to **0.012 at H=720** (Table 9); ours has only ever been measured at H=96. Assuming a
constant σ across horizons would be wrong by an order of magnitude.

**Registered in advance:** the per-horizon σ measured in P3 replaces 0.00215 in every threshold
below, at every horizon other than 96. If P3 cannot be run, all H≠96 thresholds are declared
**untestable** and those results are reported descriptively with no verdict.

---

## 2. Frozen protocol — identical to the reconstruction (C2)

Non-negotiable, and enforced in code rather than promised in prose:

- **Split**: ETTh1, chronological 12/4/4 months, train `[0,8640)`, val `[8544,11520)`,
  test `[11424,14400)`. Every arm calls `common.results.assert_split_hash('b66ee6b47e2b2eb8')` at
  L=H=96. A run that does not assert is not a result.
- **Metrics**: MSE, MAE, RMSE, MdAE from `common/metrics.py`, unchanged, float64, flat reduction
  over all 1,871,520 elements, on **z-scored** data.
- **Protocol**: rolling-origin, stride 1, one fixed model, exactly as Stage 1.
- **Everything else** copied verbatim from `TQNet/scripts/TQNet/etth1.sh`: `d_model` 512,
  4 heads, attention dropout 0.5, output dropout 0.5, batch 256, Adam lr 1e-3, `type3` schedule,
  30 epochs, patience 5, instance norm on, `--cycle 24`.
- **Seeds**: 2024 / 2025 / 2026 for every arm at every horizon. **No arm is ever judged on one seed.**
- **Test is read once per arm per horizon**, after training completes. Never during.

**Anything not named above is unchanged. No hyperparameter of the original model is tuned by us**
(PLAN §6), so nothing in this study creates a new B2 tuning surface — with the single exception of
Arm A's damping φ, which is handled in §3.1.

---

## 3. The four arms

Each arm gives: the mechanism · the derivation (this is the C1 justification) · the code change ·
**the quantitative prediction** · the abandon condition.

---

### Arm A — Damped-trend instance normalisation

**Mechanism now.** TQNet subtracts each input window's mean and divides by its standard deviation,
then reverses that on the output (`TQNet.py:44-50, 75-77`). This removes the window's **level**. It
does not remove the window's **trend**.

**Derivation.** Instance normalisation is the model's *only* defence against non-stationarity
(`TQNET_BRIEF` §6 item 9). ETTh1 is a two-year sensor series with genuine drift. A window on a clear
slope is handed to the network with the slope intact, and the network must re-derive how to continue
it, every sample. That is cheap at H = 96 and expensive at H = 720, where 30 days are projected from
a 4-day window — and **H = 336 and 720 are exactly where TQNet loses to TimeXer**, by 0.008 and 0.018
MSE respectively in the paper's own Table 5. Removing the first-order drift component before the
network sees it should free capacity for the periodic and cross-channel structure the model was
designed for.

Naively adding the fitted slope back is unstable: extrapolating a line fitted on 96 points out to 720
steps is 7.5× the fitting window. The classical remedy is **damping**, exactly as in damped-trend
exponential smoothing (damped Holt):

```
trend added back at step h  =  slope · Σ_{k=1..h} φ^k       0 < φ ≤ 1
```

φ = 1 recovers plain linear extrapolation; φ → 0 recovers the current model.

**Code change.** ~15 lines in `TQNet.forward`. Least-squares line per window per channel (closed
form on a fixed 0..95 index — no loop), subtract, then normalise the residual as now; on the output,
de-normalise as now and add the damped trend.

**φ is a hyperparameter, so B2 applies.** φ ∈ {0.8, 0.9, 0.95, 1.0}, chosen by **validation** MSE at
H = 96 only, then **frozen and reused unchanged at all four horizons**. Test is not consulted. This
choice is made once, is logged, and is the only tuning that happens anywhere in this study.

**Prediction.**

| Horizon | Predicted ΔMSE vs reconstruction | In units of σ |
|---|---|---|
| 96 | 0 ± 0.002 (null, explicitly) | within 1σ |
| 192 | −0.002 or better | ~1σ |
| 336 | **−0.004 or better** | measured vs P3's σ |
| 720 | **−0.008 or better** | measured vs P3's σ |

The *shape* — no effect at short horizon, growing effect at long horizon — is the prediction. A
uniform improvement across all four horizons would **disconfirm** the stated mechanism even if the
numbers look good, and must be reported as such rather than claimed as a win.

**Abandon if:** H = 96 degrades by more than 1σ. That means the normalisation is implemented wrong,
not that the idea is wrong — debug before running the remaining horizons.

**Honesty note for the report.** Adaptive/learned normalisation for non-stationary forecasting is a
crowded area (Non-stationary Transformers, Dish-TS, SAN). We claim a justified modification *to this
model*, not novelty against that literature. This sentence goes in F4.

---

### Arm B — Estimate the period *W* from the training split

**Mechanism now.** `--cycle 24`. One integer, typed by a human, per dataset. The entire Temporal
Query is indexed by it.

**Derivation.** This is the paper's **own first conceded limitation**: *"TQNet heavily relies on the
inherent periodicity of the data to determine the hyperparameter W. This dependency may limit its
generalization to datasets without clear periodic patterns."* And misspecification is *actively
harmful*, not merely suboptimal — their Figure 6 shows W = 167 on Electricity scoring **worse than
using no Temporal Query at all**. Replacing a hand-set constant with a quantity estimated from data
is an **applicability** improvement in C1's exact sense, and estimating it on training rows only ties
it directly to **B2**.

**Code change.** ~40 lines: an estimator (ACF peak and periodogram argmax, computed on training rows
`[0,8640)` only, agreeing or the run fails loudly) plus a `--cycle auto` path.

**The deliverable is the curve, not the estimate.** On ETTh1 the estimator will almost certainly
return 24, making the model numerically identical to the reconstruction. So the reportable artefact
is **the W-sensitivity curve the authors never ran on any ETT dataset**: train at
W ∈ {6, 8, 12, 23, 24, 25, 48, 168} and plot MSE against W.

**Prediction.**

- Estimator returns **W = 24** from ACF and from periodogram independently.
- MSE at estimated W equals the reconstruction to within ±0.0005 (it is the same model).
- **The W-curve is flat: all eight values within ±2σ of each other.**

That third prediction is the interesting one and it is a **prediction of a null**. It follows from
the ablation: if the Temporal Query contributes nothing measurable at C = 7, then *nothing indexed by
W can matter either*, so W-sensitivity must vanish. **If the curve is flat, two independent results
agree and each becomes much stronger. If the curve is NOT flat, the ablation conclusion is wrong and
that is a more interesting finding still.** Either outcome is reportable, which is why this arm is
the safe one.

**Abandon if:** ACF and periodogram disagree on the dominant period. Then the estimator is not
well-posed on this data and the arm is reported as "not attempted, and here is why", which is
legitimate F6 content.

**Blocker — clear before running this arm.** `TQNET_BRIEF` §8: **PTQNet** (*Periodic-temporal query
network for long-term multivariate time series forecasting*, Xun et al., *Information Processing &
Management* 63(7):104785, Apr 2026, doi `10.1016/j.ipm.2026.104785`) is a direct follow-up aimed at
exactly this limitation, and it is paywalled and unread. Get it through the BGU library and cite it.
**If it cannot be obtained by the time this arm runs, that fact is stated in F4 and F7 rather than
passed over in silence.**

---

### Arm C — Quantile output head with pinball loss

**Mechanism now.** The last layer is one `Linear(512 → H)` and the loss is `nn.MSELoss`. One number
per channel per future step. **No interval, no quantile, no variance — no uncertainty of any kind**
(`TQNET_BRIEF` §6 item 7).

**Derivation.** An operator deciding whether a transformer will overheat needs *how likely*, not
*how much*. A point forecast cannot answer that at any accuracy. This is an **applicability** gain in
the plainest sense: a capability the model does not have.

**Code change.** ~40 lines. `output_proj` emits `pred_len × 3` for quantiles {0.1, 0.5, 0.9}; loss
becomes the pinball loss summed over the three; **the 0.5 quantile is reported as the point forecast**
so MSE/MAE/RMSE/MdAE stay computable from `common/metrics.py` on the identical split — **C2 is
satisfied**. Interval coverage (fraction of true values inside the 10–90 band) is reported as a
*diagnostic alongside*, never as a substitute for a required metric.

**Prediction — and the direction is deliberately unflattering.** The pinball loss at τ = 0.5
optimises the conditional **median**; MSE optimises the conditional **mean**. `report/metrics.md`
already states this. So:

| Quantity | Prediction |
|---|---|
| MSE at H = 96 | **degrades** by 0.002–0.008 (1–4σ) |
| MAE at H = 96 | **improves** by 0.001–0.004 |
| MdAE | improves, by more than MAE does |
| 10–90 coverage | 80% nominal; accept **72–88%** |

**Registering that a required headline metric will get worse, and then observing it, is stronger
evidence of understanding than any accuracy win this cell can produce.** If MSE *improves*, the
implementation is suspect — check that the median head is being read and not the mean.

**Abandon if:** coverage falls outside 60–95%. That means the quantile head has not learned the
distribution at all and the capability claim is empty regardless of the MSE.

**Time-box.** This is the most code of the four arms. See §5.

---

### Arm D — Channel-count-conditional attention (efficiency)

**Mechanism now.** The Temporal Query and the channel-attention layer always run, at C = 7 and at
C = 883 alike. Attention over channels is quadratic in C, and the paper concedes the efficiency
claim is empirical and bounded to C < 1000.

**Derivation.** We *measured* that both are inert here: pure MLP scores 0.370963 versus the full
model's 0.371050, and all three variants sit inside 0.34σ (`docs/03` §3.7). Spending 37,248
parameters (5.6% of the model) and a quadratic operation on a mechanism contributing nothing
measurable is a real cost. The improvement is to make the model decide this **itself**, from a
statistic computed on the **training split only** — the off-diagonal mass of the train-split channel
correlation matrix — rather than from a human knowing the answer.

**Code change.** ~10 lines. The `--use_tq` / `--channel_aggre` flags already exist (Itay exposed
them for the ablation), so this is a criterion plus a dispatch.

**Prediction.**

- Criterion fires "drop" on ETTh1 (C = 7).
- Parameters **661,640 → 624,224** (−5.7%).
- MSE within ±1σ of the reconstruction at every horizon — i.e. **statistically indistinguishable**.
- Training wall-clock **≥ 10% faster** on the same CPU.

**Abandon if:** MSE degrades by more than 1σ at any horizon. Then the mechanism is doing something
at C = 7 after all, the ablation was a single-seed artefact, and P4 becomes the more interesting
result.

**Known objection, stated here so F4 states it too.** This is an improvement **by deletion**, and a
grader may read it as removing the paper's contribution. The brief names efficiency as a permitted
axis and the evidence is ours and measured, so the reading is defensible — but it must be argued in
F4, not assumed.

---

## 4. The selection rule — fixed now, applied without discretion later

**This section is the reason the four-arm design is legitimate. It is fixed before any run and is
not revisited.**

1. **Primary endpoint: validation MSE at H = 96, averaged over seeds 2024/2025/2026.**
   Validation, not test. Mean of three seeds, not a best-of.
2. **The arm with the lowest mean validation MSE becomes the "improved" column of the F5 table.**
   Ties inside 0.0005 are broken by the pre-stated preference order **A > B > D > C**, recorded here
   so the tiebreak is not a decision made later.
3. **Test is then read once per arm per horizon, and every arm's test result is reported** — in F5
   for the selected arm, in F6 for the other three. Nothing is run and discarded. Full disclosure of
   all four is what removes the multiplicity problem; hiding three would reintroduce it.
4. **A "win" requires the improvement to beat the reconstruction by more than 1σ on the metric and
   horizon its own §3 prediction named.** Anything smaller is reported as **"no measurable effect"** —
   not as a small improvement. At σ = 0.00215, a 0.001 gain is not a gain.
5. **Selecting by test MSE is forbidden.** If any post-hoc reasoning of the form "arm X did better on
   test so let's feature it" appears, the study is invalid and this document is the evidence of that.

**What would make us abandon the whole idea rather than one arm:** all four arms null on all four
horizons *and* the W-curve flat. That is a coherent, publishable-shaped result — *"on a 7-channel
dataset, this architecture's headline mechanism and four independent modifications to it are all
below the noise floor"* — and it is exactly the content **F6** asks for. It is not a failed project.
It is the project's actual finding, and the report should be written so that outcome needs no rescue.

---

## 5. Time-box and drop order — declared in advance

**Deadline 10.08 (PLAN.md), one working day.** Compute is not the constraint: 4 arms × 4 horizons ×
3 seeds ≈ 48 runs, well under two hours on CPU and runnable unattended. **Implementation time is the
constraint.**

Implementation order, fixed now, cheapest-per-unit-evidence first:

| Order | Arm | Est. code | If the clock runs out here |
|---|---|---|---|
| 1 | **D** — efficiency | ~10 lines (flags exist) | — |
| 2 | **A** — damped trend | ~15 lines | — |
| 3 | **B** — period estimation | ~40 lines | drop the 8-point W-curve, keep the estimator |
| 4 | **C** — quantile head | ~40 lines, 2 files | **drop the arm entirely** |

**Arm C is the declared drop.** If it is not implemented, F6 says *"Arm C was pre-registered, was
last in a pre-declared implementation order, and was not reached before the deadline"* — which is
honest and costs nothing. **A dropped arm that was declared droppable in advance is not a missing
result. An undeclared drop would be.**

**Minimum viable study, if everything goes wrong:** Arms D and A only, at H = 96 and H = 720, 3 seeds.
Twelve runs. Still a real F5 table and real F6 content.

---

## 6. Checklist to be satisfied before the first Stage-2 run

- [ ] STATUS **G5** closed — repo out of OneDrive (everything below writes to git)
- [ ] STATUS **G1** closed — `report/audit.md` + `results/audit.json` committed (B2 is PASS/FAIL)
- [ ] STATUS **G2** closed — run records for the seed spread and the ablation committed. **σ = 0.00215
      is quoted throughout this document and currently has no committed record**
- [ ] **P3** run — reconstruction at 192/336/720 × 3 seeds; per-horizon σ measured and recorded
- [ ] **P4** run — ablation at 3 seeds per variant
- [ ] **This file committed**, with its commit hash recorded in the report
- [ ] Every arm asserts split hash `b66ee6b47e2b2eb8`
- [ ] No arm reads the test split before training completes

---

## 7. What goes where in the report

| Section | Content from this study |
|---|---|
| **F4** — improved architecture, what changed and why | The selected arm's §3 derivation. Plus the two honesty notes: Arm A's crowded-literature caveat, Arm D's improvement-by-deletion objection |
| **F5** — 3-way table | paper / reconstruction / **selected arm**, one row per horizon (96/192/336/720), MSE and MAE, with σ stated per row so the reader can see what is and is not resolvable |
| **F6** — what worked, what did not, what we learned | The three non-selected arms with their pre-registered predictions beside their results. The W-curve. The 3-seed ablation. The paper/code divergences (`docs/04` §4.4). **And the methodological point: with σ = 0.00215 against a claimed effect of 0.004, this cell cannot resolve most of what we tried — which is itself the most useful thing the project measured** |
| **F7** — references | TQNet (Lin et al., ICML 2025); CycleNet; SparseTSF; TimeXer; iTransformer; Informer/ETDataset; **PTQNet if obtained**; the course decks cited per D17 (printed slide **and** PDF page) |

---

## Provenance

Every number in §1 was read from the file named beside it. Nothing in this document rests on a
Stage-2 run, because no Stage-2 run has happened. The two numbers this study is measured against —
σ = 0.002154 and the ablation triple — are quoted from `docs/03` §3.7 and are **flagged as lacking
committed run records (STATUS G2)**; §6 requires that gap closed before the first arm runs.
