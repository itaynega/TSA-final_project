# Evaluation metrics

*Report material, discharging requirements B7 and B8. Drafted alongside `common/metrics.py`;
every formula below is the one that module implements.*

---

## Which metrics, and why these

We evaluate with four metrics: **MSE**, **MAE**, **RMSE** and **MdAE**. All four are implemented
once, in `common/metrics.py`, and imported by the reconstruction, the baseline and the improved
model alike — which is what makes requirement C2's "the same metrics as the reconstruction" a
property of the code rather than an assurance in prose.

Two requirements are in play and they are discharged as follows.

**B7 asks for the paper's own metric.** TQNet reports MSE and MAE, so both are here.

**B8 asks for at least one metric studied in class.** The course defines MSE, MAE, RMSE and MdAE
on slides 47–48 of `Time-Series Forecasting.pdf`. **MAE therefore satisfies B7 and B8
simultaneously** — it is the paper's metric *and* a course metric. We nonetheless report RMSE and
MdAE as well, so that B8 does not rest on a single coincidence, and because MdAE in particular
tells us something MAE cannot (below).

The course defines the forecast error first, and we follow its notation exactly:

> If 𝑓(𝑥ₜ) is a prediction of the model for time step 𝑡, and the actual target value is 𝑦ₜ … the
> **forecast error** (also **prediction error or residual**) is the difference between the actual
> values of the target and the values our model predicts:
>
> **eₜ = yₜ − 𝑓(𝑥ₜ)**

and then, over N points:

| Metric | Formula, as the course writes it |
|---|---|
| Mean squared error | MSE = (1/N) Σ eₜ² |
| Mean absolute error | MAE = (1/N) Σ \|eₜ\| |
| Root mean squared error | RMSE = √MSE |
| Median absolute error | MdAE = median(\|eₜ\|) |

---

## What each metric measures, and where it misleads

**MSE — mean squared error.** The average squared error. Squaring means an error of 2 contributes
four times what an error of 1 does, so MSE is a statement about the *worst* parts of the forecast
more than the typical parts. It is minimised by the conditional mean, which is also the quantity a
model trained under an MSE loss is estimating — so reporting MSE on a model trained on MSE measures
the model on its own terms and flatters it slightly relative to any other criterion.

*Failure modes.* It is not in the units of the target — it is in units squared, so its magnitude has
no direct interpretation. It is dominated by a small number of badly-missed windows, which means two
models with very different typical behaviour can share an MSE. And it is scale-dependent: MSE on
z-scored data and MSE on the original series are different numbers and cannot be compared.

**MAE — mean absolute error.** The average size of the error, in the target's own units. Every
error counts in proportion to its size, so MAE describes the typical miss rather than the worst one.
It is minimised by the conditional median.

*Failure modes.* Scale-dependent, like MSE, so it cannot be compared across differently-scaled
series. It is insensitive to whether error is concentrated or spread out — a forecast that is
slightly wrong everywhere and one that is badly wrong occasionally can have the same MAE. And
because it is not differentiable at zero, a model trained on MSE and evaluated on MAE is being
judged by a criterion it was not optimising; that is not a defect, but it is worth stating, since
it is one reason a reconstruction can match a paper on one metric and not the other.

**RMSE — root mean squared error.** MSE returned to the units of the target by taking the square
root. It inherits MSE's sensitivity to large errors while being interpretable on the same axis as
the data.

*Failure modes.* All of MSE's, since it is a monotone transform of it — in particular, RMSE never
reorders two models that MSE already ordered, so it adds interpretability, not independent evidence.
RMSE ≥ MAE always, and the size of the gap is itself informative: a large gap means the error is
concentrated in a few windows.

**MdAE — median absolute error.** The median of the absolute errors: the error that half the points
beat. Robust by construction — moving a single prediction arbitrarily far cannot move it.

*Failure modes.* Robustness cuts both ways. A model that forecasts most windows well and fails
catastrophically on a few will have a good MdAE, and if those few windows are the ones that matter,
MdAE is actively misleading. It also discards information: it uses the ranking of the errors, not
their sizes. We report it *next to* MAE rather than instead of it, because **the gap between MAE and
MdAE is the finding** — if MAE is much the larger, the error is concentrated in a minority of
windows, and that is a fact about the model that neither metric states on its own.

---

## What we deliberately do not report

**MAPE and SMAPE are excluded**, and the exclusion is a decision rather than an oversight.

Both divide by |yₜ|. Long-horizon forecasting results — TQNet's included, and the whole benchmark
literature it is compared against — are computed on **z-score-normalised** data, so the series is
centred on zero and crosses it constantly. Every crossing is a near-zero denominator, and the metric
returns an arbitrarily large number that reflects the position of the zero crossing rather than the
quality of the forecast. The course's own metric table already records MAPE as *"undefined at
yₜ = 0"*; on this dataset that is not an edge case but the normal condition.

If a percentage error is ever wanted, it must be computed on the **original scale** and labelled as
such — and, following the course, written **without** a ×100 factor, as a fraction.

---

## Two conventions, stated once

Both of these produce plausible-looking numbers when they are wrong, which is why they are recorded
here rather than left implicit.

**Sign.** The course writes eₜ = yₜ − 𝑓(𝑥ₜ). TQNet's own evaluation code writes `pred - true`, the
opposite sign. All four metrics above square or take the modulus of the error, so the two
conventions give identical results; we use the course's, and the test suite pins the equivalence
rather than assuming it.

**Reduction.** Predictions for this task are shaped (windows × horizon × variables). The long-horizon
literature, TQNet included, averages over **every element at once** — a single flat mean, not a
per-window mean that is then averaged. For MSE, MAE and RMSE on a full rectangular array the two
give the same answer; for MdAE they do not, since a median of medians is not a median. We use the
flat reduction throughout, because that is the convention under which the paper's target numbers
(MSE 0.3712 / MAE 0.3928) were produced, and comparing against them under any other reduction would
be comparing two different quantities.

**A consequence worth stating in the results section:** because all numbers are computed on
z-scored data, they are dimensionless. They are comparable to the paper's table and to each other,
and they are *not* comparable to any error expressed in the units of the original series.

---

## Sources

- `Time-Series Forecasting.pdf`, printed sl. 47 (PDF p. 45) — the error term, MSE, MAE, RMSE.
- `Time-Series Forecasting.pdf`, printed sl. 48 (PDF p. 46) — MdAE, MAPE, SMAPE, NMSE, RMSLE.
  Both slides are images; the formulas were recovered by rendering the pages.
- `COURSE_NOTATION_2026-07-30.md` §2.1, which records the above verbatim.
- Implementation: `common/metrics.py`. Tests: `tests/test_metrics.py`.
