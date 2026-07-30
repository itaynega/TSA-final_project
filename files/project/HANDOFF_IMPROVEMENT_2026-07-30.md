# HANDOFF — Stage 2 (improvement) work stream, Amitay

### Paste this whole file at the start of a fresh conversation, with the repository folder mounted.

---

## 0. Your role

You are working with **Amitay** on the **Stage 2 improvement** of a university final project, in parallel
with **Itay**, who is running the Stage 1 reconstruction in a separate conversation.

**You own:** the improvement, `common/metrics.py`, the baseline, the leakage audit, and the report.
**Itay owns:** `common/data.py`, `common/split.py`, and the reconstruction run.
**Do not write into Itay's files.** One writer per file is a standing rule (D5) and this project has already
lost time to desynchronisation.

**Deadline: 10.08.** Five working days, numbered Day 1–Day 5.

---

## 1. Read these first, from the repository — not from anywhere else

The folder `Final Project\` *outside* the repo contains **stale copies** of several of these documents. A
previous session read those copies, gave advice from them, and had to withdraw it. Read only
`TSA-final_project\files\project\`.

| Order | File | Why |
|---|---|---|
| 1 | `Final_Project.pdf` | The assignment brief. The only external authority |
| 2 | `REQUIREMENTS_2026-07-30.md` | Every requirement quoted verbatim with IDs. **C1, C2, B7, B8, F4, F5 are yours** |
| 3 | `DISPATCHES_2026-07-30.md` | **The live document.** Every dispatch, its audit, and the decision log. It is ahead of the PLAN — where they disagree, DISPATCHES wins |
| 4 | `SELECTION_RUBRIC_2026-07-30.md` | Read **C6** and **G9** specifically. C6 is what a good Stage 2 looks like here |
| 5 | `COURSE_NOTATION_2026-07-30.md` | §2.1 metrics, §5 conventions, §6 baselines. This is how you cite the course |
| 6 | `AMENDMENTS_2026-07-30.md` | Four amendments Amitay approved, three open risks |
| 7 | `PLAN_2026-07-30_v2.md` | Behind the decision log. Read last, for structure only |

---

## 2. Settled. Do not reopen.

- **Paper: TQNet (ICML 2025), locked as D35.** `github.com/ACAT-SCUT/TQNet`, Apache-2.0. Selection ran across
  four scouting dispatches and a 2024–26 gap-fill. It is closed. If you find yourself evaluating whether
  TQNet was the right choice, stop — that is not this session's work.
- **Benchmark: ETTh1**, multivariate, `seq_len=96`. Target cell `pred_len=96` → **MSE 0.3712 / MAE 0.3928**.
- **Task type:** multivariate, supervised, deterministic point forecasting.
- **Evaluation protocol:** the paper's own — the forecast origin slides across the test split at stride 1
  with a fixed model. That is *rolling-origin evaluation*, which requirement B5 names as acceptable. **Do not
  add walk-forward as a second protocol**; it retrains per fold and would break comparability with the
  paper's numbers.

---

## 3. The sequencing problem — read this before you plan anything

**Decision D36 forbids the method-and-limitations analysis (dispatch #13) from recommending an improvement.**
That is deliberate: the analysis is meant to surface TQNet's real weaknesses *without* being steered toward a
conclusion, and the improvement is chosen afterwards, from that evidence.

**So: do not choose the improvement before #13 returns.** If #13 has not returned yet, the eligible parallel
work is §4. If it has, §5 is open.

Ask Amitay which state you are in before starting. Do not assume.

---

## 4. Work available now, regardless of #13

These are genuinely parallel — they block nothing and nothing blocks them.

**4a. `common/metrics.py`.** Implement MSE, MAE, RMSE, MdAE in the course's notation, using the course's
error convention `e_t = y_t − f(x_t)`.

- TQNet reports **MSE and MAE**, and both are course metrics, so **MAE discharges B7 (the paper's metric) and
  B8 (a metric studied in class) simultaneously.** Say so explicitly in the report.
- **Do not implement MAPE or SMAPE.** LTSF numbers are computed on z-score-normalised data, so the series
  crosses zero and both metrics divide by |yₜ|. `COURSE_NOTATION` §2.1 already flags MAPE as undefined at
  yₜ = 0. B8 is discharged by RMSE and MdAE, both cited to printed sl. 48 (PDF p. 46).
- The course writes MAPE and SMAPE **without ×100**. If either is ever added, match that convention.
- B7 also requires *prose explaining what the metric measures*, and its failure modes. Write that now.

**4b. Seasonal-naive baseline (B6).** Period 24 on hourly ETTh1. It is Tier-1 in `COURSE_NOTATION` §6 —
lectured *and* implemented in Assignment 2 — so it is citable.

**The trap:** it must be computed on the **z-scored scale, using the train-fitted scaler, on the identical
windows** as TQNet. A baseline on raw-scale data is not comparable to anything in the paper's table and the
error is invisible until the numbers look strange.

**4c. Environment.** Amitay runs the improvement (an ownership change from the earlier plan), so TQNet must
run on **his** machine, not only Itay's. Verify this now — Day 2, not Day 4. Repo README specifies
`conda create -n TQNet python=3.8`; Python 3.8 is EOL, so expect friction and record every deviation.

**4d. The pre-registration template.** Build the empty form now so that when the improvement is chosen there
is no delay: derivation, a **quantitative** prediction of direction and rough magnitude, pre-fixed
thresholds, a STOP condition, and what result would cause abandonment.

---

## 5. Once #13 has returned — designing the improvement

**What a strong Stage 2 looks like here** (rubric C6, weight 30 — the largest in the rubric):

> a one-run change on a **named component**, whose direction and rough magnitude can be **pre-registered
> before running**, and which is **ours to derive** rather than lifted from a follow-up paper.

"Named component" means: name the file and line, or the paper's equation number, and write what it becomes.
*"This could probably be improved"* is a fail.

**On the improvement axes.** The project has discussed a GPD / peaks-over-threshold fit and a Kalman /
state-space recursion. **These are examples, not gates.** This has been recorded twice as an error — two
dispatches were killed for turning them into hard constraints, and a later session repeated it. An
improvement outside both axes is eligible if it is well-derived. Do not let the axes narrow your search, and
do not reject a good idea because it fits neither.

**Constraints that are real:**

- **C2 is PASS/FAIL: the improved method must use the identical split and identical metrics as the
  reconstruction.** Import them from Itay's `common/`; do not build your own. Assert the split hash.
- **D10: no experiment before its pre-registration exists.** This is enforced.
- **B2 is PASS/FAIL** and names four stages — training, preprocessing, feature construction, hyperparameter
  tuning. If your improvement estimates anything from data, it estimates it from the **training split only**.
- One improvement. Not two, however good the first one's numbers look.
- The change should cost roughly **one training run**. A sweep cannot be debugged, run and written up inside
  the remaining time.

---

## 6. Coordination with Itay

- **Pull before you start, push when you stop.** Two people editing one markdown file at this deadline is a
  merge conflict nobody needs.
- You need three things from Itay before you can run anything: `common/data.py`, `common/split.py`, and the
  reconstruction's result files. Ask for the **split hash** as soon as it exists.
- **You audit his pipeline for leakage, not your own** — that separation is deliberate. Verify the scaler's
  fit window by inspecting the fitted parameters, not by reading the call site.
- Never commit datasets, checkpoints, or run outputs. See `.gitignore`.

---

## 7. Failure modes this project has already hit

Recorded so you don't repeat them.

1. **Reading the stale parent folder instead of the repo.** Cost: a withdrawn plan and a retracted
   assessment. Read from `TSA-final_project\files\project\`.
2. **Turning the user's examples into hard constraints.** Twice. See §5.
3. **Proposing an improvement from priors instead of from the method's actual code and limitations.** The
   idea sounded reasonable and was ineligible.
4. **Re-deciding settled questions.** Selection is closed.
5. **Numbers reported into a chat message instead of a result file.** Every number that reaches the report
   must be traceable to a file. A number that cannot be traced does not get printed.

---

## 8. What "done" looks like for this work stream

- `common/metrics.py`, frozen and shared with Itay.
- A baseline number on the same split and metrics.
- A pre-registration written **before** the improvement runs.
- The improvement implemented, run once, asserting the reconstruction's split hash.
- Report §F4 (what you changed and why) and §F5 — **a three-way table: paper / reconstruction / improved**,
  one row per horizon.
- A `.py` or notebook that is the Stage 2 code deliverable, separate from the reconstruction's.

---

## 9. First message back to Amitay

Before doing any work, report:

1. Which state we are in — has dispatch #13 returned?
2. What you read, and anything in those documents that contradicts anything else.
3. What you propose to do first, and why.

Then stop and wait. Do not start writing code in the same turn.
