# DISPATCHES — Final Project, Time-Series Analysis

Every dispatch, numbered, with its Acceptance line. Paste one into a **fresh conversation**; it starts cold.
Status updates are audited against the Acceptance line before the plan is updated.

| # | Task | Wave | Status |
|---|---|---|---|
| 1 | T0 — Course notation & vocabulary map | — | **ACCEPTED with one required fix** — see audit below |
| 2 | T0-b — Citation offsets + unread-image sweep | — | **DEFERRED to pre-report (D20).** Written, not sent. Selection does not need it |
| 3 | T1a — Scout: **forecasting** candidates | 1 | **ACCEPTED.** 22 considered / 5 scored / 9 rejected. Top: **DLinear 88.3** |
| 4 | T1b — Scout: **anomaly detection** candidates | 1 | **ACCEPTED.** 18 / 5 / 10. Top: telemanom 73.3 → **76.7** after amendment A3 |
| 5 | T1c — Scout: **classification** candidates | 1 | **ACCEPTED.** 24 / 3 / 11. Top: DeepConvLSTM 73.3. Found **M9** (B8 misquote) |
| 6 | T1d — Scout: **change-point detection** candidates | 1 | **ACCEPTED.** 20 / 5 / 10. Top: **changeforest 83.3** (cap binding, 93.3 uncapped) |
| ~~7~~ | ~~T1e — Red team on DLinear + changeforest~~ | ~~2~~ | **NEVER SENT — DEAD.** Both papers fail rubric v2: DLinear on **G8** (refuted by Toner & Darlow, ICML 2024), changeforest on **G9** (no sequential score for either improvement to attach to). The dispatch text is kept below as the template for the v2 red team |
| ~~8~~ | ~~T2a — probe, DLinear~~ | ~~3~~ | **DEAD with #7** |
| ~~9~~ | ~~T2b — probe, changeforest~~ | ~~3~~ | **DEAD with #7** |
| **10** | **T1f — Scout: where a GPD / extreme-value threshold is the missing piece** | **1B** | **Ready to send** |
| **11** | **T1g — Scout: where a Kalman / state-space component is the missing piece** | **1B** | **Ready to send** |
| 12 | T1h — Red team on the v2 top 2 | 2B | Blocked on #10/#11 + consolidation. Reuse #7's text, swap the papers |
| 13/14 | T2 — Clone-and-run probes | 3B | Blocked on #12 |

| ~~10~~ | ~~T1f — Scout: GPD axis~~ | ~~1B~~ | **NEVER SENT — DEAD.** I had turned one of the user's *examples* into a hard constraint after being told not to |
| ~~11~~ | ~~T1g — Scout: Kalman axis~~ | ~~1B~~ | **NEVER SENT — DEAD.** Same |
| — | 2024–26 gap-fill, run by the manager directly | — | **DONE** → `CANDIDATES_2024-26_2026-07-30.md`, 15 candidates, 3 parallel passes |
| **12** | **T2 — Clone-and-run probe: TQNet** | **2** | **Ready to send. BLOCKING — R1.** |
| **13** | **T3 — TQNet method + limitations analysis** | **2** | **Ready to send.** Parallel with #12 |

**PAPER LOCKED (D35): TQNet, ICML 2025. Send #12 and #13 together, in two fresh conversations.**
**#13 is explicitly forbidden from recommending an improvement (D36).**

---

# WAVE 2 — Paper locked. Probe it, then understand it.

## Dispatch #12 — T2: Clone-and-run probe, TQNet

> Paste below the line into a fresh conversation with the folder mounted. Nothing to fill in.

---

**Before anything else, read `WORKER_BRIEF_2026-07-30.md` in the mounted folder.**

**The decision this feeds**

The paper is locked and the whole project now rests on it. **The only question you answer is: does this code run on our machine, and does it produce the number the paper reports?** A NO-GO today costs a re-selection. A NO-GO discovered on Day 3 costs the project.

**Task**

Clone `https://github.com/ACAT-SCUT/TQNet` (Apache-2.0) and reproduce **one cell**: ETTh1 multivariate,
`seq_len=96`, `pred_len=96` → **MSE 0.3712 / MAE 0.3928**. Time everything. Return **GO** or **NO-GO**.

Do not reconstruct anything, do not improve anything, do not tune anything.

**What is already established — do not rediscover**

- The config is fully pinned in `scripts/TQNet/etth1.sh`, which I have read: `--cycle 24 --train_epochs 30 --patience 5 --dropout 0.5 --batch_size 256 --learning_rate 0.001 --random_seed 2024 --enc_in 7 --features M`.
- `sh run_main.sh` reproduces the whole main table, and **the repo ships the authors' own execution output at `./result.txt`.** Diff your numbers against that file — it is the cheapest and strongest check available, and it distinguishes "our environment differs" from "the paper's number is wrong".
- Environment per README: `conda create -n TQNet python=3.8` then `pip install -r requirements.txt`. **Python 3.8 is EOL**; expect friction and record every deviation.
- **Data.** The README points at a Google Drive bundle. **Google Drive folder listings are JS-rendered and have been unreadable from these sessions.** ETTh1 is also available directly and unbundled from `github.com/zhouhaoyi/ETDataset` at `ETT-small/ETTh1.csv` (2,589,657 bytes, hourly, 17,420 rows, 7 columns, **CC BY-ND 4.0**). Try the direct route first. If you use the Drive bundle, verify the ETTh1 file you get is byte-identical to the ETDataset one and say so.
- **Licence note for later:** CC BY-ND permits redistribution of the *unmodified* file but forbids derivatives — so we ship the raw CSV plus our preprocessing code, never a cleaned copy. Just record what you used; the packaging decision is not yours.

**Constraints**

- Work in the scratch/outputs directory. **Do not clone into the mounted `timeAnalysis` folder** — we ship patches, never vendored copies of licensed upstream code.
- Do not modify the mounted folder except to create your one output file.
- **Do not fix the science.** You may fix *environment* problems — a pin that will not install, a missing system package, a pandas API removed since 2025 — and you **must** record every one, because each is a line in the README deliverable and a threat to reproducibility.
- **Timebox: 2 hours.** If it is not running by then, that is a result. Report NO-GO with the wall you hit. Do not spend the day debugging someone else's environment.

**Method**

1. Record the hardware and software you actually have: GPU model, VRAM, driver, CUDA, Python. Not what the repo asks for — what is present.
2. Clone at a **pinned commit** and record the SHA. "Latest main" is not reproducible.
3. Build the environment. Record every deviation from `requirements.txt` and why it was needed.
4. Obtain ETTh1. Record source, byte size, and a checksum.
5. Run **only** the `pred_len=96` case from `scripts/TQNet/etth1.sh`. You do not need the other three horizons.
6. **Time it with a clock, not an estimate:** environment build, data acquisition, and the training run.
7. Compare your MSE/MAE against **0.3712 / 0.3928**, and separately against the corresponding line in the repo's shipped `result.txt`. Report all three numbers.
8. **Then run the identical command a second time with `--random_seed 2025`,** and report that MSE/MAE too. The paper reports a single seed; we need to know the run-to-run spread before we can claim any improvement is real. This is one extra short run and it is the most valuable thing you can hand us.

**Output**

Exactly one new file in the mounted folder: `PROBE_TQNet_2026-07-30.md`, **≤2 pages**, containing: hardware and software actually present; commit SHA; every environment deviation with its reason; data source and checksum; the three timings; **MSE/MAE for seed 2024 and seed 2025 against the paper's 0.3712/0.3928 and against `result.txt`**; and a one-word verdict **GO** or **NO-GO** with a one-sentence reason.

Do not copy the cloned repository into the mounted folder.

**Acceptance**

- [ ] Commit SHA recorded; clone is outside the mounted folder
- [ ] Actual hardware and software recorded, not the repo's requirements
- [ ] Every environment deviation listed with its reason
- [ ] ETTh1 source, byte size and checksum recorded; if the Drive bundle was used, byte-identity to ETDataset checked
- [ ] `pred_len=96` run completed for **seed 2024 and seed 2025**
- [ ] All three timings measured, not estimated
- [ ] Obtained MSE/MAE stated against **both** the paper's number and the repo's `result.txt`
- [ ] Verdict **GO** or **NO-GO**, one-sentence reason
- [ ] Timebox respected; if exceeded, NO-GO with the wall hit
- [ ] ≤2 pages · only `PROBE_TQNet_2026-07-30.md` created in the mounted folder

**Self-audit — required; the status update is rejected without it**

Head a section `SELF-AUDIT`. Plus, specifically:

- **A run that printed no error may have failed.** Check exit codes. This has now fired three times on this project — most recently a `curl` that returned exit 0 while printing `HTTP 403`. Confirm the output files exist and contain what you think.
- **A metric that matches the paper on the first try deserves suspicion, not celebration.** Confirm you evaluated on the split you think you did, and that you are not reading a number the repo shipped pre-computed.
- State plainly whether your verdict rests on a completed run or an extrapolation.

**Status update format:** as the other dispatches, **under one page.**

---

## Dispatch #13 — T3: TQNet method and limitations analysis

> Paste below the line into a fresh conversation with the folder mounted. Nothing to fill in.

---

**Before anything else, read `WORKER_BRIEF_2026-07-30.md` in the mounted folder.**

**The decision this feeds**

We have locked TQNet and must now **invent an improvement to it** — that is the graded half of the
assignment. **You are not choosing the improvement.** You are building the understanding from which we will
choose it. Your output has two jobs: it is the raw material for report sections F1 and F2, and it is the
limitations inventory we will brainstorm against.

**Task**

Produce a rigorous analysis of **TQNet** — Lin, Chen, Wu, Qiu, Lin, *Temporal Query Network for Efficient
Multivariate Time Series Forecasting*, **ICML 2025, arXiv 2505.12917** — covering how it works, what it
reports, and **where it is weak**. Read the code as well as the paper; they will not agree everywhere.

**Context — the lineage matters, and it is unusual**

TQNet is the third paper in a series by the same group, and **the authors publish their own limitations
table** in the repo README:

| Model | Technique | Stated limitation |
|---|---|---|
| **SparseTSF** (ICML 2024 **Oral**) | Cross-Period Sparse Forecasting, <1k params | *"Hard to model long periods without extending the input length"* — **solved by CycleNet** |
| **CycleNet** (NeurIPS 2024 **Spotlight**) | Residual Cycle Forecasting with learnable periodic vectors | *"Fails in multivariate modeling"* — **solved by TQNet** |
| **TQNet** (ICML 2025) | Those same periodic vectors used as attention **queries**; K/V from raw input | *"Hard to scale to ultra-long look-back inputs due to low SNR in multivariate histories"* — **UNSOLVED** |

**That last row is a limitation the authors nominated themselves and have not addressed.** Read all three
papers, at least well enough to understand what each inherited from the last — the improvement we eventually
choose has to be one they did not already make.

**Constraints**

- **Read-only except your one output file.** Do not clone, do not run, do not train. A separate dispatch is
  probing runnability in parallel; do not duplicate it.
- **Do NOT recommend an improvement, and do not rank the limitations by attractiveness.** Surface them
  comprehensively and neutrally. Choosing the improvement is explicitly reserved (decision D36), because
  choosing before understanding is the error that cost this project its first paper selection.
- Quote rather than paraphrase for anything that will be cited in the report.
- Where paper and code disagree, **report both and do not reconcile them.** That divergence is a finding.
- ≤6 pages. This one may be longer than recent dispatches because the report depends on it.

**Tooling — established, do not rediscover**

`api.github.com` returns empty bodies — use `raw.githubusercontent.com`. Google Drive listings are
JS-rendered and unreadable. **openreview.net is unreachable** from these sessions; ICML uses PMLR, so use
the PMLR proceedings page instead. A command that appears to return nothing may be failing with its
message swallowed — check exit codes.

**Method — five questions**

1. **How does it actually work?** The TQ mechanism: what exactly are the queries, how are the periodic
   vectors parameterised and shifted, what are K and V, how does the single attention layer combine with the
   MLP. Give the equations as the paper writes them, with their numbers.
2. **What are the training objective, the loss, and every hyperparameter?** Read `run.py`, the model file
   and `scripts/TQNet/*.sh`. Note especially anything hard-coded rather than exposed as a flag.
   Report the four quantities the assignment requires by name: **forecast horizon, sampling frequency,
   input window length, output window length.**
3. **What does it report?** The main table with the metrics and the baselines it is compared against, each
   number traceable to a table and page. Also: what the ablations show and, more usefully, **which
   ablations are absent.**
4. **Where is it weak?** The core of this task. Be exhaustive and neutral. Cover at minimum:
   - **assumptions** — what must be true of a series for TQNet to work, and when is it not true;
   - **hard-coded constants and hand-set hyperparameters** — `--cycle` is a single integer per dataset
     (24 for ETTh1); what else is fixed, and what does each one assume;
   - **what the method cannot represent** — e.g. multiple co-existing periods, non-integer periods, drifting
     or regime-switching periodicity, aperiodic series;
   - **what it does not output** — there is no uncertainty estimate of any kind; what else is missing;
   - **the authors' own conceded limitations**, quoted verbatim from the paper, README and any appendix;
   - **evaluation fragility** — the published run uses a **single seed (2024)**; what else about the
     protocol is brittle;
   - **the unsolved lineage limitation** — what "low SNR in multivariate histories at long look-back"
     actually means mechanically, and what would have to change to address it.
5. **What has the community said?** Forward citations, papers that build on or criticise TQNet/CycleNet/
   SparseTSF, GitHub issues. **Read `github.com/AIHNlab/NoChamps` and Brigato et al., *There are no
   Champions in Supervised Long-Term Time Series Forecasting* (TMLR, Jan 2026)** — it retrained ~5,000
   networks and argues LTSF rankings are not robust to setup changes. Establish what that implies for
   TQNet's reported margins specifically, and what evaluation protocol it recommends. This directly shapes
   how we will have to evaluate our improvement.

**Output**

Exactly one new file: `PAPER_ANALYSIS_TQNet_2026-07-30.md`, ≤6 pages, with sections matching the five
questions, plus a final section **"Limitations inventory"** — a flat, numbered, unranked list of every
weakness found, each with its evidence and its source. That list is the input to our brainstorm, so
completeness matters more than judgement, and **an entry you are unsure about should be included and marked
uncertain rather than omitted.**

**Acceptance**

- [ ] All five questions answered; equations given as the paper writes them, with equation numbers
- [ ] Training objective, loss and full hyperparameter list recovered **from the code**, with hard-coded values distinguished from exposed flags
- [ ] The four required quantities named explicitly: forecast horizon, sampling frequency, input window length, output window length
- [ ] Main-table numbers traceable to table and page; **absent ablations** identified, not just present ones
- [ ] All three lineage papers read to the level of what each inherited and what each left unsolved
- [ ] Authors' conceded limitations quoted **verbatim**
- [ ] Brigato et al. read, with its implication for TQNet's margins and for our evaluation protocol stated concretely
- [ ] Paper/code disagreements reported as such, unreconciled
- [ ] **Limitations inventory present, flat, numbered, unranked, each entry sourced**
- [ ] **No improvement recommended anywhere in the output**
- [ ] ≤6 pages · nothing cloned or run · only `PAPER_ANALYSIS_TQNet_2026-07-30.md` created

**Self-audit — required; the status update is rejected without it**

Head a section `SELF-AUDIT` covering what you got wrong in your own work, what you got wrong in your
tooling, anything recorded at lower confidence, and anything you could not verify — stated as unverified
rather than rounded up. **If you found few limitations, say explicitly whether that is a finding or a
failure of your search.**

**Status update format:** as the other dispatches, **under one page.**

### Wave 1 audit (manager, per handover §5 — re-measured, not taken on trust)

**Ruling: ACCEPT all four.** Every Acceptance line met. Arithmetic re-computed independently on the top five
candidates across all families — DLinear 88.3, Informer 85.0, changeforest 83.3, ClaSP 75.0, telemanom 73.3,
DeepConvLSTM 73.3 — **all six reproduce exactly.** B8's full text re-read from `Final_Project.pdf` p. 1
line 34, confirming Dispatch #5's escalation and my error M9.

**Three scouts raised criticisms of the rubric. All three ruled** (rubric v1.1 header, decision D28): A1
VALID, A2 NOT VALID as applied, A3 VALID. None changes the top two.

**What the scouts found that I would not have:** requirement G5 eliminates classification's entire state of
the art — ROCKET, MiniRocket, InceptionTime, HIVE-COTE, shapelets all headline on UCR/UEA where examples are
exchangeable and "future" is undefined (#5). Four of five anomaly candidates report point-adjusted F1 and all
five touch the test set when thresholding; Anomaly Transformer's *unadjusted* F1 is ~0.02, below the base
rate (#4). Four of five CPD candidates have a paper/code disagreement about the evaluation protocol itself
(#6). And DLinear's current HEAD provably does not reproduce its own published table (#3). **None of that was
visible from the outside**, and it is why the depth was worth buying even though the breadth was not (M11).

**Wave 1 runs in parallel — send all four at once, in four fresh conversations.** They share the frozen
rubric in `SELECTION_RUBRIC_2026-07-30.md` and write to four different files, so they cannot collide (D5).

**Every dispatch from #3 onward opens with the same line, per handover §5:**

> **Before anything else, read `WORKER_BRIEF_2026-07-30.md` in the mounted folder.** It carries the standing
> rules and this project's specific traps, and it is shorter than the time it will save you.

---

# WAVE 1B — Improvement-first re-search (Dispatches #10, #11)

*Supersedes the Wave 1 shortlists as the basis for selection. **Send both at once, in parallel.***

**Why this wave exists.** Wave 1 optimised for reproducibility and cost and returned papers that are cheap
to rebuild and weak to improve (**M12**). The rubric is now v2: two new gates — **G8** state of the art,
**G9** the improvement attaches at a named component — and C6 reweighted from 15 to **30**, the largest
weight in the rubric. Read rubric §−1 and §−0.5 before anything else; they are new and they are the point.

**The inversion.** Wave 1 asked "find good papers, then look for an improvement." This wave asks the
reverse: **"here is an improvement we can derive and defend — find the state-of-the-art methods where it is
the missing piece."** Stage 2 is the graded half of this assignment; the improvement drives the search.

**Budget: 3 candidates each, ≤4 pages, 2 hours.** Wave 1 cost 321 KB to choose nothing. Do not repeat it.

---

## Dispatch #10 — T1f: Scout where a GPD / extreme-value threshold is the missing piece

> Paste below the line into a fresh conversation with the folder mounted.

---

**Before anything else, read `WORKER_BRIEF_2026-07-30.md`, then `SELECTION_RUBRIC_2026-07-30.md` §−1, §−0.5
and C6.** Sections §−1 and §−0.5 are new and change what counts as a valid candidate.

**The decision this feeds, and the only evidence that changes it**

We must lock one paper to reconstruct and improve. **The improvement is chosen; the paper is not.** So the
only evidence that moves this decision is: does a state-of-the-art method exist where a peaks-over-threshold
GPD fit is the obvious missing piece, and can we run it. Nothing else is in scope.

**Task**

Find **3 candidate papers** where fitting a **generalized Pareto distribution to the tail of a score
distribution** — peaks-over-threshold, extreme value theory — would replace or augment a named component,
and where the base method is genuinely state of the art. Score them on rubric v2. **Do not recommend one.**

**The improvement, stated precisely so you can search for its absence**

Extreme value theory says that exceedances over a high threshold converge to a generalized Pareto
distribution regardless of the underlying distribution. Fit a GPD to the upper tail of a score computed on
**training data only**, and you obtain a detection threshold with a chosen false-alarm probability — **no
labels, no test-set contact, and it updates in a stream.** The canonical time-series instance is
SPOT/DSPOT (Siffer et al., KDD 2017); the technique is far older and is not owned by that paper.

**So you are looking for methods that produce a good score and then threshold it badly.** Wave 1 already
established that this is endemic: `CANDIDATES_anomaly_2026-07-30.md` §1.1 records that **all five** of its
candidates select their threshold with test-set labels in hand, and four of five report point-adjusted F1.
**That file is your starting evidence — read it first and do not re-derive it.** But note those five were
scored under v1 and mostly fail **G8**; treat them as a map of the failure mode, not as a shortlist.

Where to look, in rough priority: streaming and online anomaly detection; residual-based detectors built on
a forecaster; multivariate telemetry and monitoring; tail-risk and extreme-event forecasting; online
change-point detection with a threshold or penalty term. Also consider methods whose *loss* ignores the
tail — a GPD-weighted or EVT-aware objective is the same axis applied one level deeper.

**Constraints**

- **Read-only except `CANDIDATES_gpd_2026-07-30.md`.** Nothing cloned, nothing executed.
- **Do not recommend, rank by preference, or name a winner.** Gates, scores, evidence, kill reason.
- **G8 is not optional and not a formality.** A paper that merely appeared at a good venue fails it. You must
  search for refutations — forward citations, re-benchmark studies, "an analysis of…" papers — and **report
  the search you ran, not only what it found.** The gate exists because Wave 1's top candidate had been
  refuted by a 2024 ICML paper, its own scout recorded that fact, and no gate consumed it.
- **G9 must name a component**, with a file and line or a paper equation number, and say what it becomes.
- ≤4 pages, 2 hours. At the limit, return what you have with gaps named.
- Do not read the other Wave-1B scout's output file.

**Tooling — established, do not rediscover (D30)**

**OpenReview is unreachable** from these sessions — four scouts confirmed it independently. Do not spend a
minute on it; use the venue's own proceedings page and forward citations instead. `api.github.com` returns
empty bodies — use `raw.githubusercontent.com`. Google Drive folder listings are JS-rendered and unreadable.

**Method**

1. Read `CANDIDATES_anomaly_2026-07-30.md` §1.1 — the threshold-leakage table. It is the map.
2. Long-list ~10 methods that are current or near-current best on a benchmark others still publish against.
3. Apply **G8 first** — it is the cheapest gate to fail and the most expensive to get wrong. Then G9. Then G0–G7.
4. For the 3 survivors, fill all twelve evidence fields and score C1–C7 on **v2 weights (20/10/10/10/10/30/10)**.
5. In field 6, write the pre-registration sentence: what we change, from what to what, and the predicted
   direction **and rough magnitude**. Note explicitly whether the improvement is **ours to derive** or has
   already been demonstrated on this exact method by a follow-up paper — the latter caps C6 at 2.

**Output**

Exactly one file: `CANDIDATES_gpd_2026-07-30.md`. Shape per rubric §6. **≤4 pages.**
List rejections with the gate that killed each — **especially G8 rejections**, which are the informative ones.

**Acceptance**

- [ ] ~10 considered, **3** scored, ≥5 rejections with the killing gate named
- [ ] **G8 ruled per survivor with the refutation search reported** — what you searched, not only what you found
- [ ] **G9 names a specific component** (file+line or equation number) and what it becomes, for every survivor
- [ ] All ten gates G0–G9 ruled per survivor
- [ ] All twelve evidence fields per survivor
- [ ] C1–C7 on **v2 weights**, one worked weighting shown, totals to one decimal
- [ ] Field 6 states whether the improvement is ours to derive or already published on this method
- [ ] ≤4 pages · nothing cloned or run · no recommendation · no time spent on OpenReview
- [ ] Only `CANDIDATES_gpd_2026-07-30.md` created

**Self-audit — required; the status update is rejected without it.** As Dispatch #3.

**Status update format:** as Dispatch #3, **under one page.**

---

## Dispatch #11 — T1g: Scout where a Kalman / state-space component is the missing piece

> Delta against Dispatch #10. Substitute the **five** points below; everything else is used verbatim.

| # | Where | Replace with |
|---|---|---|
| 1 | Section header | `Dispatch #11 — T1g: Scout where a Kalman / state-space component is the missing piece` |
| 2 | **Task**, first sentence | "…where a **Kalman filter or state-space formulation** would replace or augment a named component…" |
| 3 | **The improvement** block | the block below |
| 4 | **Output** filename | `CANDIDATES_kalman_2026-07-30.md` |
| 5 | **Acceptance**, last line | `CANDIDATES_kalman_2026-07-30.md` |

**The improvement, stated precisely so you can search for its absence**

A Kalman filter is the optimal recursive estimator for a linear-Gaussian state-space model: it maintains a
state and its covariance, updates both with each new observation, and is **causal by construction**. Its
value here is threefold — it produces a *calibrated uncertainty* alongside the point estimate, it adapts to
drift without retraining, and it is *numerically stable* in square-root/Joseph form where naive recursions
are not.

**So you are looking for methods that estimate something sequentially with an ad-hoc rule.** Candidate
attachment points, in rough priority:

- a **hand-rolled smoothing or denoising** step — an exponential moving average, a fixed-width moving
  average, a hard-coded decay — standing where a filter belongs;
- a **variance or uncertainty estimate** that is fixed, heuristic, or absent, in a method that would benefit
  from a calibrated one;
- **online adaptation under drift**, where the method either retrains periodically or does not adapt at all;
- a **hybrid classical-deep forecaster** whose classical half is a fixed decomposition rather than a filter;
- a **deep state-space model** (S4/S5/Mamba-family applied to time series) whose recursion could be made
  numerically stable, or whose state could carry a covariance it currently does not.

Note for `COURSE_NOTATION` §7.1: the course syllabus promised state-space and ETS and **never delivered
them**, and the Kalman filter appears nowhere in any deck. That is a real explanation cost under C5 — score
it honestly — but it does not disqualify anything, and requirement B7 only asks that we explain the metric.

**One warning specific to this axis.** "Add a Kalman filter" is easy to say and hard to make falsifiable.
**G9 and C6 = 3 both require a named component and a predicted magnitude.** If the honest answer is "it
would probably help somewhere", that is C6 = 1, and say so rather than dressing it up.

---

# WAVE 1 — Family scouts (Dispatches #3–#6) — SUPERSEDED as a selection basis

> **These four ran and were accepted; their evidence is still good and is cited by Wave 1B.** They are
> superseded only as the *basis for choosing the paper*, because they were scored under rubric v1, which
> weighted cheapness over improvability (**M12**). Do not re-send them.

## Shared preamble — the manager's reasoning, so you can push back on it

Task family is **deliberately open**. Rather than guess which family yields the best paper, we scout all four
in parallel against one frozen rubric and let the scores decide. That is why your dispatch tells you not to
recommend: your family is one of four, and you cannot see the others.

`SELECTION_RUBRIC_2026-07-30.md` was written before any candidate was seen. **Read it first, in full.** It
carries the hard gates, the weighted criteria, the mandatory evidence fields, and the rules of evidence.
Everything below is only what is specific to your family.

The four scout dispatches are identical except for the family-specific blocks. That is intentional —
comparability is the whole point.

### How to assemble Dispatches #4, #5 and #6 — read before pasting

#4/#5/#6 are written as **deltas against #3**. Dispatch #3's text names *forecasting* in **five** places, not
one, so a careless paste sends an anomaly scout a task that says "forecasting" and an acceptance line naming
the forecasting output file. Substitute **all five**:

| # | Where in Dispatch #3 | Forecasting text | Replace with |
|---|---|---|---|
| 1 | Section header | `Dispatch #3 — T1a: Scout forecasting candidates` | the delta's own header |
| 2 | **Task**, first sentence | "…candidate papers in **time-series forecasting** for a course…" | the delta's family name |
| 3 | **Family — forecasting** block | the whole block, including the two family notes and the "one convention to check" paragraph | the delta's **Family** block |
| 4 | **Output** section | `CANDIDATES_forecasting_2026-07-30.md` | the delta's output filename |
| 5 | **Acceptance**, last line | "Only `CANDIDATES_forecasting_2026-07-30.md` created…" | the delta's output filename |

Then apply the delta's Acceptance replacements/additions. Everything else — Context, Constraints, Method,
Self-audit, Status update format — is used verbatim.

---

## Dispatch #3 — T1a: Scout forecasting candidates

> Paste below the line into a fresh conversation with the `timeAnalysis` folder mounted.

---

**Before anything else, read `WORKER_BRIEF_2026-07-30.md` in the mounted folder.** It carries the standing
rules and this project's specific traps — including several things about this folder that will otherwise
cost you an hour to rediscover.

**Task**

Produce a scored shortlist of **3–5 candidate papers** in **time-series forecasting** for a course final
project that reconstructs a paper and then improves it. Score them against a rubric you did not write and
must not edit. Do not recommend one.

**Context**

Course: Time-Series Analysis, instructor Havana Rika. The brief (`Final_Project.pdf`, 2 pp., read it) requires
a reconstruction of a published method, a baseline, the paper's metric plus a metric taught in class, a
temporal evaluation protocol, and then a meaningful improvement on the identical split. Five working days,
two people, deadline 10.08.

Read these files in the mounted folder before you start:

- `WORKER_BRIEF_2026-07-30.md` — standing rules and folder traps
- `SELECTION_RUBRIC_2026-07-30.md` — **the rubric. Frozen. Apply as written.**
- `Final_Project.pdf` — the brief
- `COURSE_NOTATION_2026-07-30.md` — what the course actually taught. §2 metrics, §3 models, §6 baselines, §7 NOT covered. C4, C5 and C7 are scored against this file

`REQUIREMENTS_2026-07-30.md` and `PLAN_2026-07-30_v2.md` are available if you want the requirement IDs
spelled out, but the rubric is self-contained.

**Family — forecasting**

Point forecasting of a numeric series, univariate or multivariate, single- or multi-horizon. In scope:
deep sequence models, linear and feature-based ML models, classical-statistical and hybrid methods,
and papers whose contribution is a decomposition, a normalisation, or a training objective rather than an
architecture.

Two family-specific notes, both of which affect scores:

- **The standard long-horizon benchmark suite (ETT variants, Electricity, Traffic, Weather, ILI, Exchange) is
  small, script-downloadable and permissively hosted in most repos.** Papers on it tend to score well on C3
  and often on C2. Check this rather than assume it — dataset hosting has moved more than once.
- Forecasting is the family with the **strongest course coverage**: RMSE/MAE/MAPE are defined with formulas
  (`COURSE_NOTATION` §2), Tier-1 baselines are both lectured and set as homework (§6), and walk-forward
  validation is prescribed by `ML models for TS.pdf`. Expect high C4 and C7 here — which means a *low* C4 or
  C7 for a forecasting paper is a strong signal something is off, and worth a sentence of explanation.

**One convention to check per candidate, because it silently breaks comparability:** whether the paper's
reported error is computed on **normalised** or **original-scale** data, and whether the normaliser was fitted
on train only. Record it in the evidence block. A reconstruction that gets this wrong will not match and we
will not know why.

**Constraints**

- **Read-only on every file in the folder except your own output file.**
- **Do not clone or run any repository.** Execution is a separate dispatch, deliberately, so that "looks
  runnable" and "runs" stay different claims.
- **Do not recommend, rank by preference, or write "the best candidate is".** Produce gates, scores, evidence,
  kill reasons. The manager selects.
- Do not edit the rubric. If you think a criterion is wrong, score it as written and say so separately.
- Do not read or write any other scout's output file.
- English output.

**Method**

1. Read the rubric in full, then the brief, then `COURSE_NOTATION` §2, §3, §6, §7.
2. Search broadly first — arXiv, OpenReview, Papers-with-Code, the proceedings of NeurIPS / ICLR / ICML /
   AAAI / KDD / IJCAI — and assemble **at least 10 plausible papers before filtering to 5**. A shortlist that
   was never long was never a shortlist.
3. Apply the hard gates G1–G6. Disqualified candidates go on the rejection list with the gate that killed them.
4. For survivors, open the artefacts: the paper page (record the arXiv **version**), the results table, the
   repo file tree, the dataset URL, the licence file, the issue tracker.
5. Score C1–C7. Show one worked weighting so the arithmetic can be checked.
6. Fill all seven mandatory evidence fields per candidate, including **known defects** and **kill reason**.

**Output**

Exactly one new file: `CANDIDATES_forecasting_2026-07-30.md`, in the shape given in rubric §6.
No other file created or modified.

**Acceptance**

- [ ] ≥10 papers considered; 3–5 survive to full scoring; **≥5 rejections listed with the gate that killed each**
- [ ] Every survivor passes all **eight** hard gates G0–G7, each with the evidence the rubric names
- [ ] G6's improvement is written as a complete sentence for every survivor — not "there is room to improve" — and its cost in runs is stated separately, for C6
- [ ] **All twelve mandatory evidence fields** present for every survivor — including **known defects** (issues / OpenReview / arXiv v2+ / errata), **kill reason**, implementation-detail sufficiency, the four E2 quantities, the headline number's **evaluation protocol** (for D15), how **B8** would be satisfied, and metric conventions
- [ ] arXiv version recorded, and all quoted numbers taken from that version
- [ ] C1–C7 scored with one worked weighting shown; totals out of 100, rounded only at the end, to one decimal
- [ ] No repository cloned or executed
- [ ] No recommendation anywhere in the output
- [ ] Only `CANDIDATES_forecasting_2026-07-30.md` created; nothing else modified

**Self-audit — required; the status update is rejected without it**

Head a section `SELF-AUDIT` and cover: what you got wrong in your own work and corrected; what you got wrong
in your **tooling**; anything recorded at lower confidence and why; anything you could not verify, stated as
unverified rather than rounded up.

Three specific warnings, two of them from real failures on this project:

- **A command that appears to return nothing may be failing loudly with its message swallowed.** Check exit codes.
- **A check that never fails is probably not checking anything.** If no candidate trips a gate, break one deliberately and confirm the gate catches it.
- **OpenReview and Papers-with-Code are client-rendered.** `web_fetch` returns an empty shell. Use the Claude-in-Chrome tools (`navigate`, then `get_page_text`). If you report "no reviews found" after a `web_fetch`, you have reported a tooling failure as a finding.

**Status update format**

```
1. What was done, with commands/searches and their output
2. What was measured, against the Acceptance line item by item
3. What changed in which files
4. SELF-AUDIT
5. What I could not verify, stated as unverified
6. Recommended next step  (process only — NOT which paper to pick)
```

---

## Dispatch #4 — T1b: Scout anomaly-detection candidates

> Delta against Dispatch #3. **Apply all five substitution points from the table above**, then these blocks.

**Family — anomaly detection**

Detecting anomalous points, subsequences or windows in temporal data — supervised, semi-supervised or
unsupervised. In scope: reconstruction-based (autoencoder, VAE), forecasting-residual-based, density- and
distance-based, matrix-profile, and classical statistical detectors.

Three family-specific notes that affect scores:

- **Known-defect field is load-bearing here.** A large fraction of deep TS-anomaly-detection papers report
  **point-adjusted F1**, a protocol shown in the literature to inflate scores so severely that random noise
  scores well. If your candidate uses it, that is not a disqualification — but it **must** appear in the
  known-defects field, because reconstructing a headline number produced by a discredited protocol is a
  different project from the one we think we are doing. Check what protocol the reported number uses and say so.
- **Benchmark provenance matters.** Several widely used TS-anomaly benchmarks carry documented labelling
  problems and trivially-solvable sequences. Record which benchmark, and whether the criticism exists.
- **C4 will be middling and that is expected.** Event-detection metrics (precision/recall, range-based, IoU)
  are recorded in `COURSE_NOTATION` **§2.3** and come **only** from Oudre's third-party deck — so requirement
  B8 is genuinely strained here, and evidence field 11 asks you to say how you would satisfy it. What Rika
  *does* teach is the anomaly definition itself, z-score, the point/contextual/collective taxonomy, Hampel
  (named, never defined), IQR, Isolation Forest and S-ESD. That is real course vocabulary, and it is why this
  family is scored normally: **the D16′ cap applies to CPD only.** Decision D16 as originally written bundled
  anomaly detection in with CPD; that was wrong and is superseded by D16′. Score C4 honestly rather than
  generously — expect a middling score, not a cap.

**One convention to check per candidate:** how a *point* score is turned into a *detection* — threshold rule,
window aggregation, and whether the threshold was chosen using test-set labels. Threshold selection on test
data is a leakage failure under B2 and would sink us on Day 3. Record it.

**Output file:** `CANDIDATES_anomaly_2026-07-30.md`

**Acceptance — add:**

- [ ] Reported-metric protocol recorded per candidate: **point-adjusted or not**, plus the thresholding rule and whether the threshold was chosen using test labels

---

## Dispatch #5 — T1c: Scout classification candidates

> Delta against Dispatch #3. **Apply all five substitution points from the table above**, then these blocks.

**Family — time-series classification**

Assigning a label to a whole series or to a fixed-length window. In scope: distance-based (DTW + kNN),
feature/transform-based (ROCKET family, catch22-style feature sets), deep (CNN/InceptionTime-style, RNN),
ensembles, and shapelet methods.

Three family-specific notes:

- **G5 is the gate this family fails on.** Many classification benchmarks are collections of independent,
  pre-segmented series where "future" is not defined and no chronological split exists. The brief's B2 —
  leakage forbidden in training, preprocessing, feature construction and hyperparameter tuning — has to mean
  something concrete. If the only temporal structure is *within* each example and the examples themselves are
  exchangeable, say so plainly; that is a G5 FAIL, not a technicality to route around. Prefer datasets with a
  real acquisition-time ordering, or a task where the split can be chronological.
- **C4 will be low.** `COURSE_NOTATION` §7.2: no classification metric is defined anywhere in the course.
  ROCKET, shapelets and DTW+kNN *are* lectured (§3, `Pre-precessing.pdf`, `ML models for TS.pdf`), so a
  ROCKET- or DTW-adjacent method scores better on C4 than a bespoke deep architecture. **B8 — "at least one
  metric studied in class" — is the requirement this family strains.** Note per candidate how you would
  satisfy B8, or that you could not see how.
- **C7 is the other weak point.** `COURSE_NOTATION` §6 lists no classification baseline. Argue from what the
  course would recognise: DTW + 1-NN is the field's own standard baseline and is lectured. Score honestly.

**One convention to check per candidate:** whether the benchmark's train/test split is the archive's fixed
official split or a resampled one, and whether the reported number is a single split or a mean over
resamples. These differ by more than the improvements papers claim.

**Output file:** `CANDIDATES_classification_2026-07-30.md`

**Acceptance — add:**

- [ ] For every survivor, G5 argued explicitly: what "future" means in this dataset and whether a chronological split exists. No survivor may hand-wave this
- [ ] Official-vs-resampled split convention recorded per candidate, and whether the headline number is one split or a mean over resamples

---

## Dispatch #6 — T1d: Scout change-point-detection candidates

> Delta against Dispatch #3. **Apply all five substitution points from the table above**, then these blocks.

**Family — change-point detection**

Locating times at which the generating process changes — offline or online, univariate or multivariate.
In scope: cost-function + search-method frameworks (dynamic programming, binary segmentation, PELT,
window-based), Bayesian online CPD, kernel and graph-based methods, and deep CPD.

Three family-specific notes:

- **Read this one carefully. Your family carries a scoring penalty, by decision D16.** The only CPD teaching
  material in the folder is `CPDexamples.pdf`, and that deck is **Laurent Oudre's, ENS Paris-Saclay, Master MVA
  2023-24** — not this course's. It also cross-references its own Lectures 1–4, which we do not have. So
  "use the course's vocabulary" is not available to a CPD project. **C4 is capped at 1 for every candidate in
  this family.** Apply the cap, and say in each evidence block that you applied it. This is not a reason to
  scout half-heartedly: the cap costs **10 weighted points**, not 15 — a capped C4 = 1 still earns
  (1÷3)×15 = 5 — so a CPD candidate's ceiling is **90/100**, and a strong one can still win.
- **Any citation of `CPDexamples.pdf` attributes to Oudre, never to this course.** If your notes cite it, name him.
- Rika's own deck (`Unsupervised models for TS.pdf`) contributes far less than it looks. Checked against
  `COURSE_NOTATION` §3.5: **exactly one** CPD row cites that deck — binary segmentation, sl. 22. Dynamic
  programming, sliding window, bottom-up, PELT, penalised CPD, and all four cost functions (`cML`, `cL2`,
  `cΣ`, `clinear`) cite **Oudre's** deck. So: if a candidate's method maps onto binary segmentation or the
  general "CPD components" framing, that is the difference between C4 = 1 and C4 = 0. If it maps onto PELT or
  a cost function, it maps onto Oudre, and C4 = 0. Do not credit Rika with Oudre's content.

**One convention to check per candidate:** the evaluation protocol — tolerance window (how near a detection
must be to count), whether the number of change points is given to the algorithm or estimated, and the
penalty/threshold selection procedure. CPD results are not comparable across papers without these, and a
reconstruction that guesses them will not match.

**Output file:** `CANDIDATES_cpd_2026-07-30.md`

**Acceptance — add:**

- [ ] D16′ cap applied and stated in every candidate's evidence block; no candidate scores raw C4 > 1
- [ ] Evaluation protocol recorded per candidate: tolerance window, known-vs-estimated number of change points, penalty selection
- [ ] Any use of `CPDexamples.pdf` attributed to Laurent Oudre, not to this course

---

# WAVE 2 — Red team (Dispatch #7)

*Send only after the manager has consolidated Wave 1 and named a top 2. Blocked until then.*

## Dispatch #7 — T1e: Adversarial pass on the two finalists

> Paste below the line into a fresh conversation with the folder mounted. **Nothing left to fill in.**
>
> **Manager's note on cost (D30).** Wave 1 cost 321 KB of output and 84 papers examined to choose one. This
> dispatch is deliberately narrower: **two papers, five questions, a 90-minute box, and a 3-page ceiling.**
> The scouts already did the evidence-gathering; do not repeat it.

---

**Before anything else, read `WORKER_BRIEF_2026-07-30.md` in the mounted folder.**

**The decision this feeds, and the only evidence that changes it**

The manager must choose **one** of two papers to reconstruct, today, and cannot revisit the choice after
tomorrow morning. **The only thing that changes that decision is a failure mode severe enough to make a
paper unusable.** Anything else — how good the paper is, how well it scored, what else it might have done —
is out of scope. If a line of enquiry cannot produce a *fatal* or *severe* finding, stop it and move on.

**Task**

Find the reason each of these two will fail this project. Not whether they are good papers — whether they
break on Day 3 or Day 4, when there is no time left to change course.

**Paper 1 — DLinear / LTSF-Linear** (forecasting, scored 88.3)
Zeng, Chen, Zhang, Xu, *Are Transformers Effective for Time Series Forecasting?*, AAAI-23,
arXiv **2205.13504v3**. Repo `github.com/cure-lab/LTSF-Linear`, Apache-2.0, last commit 2024-01-27.
Target number: **Table 2, p. 5 — MSE 0.375 / MAE 0.399, ETTh1 multivariate, L=336 → T=96.**
Dataset ETTh1, 2.47 MB, hourly, 17,420 rows, CC BY-ND 4.0.

**Paper 2 — changeforest** (change-point detection, scored 83.3)
Londschien, Bühlmann, Kovács, JMLR 24(216), 2023. Rust core with Python bindings.
See `CANDIDATES_cpd_2026-07-30.md` §1 for the repo, target number and protocol table.

**Context**

Five working days, deadline 10.08. Reconstruction Day 3, improvement Day 4, report Day 5. Both papers were
scored by scouts explicitly forbidden to clone or run anything — so **every claim about runnability in
`CANDIDATES_*.md` is an inference from a file tree.** Treat it as such. That is what you are here to break.

Read: `CANDIDATES_forecasting_2026-07-30.md` §A (DLinear), `CANDIDATES_cpd_2026-07-30.md` §1 and §1.1
(changeforest and the protocol table), and `Final_Project.pdf`. **You do not need to read the rubric** —
you are not scoring.

**Constraints**

- **Read-only except `REDTEAM_2026-07-30.md`.** Do not clone or run anything — that is Dispatch #8/#9.
- **Do not re-score and do not pick a winner.** Failure modes and severities only.
- **Do not re-verify what the scouts already evidenced.** Their citations are auditable; if you doubt one,
  say which and why, do not silently redo it.
- A failure mode you cannot evidence is a **hypothesis**. Label it. Do not pad the list to look thorough.
- **90 minutes.** At the limit, return what you have with the gaps named. A partial answer today beats a
  complete one tomorrow — the paper locks before you would finish.
- **Output ceiling: 3 pages.** If it is longer, you have included things that cannot change the decision.

**Tooling — already established, do not rediscover (D30)**

- **OpenReview is unreachable** from these sessions; all four Wave-1 scouts confirmed it independently. It is
  also **not applicable here** — AAAI and JMLR do not use OpenReview. Do not spend a minute on it.
- `api.github.com` returns empty bodies. Use `raw.githubusercontent.com` for file contents.
- Google Drive folder listings are JS-rendered and unreadable. The Autoformer Drive link is unverified and
  will stay that way; four repos point at it. If DLinear's data path depends on it, **that is a finding** —
  but the ETTh1 file itself is reachable directly from the ETDataset repo, so check there before concluding.

**Method — five questions, in this order. Stop early if you find something fatal.**

1. **DLinear's reproduction path is already known to be broken — establish how badly.** The scout found
   issue **#58** (`drop_last` should be `False` at test; maintainer changed the repo afterwards) and issue
   **#39** (model code changed post-publication, pointing at historical commit `9d933a805`). So current HEAD
   does not produce Table 2. **Determine what actually has to be checked out and toggled to reproduce
   0.375/0.399, and whether doing so means deliberately reproducing a known bug.** This is the single
   highest-value question in this dispatch. If the answer is "the published number is not reproducible by any
   configuration of the public repo", that is fatal and you can stop.
2. **Leakage in each paper's own protocol** — B2 is PASS/FAIL for us. For DLinear the scout verified the
   scaler is fitted on train only, but also found `train()` prints test loss every epoch, and that
   `Dataset_Pred` fits on the whole file. For changeforest, check the pseudo-permutation test and the
   α = 0.02 default. **If a paper leaks and we reproduce it faithfully we fail B2; if we fix it we no longer
   match the paper.** Surface that fork now, not on Day 3.
3. **Break the improvement.** DLinear's proposed change is closed-form OLS in place of 10 epochs of Adam —
   which comes from **Toner & Darlow, arXiv 2403.14587 (ICML 2024)**, who report the closed form superior in
   72% of settings. Ask: does taking our improvement from a published paper leave us with anything of our
   own? Is the predicted 0.375 → 0.360–0.372 effect larger than seed variance? **Also assess the declared
   fallback** — substituting STL for the fixed kernel-25 moving average — which is course-taught and is ours.
   Do the same for changeforest's improvement sentence.
4. **The dependency chain.** DLinear pins only `torch==1.9.0`; the scout found `df_stamp.drop(['date'], 1)`,
   which breaks on pandas ≥ 2.0. Establish whether a working environment exists at all in 2026. For
   changeforest, the Rust toolchain and the Python binding build are the equivalent question.
5. **Community record, bounded.** GitHub issues marked "cannot reproduce", follow-up papers that
   re-benchmark, errata. **Skip OpenReview.** For DLinear the scout already found #107, #117, #122, #123,
   #109, #70 — read what they actually say, do not re-list them.

**Output**

Exactly one new file: `REDTEAM_2026-07-30.md`, **≤3 pages**. Per paper: failure modes ordered by severity
(**fatal / severe / moderate**), each with its evidence in one line. Then, per paper, **the single most
likely way it kills the project**. Then a short section on what you looked for and did **not** find — an
absence you searched for is information; an absence you never looked for is not.

**Acceptance**

- [ ] All five questions answered for both papers, or explicitly marked "stopped early, fatal found"
- [ ] Question 1 answered concretely: the exact commit and flags needed to reproduce 0.375/0.399, or a finding that no such configuration exists
- [ ] Every failure mode carries evidence and a severity; hypotheses labelled as hypotheses
- [ ] The B2 fork ruled for both papers: does reproducing faithfully mean leaking?
- [ ] Improvement attacked for both papers, including DLinear's STL fallback
- [ ] "What I looked for and did not find" present and non-trivial
- [ ] **≤3 pages.** No winner picked, no re-scoring, nothing cloned or run
- [ ] Only `REDTEAM_2026-07-30.md` created
- [ ] No time spent on OpenReview

**Self-audit — required**

Same `SELF-AUDIT` section as the other dispatches. Plus: **if you found no fatal failure mode for either
paper, say explicitly whether that is a finding or a failure of your search**, and describe what you would
have had to find for the answer to be "fatal".

**Status update format:** as Dispatch #3, but **keep it under one page** — item 2 (measured against
Acceptance) can be a table, and item 1 does not need every command echoed.

---

# WAVE 3 — Clone-and-run probes (Dispatches #8, #9)

*Discharges plan task T2. Blocked on #7. Send both; #9 is the fallback and is not optional —
the plan requires candidate 2 to be probed before the paper is locked.*

## Dispatch #8 / #9 — T2a / T2b: GO / NO-GO probe

> One dispatch per candidate, two fresh conversations, **run in parallel**.
> **Manager: fill in the paper, repo URL, and the target table/number before sending.**

---

**Before anything else, read `WORKER_BRIEF_2026-07-30.md` in the mounted folder.**

**Task**

Clone **[REPO URL]** and run **its own example end to end**, on the machine we will actually use. Time it.
Return **GO** or **NO-GO** with evidence. Do not reconstruct anything, do not improve anything, do not tune
anything. The question is narrow: *does this code run, on this hardware, inside our schedule?*

**Context**

This paper is a candidate for a 5-day reconstruction project (deadline 10.08). Decision **D2′**: compute is
not scarce, wall-clock against the dependency chain is. A run whose result is needed the same working day
should be ≤2 h; longer runs must be launchable at day-start or overnight. Your timing measurement is what
turns that rule from a guess into a schedule.

Everything said about this repo so far was inferred from a file tree by someone who never executed it. You
are the first person to run it.

**Constraints**

- Work in the scratch/outputs directory. **Do not clone into the mounted `timeAnalysis` folder** — per D9 we
  ship patches, never vendored copies of licensed upstream code.
- Do not modify the mounted folder except to create your one output file.
- Do not fix the science. You may fix *environment* problems (a pin that won't install, a missing system
  package) — and you must record every such fix, because each one is a line in the README (deliverable D5)
  and a threat to reproducibility.
- **Timebox: 2 hours.** If it is not running by then, that is a result. Report NO-GO with the wall you hit.
  Do not spend the day heroically debugging someone else's Dockerfile.

**Method**

1. Record the hardware and software you actually have: GPU model, VRAM, driver, CUDA, Python version. Not
   what the repo asks for — what is present.
2. Clone at a **pinned commit**; record the SHA. "Latest main" is not reproducible.
3. Build the environment. Record every deviation from the repo's stated requirements and why it was needed.
4. Run the smallest example the repo ships — a quick-start, a smoke test, a single-epoch config. Confirm it
   produces output.
5. Then run **the experiment that produces our target number** (or, if it is long, run it far enough to
   extrapolate honestly, and *state that you extrapolated and show the arithmetic*).
6. **Time it with a clock, not an estimate.** Report: environment build time, smoke-test time, and full-run
   time or the extrapolation.
7. Compare whatever number you obtained against **[TARGET: Table N, p. M, metric = value]**. A mismatch here
   is not failure — it is exactly the information we need before committing.

**Output**

Exactly one new file in the mounted folder: `PROBE_<shortname>_2026-07-30.md` containing:
hardware/software actually present; pinned commit SHA; every environment deviation; each timing; the number
obtained vs the target; and a one-word verdict **GO** or **NO-GO** with the reason in one sentence.

Do not copy the cloned repository into the mounted folder.

**Acceptance**

- [ ] Commit SHA recorded; clone is outside the mounted folder
- [ ] Actual hardware and software recorded, not the repo's requirements
- [ ] Every environment deviation listed with the reason — this list is README content later
- [ ] Smoke test run and its output shown
- [ ] Target experiment run, or extrapolated **with the arithmetic shown and labelled as extrapolation**
- [ ] All three timings measured, not estimated
- [ ] Obtained number stated against the target number
- [ ] Verdict **GO** or **NO-GO**, one-sentence reason
- [ ] Timebox respected; if exceeded, NO-GO reported with the wall hit
- [ ] Only `PROBE_<shortname>_2026-07-30.md` created in the mounted folder

**Self-audit — required**

Same `SELF-AUDIT` section. Plus, specifically:

- **A run that printed no error may have failed.** Check exit codes. On this project a `pdftoppm` call that
  appeared to return nothing was in fact exiting 99 with its message swallowed. A training script that exits
  0 having silently skipped the evaluation is the same class of failure — confirm the output files exist and
  contain what you think they contain.
- **A metric that matches the paper on the first try deserves suspicion, not celebration.** Confirm you
  evaluated on the split you think you did, and that you are not reading a number the repo shipped
  pre-computed in a results file.
- State plainly whether your verdict rests on a full run or an extrapolation.

**Status update format:** as Dispatch #3.

---

## Audit of Dispatch #1 (manager, per bootstrap §5)

**Ruling: ACCEPT.** Every Acceptance item met or exceeded. Three premises in my own dispatch were wrong and
the worker corrected them (see PLAN §6 M3, R13).

**Re-measured independently — not taken on trust:**

| Claim | My check | Result |
|---|---|---|
| 10 PDFs / 8 `.md` / 4 notebooks | `ls`, `find` | **Confirmed.** `DL for TS.pdf`, `ML models for TS.pdf` stamped 30 Jul 11:25 — after my 11:06 listing |
| `CPDexamples.pdf` is Oudre's, not Rika's | `pdftotext -f 1 -l 1` | **Confirmed.** "Laurent Oudre / Master MVA / 2023-2024", 91 pp. |
| B8 metric formulas exist and are citable | Rendered PDF pp. 45–46 at 130 dpi and **read the images** | **Confirmed.** `e_t = y_t − f(x_t)`; MSE, MAE (printed `\|e\|_t`), RMSE; then MdAE, MAPE, SMAPE, NMSE, RMSLE |
| MAPE/SMAPE carry no ×100 | same render | **Confirmed.** Both are fractions |
| NMSE's σ² undefined | same render | **Confirmed** |
| Formulas absent from the text layer | `pdftotext` pp. 45–46 | **Confirmed** — returns only "Common metrics:" and "48". The rendering step was necessary, not gold-plating |

**Defect found — the one thing the self-audit missed.** Citations of the form "sl. 47–48" do not resolve.
`Time-Series Forecasting.pdf` has **46 pages**; my first render died with `Wrong page range given ... exit=99`.
The worker cited **printed slide numbers**; printed slide 47 = PDF page 45, offset 2. Defensible convention,
but a reader verifying by page number hits an error, and the offset is likely different per deck. → **D17**,
and Dispatch #2.

*Worth noting: `pdftoppm` first appeared to return nothing at all. Treating that silence as "no output, so no
data" would have been wrong — it was a non-zero exit with a real message being swallowed. §3.3, exactly as written.*

---

## Dispatch #2 — T0-b: Citation offsets and the unread-image sweep

> Paste below the line into a fresh conversation with the course folder mounted.

---

**Task**

Two narrow fixes to `COURSE_NOTATION_2026-07-30.md`, which you did not write. Both are
verification work, not authorship. Do not restructure the document or add new analysis beyond what is asked.

**Context**

That file cites lecture slides by their **printed slide number** — the number drawn on the slide. For at
least one deck this does not match the PDF page number: `Time-Series Forecasting.pdf` has 46 pages but
prints slide numbers up to 48, an offset of 2. A citation that cannot be resolved by the person checking it
is not a citation, and this document exists to be cited from a report that a grader may verify.

Separately, that file records that ~550 of ~556 embedded images are unread, and that the deep-learning
section rests on slide titles rather than content. `DL for TS.pdf` (84 images / 31 pages) and `EDA.pdf`
(76 / 42) are the least examined. Whether the project can choose a deep-learning paper depends on what is
actually in those images, so a targeted sweep is worth one task.

**Constraints**

- **Read-only** on everything except `COURSE_NOTATION_2026-07-30.md`, and within that file, edit only what
  these two fixes require.
- Do not "improve" wording, reorder sections, or normalise formulas to textbook form. The course's own
  notation and its typos are recorded deliberately.
- Do not recommend a paper.
- If a fix would change a *claim* rather than a *citation*, stop and report it instead of applying it.

**Method**

1. For **each** of the 10 lecture PDFs: get the true page count (`pdfinfo`), find the printed slide number on
   a known page, and derive the offset. **Offsets may differ per deck, and may not be constant within a
   deck** — decks with unnumbered title pages or section dividers can drift. Check at least two widely
   separated pages per deck before declaring an offset constant, and say so if it isn't.
2. Rewrite every slide citation in the document to the form `printed sl. N (PDF p. M)`.
3. **Verify by resolution, not by arithmetic**: for a sample of at least 8 rewritten citations spanning at
   least 4 different decks, actually open the cited PDF page and confirm the cited content is on it.
4. Render and read the image-heavy pages of `DL for TS.pdf` and `EDA.pdf`. Recover any formula, definition
   or metric that exists only as an image. Priority: LSTM/RNN equations, attention, any loss function, any
   evaluation metric, and Rika's BIC (recorded as unverified, image on sl. 29).

**Output**

Edits to `COURSE_NOTATION_2026-07-30.md` only:
- all citations converted to `printed sl. N (PDF p. M)`;
- a short provenance note stating the per-deck offsets and whether each is constant;
- §3.3 (deep learning) upgraded from slide titles to actual content where the images yield it, each new
  entry sourced;
- anything still unrecoverable left **explicitly marked unverified**. Do not quietly upgrade it.

**Acceptance**

- [ ] Offset derived for all 10 decks, each from **≥2 widely separated pages**, with non-constant offsets flagged.
- [ ] Every slide citation in the document carries both numbers.
- [ ] **≥8 citations across ≥4 decks verified by opening the page**, not by computing the offset. List them.
- [ ] `DL for TS.pdf` and `EDA.pdf` image-bearing pages rendered and read; recovered formulas added with sources.
- [ ] Items still unrecoverable remain marked unverified — count them and say so.
- [ ] No other file modified. No claim changed without escalating first.

**Self-audit — required; the status update is rejected without it**

Head a section `SELF-AUDIT` and cover: what you got wrong in your own work and corrected; what you got wrong
in your **tooling**; anything recorded at lower confidence and why; and anything you could not verify, stated
as unverified rather than rounded up.

Two specific warnings, both drawn from real failures on this project:

- **A command that appears to return nothing may be failing loudly with its message swallowed.** Check exit
  codes. `pdftoppm` on this repo returned empty output on a page range that did not exist; the real message
  was `Wrong page range given ... exit=99`.
- **A check that never fails is probably not checking anything.** If your citation-verification pass finds
  zero problems, verify the checker on a citation you have deliberately broken.

**Status update format**

```
1. What was done, with commands and their output
2. What was measured, against the Acceptance line item by item
3. What changed in which files
4. SELF-AUDIT
5. What I could not verify, stated as unverified
6. Recommended next step
```

---

## Dispatch #1 — T0: Course notation & vocabulary map

> **HISTORICAL RECORD — do not re-send. Three premises below are known wrong** and were corrected by the
> worker: it says 8 lectures (there are 10), 3 homework notebooks (there are 4), and it describes a folder
> layout — `Lectures/`, `HW/Assignment 1|2|3/`, `Final Project/` — that does not exist in the current mount.
> See PLAN §6 M3 and R13. Kept unedited because the decision history is the point.

> Paste everything below the line into a fresh conversation with the course folder mounted.

---

**Task**

Build a reference document that maps what this Time-Series Analysis course actually taught: its
vocabulary, its notation, the metrics it used, the models it covered, and the preprocessing and
evaluation conventions it follows. This is *not* revision material and *not* a summary for learning.
It has three concrete downstream jobs in a final project that reconstructs a research paper:

1. The final report must use **the course's terms and symbols** wherever they differ from the paper's,
   with the mapping stated once. Graders read for their own terms.
2. The brief requires evaluation using "at least one metric studied in class." We need to cite a
   **specific lecture**, not assert that a metric was probably taught.
3. The brief requires "a standard classical model when appropriate" as a baseline, and we will later
   design an improvement. Both should be drawn from what the course actually covered.

**Context**

Course: Time-Series Analysis, instructor Havana Rika. The course folder contains:

- `Lectures/` — 8 lecture decks, each present as both `.md` and `.pdf`:
  `Intro`, `EDA`, `EDA continue`, `Pre-precessing` *(note: filename is misspelled in the repo)*,
  `Time-Series Forecasting`, `Unsupervised models for TS`, `CPDexamples`,
  `Time-Series Analysis with Python`. Plus one loose file, `gemini-code-1782123087695.py`.
- `HW/Assignment 1|2|3/` — homework notebooks. These show which methods students were actually made to
  *use*, which is stronger evidence of emphasis than a slide that merely mentions something.
- `Final Project/` — the brief and planning documents. **Read `REQUIREMENTS_2026-07-30.md` for context on
  requirements B6, B7 and B8, which this task feeds.**

You do not need to know anything about the paper being reconstructed. It has not been chosen yet, and
choosing it is a different task. Do not recommend one.

**Constraints**

- **Read-only everywhere except your one output file.** Do not modify, reformat or "fix" any lecture
  file, notebook, or planning document — including the misspelled `Pre-precessing.md`.
- Do not recommend, shortlist, or evaluate candidate papers. Out of scope, and a different dispatch owns it.
- **Quote, do not paraphrase**, when recording a definition, a formula, or a symbol. Paraphrase is where
  a course's actual notation quietly becomes generic notation, which defeats the entire purpose here.
- Where the course's usage is ambiguous or a term appears with two meanings, **say so** rather than
  picking one. An ambiguity we know about is useful; a false certainty is not.
- English output (the report is in English), but preserve any Hebrew terms verbatim where they appear,
  with a translation alongside.

**Method**

Run and read rather than assume:

1. **Verify your instrument before trusting it.** The `.md` and `.pdf` files are presumably two renderings
   of the same deck — but that is an assumption, not a fact. Check it on at least two lectures (compare
   extracted text, look for content present in one and absent in the other, e.g. images, formulas, or
   speaker notes). If they diverge, say which is authoritative and work from that one.
2. Read all 8 lectures. For each, record the topics covered and the file it came from.
3. Read the 3 homework notebooks. Record which methods and metrics students were required to *implement*.
4. Build the inventories below. Every single entry carries a source: filename plus a quoted fragment or a
   section heading. **An entry you cannot source does not go in the document.**

**Output**

Exactly one file: `COURSE_NOTATION_2026-07-30.md`

It must contain these seven sections:

1. **Lecture inventory** — table: file, topics covered, and whether `.md`/`.pdf` agreed.
2. **Metrics taught** — every evaluation metric appearing in lectures or homework: the course's name for
   it, its formula **as the course writes it**, what it measures, and the source file. Flag which are
   suitable for forecasting vs. classification vs. anomaly/change-point detection.
3. **Models and algorithms taught** — grouped: classical/statistical forecasting, ML, deep learning,
   unsupervised, change-point detection, anomaly detection. Mark which appeared in homework (stronger
   signal of emphasis than a passing slide).
4. **Notation table** — symbols the course uses and what each denotes. Include the ones that are
   conventionally overloaded across the literature, since those are exactly where a paper's notation will
   collide with the course's.
5. **Preprocessing and evaluation conventions** — how *this course* handles train/test splitting, scaling,
   missing values, resampling, stationarity, outliers. Quote the guidance where it exists. This tells us
   what a grader will consider normal practice.
6. **Baseline candidates** — which specific baselines the course would recognise as "a standard classical
   model," sourced to lectures/homework.
7. **NOT covered** — an explicit list of major time-series topics this course did *not* teach. This is as
   valuable as the rest: if the chosen paper relies on something never taught, the report has to explain
   it from scratch rather than assume it, and we need to know that before we pick the paper.

**Acceptance**

I will audit against this list item by item:

- [ ] All 8 lectures and all 3 homework notebooks read; lecture inventory table complete.
- [ ] `.md` vs `.pdf` equivalence **tested on ≥2 lectures**, with the finding stated — not assumed.
- [ ] Every metric, model, symbol and convention entry cites a source file, with a quoted fragment or
      section heading. Zero unsourced entries.
- [ ] Formulas and definitions are quoted in the course's own notation, not normalised to textbook form.
- [ ] Section 7 ("NOT covered") is present and non-empty.
- [ ] Ambiguities and two-meaning terms flagged rather than silently resolved.
- [ ] No file modified other than `COURSE_NOTATION_2026-07-30.md`.
- [ ] No paper recommendation anywhere in the output.

**Self-audit — required, and the status update will be rejected without it**

End your status update with a section headed `SELF-AUDIT` covering, specifically:

- What you got **wrong in your own work** this session and corrected.
- What you got wrong **in your own tooling** — a grep that matched less than you thought, a file you
  believed you had read and had not, an extraction that silently dropped formulas or images.
- Anything you recorded with lower confidence than the surrounding text, and why.
- **Anything you could not verify — stated as unverified, not quietly rounded up to verified.**

If a check you ran never failed, consider whether it was checking anything at all.
An error with an empty message is a harness failure, not a result.

**Status update format**

```
1. What was done, with commands and their output
2. What was measured, against the Acceptance line item by item
3. What changed in which files
4. SELF-AUDIT
5. What I could not verify, stated as unverified
6. Recommended next step
```
