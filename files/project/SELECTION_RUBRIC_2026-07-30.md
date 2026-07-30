# SELECTION RUBRIC v2 — IMPROVEMENT-FIRST

> **v2 exists because v1 optimised for the wrong thing.** v1 spent 50 of 100 points on repo fidelity, run
> time and dataset access — how *cheap a paper is to rebuild* — and 15 on improvement headroom, defined
> generically. Stage 2 is the graded half of this assignment. The rubric therefore returned papers that are
> easy to reconstruct and weak to improve, and ranked first a 2022 baseline-critique paper whose method
> contribution a 2024 ICML paper had already refuted. **Manager error M12.** Corrected below.
>
> **What changed:** two new gates (**G8** state of the art, **G9** the improvement attaches), and C6
> reweighted from 15 to **30 — the largest single weight in the rubric**. Everything else in v1/v1.1 stands.

---

## −1. What the brief does and does not require about causality

This section exists because a candidate was rejected for not being causal, and I need the gate to trace to
something real rather than to my own preference. **Handover §8: any constraint I introduce that is not in
the assignment brief must trace to something a human actually told me.**

**Verbatim, `Final_Project.pdf` p. 1:**

> *"The reconstruction must respect the temporal structure of the data: future observations must not be used
> during **training, preprocessing, feature construction, or hyperparameter tuning**."*

> *"A valid temporal evaluation protocol, such as chronological train/validation/test split, rolling-origin
> evaluation, or walk-forward validation."*

**Reading this honestly: the brief constrains our *pipeline*, across four named stages. It does not require
the algorithm to be causal at inference.** A method that reads a whole test segment to score a point inside
it is not literally excluded by those words. So *"the method must be causal"* is **not** a requirement I can
derive from the assignment, and I will not pretend otherwise.

**What actually rules such a method out is narrower and firmer: the improvement has to attach to it.**
Both improvement axes this project has chosen — a Kalman/state-space recursion, and a peaks-over-threshold
GPD fit — are *sequential* techniques. They operate on a stream of residuals, scores or states. A method
that emits a single whole-series partition has no such stream, so there is nowhere for either improvement to
go. That is gate **G9**, and it is why an offline segmentation method fails: not because the brief forbids
it, but because we could not improve it in the way we intend to.

Consequence worth stating: **G9 is contingent on the improvement axis.** Change the axis and G9 changes.
That is correct behaviour for an improvement-first rubric, and it is the opposite of how v1 worked.

---

## −0.5 The two new gates

| ID | Gate | Source | Evidence required |
|---|---|---|---|
| **G8** | **The method is state of the art, operationally:** it reports beating **named recent baselines** on a benchmark that other groups still publish against, **and** no widely-cited follow-up has shown the contribution does not hold. | The project's goal, stated by the user: *"find an algorithm in a paper that is the state of the art"* | The comparison table with the baselines named and dated, **plus** a deliberate search for refutations — forward citations, "an analysis of…" papers, re-benchmark studies. **Report the search, not just the result.** |
| **G9** | **At least one chosen improvement axis attaches at a named component.** Point to the specific step — a thresholding rule, a residual, a smoothing operation, a state update, a variance estimate — that a GPD tail fit or a Kalman/state-space recursion would replace or augment. | The improvement axes, chosen by the user | Name the component, name the file and line or the paper's equation number, and write what it would become. *"This method could probably be improved"* is a FAIL. |

**G8 is the gate that would have caught DLinear.** Its own scout found Toner & Darlow (ICML 2024) reporting
that linear forecasters are *"functionally indistinguishable from standard, unconstrained linear regression"*
and that closed-form solutions win in 72% of settings — recorded faithfully in evidence field 5, and then
worth nothing because no gate consumed it. The evidence was there; the rubric had no slot for it.

---

# SELECTION RUBRIC v1.1 — the three ruled amendments (still in force)

> **Amendment log — 30 Jul, after Wave 1 returned.** Three defects were found by applying the rubric, which is
> the only way rubric defects are ever found. All three are ruled below, all three are applied **uniformly to
> all four families**, and for each one I state whether it changes the outcome. **None of them does** — the
> top two candidates are the same before and after. That is the strongest available evidence that these are
> corrections and not motivated reasoning, and it is why I am willing to amend a document I froze. See PLAN §9.
>
> | # | Raised by | Ruling | Moves totals? | Changes top 2? |
> |---|---|---|---|---|
> | A1 | Dispatch #5 | **VALID** — the rubric misquoted requirement B8 | No (evidence field, not scored) | No |
> | A2 | Dispatch #3 | **NOT VALID as applied** — CC BY-ND does not forbid redistribution | +3.3 to four forecasting candidates | No |
> | A3 | Dispatch #4 | **VALID** — C3 band 0 conflated a free signup with an approval process | +3.3 to telemanom (73.3→**76.7**), +3.3 to USAD (56.7→**60.0**) | No |

*Manager-owned. Written **before** any candidate was seen, which is the only reason it is worth anything.*
*Workers: read this, apply it, do not edit it. If you believe a criterion is wrong, say so in your status
update and score it as written anyway.*

Governs: A1–A5 (paper-selection requirements), D7, D13′, D16′, R12.
Consumed by Dispatches #3–#6 (family scouts) and #7 (red team).

> **ID namespaces collide across this project's documents — read this once.**
> `REQUIREMENTS_2026-07-30.md` uses **D1–D5** for *submission files* and **C1–C2** for *improvement
> requirements*. `PLAN_2026-07-30_v2.md` uses **D1–D22** for *decisions*. This rubric uses **C1–C7** for
> *scoring criteria* and **G0–G7** for *gates*. Both files also use **T-numbers** for different task lists.
> In this file, unqualified `C1`–`C7` and `G0`–`G7` are always this rubric's; requirement IDs are written
> as "requirement A3", decision IDs as "decision D7".

---

## 0. Why this file exists separately

Four scouts score in parallel. If each invents its own scale, the scores cannot be compared and the
selection collapses into whichever scout wrote most persuasively. The rubric is frozen before candidates
are known so that it cannot be bent toward a paper someone has already fallen in love with.

**You are not choosing the paper.** You are producing evidence and a score. The manager selects.
A scout that writes "I recommend X" has exceeded its brief.

---

## 1. Hard gates — any FAIL disqualifies, no score is computed

Gate before you score. A disqualified candidate still gets reported (one line, gate + evidence), because
knowing what we rejected is how the manager checks that filtering happened.

| ID | Gate | Source | Evidence required |
|---|---|---|---|
| **G0** | The paper's topic sits inside a family requirement A1 names: *"time-series analysis, forecasting, classification, anomaly detection, change-point detection, time-series representation learning, etc."* | **A1** — the scope gate | Name the family. If it is not one of the six, say which of A1's "etc." you are claiming and why |
| **G1** | The paper's contribution is an implementable **model, algorithm, or architecture**. Not a survey, not a benchmark/dataset paper, not a pure theory or bounds result, not a position paper. | A2 | The sentence in the paper that states the contribution, quoted |
| **G2** | A **public dataset** is reachable today. | A3 | The URL, plus confirmation you actually reached it (status, file listing, or size). "The paper says it's public" is not evidence |
| **G3** | The paper reports **at least one number on a named benchmark with a named metric**, traceable to a table and page. | A4 | `Table N, p. M: <metric> = <value> on <dataset>` |
| **G4** | A **public code repository exists**. | **D7** — no public repo ⇒ do not select | Repo URL + the file listing you actually saw |
| **G5** | The task is genuinely **temporal** — it admits a chronological split and can be violated by leakage. | B2, B5 | One sentence on what "future" means in this dataset |
| **G6** | **Some** meaningful improvement is conceivable and nameable — any improvement, at any cost. | A1/C1 | Name it in one sentence. *Whether it costs one run is **not** decided here; that is criterion C6* |
| **G7** | The headline run is **≤24 h on one GPU**. | **D2′** | The estimate and where it came from |

**On G6 and D13′.** Decision D13′ is *"relaxed but not dropped"* — a preference, not an elimination rule.
An earlier draft of this rubric made "costs one extra run" a hard gate, which both over-enforced D13′ and
charged it twice (once as elimination, once as C6). Corrected: G6 fails only when **no** improvement can be
named at all; C6 grades how expensive the named improvement is. You must still be able to finish the sentence
*"we would change ___ from ___ to ___ and expect ___"* — "there is probably room to improve" is a G6 FAIL.

**On G7.** Not a compute limit — compute is not scarce (≥60 GPU-h/week). It is a **dependency-chain** limit:
the reconstruction runs on Day 3 and the improvement on Day 4, so a run exceeding 24 h cannot be launched and
consumed anywhere in the schedule, whatever the GPU budget. Runs between 2 h and 24 h are permitted and
scored down on C2, not gated.

---

## 2. Scored criteria — 0 to 3 each, weighted to 100

Weighted points = (score ÷ 3) × weight. Report the raw 0–3 **and** the weighted total.

### C1 — Repo fidelity to the headline number · weight **20** (was 25)

Does the repo contain the *specific experiment* that produced the number we will compare against?

- **0** — repo exists but has no training or evaluation entrypoint
- **1** — generic training code; nothing ties any script or config to the paper's reported table
- **2** — a named script/config plausibly corresponds to the headline experiment; dependencies unpinned or partial
- **3** — a named script/config explicitly reproduces the headline table (README or config names the table/dataset), **and** dependencies are pinned (lockfile, `environment.yml`, or pinned `requirements.txt`)

*A README claiming reproducibility is a claim. Score from the file tree.*

### C2 — Run fits the 5-day dependency chain · weight **10** (was 15)

Per **D2′**: compute is not scarce (≥60 GPU-h/week available); **wall-clock against the dependency chain is**.

Bands are half-open so that a boundary value falls in exactly one. `t` = wall-clock for the headline run on one GPU.

- **0** — needs multiple GPUs, **or** neither the paper nor the config states a cost and you cannot estimate one
- **1** — 6 h < t ≤ 24 h  (overnight only; consumes a whole day of the chain)
- **2** — 2 h < t ≤ 6 h  (must be launched at day-start or overnight)
- **3** — t ≤ 2 h, so a same-day result is possible

*(t > 24 h is gated out by G7 and never reaches scoring.)*

State where the estimate came from: paper's stated training time, epochs × steps in the default config, or a repo issue.
If you are extrapolating, say so and give the arithmetic.

### C3 — Dataset accessibility, size, licence · weight 10

Score the *access* band first, then apply the licence adjustment. This ordering exists because an earlier
draft left a real case unscoreable: a small, public, script-downloadable dataset whose licence clearly
*forbids* redistribution matched none of the bands.

Access band:

- **0** — **gated by a human decision**: an approval process, an institutional affiliation, a data-use agreement, a licence request, or a review of any kind. The blocker is that someone else must say yes, and they may say no or take days
- **1** — public but large (>2 GB), or requires a manual multi-step download, **or requires a free self-serve account that is granted instantly** (e.g. a Kaggle signup)
- **2** — public, script-downloadable, ≤2 GB, no account
- **3** — public and **repo-bundled**, or script-downloadable and small enough to attach to a submission

> **Amendment A3, ruled VALID (raised by Dispatch #4).** Band 0 previously read *"registration-walled, gated,
> or otherwise not obtainable without a human account"*, which put a two-minute Kaggle signup in the same band
> as a three-working-day institutional review. Those are not the same risk: one costs two minutes, the other
> can return "no" after the deadline. The scout scored it as written and escalated — correctly — but it cost
> telemanom **10 weighted points** for a signup. Corrected by splitting on *who decides*: band 0 is now
> "someone else must approve", instant self-serve accounts drop to band 1. **Applied uniformly to all four
> families.** Kaggle-gated candidates move C3 = 0 → 1, worth (1÷3)×10 = 3.3: telemanom **73.3 → 76.7**
> (3,2,**1**,2,1,3,3), USAD **56.7 → 60.0** (1,3,**1**,2,1,2,3). Note the amendment deliberately does *not*
> put a signup in band 2 — an account and a ToS acceptance is real friction, just not a human gatekeeper.
> **Does not change the top two.**

Licence adjustment, applied after: **−1** (floor 0) if the licence **forbids redistribution** *or* is not
stated. Deliverable D3 then becomes "link + download script" rather than "ship the data", which is permitted
by the brief but is one more thing to get right. Name the licence, or write "not stated".

> **Amendment A2, ruled NOT VALID AS APPLIED (raised by Dispatch #3).** The scout applied the −1 to CC BY-ND
> datasets on the reading that ND forbids redistribution. It does not. **ND forbids *derivatives*; it
> explicitly permits redistribution of the unmodified work.** For C3's purpose — can we ship the dataset with
> the submission — CC BY-ND is permissive and takes no penalty. **The real constraint it imposes is
> different and worth stating:** shipping a *preprocessed* copy would be a derivative and is not permitted, so
> under CC BY-ND we ship the raw data plus our preprocessing code, never a cleaned CSV. That is a note for
> deliverable D3, not a scoring deduction. Restores +3.3 to four forecasting candidates. **Does not change the
> top two** — both already score C3 = 3 and cannot gain.

Feeds D3 (deliverable: ship the data or a link) and R9 (Moodle size limit). Name the licence or say "not stated".

### C4 — Course-vocabulary fit · weight **10** (was 15)

How much of the method and its metrics can be written in *this course's* terms and cited to a Rika deck.
Score against `COURSE_NOTATION_2026-07-30.md` — §2 (metrics), §3 (models), §4 (notation).

- **0** — neither method nor metric has any counterpart in the course material
- **1** — the metric is citable to a deck; the method is not
- **2** — both partially citable
- **3** — the method family **and** both required metrics (the paper's, per B7, and a course metric, per B8) cite cleanly to a Rika deck

> **Under v2 weights the D16′ cap costs (2÷3)×10 = 6.7 points, not 10** — C4 dropped from 15 to 10. The CPD
> family ceiling is correspondingly **93.3**, not 90.0. Note that under **G9** most offline CPD methods now
> fail a gate before C4 is ever reached, so the cap is largely moot; it stays in force for online CPD.
>
> **D16′ CAP — change-point-detection papers score at most 1 on C4.** The only substantial CPD material in
> the folder is `CPDexamples.pdf`, which is **Laurent Oudre's ENS Paris-Saclay deck, not Rika's** — cost
> functions, search methods, PELT and penalised CPD are all his. "Use the course's vocabulary" is therefore
> unavailable for CPD. When you apply this cap, say so explicitly in the candidate's evidence block. Any
> citation of that deck attributes to **Oudre**, never to this course.

**The cap applies to CPD only.** Decision D16 as originally written bundled anomaly detection in with CPD;
that was wrong and is superseded by **D16′**. Anomaly detection *is* taught in Rika's own deck
(`Unsupervised models for TS.pdf`: the anomaly definition, z-score, point/contextual/collective anomalies),
so it has genuine course vocabulary to draw on and is scored normally.

Anomaly-detection and classification papers therefore take no cap — but score C4 honestly rather than
generously. `COURSE_NOTATION` **§2.3** records that event-detection metrics (precision/recall, range-based,
IoU) appear **only** in Oudre's deck, and §7.2 records that no classification metric is defined anywhere in
the course. Those are real C4 deductions; they are just not caps.

### C5 — Explanation cost, inverted · weight 10

Per **R12**: a method the course never taught means the report explains it from scratch. This is a **cost,
not a veto**. High score = cheap.

- **0** — the report must define an entire method family from scratch. `COURSE_NOTATION` §7.2 lists these:
  Transformer/attention internals, conformal prediction, state-space/Kalman, contrastive or self-supervised
  representation learning, quantile/pinball/CRPS, spectral methods, foundation models
- **1** — one major un-taught component
- **2** — minor un-taught components only
- **3** — every component is taught **with a formula** in a deck

Note that DL is a special case: `DL for TS.pdf` covers RNN, LSTM, CNN, dilated causal convolution and
attention **as slide titles and images**, and no DL formula survives in any text layer. Treat DL internals
as score 1 at best unless you can point to a recovered formula.

### C6 — Improvement headroom · weight **30** (was 15 — see the v2 header)

**This is now the largest weight in the rubric, and it is scored against *our* two axes, not against
improvement in the abstract.** A paper with a large generic improvement space and no purchase for a GPD tail
fit or a state-space recursion scores low here. That is intended.

- **0** — no improvement visible short of a hyperparameter sweep or a new architecture; neither axis attaches (this also fails G9)
- **1** — an axis attaches, but demonstrating it needs more than one run or a sweep
- **2** — a one-run change on a named component, but its effect cannot be predicted quantitatively — weak pre-registration
- **3** — a one-run change on a named component whose **direction and rough magnitude** can be pre-registered before running, **and which is ours to derive** rather than lifted from a follow-up paper

**On band 3's last clause.** DLinear's proposed improvement was closed-form OLS — taken directly from
Toner & Darlow. It satisfied every other part of band 3 and would still have left us presenting someone
else's result as our contribution. An improvement we derive from the Kalman or EVT literature and *apply*
is ours; an improvement a follow-up paper already demonstrated on this exact method is not.

C6 = 3 requires you to write the pre-registration sentence. **D10** forbids running an experiment before its
pre-registration exists; a paper we cannot pre-register against is a paper that will stall on Day 4.

**Where the axes typically attach — use this to search, not as a checklist:**

| Axis | Attaches at | Typical target |
|---|---|---|
| **GPD / peaks-over-threshold** | any step converting a continuous score into a detection — a fixed quantile, a 3σ rule, a grid-searched threshold, an F1-maximising sweep | streaming anomaly detection; residual-based detectors; tail-risk forecasting |
| **Kalman / state-space** | a smoothing or denoising step, a variance or uncertainty estimate, a parameter update under drift, a hand-rolled recursive filter | probabilistic forecasting; online/adaptive models; hybrid classical-deep forecasters |

**Wave 1 already located one of these.** The anomaly scout established that **all five** of its candidates
select their detection threshold with test-set labels in hand. A POT/GPD threshold fitted on training
residuals removes that entirely — no labels, no test contact, streaming-compatible. That is a live,
documented, subfield-wide flaw with a principled fix, which is a stronger Stage 2 than any decimal chase.
Treat it as a lead, not as an answer: the paper still has to clear G8.

### C7 — Baseline compatibility · weight 10

Does a baseline the course would recognise (B6) apply to this paper's task without contortion?
Score against `COURSE_NOTATION_2026-07-30.md` §6.

- **0** — no course-recognised baseline applies
- **1** — a baseline applies only after redefining the task
- **2** — a Tier-2 baseline applies (lectured, not in homework): ARIMA, ARMA, AR(p), MA(q), SES, Theta, GARCH
- **3** — a **Tier-1** baseline applies directly — lectured *and* implemented in homework:
  forecasting → naive, seasonal naive, moving average, SARIMA, Holt-Winters;
  detection/clustering → IQR, Isolation Forest, S-ESD, std-rule on STL residuals, binary segmentation with L2 cost, KMeans + silhouette

**v2 total: C1 20 + C2 10 + C3 10 + C4 10 + C5 10 + C6 **30** + C7 10 = 100.**
*(v1 was 25/15/10/15/10/15/10. The 20 points moved out of C1+C2+C4 went to C6. Scores from Wave 1 were
computed under v1 weights and are **not comparable** to v2 scores — do not mix them in one table.)*

---

## 3. Mandatory evidence fields — every candidate, no exceptions

A candidate missing any field is incomplete and will be sent back. Fields 8–12 exist because each one is a
requirement that would otherwise surface on Day 3 or Day 4, when the paper is locked and there is no move left.

1. **Citation** — title, authors, venue, year, arXiv ID **and version** (v1/v2/…), DOI if any
2. **Repo** — URL, stars, last commit date, licence, and the top-level file listing you actually saw
3. **Headline number** — `Table N, p. M: <metric> = <value> on <dataset>`. This is the number we will try to reproduce
4. **Dataset** — name, URL, size, frequency, length, licence
5. **Known defects** — GitHub issues, OpenReview reviews, arXiv v2+ changes, errata, retraction. Search for these deliberately. *We must not rediscover a known bug on Day 3 and write it up as a finding.*
6. **The improvement** — the sentence from G6, plus what it would cost in runs (this is what C6 grades)
7. **Kill reason** — the single most likely way this paper wrecks the project. Every candidate has one. A candidate with no kill reason has not been examined
8. **Implementation detail sufficiency** — requirement A2 is *"a clear model, algorithm, or architecture **that can be implemented**"* and A5's second branch is *"**enough implementation details to reproduce the method**"*. Answer directly: does the paper state its layer sizes, optimiser, learning rate, batch size and training schedule, or must they be read out of the repo? **This is load-bearing exactly when C1 ≤ 1** — if the repo does not carry the headline experiment, the paper's own detail is all we have
9. **The four E2 quantities** — requirement E2 forces the report to name **forecast horizon, sampling frequency, input window length, output window length**. Record which of the four the paper actually states, and which we would have to infer. A paper that never states its input window is a problem we want on Day 1, not Day 3
10. **Evaluation protocol of the headline number** — single chronological holdout, rolling-origin, walk-forward, k-fold, or unstated. **Decision D15 turns on this**: if the paper's number is only comparable under a single holdout, we must run *both* protocols, and that cost has to be visible now
11. **How requirement B8 would be satisfied** — B8 reads, in full: *"Evaluation using at least one metric studied in class **or appropriate for the task**."* Name the metric you would report alongside the paper's — either sourced to `COURSE_NOTATION` §2, **or** justified as appropriate for the task — or state plainly that you cannot see how B8 would be met. Asked of every family.

    > **Amendment A1, ruled VALID (raised by Dispatch #5).** This field previously truncated B8 to *"evaluation
    > using at least one metric studied in class"*, dropping *"or appropriate for the task"*. Verified against
    > the brief, `Final_Project.pdf` p. 1 line 34, and `REQUIREMENTS_2026-07-30.md` line 31 — both carry the
    > full clause; the truncation was introduced here and nowhere else. **Consequence of the error:** under the
    > truncated wording no classification candidate could satisfy B8, because `COURSE_NOTATION` §7.2 records
    > that the course defines no classification metric. Under the actual wording, macro-F1 satisfies it in one
    > line. The rubric had made an entire eligible family look non-viable on my typo. Dispatch #5 scored it as
    > written and escalated rather than quietly picking, which is the behaviour WORKER_BRIEF §2.5 asks for.
    > **Does not move any total** — field 11 is evidence, not a scored criterion. Recorded as manager error M9.
12. **Metric conventions** — anything about how the paper computes its metric that would make our number differ silently. Specifically: is the error computed on **normalised or original-scale** data, was the normaliser fitted on train only, and does any MAPE/SMAPE carry a **×100**? The course writes MAPE and SMAPE **without** ×100 (PLAN §2), so a mismatch here is a 100× discrepancy that looks like a reconstruction failure

---

## 4. Rules of evidence

Drawn from this project's own recorded failures (PLAN §6, R11) — these are not hypothetical.

- **Open the artefact.** A number that came from a search-result snippet or an abstract is not evidence. Open the paper page, open the repo file.
- **A README is a claim, not evidence.** Verify against the file tree.
- **An empty command output may be a swallowed error.** Check exit codes. On this project `pdftoppm` appeared to return nothing and was in fact failing with `exit=99`.
- **A check that never fails is probably not checking anything.** If your gate pass finds zero problems across all candidates, your gates are broken — verify them on a candidate you deliberately break.
- **Re-read your own evidence before writing a number.** PLAN §6 M3 is a manager error of exactly this kind: a count contradicted by a listing taken 40 minutes earlier.
- **Do not clone or run anything.** Repo execution is Dispatch #8/#9's job and is deliberately separated from scoring, so that "it looks runnable" and "it runs" stay distinct claims.
- **Report your rejections.** At least 5, one line each, with the gate that killed them. A scout that returns only strong candidates has filtered badly or filtered invisibly.
- **Do not recommend.** Score, evidence, kill reason. The manager selects.

---

## 5. Known tooling traps for this task

- **OpenReview is client-rendered.** `web_fetch` returns a page shell with no reviews. Use the Claude-in-Chrome tools (`navigate`, then `get_page_text`) for OpenReview, and for Papers-with-Code leaderboards.
- **arXiv version matters.** v1 and v3 can report different numbers. Record the version you read and cite numbers from that version only.
- **GitHub file listings via `web_fetch`** are usually fine, but the rendered file *contents* may be truncated; prefer the `raw.githubusercontent.com` URL when you need to read a config.
- **The mounted folder is a subset of the course repo.** It contains: the 10 lecture PDFs in `lectures/`
  (lowercase — `COURSE_NOTATION` writes `Lectures/`), `Final_Project.pdf` (the brief), the planning documents,
  and `Time-Series Analysis.pdf`. It does **not** contain the homework notebooks, nor the `.md` lecture
  renderings, nor a `Final Project/` directory — all three are referenced by older documents in this folder.
- **`COURSE_NOTATION_2026-07-30.md` cites sources you cannot open.** Many of its entries are sourced to
  `HW*` notebooks or to `<Lecture>.md` files, none of which are mounted. Those claims were audited when T0 was
  accepted. **You may rely on them; you cannot re-verify them; do not claim you did.** This applies to the
  whole of §6 (baseline tiers), which C7 is scored against.
- **You are scoring C4 on what `COURSE_NOTATION` *says*, not on resolving its citations.** That document
  currently carries **zero** citations in the resolvable `printed sl. N (PDF p. M)` form (decision D20 defers
  the fix). Do not treat an unresolvable slide number as a missing source, and do not go and rebuild the
  citation map — that is a different, already-written dispatch.
- **`Time-Series Analysis.pdf` is NOT a source. Do not read it, cite it, or score against it.** It is a
  16-page synthesis of the same course material, produced by a conversation the plan does not record and never
  audited (risk R15). It looks authoritative and is not. Using it would launder an unverified claim into an
  apparently independent one.

---

## 6. Output shape

A summary table across your candidates:

| # | Paper | Gates G0–G9 | C1 | C2 | C3 | C4 | C5 | **C6** | C7 | **Total /100** |

Weights, for the weighting you must show: **20 / 10 / 10 / 10 / 10 / 30 / 10.**

Then one evidence block per candidate carrying **all twelve** §3 fields, then the rejection list.

**Arithmetic convention — follow it exactly, or four scouts produce four incomparable totals:**

- Table cells carry the **raw 0–3** score. The last column carries the **weighted total**.
- Weighted points = (raw ÷ 3) × weight. Most odd scores repeat as decimals (raw 1 on C1 → 8.333…).
- **Round only the final total, to one decimal place.** Do not round the per-criterion contributions.
- Show one worked weighting in full so the arithmetic is checkable.
- The D16′ cap is a cap on the **raw** C4 score, applied before weighting.
- The C3 licence adjustment is likewise applied to the **raw** score, before weighting.

There is no pass mark and no tie-break. Do not invent one — the manager compares across all four families,
and a threshold that made sense inside one family would corrupt that comparison.
