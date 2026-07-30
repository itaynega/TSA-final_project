# Proposed amendments — for the manager to rule on

Written 30 Jul, in a session with Amitay. **These are proposals, not plan text.** Nothing here supersedes
`DISPATCHES_2026-07-30.md` or the decision log. Paper is locked to TQNet (D35) and that is not reopened here.

Items 1–4 were **approved by Amitay** in conversation. Items 5–7 are risks the current documents do not
carry. Item 8 is housekeeping already applied to the repo.

---

## 1. Ownership splits on the reconstruction / improvement line — **approved**

Itay owns reconstruction and the shared `common/` foundation. Amitay owns the improvement, metrics, the
baseline, the report, and the leakage audit.

**This preserves and strengthens the separation-of-duties rule.** The existing plan requires the leakage
audit to be run by whoever did *not* write the pipeline; under this split that falls on a task boundary
rather than being asserted task by task.

**Consequence to rule on:** the "implement and run the improvement" task moves to **Amitay**, so he needs a
working TQNet environment on his own machine, not only Itay's. → risk **R15** below.

## 2. One evaluation protocol, not two — **approved**

The existing decision D15 wants walk-forward as primary with the paper's protocol in parallel.

**TQNet's evaluation slides the forecast origin across the test split at stride 1 with a fixed model. That
is rolling-origin evaluation, which B5 names explicitly as acceptable.** B5 is therefore satisfied by the
paper's own protocol, correctly named in the report.

The course's walk-forward *retrains* per fold. Running it in addition would cost 4 horizons × k folds and
produce numbers not comparable to the paper's, damaging report sections F3 and F5.

**Proposal: discharge the lecture-vs-homework divergence recorded in `COURSE_NOTATION` §5.1 in prose, not by
a second run.** This supersedes D15 if accepted.

## 3. Improvement pre-registration moves earlier — **approved, but see the caveat**

D10 forbids running an experiment before its pre-registration exists — it does not forbid *designing* one
earlier. Registering the improvement on Day 2 rather than Day 4 makes Day 4 implementation-only and surfaces
a dead improvement with days left rather than none.

**Caveat added after the fact:** D36 forbids dispatch #13 from recommending an improvement, and that
sequencing is deliberate — understand the method first. This amendment should be read as *"register as early
as the method analysis allows"*, not as a licence to pick an improvement before #13 returns.

## 4. Repo forensics questions — **approved as content, wherever they land**

Five questions to be answered from TQNet's code rather than the paper's prose. Dispatch #12 already covers
the run; (a), (b), (c) and (e) below are not in it and are cheap to fold in.

| | Question | Why it matters |
|---|---|---|
| a | What train/val/test ratio does the ETTh1 loader actually use? | The Informer-family convention is **12/4/4 months, not 7:1:2.** Assuming wrong manufactures a reconstruction gap we would then spend Day 3 explaining → **R16** |
| b | Is `StandardScaler` fit on the training split only? Verify by inspecting fitted parameters, not the call site | B2 is PASS/FAIL |
| c | Does early stopping / best-checkpoint selection read validation or test? | B2, "hyperparameter tuning" stage |
| d | Is the query period hand-set per dataset or learned? | `scripts/TQNet/etth1.sh` pins `--cycle 24`, so this is largely answered — confirm nothing overrides it |
| e | Does the model apply instance normalisation (per-window mean/variance), and where? | The most likely attachment point for a state-space improvement, **if** that axis is chosen after #13 |

---

## 5. R14 — MAPE and SMAPE are unusable on this benchmark

LTSF results are reported on **z-score-normalised** data, so the series crosses zero. Both metrics divide by
|yₜ| and return garbage at near-zero denominators. `COURSE_NOTATION` §2.1 already flags MAPE as undefined at
yₜ = 0; this is that flag, made concrete for this dataset.

**Proposal:** discharge B8 with **RMSE and MdAE** (both on printed sl. 48 / PDF p. 46). If MAPE is wanted,
compute it on the **original scale** and label it as such.

**Note this makes B7 and B8 cheap:** TQNet reports MSE and MAE, and both are course metrics, so **MAE
discharges the paper's-metric requirement and the course-metric requirement simultaneously.**

## 6. R15 — the improvement owner's machine

Amendment 1 moves the improvement run to Amitay. **Verify his environment during Day 2, not Day 4.**
Fallback: run on Itay's machine with Amitay pair-driving.

## 7. R16 — the split ratio

See 4(a). Settle it from code before the split module is frozen, or a self-inflicted mismatch eats Day 3.

## 8. Repo housekeeping — already applied

- **`.gitattributes`** (`* text=auto`). The working tree is CRLF and the committed blobs are LF; without this
  a future commit from a tool that does not convert would rewrite every line of every file and make diffs
  and merges unreadable.
- **`.gitignore`** — datasets, checkpoints, run outputs, `__pycache__`, `.DS_Store`. What ships as the
  dataset deliverable is decided at T5 regardless.
- **`WORKER_BRIEF_2026-07-30.md`** copied in from the parent folder — it was missing from the repo.
  **Flagged:** that copy came from the parent folder, which is stale in at least one other file, so it should
  be diffed against the manager's current version before anyone relies on it.

---

## Two things the manager should know about this session

**The parent folder is stale and was mistaken for current.** `Final Project\` outside the repo holds older
copies of `DISPATCHES`, `PLAN` and others. Advice in this session was given from those copies before the
repository versions were read. Two consequences worth recording:

1. A draft "PLAN v3" was written that described TQNet as *provisional* and proposed re-ruling a decision the
   manager had already closed as **D35**. It was never committed and has been deleted. The only parts worth
   keeping are items 1–7 above.
2. That draft also made the GPD and Kalman axes a **hard constraint** on the improvement. Dispatch rows
   22–23 record that dispatches #10 and #11 were killed for exactly that — *"I had turned one of the user's
   examples into a hard constraint after being told not to."* The same error, made a second time, by a
   session that had not read the row recording the first. **The axes are examples. They are not gates.**
3. The five-paper assessment that opened this session (PaAno, RFF-MMD, TSPulse, KAN-AD, TQNet) was scored
   against `REQUIREMENTS` only. **`SELECTION_RUBRIC` v2's gates G8 and G9 were never applied.** Selection is
   closed and this changes nothing, but the assessment should not be cited as if it were rubric-scored.

**`CANDIDATES_2024-26_2026-07-30.md` is referenced by `DISPATCHES` as DONE but is not in this repository.**
Either it was never committed or it lives only on the manager's machine. It is the evidence base for D35.
