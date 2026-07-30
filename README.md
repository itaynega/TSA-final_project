# TSA-final_project

Time Series Analysis final course project — Amitay & Itay. Instructor: Havana Rika. **Deadline 10.08.**

> This README orients the two of us. It is **not** the README the assignment asks for (requirement D5) —
> that one covers the code deliverable and is written on Day 5 as task T18.

## Read these, in this order

| File | What it is |
|---|---|
| `files/project/Final_Project.pdf` | The assignment brief. The only external authority |
| `files/project/REQUIREMENTS_2026-07-30.md` | Every requirement quoted verbatim, with IDs (A1–A5, B1–B10, C1–C2, D1–D5, E1–E3, F1–F7) |
| **`files/project/DISPATCHES_2026-07-30.md`** | **The live document. Start here.** Every dispatch, its audit, and the decision log — currently ahead of the PLAN |
| `files/project/SELECTION_RUBRIC_2026-07-30.md` | Paper-selection gates and scoring. Hard gates **G8, G9** |
| `files/project/PLAN_2026-07-30_v2.md` | The plan. Behind the decision log — read DISPATCHES alongside it |
| `files/project/COURSE_NOTATION_2026-07-30.md` | What the course actually taught — metrics, baselines, notation, with slide citations |
| `files/project/AMENDMENTS_2026-07-30.md` | Proposed amendments awaiting a manager ruling. Not plan text |
| `files/project/WORKER_BRIEF_2026-07-30.md` | Brief for an execution session |
| `files/project/PM_HANDOVER_2026-07-30.md` | Brief for a manager session |

## Where things stand

**Paper: TQNet (ICML 2025) — LOCKED (D35).** Selection is closed. Next: dispatches **#12** (clone-and-run
probe — blocking) and **#13** (method and limitations), sent together in two fresh conversations.

**Ownership (D22):** Itay owns reconstruction and the shared `common/` foundation. Amitay owns the
improvement, metrics, baseline, report, and the leakage audit — the audit deliberately sits with whoever
did *not* write the pipeline.

**Two PASS/FAIL requirements govern everything:** B2 (no future information in training, preprocessing,
feature construction, or hyperparameter tuning) and C2 (the improved method on the *identical* split and
metrics as the reconstruction).

## Working rules

- **This repo is the single source of truth (D23).** Copies elsewhere are stale.
- **One writer per file (D5).** Check PLAN §3 for who owns what before editing.
- **No experiment before its pre-registration exists (D10).**
- Never commit datasets, checkpoints or run outputs — see `.gitignore`. What ships as the dataset
  deliverable is decided at T5.
- Pull before you start, push when you stop. Two people editing one markdown file is a merge conflict
  nobody needs at this deadline.

## Setup

```
git clone https://github.com/itaynega/TSA-final_project.git
```

**Do not put your clone inside a OneDrive- or Dropbox-synced folder.** Sync clients lock and re-write files
under `.git/` while git is using them, which corrupts the object store. Somewhere like `C:\dev\` is fine.
