# Status — open gaps

**Re-audited 2026-08-09 19:05 IDT against `HEAD` = `3d807b0d064c063488a53dcae6d8d4a9cd8278e7`,
by reading the repository rather than a status report.**

> **This file was previously audited on 2026-07-31 at commit `6b6777e` and had gone badly stale:
> it still listed G1, G3 and G4 as open when all three had closed, and it named Itay as owner of
> two gaps after his machine became unavailable.** The 07-31 text is preserved in git history at
> `6b6777e:docs/STATUS.md`. Anyone who read this file between 13:00 and 19:00 on 09.08 got a
> wrong picture of the project.

**None of these are design problems.** Stage 1 reproduces on two architectures, the tests pass, and
the numbers are internally consistent. What was missing was the *artefact trail*. Most of it now
exists.

---

## Summary — 09.08 19:05

| # | Gap | State | Evidence |
|---|---|---|---|
| **G1** | Leakage-audit artefacts never committed | **CLOSED 12:41** | Commit `9663bcd`. `report/audit.md` (4,067 B) and `results/audit.json` (5,091 B). 10 rulings, 7 CLEAN / 3 DISCLOSE, re-read from the committed blob |
| **G2** | 5 of 7 runs have no committed run record | **CLOSED 2026-08-10** — was recorded as half-uncloseable on 09.08 on a premise that proved false | Commit `8ac3f9b`. The arm64 machine was reachable and its artefacts intact; the two missing ablation variants were ingested. See below |
| **G3** | Only horizon 96 exists; F5 needs four | **CLOSED 13:50** | Commit `ee6e334`. 12 new records at H = 96/192/336/720 × seeds 2024/2025/2026. 14 records total |
| **G4** | Neither pre-registration exists | **CLOSED for the improvement; permanently open, by design, for the reconstruction** | Commit `ac426c7` (frozen text) + `3d807b0` (Amendments). No arm had run when it landed |
| **G5** | Repo lives inside OneDrive, against D23 | **OPEN, knowingly accepted** | Amitay's decision, 12:00. `git fsck` silent, no corruption ever observed |
| **G6** | Stage 2 not started | **IN PROGRESS — Gates 0–2 complete, Gate 3 next** | `STAGE2_WORKPLAN_2026-08-09.md` §-1 carries the live state |

---

## G2 — CLOSED 2026-08-10

> **Correction, 2026-08-10.** Everything below the rule was written on 2026-08-09 and asserted that
> half of this gap *"cannot close, ever"*. **That was wrong, and the premise was wrong rather than
> the reasoning.** The original text is retained verbatim underneath, per this project's norm of
> marking superseded claims instead of deleting them.
>
> **The machine was never gone.** The macOS/arm64 machine is reachable; its five Stage-1 checkpoints
> and all five sets of `pred.npy` / `true.npy` have been on disk untouched since 2026-07-30
> 18:44–18:51. They were invisible to every audit conducted from git because `.gitignore` excludes
> `TQNet/results/` wholesale — the artefacts were never lost, only unreadable from the one place
> anyone looked.
>
> **What closed it.** `tools/collect_results.py` ingested the two variants that had no record, at
> commit `8ac3f9b`:
>
> | Variant | MSE `[test]` | Record |
> |---|---|---|
> | no-TQ (self-attention) | `0.3717761290076582` | `results/runs/reconstruction-TQNet-s2024-h96-1786379059318751000.json` |
> | pure MLP | `0.37096275348328595` | `results/runs/reconstruction-TQNet-s2024-h96-1786379059385966000.json` |
>
> Both passed the split-hash assertion (`b66ee6b47e2b2eb8`), the 2,785-window check, and agreement
> with TQNet's own float32 metrics to under `2e-7` relative. The published variant at seed 2024 was
> already committed on 2026-07-30 (`reconstruction-TQNet-s2024-h96-1785426343196465000.json`), so the
> ablation triple is now complete and `tools/make_report.py`'s ablation table renders from committed
> records for the first time. The deltas — `+0.000726` (0.34σ) for no-TQ and `−0.000087` (0.04σ) for
> pure MLP — reproduce `docs/03` §3.7 and `report/prereg-improvement.md` §1 exactly.
>
> **Three seeds of the full model were deliberately NOT re-ingested.** `tools/make_report.py` keys
> its seed-spread table by seed and takes the most recent record, so fresh arm64 records would have
> displaced the x86 re-baseline and reported a σ mixing two architectures — the conflation
> `tools/horizon_sigma.py` exists to prevent. The x86 σ stands.
>
> **Job J-07′ is therefore unnecessary** as a rerun; the ablation no longer needs superseding. **T15
> is satisfied for these numbers** — they are traceable and may be printed. The `M6` reading below
> still holds as a lesson about *where the error started*, but the bill it describes did not arrive.

---

*Superseded text, 2026-08-09, retained:*

The 07-31 text said "rerun and ingest, ~5 minutes". That fix is **no longer available for half of
this gap**, and the reason matters enough to state plainly.

**The half that closed.** The seed-spread numbers now have committed records — twelve of them, at
`HEAD~2` (`ee6e334`), run on Amitay's x86 machine. The x86 measurement of σ at H = 96 came out at
**0.002154**, reproducing the original arm64 figure exactly.

**The half that cannot close.** The ETTh1 ablation triple and the original seed spread in
`docs/03` §3.7 were produced on **Itay's macOS/arm64 machine, and that machine became unavailable on
2026-08-09**. Their run records were never committed. They therefore cannot be traced, ever — not
because anyone lost a file, but because the machine that would have to re-run them is gone.

**Under T15 those numbers do not get printed.** They are being *marked in place and retained*, not
deleted (job J-02b in the work plan), because retaining them marked is honest and because they are a
concrete worked example of why the traceability rule exists. **M6** in `PLAN.md` §8 is where the
error started; this is the bill arriving.

Their x86 replacements: the seed spread is superseded by `ee6e334`; the ablation is superseded by
job J-07′, which reruns it at three seeds per variant on x86.

**Note for anyone rerunning the ablation:** `repro/run_etth1_ablation.sh` takes `SEED` **singular**
and its `VARIANTS` array holds **two** entries. Passing `SEEDS=` is silently ignored. Loop outside
the script:

```bash
for s in 2024 2025 2026; do SEED=$s bash repro/run_etth1_ablation.sh; done
```

---

## G4 — closed for the improvement, and it stays open for the reconstruction

**Improvement (T13′) — CLOSED.** `report/prereg-improvement.md` was committed as `ac426c7` at
2026-08-09T18:36+03:00, **before any arm ran**, which is what D10 requires. Its text is frozen; the
one correction since (Arm D's parameter count, 37,248 → 37,416) was appended as a dated
`## Amendments` block in `3d807b0` with zero deletions, so the frozen text is still byte-identical
at its own commit.

**Reconstruction — permanently open, and this is the correct outcome.** The reconstruction had
already run before any pre-registration was written, so one cannot now be written honestly.
**It is not backdated.** The report says plainly that the reconstruction's threshold was set after
the fact. A fabricated pre-registration would be worse than none.

---

## G5 — OneDrive, and what actually happened

The clone is at `…/OneDrive/…/Final Project/TSA-final_project`, against D23. Amitay elected at 12:00
to stay there for the remainder of the project and accept the risk.

**The `index.lock` question is settled, and it was not OneDrive.** Two workers independently
diagnosed stranded `.git/index.lock` files as OneDrive corruption. They were wrong. The Cowork
session's mounted view of the filesystem refuses `unlink`, so any git command run *through the mount*
strands its lock. The discriminating observation: **no lock existed at 12:43, two minutes after a
commit made from a real terminal.** `git fsck` has been silent at every check.

**As of 19:05 the fix is stronger still: no git command runs from a terminal at all.** All git goes
through **GitHub Desktop** (D25), which is a native Windows application and touches the repository
the way git expects. The mount is used for reads only.

---

## G6 — Stage 2, live state

Gates 0, 1 and 2 are complete apart from two off-critical-path jobs (J-07′, J-02b). **Gate 3 — the
arms — is next and has not started.** Arm C was dropped at its pre-declared 17:00 go/no-go.

`STAGE2_WORKPLAN_2026-08-09.md` is the live plan; its **§-1 STATE** block is the one place to look
for where the project is. This file describes repository gaps only.
