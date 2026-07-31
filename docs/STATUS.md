# Status — open gaps

Audited **2026-07-31** against the repository at commit `6b6777e`, by reading the files rather
than by reading a status report. Every gap below names the command that closes it.

**None of these are design problems.** Stage 1 reproduces, the tests pass, and the numbers are
internally consistent. What is missing is the *artefact trail* — results that exist in prose but
not in committed files. That distinction matters because the project's own rule (PLAN §5 T15) is
"a number that cannot be traced does not get printed."

---

## Summary

| # | Gap | Severity | Owner | Cost |
|---|---|---|---|---|
| **G1** | Leakage-audit artefacts never committed | **Highest — B2 is the brief's only PASS/FAIL** | Amitay | ~1 min |
| **G2** | 5 of 7 runs have no committed run record | High — breaks traceability | Itay | ~5 min |
| **G3** | Only horizon 96 exists; F5 needs four | Medium | Itay | ~3 min |
| **G4** | Neither pre-registration exists, though D10 requires them | Medium — process, and it is a marked item | Amitay | ~30 min |
| **G5** | Repo lives inside OneDrive, against D23 | High — silent corruption risk | Both | ~10 min |
| **G6** | Stage 2 not started | Expected, not a defect | Amitay | — |

---

## G1 — The leakage audit has no committed evidence

**This is the most important item on the page.** Requirement **B2** is the only PASS/FAIL
requirement in the brief:

> "The reconstruction must respect the temporal structure of the data: future observations must
> not be used during training, preprocessing, feature construction, or hyperparameter tuning."

**The audit was genuinely run.** `docs/03` §3.2 reports all ten checks with their rulings — seven
CLEAN, three DISCLOSE — including the scaler check done properly (confirming `scaler.mean_` matches
the training rows *and* differs measurably from the whole-series statistics, which is what makes the
first half of the check meaningful).

**But neither output file exists:**

- `report/audit.md` — **missing**, despite `README.md` stating it is written
- `results/audit.json` — **missing**

So the strongest single piece of evidence in the project is currently unverifiable by a grader.

**Fix — needs the dataset, needs no GPU and no torch** (`tools/audit_split.py` imports only
numpy, pandas and sklearn):

```bash
python3 tools/get_data.py
python3 tools/audit_split.py --markdown report/audit.md --json results/audit.json
git add report/audit.md results/audit.json && git commit -m "Commit the leakage audit artefacts"
```

The script exits non-zero if any check fails, so a clean exit is itself the result.

---

## G2 — Five of seven runs have no run record

`results/runs/` holds **two** JSONs: the seasonal-naive baseline and the target cell. But
`docs/03` §3.7 and `README.md` quote numbers from **seven** runs:

| Run | Quoted in | Record in `results/runs/`? |
|---|---|---|
| Baseline, seasonal-naive 24 | `report/results.md` | **yes** |
| TQNet seed 2024 (target cell) | everywhere | **yes** |
| TQNet seed 2025 | `docs/03` §3.7 | **no** |
| TQNet seed 2026 | `docs/03` §3.7 | **no** |
| `--use_tq 0` | `docs/03` §3.7 | **no** |
| `--use_tq 0 --channel_aggre 0` | `docs/03` §3.7 | **no** |

Both headline secondary findings — **the 0.00215 seed sd** and **the ablation showing the Temporal
Query is unmeasurable on ETTh1** — rest entirely on runs with no committed record. The seed sd is
the bar Stage 2 is measured against, so this is not a bookkeeping detail.

`README.md` says of `results/runs/`: *"One JSON per run. Committed — the report is assembled from
these."* That is currently not true.

**Fix — rerun and ingest** (each run is ~33 s on CPU):

```bash
SEEDS="2025 2026" bash repro/run_reconstruction.sh
bash repro/run_etth1_ablation.sh
python3 tools/collect_results.py
git add results/runs/ && git commit -m "Commit run records for the seed spread and the ETTh1 ablation"
```

If any rerun disagrees with the number already written in `docs/03`, **the run wins** — correct the
document, and say so in the report's discussion.

---

## G3 — Only horizon 96 exists

PLAN §5 requires **F5 to be a three-way table (paper / reconstruction / improved) with one row per
horizon**, and the authors publish ETTh1 at 96 / 192 / 336 / 720. Only 96 has been run.

`TQNet/result_authors_reference.txt` already carries the reference numbers at full precision for all
four, so the comparison column costs nothing:

| Horizon | Authors' MSE | Authors' MAE |
|---|---|---|
| 96 | 0.3712165653705597 | 0.3928201496601105 |
| 192 | 0.4283985197544098 | 0.4260946214199066 |
| 336 | 0.4757070839405060 | 0.4460628032684326 |
| 720 | 0.4874295890331268 | 0.4697666168212890 |

Three extra runs turn F5 from one row into four, and a pattern across horizons is far more
convincing than a single cell. Roughly two minutes of compute.

```bash
PRED_LENS="192 336 720" bash repro/run_reconstruction.sh
python3 tools/collect_results.py && python3 tools/make_report.py
```

*Verified: `repro/run_reconstruction.sh` reads `PRED_LENS` (default 96) and `SEEDS` (default 2024),
and loops over both. `repro/run_etth1_ablation.sh` runs all three variants by default.*

---

## G4 — Neither pre-registration exists

**D10: "No experiment before its pre-registration exists."** Experiments have run. Neither
pre-registration is in the repository:

- **Reconstruction pre-registration** — the predicted number and the reproduced/not-reproduced
  threshold, fixed before seeing anything. The reconstruction has already run, so this one can no
  longer be written honestly. **Do not backdate it.** Say plainly in the report that the threshold
  was set after the fact, or omit the claim. A fabricated pre-registration is worse than none.
- **Improvement pre-registration (T13′)** — **not yet compromised, because Stage 2 has not started.**
  Write it before the first improvement run. It needs: the derivation, a **quantitative** prediction,
  pre-fixed thresholds, a STOP condition, and what result would make you abandon the idea.

The improvement's threshold must clear **our** seed sd of 0.00215, not the paper's 0.001 — and note
that 0.00215 is already half the 0.004 margin TQNet claims over CycleNet. Suggested home:
`report/prereg-improvement.md`.

---

## G5 — The repository is inside OneDrive

`README.md` says, correctly:

> "Do not put your clone inside a OneDrive- or Dropbox-synced folder. Sync clients lock and rewrite
> files under `.git/` while git is using them, which corrupts the object store."

The clone is currently at `…/OneDrive/שולחן העבודה/…/Final Project/TSA-final_project`.

**This is already happening.** As of the audit, `.git/` contained stale artefacts of an interrupted
write, all timestamped the same minute:

```
.git/HEAD.lock
.git/index.lock
.git/objects/maintenance.lock
.git/objects/{0d,4b,6b,bf}/tmp_obj_*
```

`git fsck` reports **no corruption yet** and `HEAD` matches `origin/main`, so nothing is lost. That
is luck, not safety — and stale `index.lock` will eventually block a commit with
`Another git process seems to be running`.

**Fix.** Everything is pushed, so re-cloning outside OneDrive is clean and loses nothing:

```bash
cd ~/Desktop            # any path NOT under OneDrive
git clone https://github.com/itaynega/TSA-final_project.git
```

Keep the planning documents (`PLAN.md`, `REQUIREMENTS_…md`, `COURSE_NOTATION_…md`) in OneDrive if
that is convenient — they are prose and sync fine. It is `.git/` that must not be synced.

If you prefer to stay put for now, at minimum delete the stale locks before the next commit:

```bash
rm -f .git/HEAD.lock .git/index.lock .git/objects/maintenance.lock
find .git/objects -name 'tmp_obj_*' -delete
git fsck        # expect silence
```

---

## G6 — Stage 2 not started

Expected, not a defect: the improvement is chosen *after* the method is understood. See PLAN §5.

The ablation finding in `docs/03` §3.7 bounds the choice and should be read before committing to a
direction — **on ETTh1, neither the Temporal Query nor the channel-attention layer is measurable
above run-to-run noise**, so an improvement aimed at the TQ mechanism would be tuning a component
this dataset cannot resolve.

---

## Suggested order

1. **G5** — do it first. Everything else writes to git, and this protects the writes.
2. **G1** — one command, closes the brief's only PASS/FAIL requirement.
3. **G2** and **G3** — one sitting, ~10 minutes of compute, together they make every number traceable.
4. **G4** — write the improvement pre-registration *before* the first Stage 2 run.
5. **G6** — Stage 2.
