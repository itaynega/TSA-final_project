# TSA-final_project

Time Series Analysis final course project — Amitay & Itay. Instructor: Havana Rika.
**Deadline 10.08.**

Reconstruction and improvement of **TQNet** (Lin et al., *Temporal Query Network for
Efficient Multivariate Time Series Forecasting*, ICML 2025) on **ETTh1**.

> This README orients the team and gets the code running. The documentation the
> assignment asks for is in **`docs/`**; the report material is in **`report/`**.

---

## Quick start

```bash
python3 -m pip install -r requirements.txt

python3 tools/get_data.py                                # fetch + verify ETTh1.csv
python3 tools/check_env.py --json results/environment.json
python3 -m pytest -q
python3 tools/audit_split.py --markdown report/audit.md   # leakage audit, must PASS

python3 tools/run_baseline.py                            # seasonal-naive baseline
bash repro/run_reconstruction.sh                         # train TQNet (~30 s on CPU)
python3 tools/collect_results.py                         # ingest + cross-check
python3 tools/make_report.py                             # tables + figures
```

No GPU needed. The whole Stage-1 pipeline is under three minutes on a laptop CPU.

## Documentation

Written to answer the three questions the project brief asks. Read them in order.

| Document | Contents |
|---|---|
| **[`docs/01-paper-and-method.md`](docs/01-paper-and-method.md)** | The paper, the problem it attacks, the Temporal Query algorithm, and how all of it maps onto the course material |
| **[`docs/02-architecture-and-implementation.md`](docs/02-architecture-and-implementation.md)** | The architecture step by step, how the code implements it, six paper/code disagreements, and **every change we made to the vendored code** |
| **[`docs/03-running-the-experiments.md`](docs/03-running-the-experiments.md)** | How the training data is generated, how training and validation run, and how to reproduce every number |
| **[`docs/STATUS.md`](docs/STATUS.md)** | **Open gaps, each with the command that closes it.** Read this before picking up work |

Supporting material:

- [`files/project/TQNET_BRIEF.md`](files/project/TQNET_BRIEF.md) — the prior deep read of
  the paper and repository, with a 20-item limitations inventory for Stage 2.
- [`report/metrics.md`](report/metrics.md) — the four metrics in the course's notation,
  and why MAPE/SMAPE are excluded.
- [`report/results.md`](report/results.md) — generated tables and figures.
- [`report/paper_code_divergences.md`](report/paper_code_divergences.md) — the six places
  the TQNet paper and the TQNet code disagree, and which one we followed.

## Where things stand

**Stage 1 is reproduced.** ETTh1, multivariate, *L*=96 → *H*=96, seed 2024:

| | MSE | MAE |
|---|---|---|
| Paper (authors' own run, full precision) | 0.3712166 | 0.3928201 |
| Ours | 0.3710499 | 0.3927240 |
| Difference | **−0.000167** (−0.045%) | −0.000096 (−0.024%) |

The MSE gap is **0.17× the paper's own three-seed sigma** of 0.001 — inside the noise the
authors measured, despite a different Python, PyTorch, numpy and a CPU instead of an
RTX 4090.

Two results beyond the reproduction, both in `docs/03` §3.7:

- **Our own seed spread is 0.00215 MSE** over seeds 2024/2025/2026 — about twice the
  paper's. That, not the paper's figure, is the bar Stage 2 has to clear.
- **On ETTh1 the Temporal Query is not measurable.** Removing it costs +0.00073 MSE and
  the pure MLP is nominally *better* than the full model — every variant inside one seed
  sigma. Not a refutation (the paper's claims are made on >100-channel datasets), but a
  hard limit on what our cell can demonstrate, and directly relevant to choosing an
  improvement.

> ⚠️ **Both findings above are currently untraceable.** They come from five runs whose
> record files were never committed to `results/runs/`, which holds only the baseline and
> the target cell. Rerunning and ingesting them is ~5 minutes — see
> [`docs/STATUS.md`](docs/STATUS.md) **G2**.

**Stage 2 is not started.** The improvement is chosen *after* the method is understood,
not before.

**Open gaps before Stage 2.** The leakage audit was run and ruled (`docs/03` §3.2) but its
artefacts were never committed, so B2 — the brief's only PASS/FAIL requirement — has no
evidence a grader can check. That plus four other gaps are listed with their fix commands
in [`docs/STATUS.md`](docs/STATUS.md). **Read it before picking up work.**

| Track | Owner | Owns |
|---|---|---|
| Reconstruction and pipeline | Itay | `TQNet/` adaptations, `common/data.py`, `common/split.py`, `repro/` |
| Improvement, baseline, leakage audit, report | Amitay | `common/metrics.py`, `report/`, the improvement |

The leakage audit is deliberately owned by whoever did **not** write the pipeline.

## Settled — do not reopen

- **Paper: TQNet.** Selection is closed.
- **Benchmark: ETTh1**, multivariate, `seq_len=96`; target cell `pred_len=96`, seed 2024.
- **Task type:** multivariate, supervised, deterministic point forecasting.
- **Evaluation protocol: the paper's own** — rolling-origin, stride 1, fixed model.
  **Do not add walk-forward as a second protocol:** it refits per fold, so its numbers
  would not be comparable to the paper's.
- **Metrics: MSE, MAE, RMSE, MdAE**, implemented once in `common/metrics.py`. MAE is both
  the paper's metric and a course metric. **MAPE and SMAPE are excluded** — the data is
  z-scored and crosses zero.
- **Baseline: seasonal-naive, period 24**, on the z-scored scale, train-fitted scaler,
  identical windows. Enforced by `tools/run_baseline.py` taking a `Windows` object.
- **One improvement**, costing roughly one training run.
- The improvement imports the split and metrics from `common/` and **asserts the split
  hash** (`b66ee6b47e2b2eb8` for L=H=96 on the pinned CSV).

## Layout

| Path | What it is |
|---|---|
| `TQNet/` | **Vendored upstream** at commit `15e19cb2`, Apache-2.0, `LICENSE` included, plus our documented adaptations. `README_UPSTREAM.md` is the authors' README |
| `TQNet/result_authors_reference.txt` | The authors' published numbers at full precision. Never written to |
| `common/` | The shared frozen foundation: `split.py`, `data.py`, `metrics.py`, `results.py` |
| `tools/` | Helper scripts: data fetch, env check, leakage audit, baseline, result ingest, report builder |
| `repro/` | `run_reconstruction.sh` (Stage 1) and `run_etth1_ablation.sh` (the unpublished ETTh1 ablation) |
| `docs/` | The three documents above, plus `STATUS.md` (open gaps) |
| `report/` | Report material. `metrics.md` and `paper_code_divergences.md` are written; `results.md` is generated. **`audit.md` is not yet generated — STATUS.md G1** |
| `results/runs/` | One JSON per run. Committed — the report is assembled from these. **Currently 2 of 7 runs — STATUS.md G2** |
| `tests/` | `pytest` suite over `common/` and `tools/`. Run with `python3 -m pytest` from the root |
| `files/project/` | The assignment brief, the paper, and `TQNET_BRIEF.md` |
| `files/lectures/` | The ten course decks, plus `CPDexamples.pdf` (which is **Laurent Oudre's** ENS deck, not this course's — cite him, never Rika) |

## Two decisions that changed since the earlier plan

**We now vendor upstream instead of shipping patches.** The earlier note said "clone
TQNet outside this repository — we ship patches, never vendored copies". That is
reversed: `TQNet/` is a full copy at a pinned commit. Apache-2.0 permits redistribution,
the `LICENSE` is included, and the reasons are practical — the repository is now
self-contained and runnable, there is no patch that can go stale against a moving clone,
and `git diff` shows exactly what we changed. Every change is listed in `docs/02` §2.6,
and §2.6 also gives the command to regenerate a diff against pristine upstream.
`repro/tqnet-ablation-flags.patch` was deleted as superseded.

**The dataset is downloaded, not committed.** `tools/get_data.py` fetches ETTh1.csv from
a pinned ETDataset commit and verifies it against a sha256, a row count, the column
names and the date range. The assignment permits a documented download, and a pinned
digest proves we evaluated on the same bytes rather than claiming to.

## Working rules

- **This repo is the single source of truth.** If a document contradicts this README or
  `docs/`, the document is stale.
- **One writer per file.**
- **No experiment before its pre-registration exists** — the predicted number is written
  down first.
- The leakage audit is run by whoever did **not** write the pipeline.
- Never commit datasets, checkpoints or raw run arrays — see `.gitignore`. Do commit
  `results/runs/*.json`.
- Pull before you start, push when you stop.
- **Do not put your clone inside a OneDrive- or Dropbox-synced folder.** Sync clients
  lock and rewrite files under `.git/` while git is using them, which corrupts the
  object store. ⚠️ **Amitay's clone currently breaks this rule** and stale
  `.git/*.lock` files prove OneDrive is interfering. No corruption yet — see
  [`docs/STATUS.md`](docs/STATUS.md) **G5** for the fix.

## Environment

Validated on Python 3.13.5 / torch 2.9.1 / numpy 2.2.6 / pandas 2.3.3 /
scikit-learn 1.8.0, macOS arm64, CPU. Pins are in `requirements.txt`;
`python3 tools/check_env.py` reports drift.

Note that **unmodified upstream code cannot run on any of these versions** — it calls a
pandas API removed in 2.0 and a numpy alias removed in 2.0. The vendored copy is fixed;
see `docs/02` §2.6.

## Setup

```bash
git clone https://github.com/itaynega/TSA-final_project.git
```
