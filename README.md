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
| **[`docs/04-paper-vs-upstream-vs-ours.md`](docs/04-paper-vs-upstream-vs-ours.md)** | The three implementations side by side — the paper as *described*, the authors' repository as *executed*, and our tree — and why we follow the code wherever the two disagree |

Supporting material:

- [`files/project/TQNET_BRIEF.md`](files/project/TQNET_BRIEF.md) — the prior deep read of
  the paper and repository, with a 20-item limitations inventory for Stage 2.
- [`report/metrics.md`](report/metrics.md) — the four metrics in the course's notation,
  and why MAPE/SMAPE are excluded.
- [`report/results.md`](report/results.md) — generated tables and figures.

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

**Stage 2 is not started.** The improvement is chosen *after* the method is understood,
not before.

| Track | Owner | Owns |
|---|---|---|
| Reconstruction and pipeline | Itay | `TQNet/` adaptations, `common/data.py`, `common/split.py`, `repro/` |
| Improvement, baseline, leakage audit, report | Amitay | `common/metrics.py`, `report/`, the improvement |

The leakage audit is deliberately owned by whoever did **not** write the pipeline.

---

## Next move

**Deadline 10.08. Stage 1 is done; everything remaining is Stage 2 and the report.**

### The decision that unblocks everything: which axis

The brief permits improving *performance, robustness, interpretability, efficiency **or**
applicability*. Only the first is closed off by our own evidence, and it is worth being
blunt about why:

- The margin TQNet claims on this cell is **0.004 MSE**.
- **Our** three-seed standard deviation is **0.00215 MSE** — half the margin, and twice
  the paper's own 0.001.
- The ablation shows that on ETTh1 **neither the Temporal Query nor the channel attention
  is measurable above that noise**, and the pure MLP is nominally best.

So an accuracy improvement here would have to beat a mechanism that itself cannot be
detected, using a metric whose noise floor is half the effect size. **Do not target MSE.**
Anything we do target must be evaluated over multiple seeds with a paired test.

Three candidate axes survive that filter, all traceable to the limitations inventory in
[`files/project/TQNET_BRIEF.md`](files/project/TQNET_BRIEF.md) §6 and the component list
in §7:

| Axis | Attaches at | Evidence it is real | Risk |
|---|---|---|---|
| **Uncertainty / probabilistic output** | the output projection, `TQNet.py:35-38` | Inventory item 7: the model has **no** uncertainty output of any kind — point forecasts only, plain MSE loss. Nothing to beat, so a result is guaranteed to exist | Must not be judged on MSE; needs its own metric (e.g. pinball loss, coverage) declared up front |
| **Robustness to a misspecified period *W*** | the query gather, `TQNet.py:53-54` | Inventory item 6: the paper's own Figure 6 shows *W*=167 scoring **worse than using no TQ at all**. A conceded, published failure | Touches periodicity → **reading PTQNet first is a prerequisite, not a nicety** |
| **Applicability to non-integer / drifting periods** | `θ_TQ`, `TQNet.py:21` | Inventory item 5: *W* must be an integer; item 1: it is hand-set per dataset | Same PTQNet prerequisite; also the largest implementation cost |

The uncertainty axis is the recommendation on a two-day clock: it has no originality
prerequisite blocking it, it attaches at a single layer, it costs about one training run,
and "the method emits no uncertainty at all" is an unarguable starting point. **The choice
is still the team's to make** — this README records the evidence, not a decision.

### Order of work

1. **Choose the axis** (above). Everything else is blocked on this.
2. **Pre-register** the improvement before running it — the predicted direction *and* a
   rough magnitude, written down first. This is a hard project rule, and requirement C1
   asks for the justification anyway.
3. **Run it** on the frozen split. `common.results.assert_split_hash` enforces
   `b66ee6b47e2b2eb8`; requirement C2 is pass/fail on this.
4. **Multiple seeds plus a paired test.** Not optional given the noise floor above.
5. **Write the report** — seven mandated sections, F1–F7, in order, as a PDF.

### Things the report must not forget

- **F5 requires a table**, and it must be three-way: paper / reconstruction / improved.
  `tools/make_report.py` assembles it from `results/runs/`, so no number is retyped.
- **F6 requires "what did not work."** Negative results are graded content here. The ETTh1
  ablation — the paper's own headline mechanism being unmeasurable at seven channels — is
  the strongest thing we have for that section, and it is already measured.
- **Say plainly what "reconstruction" means in our case.** We vendored and re-ran the
  authors' model; what we reimplemented independently is the data path and the metrics.
  That is defensible and it is *why* the 0.045% match is meaningful, but a reader must not
  be left to assume we rewrote the architecture from scratch.
  [`docs/04`](docs/04-paper-vs-upstream-vs-ours.md) §4.3 states this in the form the report
  can reuse.
- **Six paper/code disagreements** are a finding worth reporting in their own right, not an
  implementation footnote. See [`docs/04`](docs/04-paper-vs-upstream-vs-ours.md) §4.1.

### Still open

- **PTQNet** (Xun et al., *Inf. Process. Manage.* 63(7):104785, Apr 2026) is paywalled and
  unread. It blocks the two period-related axes above. University library access should
  get it.
- Five submission artefacts are required (D1–D5): reconstruction code, improvement code,
  dataset or documented download, the PDF report, and this README.

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
| `docs/` | The three documents above |
| `report/` | Report material. `metrics.md` and `audit.md` are written; `results.md` is generated |
| `results/runs/` | One JSON per run. Committed — the report is assembled from these |
| `tests/` | `pytest` suite over `common/` and `tools/`. Run with `python3 -m pytest` from the root |
| `files/project/` | The assignment brief, the paper, and `TQNET_BRIEF.md` |
| `files/lectures/` | The ten lecture PDFs, and nothing else. One of them, `CPDexamples.pdf`, is **Laurent Oudre's** ENS deck rather than this course's — cite him, never Rika |

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
  object store.

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
