# 3. Running the experiments

Everything in this document is runnable from the repository root. Total compute for the
whole Stage-1 pipeline is **under three minutes on a laptop CPU** — the target cell
trains in about 30 seconds — so nothing here needs a GPU or a queue.

---

## 3.0 The short version

```bash
python3 -m pip install -r requirements.txt

python3 tools/get_data.py                              # fetch + verify ETTh1.csv
python3 tools/check_env.py --json results/environment.json
python3 -m pytest -q                                   # 40 tests
python3 tools/audit_split.py --markdown report/audit.md # leakage audit, must PASS

python3 tools/run_baseline.py                          # seasonal-naive baseline
bash repro/run_reconstruction.sh                       # train TQNet, the target cell
python3 tools/collect_results.py                       # ingest + cross-check
python3 tools/make_report.py                           # tables + figures
```

The last step writes `report/results.md` and `report/figures/*.png`.

---

## 3.1 Where the data comes from

There is **no synthetic data and no feature engineering.** The entire input is one CSV
of real hourly measurements, and the "training data" is windows cut from it. This is
worth being explicit about, because "how do we generate the training data" has a
slightly surprising answer for this kind of paper: the generation step is *windowing*,
and everything that can go wrong is in how the windows are cut and scaled.

### The file

**ETTh1** — Electricity Transformer Temperature, hourly, station 1. Introduced by
Informer (Zhou et al., 2021), and the standard long-horizon forecasting benchmark.

| Property | Value |
|---|---|
| Rows | 17,420 |
| Timestamp range | 2016-07-01 00:00:00 .. 2018-06-26 19:00:00 |
| Interval | 1 hour, strict grid, no gaps |
| Missing values | **0 cells** |
| Channels | `HUFL HULL MUFL MULL LUFL LULL OT` — six power-load measurements plus oil temperature |
| Source | [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset), `ETT-small/ETTh1.csv`, CC BY-ND 4.0 |
| sha256 | `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` |

```bash
python3 tools/get_data.py            # download, then verify
python3 tools/get_data.py --verify   # verify only, no network
```

The dataset is **not committed** — `.gitignore` excludes `*.csv`. The assignment permits
a documented download, and a pinned URL plus a digest is a stronger guarantee than a copy
in git: it proves we evaluated on the same bytes rather than merely claiming to. The URL
is pinned to the commit that last touched the file (2020-12-09), so it cannot start
serving something else.

`get_data.py` checks **four** things and treats any mismatch as fatal: the sha256, the
row count, the column names, and the first and last timestamps. A silently different CSV
is the one failure mode that yields a plausible-looking MSE that cannot be compared to
0.3712.

### Preprocessing, in full

The complete list, because the short list is the point:

1. **Parse the `date` column** as a timestamp.
2. **Take the seven numeric columns** in file order.
3. **Fit a `StandardScaler` on training rows `[0, 8640)` only**, then apply it to all
   17,420 rows.
4. **Cut windows** at stride 1.

That is all. Specifically **not** done, and each omission is deliberate:

- *No resampling* — the series is already on a strict hourly grid.
- *No missing-value handling* — there are no missing values. Imputing a complete series
  would be a silent deviation from the paper.
- *No outlier removal* — the extreme values are the phenomenon TQNet claims robustness
  to. Removing them would erase the effect being tested.
- *No feature construction* — the loader computes calendar features (month, day,
  weekday, hour) but `TQNet.forward(x, cycle_index)` never receives them. The only
  derived input is the cycle index, which is `row mod 24`.

### The split

Chronological, 12 / 4 / 8-month boundaries taken from the loader
(`data_loader.py:49-50`), inherited from Informer and shared by every model in the
paper's comparison table.

| Split | CSV rows | Length | First target row | Windows (L=H=96) |
|---|---|---|---|---|
| train | `[0, 8640)` | 8,640 | 96 | **8,449** |
| val | `[8544, 11520)` | 2,976 | 8,640 | **2,785** |
| test | `[11424, 14400)` | 2,976 | 11,520 | **2,785** |
| never used | `[14400, 17420)` | 3,020 | — | — |

Three things to notice:

- **2,785 test windows** is the denominator behind the paper's 0.3712. Each is a 96×7
  matrix in, 96×7 out; the origin slides forward one hour for the next.
- **3,020 rows are discarded.** 12+4+4 = 20 months of 30 days = 14,400 rows, and the
  loader stops there. The paper's Table 1 reports ETTh1 as 14,400 timesteps without
  mentioning that anything was dropped. Reproducing the paper means dropping them too.
- **Validation and test start 96 rows early**, so that each split's *first target* lands
  exactly on its month boundary with real history behind it. Those early rows are inputs
  only — never targets, never fitted on. The audit reports this explicitly as a
  `DISCLOSE` item so it is not mistaken for leakage later.

### How windows become training examples

For window *i* of a split beginning at CSV row `b`:

```
inputs   x = rows[b+i        : b+i+96]     shape (96, 7), z-scored
targets  y = rows[b+i+96     : b+i+192]    shape (96, 7), z-scored
phase      = (b + i + 96) mod 24           the clock hour of the first forecast step
```

Inputs and targets are adjacent and disjoint — no gap, no overlap. `label_len` is fixed
at 0 for this model, so there is no decoder-style teacher-forcing prefix.

`common/data.py` reimplements this in numpy so the baseline and the audit can see
exactly these arrays without a DataLoader, and
`tests/test_data.py::test_windows_match_the_upstream_dataset_exactly` checks the
reimplementation against the real vendored `Dataset_ETT_hour` element for element on all
three splits. That test is what licenses every later claim that the baseline is scored on
"the same windows".

---

## 3.2 No future information: how it is checked

This is the assignment's pass/fail requirement, so it is verified rather than asserted:

```bash
python3 tools/audit_split.py --markdown report/audit.md --json results/audit.json
```

Ten checks, each recomputed from the CSV and from the *fitted* scaler object — not by
re-reading the source and agreeing with it. The audit exits non-zero if any check fails.

| Stage | What is checked | Result |
|---|---|---|
| Grid integrity | strict hourly grid, no gaps, no NaN — otherwise `row mod 24` is not clock phase | CLEAN |
| Chronology | splits ordered in time; target ranges non-overlapping | CLEAN |
| Scaler provenance | `scaler.mean_` / `scaler.scale_` equal the training-rows statistics **and** differ measurably from the whole-series statistics | CLEAN |
| Target disjointness | no row is a target in two splits | CLEAN |
| Input reach-back | val/test inputs read 96 rows before their boundary — by design | DISCLOSE |
| Model selection | early stopping and checkpointing key on validation loss, never test | CLEAN |
| Observation hygiene | upstream evaluates and prints test loss every epoch | DISCLOSE |
| Feature construction | cycle index reproducible from the row number alone | CLEAN |
| Covariates | calendar features computed but never consumed by the model | CLEAN |
| Metric scale | de-normalisation left commented out, so metrics are in z-scored units | DISCLOSE |

The scaler check deserves a note, because it is the one that usually fails in projects
like this and the naive version of the test is worthless. Confirming that
`scaler.mean_` matches the training rows is not enough on its own: if the training mean
and the whole-series mean happened to coincide, a leaking scaler would pass. So the
audit also confirms they differ — here by up to 3.80 in raw units — which is what makes
the first half of the check meaningful.

The three `DISCLOSE` items are all upstream's design, not ours. None is leakage; all
three belong in the report.

---

## 3.3 The baseline

```bash
python3 tools/run_baseline.py               # seasonal-naive, period 24
python3 tools/run_baseline.py --period 1    # degenerates to the persistence naive
```

Seasonal-naive at period 24: the forecast for step *h* is the observation 24 hours
before the corresponding future time, i.e. "tomorrow looks like today". Since *H* = 96 =
4×24, the last observed day is tiled four times.

It is the right baseline here rather than a conventional one, because TQNet's `--cycle
24` encodes **the same** daily-periodicity assumption. The gap between them is therefore
what the network adds *on top of* knowing that ETTh1 repeats daily, rather than a
comparison between a model that knows about seasonality and one that does not.

Three properties make it comparable to the paper's number, and all three are easy to get
wrong in ways that leave a plausible result behind:

- computed on **z-scored** data with the **train-fitted** scaler;
- computed on the **identical 2,785 windows**, by taking a `Windows` object rather than
  re-deriving one;
- scored by the **same** `common/metrics.py` the model is scored by.

---

## 3.4 Training and validation

```bash
bash repro/run_reconstruction.sh                          # target cell, seed 2024
SEEDS="2024 2025 2026" bash repro/run_reconstruction.sh   # seed spread
PRED_LENS="96 192 336 720" bash repro/run_reconstruction.sh
ACCELERATOR=cpu bash repro/run_reconstruction.sh          # force a device
```

Every flag in the script is copied verbatim from `TQNet/scripts/TQNet/etth1.sh`. **Nothing
is tuned.** That is the reason this cell was chosen: the authors pinned every
hyperparameter, so there is nothing to guess, and any difference from their number is a
fact about our environment rather than about our choices.

### The training loop

Per epoch:

1. Iterate the 8,449 training windows in shuffled batches of 256 (33 steps, last partial
   batch dropped).
2. Forward, MSE loss on z-scored values, backward, Adam step.
3. Evaluate mean MSE on the validation split.
4. `EarlyStopping` saves `checkpoint.pth` whenever validation loss improves, and stops
   after 5 epochs without improvement.
5. Adjust the learning rate: constant for 3 epochs, then ×0.8 per epoch.

Up to 30 epochs. After the loop, the **best-validation** checkpoint is reloaded and
evaluated once on the test split.

**Validation is used for two things and only two things:** early stopping and choosing
which checkpoint to keep. It is never used to tune a hyperparameter, because no
hyperparameter is being tuned.

One upstream quirk to be aware of when reading the logs: the validation loader inherits
`shuffle=True, drop_last=True` from `data_factory.py`, so 2785 mod 256 = 225 validation
windows (8%) are dropped each epoch and never reach the early-stopping signal. That is
not leakage, but it does mean early stopping is computed on 92% of validation. It is
upstream's behaviour and is left alone.

### The evaluation protocol

**Rolling-origin.** One model, trained once, then evaluated with the forecast origin
advancing one hour at a time across the 2,785 test windows. Metrics are a single flat
mean over 2,785 × 96 × 7 = **1,871,520 predictions**.

Deliberately **not** walk-forward validation, which refits per fold. Walk-forward is a
valid protocol and the assignment names it, but the paper's numbers were produced under
rolling-origin, and mixing the two would make our table incomparable to theirs.

### Outputs

| Path | What |
|---|---|
| `TQNet/checkpoints/<setting>/checkpoint.pth` | best-validation weights |
| `TQNet/results/<setting>/pred.npy`, `true.npy` | the 2,785 test windows |
| `TQNet/results/<setting>/metrics.json` | upstream's own metrics, params, device |
| `TQNet/result_ours.txt` | appended one-line summary |
| `TQNet/result_authors_reference.txt` | the authors' published numbers — **never written to** |

`<setting>` is e.g. `ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024`.

---

## 3.5 Ingesting a run

```bash
python3 tools/collect_results.py                 # label as reconstruction
python3 tools/collect_results.py --arm improved  # label as Stage 2
python3 tools/collect_results.py --dry-run
```

This converts raw run output into a traceable record under `results/runs/`, and performs
three checks that a printed MSE cannot:

1. **Window count** must equal the 2,785 the split defines. A run that evaluated a
   different number of windows did not evaluate this split.
2. **Directory name and `metrics.json`** must agree on horizon, seed and cycle — this
   catches a stale directory being attributed to a new run.
3. **Our metrics must agree with upstream's** on the same arrays. Both square or take a
   modulus of the error, so the opposite sign conventions cancel and they must match.

Check 3 has an instructive tolerance. The two agree to about **1.2×10⁻⁷ relative**, not
exactly, and the reason is that the saved arrays are float32: upstream's `np.mean`
accumulates in float32 over 1.87M elements, while `common/metrics.py` casts to float64
first. Ours is the more accurate. The tolerance is sized for float32 accumulation, which
is still three orders of magnitude below the third decimal the paper prints. Anything
larger is not rounding — it means one implementation is reducing over the wrong axis —
so it is fatal rather than a warning.

---

## 3.6 Tables and figures

```bash
python3 tools/make_report.py
python3 tools/make_report.py --no-figures
```

Reads only the recorded runs, so a chart cannot disagree with the table above it. Writes
`report/results.md` plus:

| Figure | Shows |
|---|---|
| `fig1_reproduction.png` | Ours vs the authors' run, with their three-seed spread drawn to scale |
| `fig2_error_by_horizon_step.png` | MSE against how far ahead the forecast is, model vs baseline |
| `fig3_per_channel.png` | Per-channel MSE — the headline mean is very uneven across the seven |
| `fig4_forecast_examples.png` | Three real forecasts on `OT`, at the 5th, 50th and 95th percentile of difficulty |
| `fig5_error_distribution.png` | Absolute-error density and log-scale tail |
| `fig6_horizon_sweep.png` | Ours vs the paper at each horizon — only if more than one has been run |

The report states a **verdict**, not just a comparison, and the yardstick is the paper's
*own* three-seed standard deviation (Table 9): within 1σ is reproduced, within 3σ is
ordinary environment drift, beyond that means look for the fault in our setup first.
"Our MSE is 0.3710 and theirs is 0.3712" is not a finding until it is measured against
something.

---

## 3.7 What we have observed so far

Recorded here as run history. Reproduce with the commands above; re-run
`tools/collect_results.py` and `tools/make_report.py` to regenerate the tables.

**The target cell, seed 2024** — trained in 33 s on CPU, early-stopped at epoch 14:

| | MSE | MAE |
|---|---|---|
| Authors' run (`result.txt`) | 0.3712166 | 0.3928201 |
| Ours | 0.3710499 | 0.3927240 |
| Difference | **−0.000167** (−0.045%) | −0.000096 (−0.024%) |

The MSE gap is **0.17× the paper's own seed sigma** of 0.001 — inside the run-to-run
noise the authors themselves measured, on a different Python, a different PyTorch, a
different numpy, and a CPU instead of an RTX 4090. The cell reproduces.

**Our own seed spread**, three seeds at *H* = 96:

| Seed | MSE | MAE |
|---|---|---|
| 2024 | 0.371050 | 0.392724 |
| 2025 | 0.375112 | 0.393156 |
| 2026 | 0.371837 | 0.393472 |
| mean | 0.372666 | 0.393117 |
| **sd** | **0.002154** | 0.000375 |

Our MSE spread is about twice the paper's reported 0.001. **This number, not the
paper's, is the bar any Stage-2 improvement has to clear on our hardware** — and it is
already half the size of the 0.004 margin TQNet claims over CycleNet.

**The ETTh1 ablation the paper never ran** (seed 2024, all other flags identical):

| Variant | Params | MSE | vs published | vs our seed sd |
|---|---|---|---|---|
| published (TQ + attention) | 661,640 | 0.371050 | — | — |
| `--use_tq 0` (self-attention) | 661,472 | 0.371776 | +0.000726 | 0.34× |
| `--use_tq 0 --channel_aggre 0` (pure MLP) | 624,224 | 0.370963 | **−0.000087** | 0.04× |

This is the most interesting result so far, and it needs stating carefully.

**On ETTh1, every ablation lands well inside one seed standard deviation of the full
model, and the pure MLP is nominally the best of the three.** Neither the Temporal Query
nor the channel-attention layer is measurable above run-to-run noise here. The MLP alone
accounts for essentially all of the reported accuracy.

That is **not** a refutation of the paper. TQNet's claims are made on datasets with more
than 100 channels, where a channel-correlation mechanism has something to work with; at
*C* = 7 the attention map is 7×7 and there is very little structure available. The
paper's own decision to ablate only on Electricity and PEMS is consistent with this. But
it does bound what our chosen cell can demonstrate, and it is directly relevant to Stage
2: an improvement targeting the TQ mechanism on ETTh1 would be tuning a component that
this dataset cannot resolve.

Caveat: these are single seeds per variant. Confirming the conclusion properly means
three seeds per variant, which is nine runs and about five minutes.

---

## 3.8 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `TypeError` in `data_loader.py` about `drop` | pandas ≥ 2.0 against unpatched upstream | Use the vendored `TQNet/`, not a fresh clone — see `docs/02` §2.6 |
| `AttributeError: np.Inf` | numpy ≥ 2.0 | Same |
| `Matplotlib ... not a writable directory` | no writable `~/.matplotlib` | `export MPLCONFIGDIR=/tmp/mplcache`. The run scripts already set this |
| `--accelerator mps requested but ... is False` | torch without Metal, or a sandbox blocking the GPU | Use `ACCELERATOR=cpu`; the cell trains in ~30 s either way |
| `No completed runs found` | run did not reach the test phase, or `--save_outputs 0` | Re-run `repro/run_reconstruction.sh` |
| Collector reports a metric disagreement | reducing over the wrong axis, or a mis-shaped array | Real bug — do not report the number. See `docs/03` §3.5 |
| Tests skip with "ETTh1.csv not present" | dataset not downloaded | `python3 tools/get_data.py` |
