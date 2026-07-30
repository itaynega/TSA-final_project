# TSA-final_project

Time Series Analysis final course project — Amitay & Itay. Instructor: Havana Rika. **Deadline 10.08.**

> This README orients the team. It is **not** the README the assignment asks for — that one explains
> how to run the code we ship, and gets written at the end.

## The assignment, in two stages

1. **Reconstruct** a published time-series method and try to reproduce its main result, with a
   reproducible data pipeline, a valid temporal evaluation protocol, at least one simple baseline,
   the paper's metric and one metric taught in class.
2. **Improve** it in a meaningful, justified way — evaluated on the **identical split and metrics**.

Five things get submitted: reconstruction code, improvement code, the dataset (or a documented
download), a PDF report with seven mandated sections, and a short README with pinned versions.

Two requirements are pass/fail and govern every decision: **no future information** anywhere in
training, preprocessing, feature construction or hyperparameter tuning; and the improvement must run
on the **same split and metrics** as the reconstruction.

## The paper

**TQNet** — Lin, Chen, Wu, Qiu, Lin, *Temporal Query Network for Efficient Multivariate Time Series
Forecasting*, **ICML 2025**, arXiv 2505.12917. Repo [ACAT-SCUT/TQNet](https://github.com/ACAT-SCUT/TQNet),
Apache-2.0.

**Target cell:** ETTh1, multivariate, `seq_len=96` → `pred_len=96`, seed 2024 →
**MSE 0.3712 / MAE 0.3928**.

### What "target cell" means

The paper reports 12 datasets × 4 horizons × 2 metrics — 96 numbers. We reproduce **one**, and the
rest of the table becomes context rather than work. Reading the spec:

- **ETTh1** — *Electricity Transformer Temperature, hourly, station 1*. A CSV of hourly readings with
  seven numeric channels (six power-load measurements plus oil temperature). 17,420 rows, of which
  the loader uses only the first 14,400.
- **multivariate** (`--features M`) — all seven channels in, all seven out. The alternatives are `S`
  (one in, one out) and `MS` (all in, one out). This is the hardest of the three and the one the
  paper's headline table uses.
- **96 → 96** — from the last 96 hours (four days) of all seven channels, predict the next 96 hours of
  all seven channels. One example is a 96×7 matrix in, 96×7 out; the window slides forward one hour
  for the next example, which is how 2,976 test rows become **2,785 test windows**.
- **seed 2024** — fixes weight initialisation and batch shuffling. The paper reports a single seed.
- **MSE / MAE** — averaged over 2,785 windows × 96 steps × 7 channels, on **z-scored** data. That is
  why they are ≈0.37 and not degrees Celsius, and why MAPE is unusable here: the normalised series
  crosses zero.

### Why this paper is reproducible at all

`scripts/TQNet/etth1.sh` pins every flag — learning rate, batch size, dropout, epochs, patience, seed
— so nothing has to be guessed. More unusually, the authors ran `sh run_main.sh` on their own machine
and **committed the raw output** to `result.txt` at full float precision:

```
ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024
mse:0.3712165653705597, mae:0.3928201496601105
```

The paper's Table 5 only prints `0.371` / `0.393`; the four-decimal figures above come from
`result.txt`. That gives us a **three-way** check instead of a two-way one, and the three outcomes
mean different things:

| Our result | Reading |
|---|---|
| Matches `result.txt` to many decimals | Our environment is effectively theirs. Reconstruction done |
| Close but not exact | Ordinary hardware/library drift — different GPU, cuDNN, PyTorch version. Expected, and reportable |
| Far off | The fault is in **our** setup. Go looking there before concluding the published number is wrong |

Without `result.txt` a mismatch would be ambiguous between "we made a mistake" and "the number is not
reproducible", and the report would have to hedge. This is why the backup step comes first in the
quick start below — `exp_main.py` opens that same file in append mode.

## Where things stand

Two tracks, running in parallel.

| Track | Who | What |
|---|---|---|
| **Reproduce the paper's result** | Itay | Get the target cell running and matching, on our hardware, and record every environment deviation |
| **Design the improvement** | Amitay | Understand the method and its limitations well enough to choose a change we can defend. Not chosen yet, deliberately |

The improvement is picked **after** the method is understood, not before. Choosing first and
justifying afterwards is the failure mode this sequencing exists to prevent.

## What is in here

| Path | What it is |
|---|---|
| `files/project/Final_Project.pdf` | The assignment brief. The only external authority |
| `files/project/TQnet.pdf` | The paper |
| **`files/project/TQNET_BRIEF.md`** | **Start here.** The method, the code as it actually is, six paper/code disagreements, the verified split, the leakage audit, and a 20-item limitations inventory with sources |
| `files/lectures/` | The ten course lecture decks, plus `CPDexamples.pdf` (which is **Laurent Oudre's** ENS deck, not this course's — cite him, never Rika) |
| `repro/tqnet-ablation-flags.patch` | Turns TQNet's hard-coded ablation switches into `--use_tq` / `--channel_aggre` flags. Defaults reproduce the published model exactly |
| `repro/run_etth1_ablation.sh` | Runs the two ETTh1 ablations the paper never published — does TQ help at all at 7 channels? |

Earlier planning documents (plan, dispatches, rubric, requirements table) were removed to get back to
a clean point. They are still in git history if anything needs recovering.

## Reproduction quick start

Clone TQNet **outside** this repository — we ship patches, never vendored copies of licensed upstream
code.

```
git clone https://github.com/ACAT-SCUT/TQNet.git
cd TQNet
git checkout 15e19cb23483ed52398566c4baa959168cfffa57

cp result.txt result_authors_reference.txt   # exp_main.py APPENDS to this file
git apply /path/to/TSA-final_project/repro/tqnet-ablation-flags.patch

# ETTh1.csv goes in ./dataset/ ; it is also available unbundled from
# github.com/zhouhaoyi/ETDataset at ETT-small/ETTh1.csv (CC BY-ND 4.0)
sh scripts/TQNet/etth1.sh
```

Two things that will bite, both verified:

- **The loader breaks on pandas ≥ 2.0.** `data_loader.py` calls `df_stamp.drop(['date'], 1)`, a
  positional form pandas removed. The README asks for Python 3.8, which is end-of-life. Fix the
  environment, don't fix the science, and **record every deviation** — each one is a line in the
  README we have to ship.
- **Running the code overwrites the evidence.** `exp_main.py:348` appends to `result.txt` in the
  working directory — the file holding the authors' reference numbers. Back it up first.

## Working rules

- **This repo is the single source of truth.** Copies elsewhere are stale.
- **One writer per file.** Two people editing one markdown file is a merge conflict nobody needs at
  this deadline.
- **No experiment before its pre-registration exists** — the predicted number is written down first.
- The leakage audit is run by whoever did **not** write the pipeline.
- Never commit datasets, checkpoints or run outputs — see `.gitignore`.
- Pull before you start, push when you stop.
- **Do not put your clone inside a OneDrive- or Dropbox-synced folder.** Sync clients lock and
  re-write files under `.git/` while git is using them, which corrupts the object store.

## Setup

```
git clone https://github.com/itaynega/TSA-final_project.git
```
