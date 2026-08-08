# 2. Architecture, implementation, and every change we made

All line references are to the **vendored** tree in `TQNet/`, which is upstream commit
`15e19cb23483ed52398566c4baa959168cfffa57` plus the changes listed in §2.6. Where a
line number has moved because of one of our edits, the section says so.

---

## 2.1 The architecture at a glance

TQNet is deliberately small: **one attention layer, one two-layer MLP, one output
projection.** For the target cell that is **661,640 trainable parameters**, verified by
instantiating the model (`docs/03` shows the command).

```
x  (batch, 96, 7)                     96 hours of 7 channels, already z-scored by the loader
 │
 ├─ instance normalise ───────────────  subtract this window's mean, divide by its sd     [TQNet.py:47-50]
 │
 ├─ transpose to (batch, 7, 96) ──────  CHANNELS BECOME TOKENS; each token's embedding
 │                                      vector IS that channel's 96-step history          [TQNet.py:53]
 │
 ├─ TQ-MHA ───────────────────────────  Q = θ_TQ slice by clock phase                     [TQNet.py:56-57]
 │    │                                 K = V = the input window
 │    │                                 4 heads, attention dropout 0.5                    [TQNet.py:27, 59]
 │    └─ residual: + x_input ─────────  the attention output is ADDED to the input        [TQNet.py:68]
 │
 ├─ input_proj: Linear(96 → 512) ─────                                                    [TQNet.py:29]
 │
 ├─ MLP: Linear→GELU→Linear→GELU ─────  512 → 512 → 512                                   [TQNet.py:31-36]
 │    └─ residual: + input ───────────                                                    [TQNet.py:72]
 │
 └─ output_proj: Dropout(0.5) ────────
    then Linear(512 → 96)                                                                 [TQNet.py:38-41]
      │
      ├─ transpose back to (batch, 96, 7)                                                 [TQNet.py:72]
      └─ instance de-normalise ───────  multiply by sd, add mean back                     [TQNet.py:75-76]
```

### The one thing to understand about this model

**The attention runs over the channel axis, not the time axis.** After the transpose at
line 53, the sequence dimension that attention sees has length 7 — the number of
variables — and each token's 512-dimensional... no, each token's **96-dimensional**
embedding is that channel's raw history. So `embed_dim=96` at line 27 is `seq_len`, not
`d_model`.

The consequence is that the attention map is **7×7**. It is a learned inter-variable
correlation matrix and nothing else. Temporal structure is modelled entirely by the MLP
that follows, which operates on the 96→512→96 axis. This is the iTransformer
("inverted transformer") arrangement, and it is why the model is cheap: attention is
quadratic in 7, not in 96.

It also explains something about our ablation results in `docs/03`: at *C* = 7 there is
very little for a 7×7 correlation matrix to discover.

## 2.2 The Temporal Query, in code

Three lines carry the paper's entire contribution.

**The parameter block** (`TQNet.py:24`):

```24:24:TQNet/models/TQNet.py
            self.temporalQuery = torch.nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)
```

Shape `(W, C)` = (24, 7) = **168 parameters**, initialised to **zeros**. The zero
initialisation is worth noting: at step 0 every query is identical, so the attention map
is uniform across channels and the correlation structure is learned from a degenerate
start.

**The cyclic gather** (`TQNet.py:56-57`):

```56:57:TQNet/models/TQNet.py
            gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
            query_input = self.temporalQuery[gather_index].permute(0, 2, 1)  # (b, c, s)
```

This is equation 9 of the paper. For each sample, take its phase, add `[0, 1, ..., L-1]`,
wrap modulo *W*, and index the parameter block. Because *W* = 24 divides *L* = 96, the
resulting 96-long query is just the 24 learned vectors tiled four times.

**The attention call** (`TQNet.py:59`):

```59:59:TQNet/models/TQNet.py
                channel_information = self.channelAggregator(query=query_input, key=x_input, value=x_input)[0]
```

`query=` from the parameter block, `key=` and `value=` from the data. That asymmetry is
the paper.

### Where the phase comes from

`cycle_index` is built in the data loader, not the model
(`TQNet/data_provider/data_loader.py:84, 97`):

```84:84:TQNet/data_provider/data_loader.py
        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]
```

Two details that are easy to get wrong and that `common/data.py` reproduces exactly:

- The index is the **absolute CSV row number** mod *W*, not an offset within the split.
  Getting this wrong shifts every query by a constant phase, which degrades the model
  quietly rather than crashing.
- The value attached to a window is the phase at `s_end` — the **first forecast step**,
  not the first input step. The paper's text reads as though it were the window start.
  The two coincide only when *W* divides *L*, which is true here and false on
  Electricity (*W* = 168 > *L* = 96).

## 2.3 Loss, objective, and optimisation

| Component | Value | Where |
|---|---|---|
| Loss | `nn.MSELoss` — plain L2 on z-scored values, weighted equally across all 96 steps and all 7 channels | `exp/exp_main.py:_select_criterion` |
| Optimiser | Adam, lr 1e-3 | `exp/exp_main.py:_select_optimizer` + script |
| LR schedule | `type3`: constant for the first 3 epochs, then ×0.8 per epoch | `utils/tools.py:adjust_learning_rate` |
| Epochs / patience | 30 max, early stop after 5 epochs without validation improvement | script |
| Model selection | checkpoint with the lowest **validation** MSE | `utils/tools.py:EarlyStopping` |

Note that a `OneCycleLR` scheduler is constructed in `train()` but only *used* when
`--lradj TST`, which this configuration does not set. With `type3`, the `OneCycleLR`
object is inert. This is upstream's code, left as-is.

## 2.4 Full hyperparameters for the target cell

Every value is pinned by `TQNet/scripts/TQNet/etth1.sh` or is a `run.py` default that no
script overrides — nothing here was chosen by us. "hard-coded" means the value is a
literal in the model source with no flag, so it **cannot** be reported as a tunable
hyperparameter.

| Quantity | Value | Source |
|---|---|---|
| Sampling frequency | 1 hour | ETTh1 |
| Input window *L* (`seq_len`) | 96 | script |
| Forecast horizon *H* (`pred_len`) | 96 | script |
| Channels *C* (`enc_in`) | 7 | script |
| Cycle *W* (`cycle`) | 24 | script |
| `features` | `M` (7 in → 7 out) | script |
| `d_model` | 512 | `run.py` default |
| Attention heads | 4 | **hard-coded**, `TQNet.py:27` |
| Attention dropout | 0.5 | **hard-coded**, `TQNet.py:27` |
| Output dropout | 0.5 | script (`run.py` default is 0) |
| Instance norm (`use_revin`) | 1 = on | `run.py` default |
| Batch size | 256 | script |
| Learning rate | 0.001 | script |
| Epochs / patience | 30 / 5 | script |
| Seed | 2024 | script |
| Trainable parameters | 661,640 | measured |
| Test windows | 2,785 | measured, see `docs/03` |

`--model_type` is parsed and assigned to `self.model_type` at `TQNet.py:12` and then
**never used**. It is a dead flag and must not be reported as a hyperparameter.

## 2.5 Six places the paper and the code disagree

Found by reading both. Reported as findings, unreconciled. **Item 1 changes the model**:
anyone reimplementing from the equations gets something different from what produced
0.3712.

> This section and §2.6 are the two halves of a three-way picture — paper, upstream
> repository, our tree. [`04-paper-vs-upstream-vs-ours.md`](04-paper-vs-upstream-vs-ours.md)
> puts them in one table and states the rule we followed: **where the paper and the code
> disagree, we follow the code**, because the code is what produced the published number.

| # | Paper says | Code does | Why it matters |
|---|---|---|---|
| 1 | Attention scaled by $1/\sqrt{L} = 1/\sqrt{96}$ (eq. 3) | `nn.MultiheadAttention` scales by $1/\sqrt{d_{head}} = 1/\sqrt{24}$ | Logits are 2× larger than the equation implies |
| 2 | MLP is `Linear(GeLU(Linear(·)))` (eq. 5) — no trailing activation | `Linear → GELU → Linear → GELU` (`TQNet.py:31-36`) | An extra nonlinearity sits between the MLP and the residual add |
| 3 | Instance norm is "optional" (§3.2) | On by default for ETT, off for PEMS/Solar via `--use_revin 0` | It is a per-dataset switch, not an option. Despite the flag name, `layers/RevIN.py` is never imported — there are no affine parameters |
| 4 | Table 1: ETTh1 has 14,400 timesteps | The CSV has 17,420; the loader hard-stops at 14,400 | 3,020 rows (~4 months) silently discarded |
| 5 | $\theta_{TQ} \in \mathbb{R}^{C \times W}$, indexed from "time step *t*" | Parameter is `(W, C)`; index is the phase at `s_end` | Only coincidentally equivalent when *W* divides *L* |
| 6 | `--model_type` documented as `[linear, mlp]` | Read and never used | Dead flag |

## 2.6 Every change we made to the vendored code

The rule applied throughout: **fix the environment, never the science.** No change below
alters the model, the data, the split, the loss, the optimiser or the metric. Each is
either a compatibility fix for a library that moved on, or new output plumbing that
upstream had commented out.

### Compatibility — without these the code does not run at all

| # | File | Change | Why |
|---|---|---|---|
| 1 | `data_provider/data_loader.py` (4 sites) | `df_stamp.drop(['date'], 1)` → `drop(columns=['date'])` | The positional form was **removed in pandas 2.0**. Raises `TypeError` on pandas 2.3.3. This is a hard blocker, not a warning |
| 2 | `utils/tools.py` | `np.Inf` → `np.inf` | The capitalised alias was **removed in numpy 2.0**. `EarlyStopping.__init__` raises `AttributeError` |
| 3 | `exp/exp_main.py` | `torch.load(path)` → `torch.load(path, map_location=device, weights_only=True)` in 3 places | `weights_only=True` became the default in torch 2.6; making it explicit pins the meaning. `map_location` lets a checkpoint trained on one device be evaluated on another |

### Portability — running without an NVIDIA GPU

| # | File | Change | Why |
|---|---|---|---|
| 4 | `run.py`, `exp/exp_basic.py` | New `--accelerator {auto,cuda,mps,cpu}`; device resolved once in `run.py` and read by `_acquire_device` | Upstream knew only about CUDA: `args.use_gpu = torch.cuda.is_available() and ...`, so on any Mac it silently fell to CPU with no way to request Apple Silicon, and `--accelerator cuda` on a CPU-only box failed obscurely rather than immediately |
| 5 | `run.py` | `torch.cuda.empty_cache()` → `release_cache()`, which dispatches on the resolved device | The unconditional CUDA call is meaningless off CUDA |

### Reproducibility

| # | File | Change | Why |
|---|---|---|---|
| 6 | `run.py` | Added `torch.cuda.manual_seed_all(seed)`, `cudnn.deterministic = True`, `cudnn.benchmark = False` | Upstream seeded only `random`, `torch` and `numpy`, leaving CUDA generators and cuDNN kernel selection free. **These are no-ops on CPU and do not consume the CPU RNG stream, so the published configuration is bit-unaffected** |
| 7 | `run.py`, `data_provider/data_factory.py` | `--num_workers` default 10 → 0; added `worker_init_fn` seeding when workers > 0 | 10 worker processes cost more in startup than they save on a dataset this small, and their RNG was unseeded. Sample order is drawn by the sampler in the parent process, so **0 workers yields the same batches as 10** — verified by the reproduction matching |

### Output plumbing — upstream finished a run and left almost nothing behind

| # | File | Change | Why |
|---|---|---|---|
| 8 | `run.py`, `exp/exp_main.py` | New `--result_path`, default `./result_ours.txt`, replacing the hard-coded `result.txt` | `exp_main` opened `result.txt` in **append** mode — the same file holding the authors' full-precision published numbers, which is our best reference. A single run would have contaminated it |
| 9 | `exp/exp_main.py` | New `--save_outputs` (default on): writes `pred.npy`, `true.npy` and `metrics.json` under `results/<setting>/`. Upstream had these as commented-out lines | Without the window-level arrays there is no way to compute the course's metrics, draw any chart, or score the baseline on identical windows. This is what makes the report possible |
| 10 | `models/TQNet.py`, `run.py` | Ablation booleans at `TQNet.py:20-21` read from `--use_tq` / `--channel_aggre`; non-default runs get a `_tq<N>ca<N>` suffix in every output path | Upstream expected you to **edit the source file** between ablation variants, which is unreproducible and risks overwriting the reconstruction's own checkpoints. Defaults are the published values, verified by parameter count (see below) |

### Housekeeping

| # | Change | Why |
|---|---|---|
| 11 | `README.md` → `README_UPSTREAM.md` | Avoids two README files claiming to describe the project |
| 12 | `result.txt` copied to `result_authors_reference.txt` | A second, never-written copy of the published numbers. `repro/run_reconstruction.sh` refuses to start if it is missing |
| 13 | Upstream `.gitignore` removed | Superseded by the repository-root one, which has to reason about both trees |
| 14 | `dataset/` created empty | Where `tools/get_data.py` puts ETTh1.csv, which is where the scripts look |

### Evidence that changes 1–14 did not alter the model

- **Parameter counts match the pre-change model exactly** in all three configurations:
  661,640 (published), 661,472 (`--use_tq 0`), 624,224 (`--use_tq 0 --channel_aggre 0`).
  The first is the number the unpatched model produces.
- **The reconstruction matched the published result to 0.00017 MSE**, which is 0.17× the
  paper's own seed standard deviation. A change that had altered the science would not
  land there.
- **`tests/test_data.py` compares our numpy reimplementation of the loader against the
  real vendored `Dataset_ETT_hour`, element for element, on all three splits.** If a
  change had disturbed the windows, that test fails.

### Regenerating the diff

There is no patch file, deliberately — the changes are applied in the tree, so the tree
is the single source of truth. To see them as a diff against pristine upstream:

```bash
curl -sSL -o /tmp/tqnet.tar.gz \
  https://codeload.github.com/ACAT-SCUT/TQNet/tar.gz/15e19cb23483ed52398566c4baa959168cfffa57
mkdir -p /tmp/tqnet-pristine && tar -xzf /tmp/tqnet.tar.gz -C /tmp/tqnet-pristine --strip-components=1
diff -ru /tmp/tqnet-pristine TQNet
```

## 2.7 What we added alongside, rather than inside, the vendored code

`TQNet/` imports nothing from us. Our code reads its outputs. That separation is
deliberate: it keeps the vendored tree a faithful copy of a licensed upstream project,
and it means a bug in our tooling cannot change a training result.

| Path | What it does |
|---|---|
| `common/split.py` | The ETTh1 split stated once, plus a fingerprint runs assert against |
| `common/data.py` | The loader reimplemented in numpy, so the baseline and the audit see the model's exact windows; plus the seasonal-naive forecast |
| `common/metrics.py` | MSE, MAE, RMSE, MdAE in the course's notation |
| `common/results.py` | One JSON record per run, with git commit and split hash |
| `tools/get_data.py` | Downloads ETTh1.csv from a pinned commit and verifies it four ways |
| `tools/check_env.py` | Environment and device report; flags version drift |
| `tools/audit_split.py` | The leakage audit — ten checks, recomputed from the fitted objects |
| `tools/run_baseline.py` | Seasonal-naive on the identical windows and scale |
| `tools/collect_results.py` | Ingests a finished run; cross-checks our metrics against upstream's |
| `tools/paper_reference.py` | The paper's numbers, transcribed once, with page citations |
| `tools/make_report.py` | Builds `report/results.md` and the figures |
| `repro/run_reconstruction.sh` | Stage 1, the target cell |
| `repro/run_etth1_ablation.sh` | The ETTh1 ablation the paper never published |
