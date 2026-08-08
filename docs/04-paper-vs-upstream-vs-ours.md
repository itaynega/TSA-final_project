# 4. Three implementations: the paper, the authors' repository, and ours

There are **three** descriptions of TQNet in play, not two, and they do not agree with
each other. Keeping them separate is the difference between a reproduction and a
coincidence.

| | Artefact | What it is | Authority |
|---|---|---|---|
| **P** | **The paper** | Lin et al., *Temporal Query Network for Efficient Multivariate Time Series Forecasting*, ICML 2025 (PMLR 267), arXiv:2505.12917v2. Local copy `files/project/TQnet.pdf` | The method as *described*: equations 1–9, Tables 1–9 |
| **U** | **Upstream** | `github.com/ACAT-SCUT/TQNet` at commit `15e19cb23483ed52398566c4baa959168cfffa57`, Apache-2.0. Vendored under `TQNet/` | The method as *executed*. **This is what produced the published numbers** |
| **O** | **Ours** | U plus the 14 changes in §4.2, plus `common/`, `tools/`, `repro/` and `tests/`, which are ours alone | What produced our numbers |

---

## 4.0 The one rule that governs every difference

**Where P and U disagree, we follow U.**

This is a deliberate choice and it needs defending, because the assignment asks us to
"implement the method described in the paper" and here the paper describes something
subtly different from what ran.

The reason is that our Stage-1 target is a *number*: MSE 0.3712 on ETTh1, *L*=96 → *H*=96,
seed 2024. That number was produced by U, not by P. An implementation faithful to the
equations would be a different model — item 1 in §4.1 changes the attention logits by a
factor of two — and it would not land on 0.3712 except by accident. Reproducing a paper's
reported result means re-running the thing that reported it.

So the six disagreements in §4.1 are reported as **findings about the paper**, not as bugs
we fixed. We changed none of them. Two of them are live candidates for Stage 2, which is a
much better use of them than silently correcting them in Stage 1 and losing the comparison.

---

## 4.1 Where the paper and the repository disagree

Six places, found by reading both. **Ours follows the code in all six.**

| # | **P** — the paper says | **U** — the code does | **O** — ours | Consequence |
|---|---|---|---|---|
| 1 | Attention scaled by $1/\sqrt{L}=1/\sqrt{96}$ (eq. 3) | `nn.MultiheadAttention` scales by $1/\sqrt{d_{head}}=1/\sqrt{24}$ | follows U | Logits are **2× larger** than the equation implies. Anyone reimplementing from eq. 3 gets a different model |
| 2 | MLP is $\mathrm{Linear}(\mathrm{GeLU}(\mathrm{Linear}(\cdot)))$ (eq. 5) — no trailing activation | `Linear → GELU → Linear → GELU` (`TQNet.py:31-36`) | follows U | An extra nonlinearity sits between the MLP and the residual add |
| 3 | Instance normalisation is "optional" (§3.2) | A per-dataset switch: on for ETT, off for PEMS/Solar via `--use_revin 0`. Despite the flag name, `layers/RevIN.py` is **never imported** — there are no affine parameters | follows U (on, no affine) | It is not an option, it is a dataset-dependent part of the method |
| 4 | Table 1: ETTh1 has 14,400 timesteps | The CSV has 17,420; the loader hard-stops at 14,400 | follows U, and `common/data.py` reproduces the same truncation | **3,020 rows (~4 months) are silently discarded.** The paper misreports the series length |
| 5 | $\theta_{TQ}\in\mathbb{R}^{C\times W}$, indexed from "time step *t*" | Parameter is `(W, C)`; the index is the **absolute CSV row number mod *W*** taken at `s_end`, the first *forecast* step | follows U exactly | Equivalent only because *W*=24 divides *L*=96. On Electricity (*W*=168 > *L*=96) the two readings differ |
| 6 | `--model_type` documented as `[linear, mlp]` | Parsed, assigned, **never used** (`TQNet.py:12`) | follows U; excluded from our hyperparameter table | A dead flag. Reporting it as a hyperparameter would be wrong |

Items **1** and **3** are named in `files/project/TQNET_BRIEF.md` §7 as components an
improvement could attach to. Item **5** is the one most likely to bite a reimplementation,
because getting the phase wrong shifts every query by a constant and degrades the model
quietly instead of crashing.

---

## 4.2 Where our tree differs from upstream

Fourteen changes, all listed in full with line references in
[`02-architecture-and-implementation.md`](02-architecture-and-implementation.md) §2.6.
The governing rule there: **fix the environment, never the science.**

| Group | # | What changed | Alters the model? |
|---|---|---|---|
| **Compatibility** — without these the code does not run at all | 1–3 | `df_stamp.drop(['date'], 1)` → `drop(columns=['date'])` (removed in pandas 2.0); `np.Inf` → `np.inf` (removed in numpy 2.0); `torch.load` given explicit `map_location` and `weights_only=True` | No |
| **Portability** — running without an NVIDIA GPU | 4–5 | New `--accelerator {auto,cuda,mps,cpu}`; `torch.cuda.empty_cache()` → a device-aware `release_cache()` | No |
| **Reproducibility** | 6–7 | Added CUDA seeding and cuDNN determinism (no-ops on CPU, and they do not consume the CPU RNG stream); `--num_workers` default 10 → 0, with worker seeding when > 0 | No |
| **Output plumbing** — upstream finished a run and left almost nothing behind | 8–10 | New `--result_path` so a run cannot append to the authors' `result.txt`; new `--save_outputs` writing `pred.npy`/`true.npy`/`metrics.json` (upstream had these commented out); ablation booleans exposed as `--use_tq`/`--channel_aggre` instead of requiring a source edit | No |
| **Housekeeping** | 11–14 | `README.md` → `README_UPSTREAM.md`; `result.txt` copied to a never-written `result_authors_reference.txt`; upstream `.gitignore` removed; empty `dataset/` created | No |

### Why "alters the model? No" is a measurement and not a claim

- **Parameter counts are identical to the pre-change model** in all three configurations:
  **661,640** published, **661,472** with `--use_tq 0`, **624,224** with
  `--use_tq 0 --channel_aggre 0`. The first is what the unpatched model produces.
- **The reconstruction landed 0.000167 MSE from the authors' own run** — 0.17× the paper's
  three-seed sigma. A change that had touched the science would not land there.
- **`tests/test_data.py` compares our numpy reimplementation of the loader against the real
  vendored `Dataset_ETT_hour`, element for element, on all three splits.**

To see the changes as a diff against pristine upstream, `docs/02` §2.6 gives the
`curl` + `diff -ru` recipe. There is no patch file by design: the tree is the source of
truth, so no patch can go stale.

---

## 4.3 What is ours alone, with no counterpart in P or U

`TQNet/` imports nothing from us. Our code reads its outputs. That separation is the
reason a bug in our tooling cannot change a training result.

| Path | What it is | Why upstream has no equivalent |
|---|---|---|
| `common/split.py` | The ETTh1 split stated once, with a fingerprint (`b66ee6b47e2b2eb8` at *L*=*H*=96) that runs assert against | Upstream's split is implicit in the loader; nothing checks that two runs used the same one |
| `common/data.py` | The loader **reimplemented in numpy**, so the baseline and the audit see the model's exact windows | This is the one place we implemented the method ourselves rather than vendoring it — see the note below |
| `common/metrics.py` | MSE, MAE, RMSE, MdAE in the course's notation, float64 | Upstream computes MSE/MAE/RMSE in float32 and reports nothing else |
| `common/results.py` | One JSON record per run, carrying git commit, split hash and metrics | Upstream appends a line to a text file |
| `tools/get_data.py` | Downloads ETTh1.csv from a pinned commit and verifies sha256, row count, columns and date range | Upstream links an unversioned Google Drive folder |
| `tools/audit_split.py` | The leakage audit — ten checks recomputed from the fitted objects | Nothing comparable exists upstream |
| `tools/run_baseline.py` | Seasonal-naive, period 24, on identical windows and scale | The paper copies baseline numbers from other papers rather than running them (`TQNET_BRIEF` §6 item 15) |
| `tools/collect_results.py` | Ingests a run and **cross-checks our metrics against upstream's on the same arrays** | — |
| `tools/paper_reference.py` | The paper's numbers, transcribed once, with page citations | — |
| `tools/make_report.py` | Builds `report/results.md` and the figures from recorded runs only | — |
| `repro/run_etth1_ablation.sh` | The ETTh1 ablation **the paper never published** | The authors ablate only on Electricity, PEMS03 and PEMS04 |
| `tests/` | pytest over `common/` and `tools/` | — |

### The honest statement of what "reconstruction" means here

**We did not rewrite the model.** `TQNet/models/TQNet.py` is upstream's, changed only to
expose two ablation booleans that were already there as hard-coded literals. What we
independently reimplemented is the **data path** (`common/data.py`) and the **metrics**
(`common/metrics.py`), and both are validated against upstream: the loader element for
element in `tests/test_data.py`, the metrics on the same arrays to 1.2×10⁻⁷ relative in
`tools/collect_results.py`.

This is a defensible reading of "reproduce its main experimental results", and it is the
reading that made the 0.045% match meaningful. But it is **not** the same as writing the
architecture from scratch, and the report must say so in plain words rather than let a
reader assume otherwise. See §4.4.

---

## 4.4 What the three-way distinction buys the report

The brief (requirement **F5**) demands a table comparing *paper / reconstruction /
improved*. The P/U/O split above is what lets that table mean something:

- The **paper** column is transcribed, with page citations, by `tools/paper_reference.py`.
- The **reconstruction** column is O running U's published configuration unchanged.
- The **improved** column will be O running a change we make, on the same split hash and
  the same metrics — enforced, not promised, by `common.results.assert_split_hash`.

Two things belong in **F6** ("what worked, what did not, and what you learned") and come
directly out of this document:

1. **A published paper and its own official code describe different models in six places,
   one of which changes the arithmetic by a factor of two.** We found this by reading both,
   and it is the single most transferable thing the project taught.
2. **Reproducing the number required following the code, not the equations.** That is worth
   stating as a methodological finding rather than buried as an implementation note.

---

## 4.5 Numbers, so this page is checkable

| Quantity | Value |
|---|---|
| Upstream commit | `15e19cb23483ed52398566c4baa959168cfffa57` |
| Licence | Apache-2.0, `TQNet/LICENSE` included |
| Cell | ETTh1, multivariate, *L*=96 → *H*=96, *W*=24, seed 2024 |
| Test windows / predictions | 2,785 / 1,871,520 |
| Split hash | `b66ee6b47e2b2eb8` |
| Trainable parameters | 661,640 |
| Authors' run (`result.txt`, full precision) | MSE 0.3712166 · MAE 0.3928201 |
| Ours | MSE 0.3710499 · MAE 0.3927240 |
| Difference | −0.000167 (−0.045%) · −0.000096 (−0.024%) |
| Paper's own 3-seed sigma (Table 9) | 0.001 MSE |
| **Our** 3-seed sigma (2024/2025/2026) | **0.002154 MSE** |
