# TQNet — brief for Amitay, for designing the improvement

**Written 2026-07-30.** For Amitay, to work in parallel with the reproduction run.
This document does **not** choose an improvement. It gives you the method, the code as it actually
is, and an unranked inventory of everything that is weak, so the choice is made after understanding
rather than before it.

**Paper.** Lin, Chen, Wu, Qiu, Lin, *Temporal Query Network for Efficient Multivariate Time Series
Forecasting*, ICML 2025, PMLR 267. arXiv 2505.12917v2 (11 Sep 2025). Local copy: `files/project/TQnet.pdf`.
**Code.** <https://github.com/ACAT-SCUT/TQNet>, Apache-2.0. Read at commit
`15e19cb23483ed52398566c4baa959168cfffa57` (6 May 2026). All file:line references below are that commit.

---

## Status — read before relying on §9 or §12

This brief was written **before** anything was run and is kept as the record of what was known
then. Three parts of it have since been overtaken; the rest stands.

| Section | Status now |
|---|---|
| **§9**, "the two runs that are prepared and waiting" | **Done, and the answer is surprising.** The ablation ran. On ETTh1 every variant lands inside one seed standard deviation and the **pure MLP is nominally the best of the three** — neither the Temporal Query nor the channel attention is measurable at *C*=7. Numbers in `docs/03` §3.7. §9 also refers to `repro/tqnet-ablation-flags.patch`, which **no longer exists**: the flags are applied directly in the vendored tree and the runner is `repro/run_etth1_ablation.sh`. |
| **§12 Q2**, "does TQ help on ETTh1 at all?" | **Answered: not measurably.** See above. |
| **§12 Q3**, "our own seed spread" | **Measured: 0.002154 MSE** over seeds 2024/2025/2026, about twice the paper's reported 0.001. That is the bar Stage 2 must clear. |

Still open and still blocking: **§12 Q1** (PTQNet, paywalled — a prerequisite for any
period-related improvement) and **§12 Q4** (which of the five permitted axes we target).

The limitations inventory in **§6** and the component list in **§7** are unaffected and are still
the material to brainstorm against.

---

## 0. Read this first

**On ETTh1 — the dataset we are reconstructing — TQNet is not the winning model.** Averaged over the
four horizons it loses to TimeXer on MSE in the paper's own headline table (0.441 vs 0.437, Table 2
p. 6), and it is beaten outright at horizons 336 and 720. It wins only at 96 and 192.

The cell we reproduce (horizon 96) is a real win, but the margin over the nearest sibling model is
**0.004 MSE**, and the paper's own three-seed study puts run-to-run standard deviation at **0.001**
(Table 9, p. 18). An improvement aimed at MSE on this cell is aiming at a target barely above the
measurement error. The brief permits four other axes — robustness, interpretability, efficiency,
applicability — and that is where the room is.

**Second thing: the paper never ablates TQNet on any ETT dataset.** Every ablation and integration
study runs on Electricity, PEMS03 and PEMS04 only, all with more than 100 channels. ETTh1 has 7.
There is no published evidence that the Temporal Query mechanism, or the attention layer at all,
contributes anything at that width. See §9 — this is two cheap runs and it is prepared and waiting.

---

## 1. The cell we reproduce

ETTh1, multivariate (7 channels in, 7 out), look-back 96 → horizon 96, seed 2024.
Config fully pinned in `scripts/TQNet/etth1.sh`. The repo also ships the authors' own execution
output in `result.txt`, and it matches the paper to four decimals:

```
ETTh1_96_96_TQNet_ETTh1_ftM_sl96_pl96_cycle24_seed2024
mse:0.3712165653705597, mae:0.3928201496601105
```

| Quantity | Value | Source |
|---|---|---|
| Input window length *L* (`seq_len`) | 96 | script |
| Forecast horizon / output length *H* (`pred_len`) | 96 | script |
| Sampling frequency | 1 hour | ETTh1; paper Table 1 |
| Channels *C* (`enc_in`) | 7 | script |
| Cycle length *W* (`cycle`) | 24 | script — one hand-set integer |
| `d_model` | 512 | `run.py` default, never overridden by any script |
| Attention heads | 4 | **hard-coded**, `models/TQNet.py:24` |
| Attention dropout | 0.5 | **hard-coded**, `models/TQNet.py:24` |
| Output dropout | 0.5 | script (`--dropout`); `run.py` default is 0 |
| Instance norm (`use_revin`) | 1 = on | `run.py` default |
| Loss | `nn.MSELoss` (L2) | `exp_main.py:61` |
| Optimiser | Adam, lr 1e-3 | `exp_main.py:57` + script |
| LR schedule | `type3`: constant for 2 epochs, then ×0.8 each epoch | `utils/tools.py:19` |
| Batch size | 256 | script |
| Epochs / patience | 30 / 5, early stopping on validation MSE | script |
| Seed | 2024, single seed | script |
| Trainable parameters | **661,640** (verified by instantiating the model) | — |
| Test windows evaluated | **2,785** | verified, see §5 |

Metrics are computed on **z-score-normalised** data, not the original scale — the de-normalisation
lines are commented out at `exp_main.py:333-335`. So 0.3712 is in normalised units. This is also why
MAPE and SMAPE are unusable on this benchmark: the normalised series crosses zero.

---

## 2. What the method actually does

TQNet is a channel-attention model in the iTransformer family. The attention runs over the
**channel** axis: each of the 7 variables is one token, and that token's embedding vector *is* its
96-step history. The attention map is therefore 7×7 — a learned inter-variable correlation matrix,
nothing more.

The one novel idea: instead of deriving the queries from the input sample, they come from a learnable
parameter block `θ_TQ ∈ R^(W×C)` — here 24×7 = **168 parameters**, initialised to zero — sliced
cyclically according to the sample's phase in the 24-hour cycle. Keys and values still come from the
raw input. Because every sample sitting at the same clock position shares the same query, that query
converges towards an average of correlation structure over many samples. That averaging is the whole
robustness argument (Appendix A.4, eqs 10–13).

Forward pass, as the code runs it:

| # | Step | Code | Shape |
|---|---|---|---|
| 1 | Instance normalise: subtract per-window mean, divide by per-window std | `TQNet.py:44-47` | (b, 96, 7) |
| 2 | Transpose so channels become tokens | `TQNet.py:50` | (b, 7, 96) |
| 3 | Gather the TQ query slice by cycle phase | `TQNet.py:53-54` | (b, 7, 96) |
| 4 | Single MHA layer, Q = TQ and K = V = input; then residual + input | `TQNet.py:56, 65` | (b, 7, 96) |
| 5 | Project 96 → 512, two Linear+GELU blocks, residual | `TQNet.py:26-33, 67` | (b, 7, 512) |
| 6 | Dropout, project 512 → 96, transpose back, de-normalise | `TQNet.py:35-38, 69-73` | (b, 96, 7) |

**The structural consequence worth thinking about.** `θ_TQ` has exactly *W×C* entries and is indexed
by clock phase alone. It cannot represent a second co-existing period, a non-integer period, a period
that drifts, or any state that depends on the recent past rather than the calendar. Everything
sample-specific has to arrive through K and V. Most of the limitations in §6 follow from that asymmetry.

---

## 3. Where the paper and the code disagree

Reported unreconciled, as findings. Item 1 changes the model.

| # | Paper says | Code does | Why it matters |
|---|---|---|---|
| 1 | Attention scaled by 1/√*L* = 1/√96 (eq 3) | `nn.MultiheadAttention` scales by 1/√head_dim = 1/√24 | Logits are 2× larger than the equation implies. Reimplement from the paper and you get a different model. |
| 2 | MLP is `Linear(GeLU(Linear(·)))` (eq 5) — no trailing activation | `Linear → GELU → Linear → GELU` (`TQNet.py:28-33`) | An extra nonlinearity sits between the MLP and the residual add. |
| 3 | Instance normalisation is "optional" (§3.2) | On by default for ETT; off for PEMS03/04/07 and Solar via `--use_revin 0` | It is a per-dataset switch, not an option. Despite the flag name, `layers/RevIN.py` is never used — there are no affine parameters. |
| 4 | Table 1 lists ETTh1 as 14,400 timesteps | `ETTh1.csv` has 17,420 rows; the loader hard-stops at 14,400 | 3,020 rows (~4 months) silently discarded. Stated nowhere in the paper. |
| 5 | `θ_TQ ∈ R^(C×W)`, queries indexed from "time step *t*" | Parameter is (W, C); index is the phase at `s_end` — the first *forecast* step, not the window start | Only coincidentally equivalent when *W* divides *L* (true for ETTh1: 96 = 4×24). Differs on Electricity, where *W*=168 > *L*=96. |
| 6 | `--model_type` documented as `[linear, mlp]` | Read into `self.model_type` at `TQNet.py:12` and never used | Dead flag. Do not report it as a hyperparameter. |

---

## 4. TQNet on ETTh1, against its own reported baselines

Test MSE, normalised ETTh1, multivariate, *L* = 96, single seed 2024. Lower is better.
Source: paper Table 5, p. 15. Note the baseline numbers were **copied** by the authors from TimeXer,
iTransformer and CycleNet, not re-run (Table 5 caption).

| Horizon | TQNet | TimeXer | CycleNet | iTransformer | DLinear |
|---|---|---|---|---|---|
| 96 | **0.371** | 0.382 | 0.375 | 0.386 | 0.386 |
| 192 | **0.428** | 0.429 | 0.436 | 0.441 | 0.437 |
| 336 | 0.476 | **0.468** | 0.496 | 0.487 | 0.481 |
| 720 | 0.487 | **0.469** | 0.520 | 0.503 | 0.519 |
| Avg | 0.441 | **0.437** | 0.457 | 0.454 | 0.456 |

### Run-to-run spread vs. the margin being claimed

The main table is one seed; the three-seed study is in Table 9, p. 18. Putting them side by side is
the sanity check that has to happen before any improvement is called real.

| Horizon | Seed 2024 | 3-seed mean ± std | Best baseline | Margin | Verdict |
|---|---|---|---|---|---|
| 96 | 0.371 | 0.371 ± 0.001 | 0.375 (CycleNet) | −0.004 | Real: 4× the seed std |
| 192 | 0.428 | 0.429 ± 0.001 | 0.429 (TimeXer) | −0.001 | Inside the noise |
| 336 | 0.476 | 0.478 ± 0.003 | 0.468 (TimeXer) | +0.008 | TQNet loses |
| 720 | 0.487 | 0.496 ± 0.012 | 0.469 (TimeXer) | +0.018 | TQNet loses, and is noisiest |

---

## 5. The split, recomputed from the loader

Appendix A.2 says "6:2:2 for the ETT and PEMS series". That is true only of the 14,400 rows the
loader chooses to use; in absolute terms it is the Informer convention of **12 / 4 / 4 months**.
Validation and test windows start `seq_len` early so the first target lands exactly on the boundary —
legitimate, since it only feeds past data in.

| Split | CSV rows | Length | Windows (L=H=96) | Share of used data |
|---|---|---|---|---|
| train | `[0 : 8640]` | 8,640 | 8,449 | 60% |
| val | `[8544 : 11520]` | 2,976 | 2,785 | 20% |
| test | `[11424 : 14400]` | 2,976 | 2,785 | 20% |
| **never used** | `[14400 : 17420]` | **3,020** | — | — |

Derived from `data_provider/data_loader.py:49-50`, then executed to confirm. 12·30·24 = 8640.

### Leakage audit (the brief's pass/fail requirement)

The pipeline is clean on the stage that usually fails. None of the findings is fatal.

| Stage | Finding | Evidence | Ruling |
|---|---|---|---|
| Scaling | `StandardScaler` fit on rows `[0:8640]` only, then applied to the whole series | `data_loader.py:61-63` | **Clean** |
| Model selection | Early stopping and best-checkpoint both keyed on validation loss, never test | `exp_main.py:225, 230`; `tools.py:52-71` | **Clean** |
| Feature construction | Cycle index is calendar phase (`row mod 24`) — no lookahead. Timestamp features are computed but TQNet never consumes them | `data_loader.py:84`; `TQNet.forward(x, cycle_index)` | **Clean** |
| Observation hygiene | Test loss evaluated and printed every epoch alongside validation loss | `exp_main.py:226, 228-229` | Does not enter model selection, but it is a human-in-the-loop channel and must be disclosed |
| Validation completeness | The val loader inherits `shuffle=True, drop_last=True`, so 225 of 2,785 val windows (8%) never reach the early-stopping signal | `data_factory.py:30-33` | Not leakage, but early stopping is computed on 92% of validation |
| Metric scale | De-normalisation commented out; MSE/MAE are in z-scored units | `exp_main.py:333-335, 344-345` | Explains why MAPE/SMAPE are unusable here |

Verify the scaler by inspecting `scaler.mean_` and `scaler.scale_` against the train slice, not by
re-reading the call site — the audit is supposed to check the fitted object.

---

## 6. Limitations inventory

Flat, numbered, **unranked**. This is material to brainstorm against, not a recommendation.
"Conceded" marks limitations the authors state themselves — safe to cite, but also the obvious
targets everyone else will reach for.

| # | Limitation | Evidence | Source |
|---|---|---|---|
| 1 | Depends on the data having clear periodicity; *W* must be supplied by hand per dataset | "TQNet heavily relies on the inherent periodicity of the data to determine the hyperparameter W. This dependency may limit its generalization to datasets without clear periodic patterns." §5 | Conceded |
| 2 | Forcing multivariate modelling can hurt when correlations are genuinely weak | "enforcing strong multivariate modeling may introduce unnecessary complexity and even negatively impact performance" §5 | Conceded |
| 3 | The benefit of multivariate modelling shrinks as the look-back grows; longer windows add noise and overfitting risk | §5 third bullet; Figure 8 p. 16. The repo README states it as "hard to scale to ultra-long look-back inputs due to low SNR in multivariate histories" and lists it as **unsolved** | Conceded — the lineage's open problem |
| 4 | One period per dataset only. Irregularly interwoven periods (weekly + monthly) are not handled; the authors' own suggested fix is to ensemble several TQNet models | Appendix A.3 case 2, p. 14 | Conceded |
| 5 | *W* must be an integer number of samples; a non-integer or slowly drifting period cannot be expressed | `θ_TQ` indexed by `t mod W` with integer *W*, `TQNet.py:53` | Code |
| 6 | Misspecifying *W* is actively harmful, not merely suboptimal: *W*=23 or 167 scored **worse than using no TQ at all** on Electricity | Figure 6 p. 8 — *W*=167 gives MSE 0.180 vs 0.175 with no TQ | Paper |
| 7 | No uncertainty output of any kind — point forecasts only, no interval, quantile or variance | Output is a single `Linear` to *H*; loss is plain MSE | Code |
| 8 | `θ_TQ` is a static lookup keyed on clock phase; no memory of the recent past, cannot adapt online to drift without retraining | `TQNet.py:21, 53-54` | Code |
| 9 | The only defence against distribution shift is per-window mean/variance removal — no affine RevIN, no trend or frequency-adaptive normalisation | `TQNet.py:44-47`; `layers/RevIN.py` present but unused | Code |
| 10 | Attention heads (4) and attention dropout (0.5) are hard-coded, not exposed, and identical for a 7-channel and an 883-channel dataset | `TQNet.py:24` | Code |
| 11 | Ablations and integration studies run only on Electricity, PEMS03, PEMS04. **No ablation on any ETT dataset** | `scripts/Ablation/*.sh` reference only `electricity.csv`, `PEMS03.npz`, `PEMS04.npz` | Code — an absent ablation |
| 12 | Consequently, on ETTh1 there is no published evidence that TQ or the attention layer contributes anything | follows from 11 | Our inference |
| 13 | On ETTh1 average MSE, TQNet is beaten by TimeXer (0.441 vs 0.437) in the paper's own headline table | Table 2, p. 6 | Paper |
| 14 | Main table is a single seed (2024); the 3-seed study is in Appendix B.6 and varies seeds only, never the data split | Table 9, p. 18 | Paper |
| 15 | Baseline numbers are copied from TimeXer / iTransformer / CycleNet rather than re-run, so every baseline inherits those papers' setups | Table 5 caption, p. 15 | Paper |
| 16 | Stacking more TQ-MHA + MLP layers makes results slightly **worse** on 9 of 12 datasets — capacity is not the bottleneck, so "make it bigger" is a dead direction | Table 7, p. 16 | Paper |
| 17 | Attention is quadratic in channel count; the efficiency claim is empirical and bounded to *C* < 1000 | §4.3 Efficiency Analysis, p. 9 | Conceded |
| 18 | Seeding is incomplete: `torch.cuda.manual_seed_all`, cuDNN determinism and dataloader worker seeds are never set, with `num_workers` defaulting to 10 | `run.py:104-107`; `data_factory.py:52` | Code |
| 19 | 3,020 rows of ETTh1 discarded without comment; the paper misreports the series length | §5 above | Our arithmetic |
| 20 | `θ_TQ` initialised to zeros, so at step 0 every query is identical and the attention map is uniform across channels — correlation structure is learned from a degenerate start | `TQNet.py:21` | Code, low confidence on whether it matters |

---

## 7. Named components an improvement could attach to

In file order, **not** preference order. The point of listing these before choosing is to keep the
design conversation concrete about *what line becomes what*.

| Component | Location | What it is now | What it opens |
|---|---|---|---|
| `θ_TQ`, the query table | `TQNet.py:21` | A single (W, C) zero-initialised parameter block indexed by `t mod W` | Multiple co-existing periods; a continuous or learned phase; a period that adapts |
| The query gather | `TQNet.py:53-54` | Integer modular indexing — a hard lookup on calendar phase | Interpolation between neighbouring phases; soft or probabilistic phase assignment |
| The single MHA layer | `TQNet.py:24, 56` | 4 heads, dropout 0.5, both hard-coded; scaling is 1/√24 not the paper's 1/√96 | Correcting the scale; sparsifying or regularising the 7×7 channel attention |
| Instance normalisation | `TQNet.py:44-47, 72-73` | Plain per-window mean/std removal, no affine terms, no trend handling | The natural home for anything about drift, non-stationarity or state |
| The output projection | `TQNet.py:35-38` | Dropout then one `Linear(512 → H)`. One number per step, no spread | Any distributional or interval output; quantile or likelihood heads |
| The training objective | `exp_main.py:61` | `nn.MSELoss` on normalised values, weighted equally across all steps and channels | Horizon weighting; robust losses; anything that changes what "good" means |

---

## 8. Originality risk — what is already taken

**A direct follow-up paper exists and I could not read it.** *PTQNet: Periodic-temporal query network
for long-term multivariate time series forecasting* — Xun, Yan, Yang, Cai, Wang, *Information
Processing & Management* 63(7):104785, published 11 Apr 2026, doi `10.1016/j.ipm.2026.104785`. The
title points squarely at the periodicity limitation. It is paywalled and the abstract could not be
retrieved. **If our improvement touches periods or multi-period modelling, reading this is a
prerequisite, not a nicety.** University library access should get it.

**The authors have already pre-empted the obvious multi-period fix.** Appendix A.3 proposes
ensembling several TQNet models at different *W*. Doing exactly that implements their suggestion
rather than improving on them.

**The lineage has already consumed two adjacent ideas.** SparseTSF (ICML 2024 Oral) → CycleNet
(NeurIPS 2024 Spotlight) → TQNet is the same group solving each predecessor's stated limitation.
CycleNet already did "learnable periodic vectors as a residual"; TQNet already did "those vectors as
attention queries". The README's own table is the cleanest statement of what is left, and what is
left is inventory item 3: low SNR in multivariate histories at long look-back.

**Community record.** Only two GitHub issues ever filed (#1, #3), both about PEMS prediction lengths,
both closed. No reproduction complaints, no published refutation of TQNet specifically. 114 stars,
14 forks. ICML reviews are behind an OpenReview browser check and could not be read.

---

## 9. The two runs that are prepared and waiting

Answering "does TQ help on ETTh1 at all?" needs two extra runs and no new code. The switches already
exist in the source as plain booleans at `TQNet.py:17-18`, but the repo expects you to *edit the file*
between variants, which is unreproducible. So:

- **`repro/tqnet-ablation-flags.patch`** turns them into `--use_tq` and `--channel_aggre` flags.
  Defaults are the published values, verified: the default build has 661,640 parameters, identical to
  unpatched. Non-default runs get a `_tq<N>ca<N>` suffix in the checkpoint path and the `result.txt`
  label, so they cannot collide with the reproduction. Patch confirmed to apply cleanly to a fresh
  clone at the pinned commit.
- **`repro/run_etth1_ablation.sh`** runs the two decisive variants with every other flag copied
  verbatim from `scripts/TQNet/etth1.sh`.

| Variant | Flags | Params | What the gap from 0.3712 tells us |
|---|---|---|---|
| baseline | (default) | 661,640 | the reproduction itself |
| `tq0ca1` | `--use_tq 0` | 661,472 | TQ removed, attention kept → **this gap is TQ's entire contribution on ETTh1** |
| `tq0ca0` | `--use_tq 0 --channel_aggre 0` | 624,224 | pure MLP → the floor: how much of 0.3712 needs neither mechanism |

If TQ contributes nothing at *C* = 7, that reframes the whole project — and it is cheap to check while
the environment is already up.

---

## 10. Two things that will bite the reproduction run

**The loader breaks on modern pandas.** `data_loader.py` calls `df_stamp.drop(['date'], 1)`, a
positional-argument form removed in pandas 2.0. Confirmed to raise `TypeError` on pandas 2.3.3. The
README asks for Python 3.8, which is end-of-life. Record every deviation — each one is a line in the
README deliverable.

**Running the code destroys the evidence.** `exp_main.py:348` opens `result.txt` in append mode in the
working directory — the same file that ships the authors' reference numbers. Copy it aside before the
first run or the cleanest available check on our reconstruction is gone.

---

## 11. What the brief forces on whatever gets chosen

| Constraint | Consequence |
|---|---|
| The improved method must use the **same split and the same metrics** as the reconstruction | The 12/4/4-month ETTh1 split and MSE/MAE on normalised data stay frozen. An improvement that only looks good under a different protocol cannot be reported. |
| The change must improve performance, robustness, interpretability, efficiency **or applicability** | Accuracy is one of five permitted axes and, given the 0.004 margin, the hardest to defend. |
| A quantitative prediction is registered **before** the run | The idea must be specific enough to predict a direction and a rough magnitude. "It would probably help" is not registrable. |
| The report needs a 3-way table: paper / reconstruction / improved | One improvement, same cell. Breadth is not rewarded. |
| "What did not work" is **required** report content | An improvement that fails but is measured honestly is gradeable. This lowers the risk of an ambitious choice. |

---

## 12. Open questions

1. **PTQNet's contribution.** Paywalled. Blocks any period-related improvement until read.
2. **Does TQ help on ETTh1 at all?** Unknown and unpublished. §9 answers it in two runs.
3. **Our own seed spread.** The paper reports ±0.001 at *H*=96 on their hardware. We need our own
   two or three seeds before any claim.
4. **Which of the five permitted axes we target.** The decision that determines everything else.

---

## Provenance

Everything above was read from `files/project/TQnet.pdf` (all 18 pages, appendices included) and from
a clone at commit `15e19cb23483ed52398566c4baa959168cfffa57`, kept outside this repository. The
parameter counts, the split arithmetic, the pandas failure and the patch's applicability were
**executed**, not estimated. Nothing was trained, so no claim here rests on a run of ours.

Not verified, and stated as unverified: PTQNet's actual contribution; the ICML reviews; whether
inventory item 20 (zero initialisation) has any practical effect.
