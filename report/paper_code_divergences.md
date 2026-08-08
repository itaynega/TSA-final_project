# Where the TQNet paper and the TQNet code disagree

**Written 2026-07-31 by Amitay.** For the reconstruction work stream as much as for the improvement
work stream — several items below change what a faithful reimplementation would produce.

**Paper.** Lin, Chen, Wu, Qiu, Lin, *Temporal Query Network for Efficient Multivariate Time Series
Forecasting*, ICML 2025. Local copy `files/project/TQnet.pdf`, 18 pages. Page numbers below are the
paper's own printed numbers.
**Code.** Vendored under `TQNet/`, upstream `github.com/ACAT-SCUT/TQNet`, Apache-2.0.

Every item was checked against the PDF and the source directly. Nothing here is taken from a summary.

---

## The one ruling that matters most

**Reimplement from Appendix A.1 (Algorithm 1, p. 13), not from Section 3.**

Section 3 is the paper's main method description, and on its own it is not sufficient to rebuild the
model. Equations 5 and 6 omit both residual connections, and they omit the 96→512 projection
entirely. Someone working only from Section 3 builds a materially different, smaller network.
Appendix A.1 gives the full forward pass, and the code matches it.

**And for the reconstruction specifically: follow the code, not the paper, wherever item 1 below
applies.** The target number we are reproducing — MSE 0.3712 / MAE 0.3928 — was produced *by the
code*. It is a property of the implementation, not of the equations. A reconstruction that
faithfully implements Equation 3 as written will not land on that number, because the attention is
scaled differently. Follow the code to hit the target, and report the divergence as a finding.

---

## Part 1 — Divergences that change the model's behaviour

### 1. The attention is scaled twice as strongly as the paper says

| | |
|---|---|
| **Paper** | Equation 3, p. 3: attention scores are divided by `√L`, where `L` is the look-back length. `L = 96`, so the divisor is `√96 ≈ 9.80`. Equation 10, p. 14, repeats the same `√L`. |
| **Code** | `models/TQNet.py:24` builds `nn.MultiheadAttention(embed_dim=96, num_heads=4)`. PyTorch divides by the square root of the *per-head* width, which is `96 / 4 = 24`. The divisor is `√24 ≈ 4.90`. |
| **Effect** | The scores fed into the softmax are **exactly 2× larger** in the code than the equation specifies. Larger scores make the softmax sharper, so attention concentrates on fewer channels than the paper describes. |

The paper states `√L` in two separate places, so this is not a typographical slip in one equation.

**This is the single most consequential item in this document.** It is the reason the ruling above
exists: implement Equation 3 literally and you get a different model from the one that produced
0.3712.

### 2. There is an extra activation function at the end of the MLP

| | |
|---|---|
| **Paper** | Equation 5, p. 4: `h_mlp = Linear(GeLU(Linear(h_attn)))`. Appendix A.1, p. 13, describes "a two-layer multi-layer perceptron (MLP) with GeLU activation". Both give: Linear → GELU → Linear, ending on a Linear. |
| **Code** | `models/TQNet.py:28-33`: `Sequential(Linear, GELU, Linear, GELU)` — ending on a GELU. |
| **Effect** | The extra GELU sits between the MLP and the residual addition at line 69. GELU outputs are bounded below at roughly −0.17, so the MLP branch can only push the residual in a near-one-sided way. The paper's version has no such constraint. |

### 3. About four months of ETTh1 are silently discarded, and the paper misstates the length

| | |
|---|---|
| **Paper** | Table 1, p. 5, lists ETTh1 as **14,400** timesteps. |
| **Code** | `ETTh1.csv` actually contains **17,420** rows. `data_provider/data_loader.py:49-50` sets the split boundaries with hard-coded arithmetic — `12*30*24` for train, then `+4*30*24`, then `+8*30*24` — which stops at row 14,400. |
| **Effect** | **3,020 rows, about 4.1 months of data, are never used by any split.** The paper reports the truncated figure as if it were the dataset length, and does not mention the truncation anywhere. |

This also explains the "6:2:2" split ratio in Appendix A.2: it is 6:2:2 of the 14,400 rows the loader
chooses to use, which in absolute terms is the 12 / 4 / 4 month convention inherited from Informer.

### 4. The daily-phase lookup is anchored at the wrong end of the window

| | |
|---|---|
| **Paper** | Equation 9, p. 13, and Algorithm 1: the phase index comes from `t`, the timestep of the look-back window `X_t` — that is, where the input window *starts*. |
| **Code** | `data_provider/data_loader.py:97`: `cycle_index = torch.tensor(self.cycle_index[s_end])`. `s_end` is where the input window *ends* — equivalently, the first step being forecast. |
| **Effect on ETTh1** | **None.** The two differ by exactly `L = 96` positions, and `96 mod 24 = 0`, so on ETTh1 they select the identical row. |
| **Effect elsewhere** | Real. On Electricity the period is 168 and `96 mod 168 = 96 ≠ 0`, so the two conventions pick different rows. |

Harmless for our benchmark, but worth recording: it is a genuine mismatch that happens to cancel on
the one dataset we use, and anyone generalising this code to another dataset inherits a real bug.

---

## Part 2 — Cosmetic differences

Record them so nobody re-derives them; do not weight them.

**5. The query table is stored transposed.** The paper writes `θ_TQ ∈ R^(C×W)` — channels by period.
`models/TQNet.py:21` allocates `(W, C)` = `(24, 7)`. Both index the period axis. Functionally
identical.

**6. `model_type` is a dead flag.** Read into `self.model_type` at `models/TQNet.py:12` and never
used again. The paper does not mention it. Do not list it as a hyperparameter.

---

## Part 3 — Things that look like divergences but are not

Each of these was checked because it looked like a mismatch. None of them is one. Recorded so the
checks are not repeated.

**7. The hard-coded attention dropout of 0.5 and 4 heads match the paper.** Appendix A.2, p. 13,
states it outright: *"the number of attention heads in the multi-head attention mechanism is fixed
at 4, and the dropout rate is set to 0.5 by default."* That these are hard-coded at
`models/TQNet.py:24` rather than exposed as command-line options is a code-quality complaint. It is
**not** a paper/code divergence and must not be reported as one.

**8. The absence of learnable scale/shift terms in the normalisation matches the paper.** Section
3.2, Equations 7–8, p. 4, define instance normalisation as plain removal of the window's mean and
variance, with no learnable parameters. RevIN (Kim et al., 2021) is cited only as related work; the
paper explicitly adopts instead *"a simple yet effective IN method that used in iTransformer and
CycleNet."* The code's hand-rolled normalisation at `models/TQNet.py:44-47` matches. The presence of
an unused `layers/RevIN.py` in the repository is therefore consistent with the paper, not evidence
against it.

**9. The residual connections match — against the appendix.** Main-text Equations 5 and 6 show no
residuals. But Section 3.1 prose asserts them, and Algorithm 1, p. 13, specifies them precisely:
line 8 `h_attn ← MHA(...) + X_t`, line 10 `h_mlp ← MLP(h'_attn) + h'_attn`. The code matches
Algorithm 1 exactly. This is an **internal inconsistency within the paper**, between its main text
and its appendix — not a disagreement between paper and code.

**10. The 96→512 projection matches — again against the appendix.** It appears nowhere in Section 3.
Algorithm 1 line 9 has it: `h'_attn ∈ R^(C×d) ← Linear(h_attn)`. The code's `input_proj` matches.

---

## Where the parameters actually live

Counted from the layer definitions and confirmed against the model's total.

| Component | Parameters | Share |
|---|---|---|
| MLP block (`input_proj` + `model` + output `Linear`) | 624,224 | **94.3%** |
| Multi-head attention | 37,248 | 5.6% |
| **The Temporal Query table — the paper's entire novelty** | **168** | **0.025%** |
| **Total** | **661,640** | |

The 168 figure is `24 × 7`: one row per hour of the day, one column per channel.

Useful context for both work streams: the mechanism the paper is *about* is a rounding error in the
parameter budget, and the paper's own Appendix B.3 reports that adding more of the 94% makes results
worse on 9 of 12 datasets.

---

## Provenance

Paper text extracted with `pdfplumber` from `files/project/TQnet.pdf` and read in full, appendices
included. Code read at the vendored `TQNet/` tree. Parameter counts computed from the layer
definitions and reconciled against the known total of 661,640. Every quotation above is verbatim;
every page and line reference was resolved before being written down.

Nothing in this document rests on a training run. No model was trained to produce it.
