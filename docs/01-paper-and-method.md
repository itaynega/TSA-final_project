# 1. The paper, the problem, and the algorithm

**Paper.** Shengsheng Lin, Haojun Chen, Haijie Wu, Chunyun Qiu, Weiwei Lin,
*Temporal Query Network for Efficient Multivariate Time Series Forecasting*,
ICML 2025, PMLR 267. arXiv:2505.12917v2. Local copy: `files/project/TQnet.pdf`.
**Code.** <https://github.com/ACAT-SCUT/TQNet>, Apache-2.0, vendored here at commit
`15e19cb23483ed52398566c4baa959168cfffa57`.

---

## 1.1 The task

Multivariate time series forecasting (MTSF). Given the last *L* observations of *C*
variables, predict the next *H* observations of all *C* variables:

$$X_t \in \mathbb{R}^{C \times L} \longmapsto Y_t \in \mathbb{R}^{C \times H}$$

The variables are called **channels** in this literature. Our target cell is ETTh1
with *C* = 7 channels, *L* = 96 hours in, *H* = 96 hours out. In the course's
vocabulary this is **multivariate, supervised, deterministic point forecasting** over
a fixed horizon: one number per channel per future step, no interval and no
distribution.

## 1.2 The problem the paper attacks

Everything in TQNet follows from one observation, and it is worth stating carefully
because the whole design is a response to it.

To forecast several correlated variables well, a model should exploit the
correlations between them. The obvious way to learn those correlations is from the
input sample: look at the 96×7 matrix in front of you, work out how the seven
channels relate, and use that. Attention over the channel axis does exactly this, and
it is what iTransformer and TimeXer do.

The paper's objection is that **a single sample is a bad estimator of a correlation
structure.** A 96-step window of a real sensor series contains extreme values,
dropouts and noise. The correlation matrix you compute from one such window can look
quite unlike the correlation matrix of the process, and Figure 1 of the paper makes
this concrete on the Traffic dataset: panel (a) is the correlation computed over the
whole training set, panel (c) is the correlation from individual samples, and they
disagree substantially. So a channel-attention model derived entirely from the current
sample is estimating a stable quantity from an unstable measurement, sample by sample,
and inheriting the instability.

At the same time, you cannot simply replace the sample correlation with the global
one. A fixed global matrix would ignore what is actually happening right now, which is
the only thing that distinguishes one forecast from another.

So the problem is: **how do you use a stable, dataset-level estimate of inter-variable
structure without discarding the sample-specific information you need to actually make
a forecast?**

## 1.3 The answer: the Temporal Query

The mechanism is a single idea applied to the attention mechanism, and it is easiest to
see by recalling what the three attention inputs do. In
$\mathrm{Attention}(Q,K,V) = \mathrm{Softmax}(QK^\top/\sqrt{d})V$, the queries and keys
together decide *which* things attend to which — they produce the correlation map — and
the values supply the content that gets mixed.

Conventional channel attention derives all three from the input, so the correlation map
is a per-sample estimate. TQNet's change is minimal and surgical:

- **Queries come from a learnable parameter block**, not from the input. Keys and values
  still come from the raw input.

Because the queries are shared parameters trained across the whole dataset, the
correlation map is no longer estimated from one window. It is a learned quantity, and
gradient descent drives it towards something like an average of the per-sample maps —
the paper's Appendix A.4 argues this explicitly, ending at

$$\mathrm{Corr}(Q^i) \approx \frac{1}{N}\sum_{n=0}^{N-1}\mathrm{Corr}(K^{i+nW})$$

which is the formal version of "the query converges to the average correlation
structure over many samples". Meanwhile K and V still carry the current window, so the
sample-specific content is preserved. Global structure and local content enter through
different doors.

The paper's Table 3 tests all three options and the ordering supports the argument:
Q from TQ with K from raw data is best, Q and K both from raw data is second, and Q and
K both from TQ — pure global, no sample information — is worst. Both halves are needed.

### Why "temporal", and what *W* is for

If the query block were a single fixed matrix, every sample in the dataset would share
one correlation map. But inter-variable relationships in real data depend on where you
are in the cycle: at 3am the relationship between load channels is not what it is at
6pm. So the parameter block is

$$\theta_{TQ} \in \mathbb{R}^{C \times W}$$

where *W* is the **period of the dataset**, and the query for a sample is a slice of
length *L* taken from it cyclically, starting at position `t mod W`:

$$\text{index}_{TQ} = (t \bmod W + \text{Range}(L)) \bmod W$$

For ETTh1, *W* = 24, so there are 24 learnable query vectors, one per hour of the day,
and all samples at the same clock hour share the same query. This is what makes the
averaging in the equation above happen over *comparable* samples — the same phase of
the cycle — rather than over everything indiscriminately.

Two consequences worth carrying forward:

- **The parameter cost is tiny.** For ETTh1, `θ_TQ` is 24×7 = **168 parameters**, out
  of 661,640 in the whole model. The paper's central mechanism is 0.025% of the model.
- ***W* has to be supplied by hand**, per dataset, from domain knowledge or from an
  ACF plot. The paper concedes this as its first limitation, and Figure 6 shows the
  cost of getting it wrong: on Electricity, a misspecified *W* = 167 scored *worse
  than using no Temporal Query at all*.

## 1.4 Where TQNet sits in the field

The paper organises prior work by how it treats the channel axis, and the taxonomy is
useful because it explains why a 168-parameter change is publishable at all.

| Strategy | Idea | Weakness | Examples |
|---|---|---|---|
| **Channel Mixing (CM)** | embed all variables at a time step together | variables become indistinguishable inside the model; often loses to linear baselines | Informer, Autoformer, FEDformer |
| **Channel Independence (CI)** | forecast each variable separately with shared weights | robust, but cannot represent inter-variable structure at all | PatchTST, DLinear, SparseTSF, CycleNet |
| **Channel Dependence (CD)** | model inter-variable structure explicitly | more capacity, but correlation estimates are noisy | iTransformer, TimeXer, Crossformer, **TQNet** |

TQNet is a CD method whose contribution is to make the CD correlation estimate more
robust — which is precisely the weakness the column above names. It is also the third
paper in a line by the same group, each solving its predecessor's stated limitation:
SparseTSF (ICML 2024) → CycleNet (NeurIPS 2024) → TQNet. CycleNet introduced learnable
periodic vectors as an additive residual; TQNet reuses the same periodic-parameter idea
in the attention queries instead.

## 1.5 What the paper claims

- **Accuracy.** Top-2 on 22 of 24 dataset/metric combinations across 12 datasets
  (Table 2), with the largest gains on high-dimensional data — Electricity (321
  channels) and the PEMS series (170–883 channels).
- **Efficiency.** Accuracy competitive with heavy transformers at a parameter count and
  training time comparable to DLinear, a purely linear model (Figure 7).
- **Interpretability.** t-SNE of the learned TQ vectors groups channels whose raw series
  look alike (Figure 4).
- **Robustness.** Low standard deviation across seeds and learning rates (Table 9).

### The claim that matters most for us

We reproduce **one** cell: ETTh1, multivariate, *L*=96 → *H*=96, seed 2024, where the
paper reports MSE 0.371 / MAE 0.393. Two facts about that cell shape the whole project
and are documented here so they are not discovered late:

1. **On ETTh1 specifically, TQNet is not the strongest model in its own table.**
   Averaged over the four horizons it loses to TimeXer on MSE (0.441 vs 0.437, Table 2)
   and is beaten outright at *H* = 336 and 720. It wins at 96 and 192. Our cell is a
   genuine win, but a narrow one.
2. **The margin is 0.004 MSE and the paper's own seed noise is 0.001** (Table 9). The
   claimed effect is four times the measurement error — real, but small enough that any
   Stage-2 improvement aimed at MSE on this cell is aiming at a target barely above
   noise.

Also worth knowing: **the paper never ablates TQNet on any ETT dataset.** Every
ablation and integration study runs on Electricity, PEMS03 and PEMS04, all with more
than 100 channels. ETTh1 has 7. There is therefore no published evidence that the
Temporal Query contributes anything at that width, which is why
`repro/run_etth1_ablation.sh` exists.

## 1.6 How this connects to the course material

This is not a tangential connection; the paper is built out of course primitives.

**Forecasting, and the metrics.** The task is the course's forecasting setting, and the
paper's metrics are the course's metrics. `Time-Series Forecasting.pdf` defines the
forecast error as $e_t = y_t - f(x_t)$ and then MSE, MAE, RMSE and MdAE from it
(printed sl. 47–48). TQNet reports MSE and MAE, so **MAE alone discharges both the
"paper's metric" and the "metric taught in class" requirements.** We add RMSE and MdAE.
The full treatment, including why MAPE and SMAPE are excluded here, is in
`report/metrics.md`.

**Seasonality and periodicity.** *W* is a seasonal period, and the paper's advice for
choosing it is the course's method: domain knowledge, or read it off the
autocorrelation function. Upstream ships `TQNet/acf_plot.ipynb` for exactly that. The
entire Temporal Query is a parameterisation of seasonality — one learnable vector per
phase of the cycle — which makes this paper an unusually direct application of a
classical idea inside a neural architecture.

**The seasonal-naive baseline.** Our required baseline is seasonal-naive at period 24,
a course-standard method. It is the right baseline here rather than merely a
conventional one: it encodes *the same* assumption as TQNet's `--cycle 24`, so the gap
between them isolates what the network adds beyond knowing that ETTh1 repeats daily.

**Stationarity and normalisation.** Instance normalisation — subtract the window's mean
and divide by its standard deviation, then invert on the output — is the course's
differencing/detrending intuition applied per window, to handle distribution shift
between training and test periods. The paper's equations 7–8; `Pre-precessing.pdf` and
`Time-Series Analysis.pdf` cover the underlying stationarity requirement.

**Correlation structure.** The 7×7 attention map is a learned cross-correlation matrix
between channels. Reading it is the course's cross-correlation analysis, learned rather
than computed.

**Temporal validation.** The evaluation protocol is **rolling-origin**: a single fixed
model, with the forecast origin advancing one hour at a time across the test months.
This is one of the protocols the assignment names as valid, and it is the paper's own.
The no-future-information requirement shows up in three concrete places, all audited in
`docs/03` and by `tools/audit_split.py`: the split is chronological, the scaler is
fitted on training rows only, and the cycle index is arithmetic on the row number
rather than anything observed.

**Deep learning for time series.** `DL for TS.pdf` covers the attention-based family
this model belongs to. TQNet is a good case study in the direction that deck's material
points: the winning design here is a single attention layer plus a two-layer MLP, and
the paper's own Table 7 shows that stacking three of these blocks makes results
*slightly worse* on 9 of 12 datasets. Capacity is not the bottleneck.

## 1.7 Reading order

- `docs/02-architecture-and-implementation.md` — the architecture step by step, how the
  code implements it, where the paper and the code disagree, and every change we made.
- `docs/03-running-the-experiments.md` — data, training, validation, and how to
  reproduce every number.
- `files/project/TQNET_BRIEF.md` — the prior deep read of the paper and repository,
  including a 20-item limitations inventory for Stage 2.
- `report/metrics.md` — the metrics, in the course's notation.
