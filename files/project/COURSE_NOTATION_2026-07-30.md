# Course notation, metrics, models and conventions — Time-Series Analysis (Havana Rika)

Built 2026-07-30. Purpose: supply requirements **B6** (baseline), **B7/B8** (metrics), and the report's
term/symbol mapping, from what the course actually delivered.

**Sourcing rule applied:** every entry below names a file and carries either a quoted fragment or a
slide/section heading. Formulas are transcribed in the course's own notation, including its typos.
Where I read a formula off a rendered slide image rather than extracted text, the entry says so.

---

## 0. Provenance — read this before trusting any `.md` file

Three facts established by running commands, not assumed:

**0.1 The `.md` files are machine-generated from the PDFs.** The loose file
`Lectures/gemini-code-1782123087695.py` is the generator. It reads a folder and converts every
non-`.md` file with Microsoft's MarkItDown:

> `md = MarkItDown()` … `result = md.convert(file_path)` … `f.write(result.text_content)`

with `folder_path = r"C:\...\ניתוח סדרות עיתיות\הרצאות"`. **The PDFs are authoritative. The `.md`
files are a lossy derivative.** This is documented rather than inferred.

**0.2 Equivalence test result (run on all 8 pairs, not just 2).** Whitespace- and
markup-stripped MD5 of `.md` vs `pdftotext -layout` output:

| Lecture pair tested | Byte count `.md` vs PDF-extract | Hash | Verdict |
|---|---|---|---|
| `EDA.md` vs `EDA.pdf` | 4466 vs 4466 | equal | **Identical** |
| `EDA continue.md` vs `EDA continue.pdf` | 9840 vs 9840 | equal | **Identical** |
| `Pre-precessing.md` vs `Pre-precessing.pdf` | 17545 vs 17545 | equal | **Identical** |
| `Intro.md` vs `Intro.pdf` | 7356 vs 7356 | differ | Same character multiset — pure reordering |
| `Time-Series Forecasting.md` vs `.pdf` | 9424 vs 9424 | differ | Same character multiset — pure reordering |
| `Unsupervised models for TS.md` vs `.pdf` | 7709 vs 7709 | differ | Same character multiset — pure reordering |
| `Time-Series Analysis with Python.md` vs `.pdf` | 13318 vs 13318 | differ | Same character multiset — pure reordering |
| `CPDexamples.md` vs `CPDexamples.pdf` | 35475 vs 34938 | differ | **Genuine content difference** |

Reordering is an artefact of two-column extraction, confirmed by sorting characters and re-hashing
(identical for all four "reordering" rows). CPDexamples is the one real divergence: `CPDexamples.md`
contains **76 `(cid:NN)` glyph-mapping failures** (`grep -c '(cid:[0-9]*)'`); every other `.md` has 0.
For that lecture, `pdftotext` output is strictly better than the shipped `.md`.

**0.3 Text extraction drops the formulas.** Neither `.md` nor `pdftotext` captures image content, and
the decks are image-heavy (`pdfimages -list`): CPDexamples 63 images/91 pages, DL for TS 84/31,
EDA 76/42, Pre-precessing 74/74, Intro 51/40, ML models for TS 66/51. **Several of the most important
formulas in this course — including the entire forecast-metrics table — exist only as images.** They
are recovered below by rendering the page (`pdftoppm`) and reading it.

**0.4 The task brief's file list was incomplete.** The folder holds **10 PDFs, not 8**.
`DL for TS.pdf` and `ML models for TS.pdf` have no `.md` counterpart (both PDFs dated Jul 30; the
`.md` files Jun 22 — generated before those decks existed). `HW/Assignment 1/` holds **two**
notebooks, so there are **4 homework notebooks, not 3**. All are covered below.

---

## 1. Lecture inventory

`.md`/`.pdf` column reports the §0.2 test. "n/a" = no `.md` exists.

| # | File | Pages | Topics covered | `.md`/`.pdf` |
|---|---|---|---|---|
| 1 | `Lectures/Intro.pdf` / `.md` | 40 | What is a time series; regular vs irregular; Data Generating Process (DGP); synthetic series generation (TimeSynth); white noise, red noise; cyclical/seasonal signals; autoregressive signals; stationary vs non-stationary; change in mean, change in variance (heteroscedasticity); what can be forecast; **assessment weights and 13-topic syllabus** | Reordering only |
| 2 | `Lectures/EDA.pdf` / `.md` | 42 | Libraries (scipy, numpy, pandas, seaborn, scikit-learn); datetime/strftime; pandas basics; understanding variables; feature leakage and collinearity; Pearson vs Spearman correlation; trend, seasonality, cyclic variation; stationarity; autocorrelation plots; resampling; seasonal box plots; calendar heatmaps | Identical |
| 3 | `Lectures/EDA continue.pdf` / `.md` | 41 | Seasonal decomposition (detrend → deseasonalize); moving averages; LOESS; period-adjusted averages; additive vs multiplicative seasonality; `seasonal_decompose`; STL; Fourier series/terms; MSTL; outlier detection: standard deviation, IQR, Isolation Forest, ESD/S-ESD; treating outliers | Identical |
| 4 | `Lectures/Pre-precessing.pdf` / `.md` *(filename misspelled in repo; left untouched)* | 74 | Feature transforms; min-max scaling; z-score normalization; log transform; power transforms (Box-Cox, Yeo-Johnson); quantile transformation; normality testing; imputation; handling missing data; ffill/bfill/mean fill; linear/nearest/spline/polynomial interpolation; seasonal interpolation; compact/expanded/wide data forms; enforcing regular intervals; feature engineering; date/time/calendar/holiday/payday/season/sun-moon/business-day features; tsfresh, featuretools; **ROCKET**; **shapelets**; causality and leakage | Identical |
| 5 | `Lectures/Time-Series Forecasting.pdf` / `.md` | 46 | MA(q); AR(p); Wold's decomposition; ARMA; ARIMA; differencing; seasonal naive; SARIMA; model selection AIC/BIC; exponential smoothing (SES); Theta method; Holt-Winters; additive vs multiplicative seasonality; ARCH; GARCH; VAR; libraries; statsmodels classes; statistical tests; **forecast evaluation metrics (image only)** | Reordering only |
| 6 | `Lectures/ML models for TS.pdf` | 51 | **No `.md` exists.** Motivation; libraries (TSFresh, SKTime, GreyKite, Kats); **validation in time-series**; **walk-forward validation**; kNN; DTW+kNN; Robot Execution Failures dataset (classification); **forecast evaluation incl. R² and correlation**; linear regression assumptions; FeatureConfig; ridge; lasso; decision trees; random forest; **XGBoost**; owid-covid-data; date feature transformer; **Silverkite/Greykite** | n/a |
| 7 | `Lectures/DL for TS.pdf` | 31 | **No `.md` exists.** Encoder-decoder paradigm; feed-forward networks; loss function, forward/backward propagation, gradient descent, learning rate; activations (ReLU, tanh, sigmoid, softmax); embedding layer; RNN; RNN in PyTorch (full parameter list); LSTM; CNN; padding/stride/dilation; convolution in PyTorch; **dilated causal CNN**; attention; forecasting with Transformers | n/a |
| 8 | `Lectures/Unsupervised models for TS.pdf` / `.md` | 30 | Why unsupervised; challenges; anomaly detection (definition, **z-score**, Hampel filter); point/contextual/collective anomalies; change point detection; CPD applications; CPD components (cost functions, search methods, constraints); binary segmentation; anomalies vs change points; clustering; distance measures (Euclidean, Manhattan, correlation, **DTW**); **silhouette score** | Reordering only |
| 9 | `Lectures/CPDexamples.pdf` / `.md` | 91 | **Not Rika's deck — see §1.1.** Problem statement; stationarity; CPD cost functions (`cML`, `cL2`, `cΣ`, `clinear`); search methods (dynamic programming, sliding window, binary segmentation, bottom-up); penalized CPD; **PELT**; AIC/BIC for CPD; anomaly detection (histogram, mu/sigma, med/MAD, model-based, **matrix profile**, clustering); **evaluation of event detection (precision/recall, range-based, IoU)** | **Genuine divergence** — use the PDF |

### 1.1 CPDexamples is third-party material

`CPDexamples.pdf` title page reads:

> "Machine Learning for Time Series / **Lecture 5: Change-point and Anomaly Detection** /
> laurent.oudre@ens-paris-saclay.fr / **Master MVA** / 2023-2024"

Every page footer reads "Laurent Oudre  Machine Learning for Time Series  2023-2024  N / 91". This
deck is by a different author, for a different course, and uses **different notation** from Rika's own
decks (§4). Two consequences:

- It cross-references its own course's lectures — "as seen in **Lecture 2**", "already described in
  **Lecture 4**", "just like DTW in **Lecture 1**", "dictionary learning (Lecture 3 & 4)". Those are
  **Oudre's** lecture numbers. Those decks are **not in this folder**. Any concept CPDexamples treats
  as already-known (DFT, dictionary learning, matrix profile background, pattern extraction) was
  never actually delivered to this class in that form.
- When the report cites CPD notation, cite it as Oudre's, not as "the course's". Rika's own CPD
  notation is the much lighter treatment in `Unsupervised models for TS.pdf`.

### 1.2 Syllabus vs delivery

`Intro.md` slide 40 lists a 13-item "Course Content". Ten lecture PDFs exist. Items listed but with
**no corresponding deck**: (8) "Probabilistic forecasting (quantiles, Prophet/BSTS)",
(11) "Multivariate time series and causality (VAR/VECM, Granger)" — VAR alone appears in
Time-Series Forecasting, VECM and Granger nowhere; (12) "Online/streaming learning and drift
detection"; (13) "Special topics and deployment (hierarchical/grouped, intermittent demand,
packaging/APIs)". Also, item (10) names "N-BEATS" and item (9) names "GRU"; neither appears in
`DL for TS.pdf`. See §7.

`Intro.md` slide 39 says "**4 homework assignments (40%)**". Only 3 assignment folders exist
(containing 4 notebooks). Unresolved — flagged, not reconciled.

---

## 2. Metrics taught

Task column: **F** = forecasting/regression, **C** = classification, **A/CPD** = anomaly or
change-point detection, **Cl** = clustering, **MS** = model selection (not an error metric).

### 2.1 Forecast error metrics — `Time-Series Forecasting.pdf`, slides 47–48

These two slides are the **single most citable source for B8**. Both are images; formulas below were
read from rendered pages 45 and 46 (`pdftoppm -r 150`). Slide 47 first defines the error term:

> "If 𝑓(𝑥ₜ) is a prediction of the model for time step 𝑡, and the actual target value is 𝑦ₜ,
> intuitively, for a particular point, 𝑡, of our dataset, the **forecast error** (also **prediction
> error or residual**) is the difference between the actual values of the target and the values our
> model predicts:"
>
> **eₜ = yₜ − f(xₜ)**

| Course's name | Formula, as the course writes it | Measures | Source | Task |
|---|---|---|---|---|
| **Mean squared error** (MSE) | `MSE = (1/N) Σ_{t=1}^{N} e²_t` | Mean squared forecast error; penalises large errors quadratically | `Time-Series Forecasting.pdf` sl. 47 "Forecast Evaluation → Common metrics:" (image) | F |
| **Mean absolute error** (MAE) | `MAE = (1/N) Σ_{t=1}^{N} \|e\|_t` | Mean absolute forecast error; same units as the target | ibid. | F |
| **Root mean squared error** (RMSE) | `RMSE = √MSE` | MSE returned to the target's units | ibid. | F |
| **Median absolute error** (MdAE) | `MdAE = median(\|e\|_t)` | Median absolute error; outlier-robust | `Time-Series Forecasting.pdf` sl. 48 "More metrics for regression" (image) | F |
| **Mean absolute percentage error** (MAPE) | `MAPE = (1/N) Σ_{t=1}^{N} \|e_t\| / \|y_t\|` | Scale-free relative error; undefined at yₜ = 0 | ibid. | F |
| **Symmetric mean absolute percentage error** (SMAPE) | `SMAPE = (1/N) Σ_{t=1}^{N} \|e_t\| / ((\|y_t\| + \|f(x_t)\|)/2)` | Relative error symmetrised over actual and predicted | ibid. | F |
| **Normalized mean squared error** (NMSE) | `NMSE = MSE / σ²` | MSE relative to series variance | ibid. | F |
| **Root mean squared logarithmic error** (RMSLE) | `RMSLE = √( (1/N) Σ_{t=1}^{N} (log(f(x_t) + 1) − log(y_t + 1))² )` | Error on the log scale; penalises under-prediction more | ibid. | F |

Note the course writes MAPE and SMAPE **without** a ×100 factor, so both are fractions, not
percentages, despite the name. `sklearn.metrics.mean_absolute_percentage_error` (used in HW2) also
returns a fraction, so lecture and homework agree.

Note also: `\|e\|_t` is how the slide sets it — modulus bars around `e` with the subscript outside.
Read as |eₜ|. Transcribed as printed rather than normalised.

### 2.2 Additional metrics — `ML models for TS.pdf`

Slide "Forecast Evaluation" (bulleted text, extracted cleanly):

> "• Compare forecast vs actual values • **RMSE** • **MSE** • **MAE** • **R²** • **Correlation metrics**"

| Course's name | Formula as given | Measures | Source | Task |
|---|---|---|---|---|
| **R²** | **No formula given** — named only | (not defined in the deck) | `ML models for TS.pdf`, "Forecast Evaluation" | F |
| **Correlation metrics** | **No formula given** — named only, plural and unspecified | (not defined) | ibid. | F |

Low confidence on both: they are bullet points with no definition anywhere in the deck. R² is safe to
call "named in class"; it is **not** safe to call "defined in class". "Correlation metrics" is too
vague to cite as a specific metric.

### 2.3 Event-detection metrics — `CPDexamples.pdf` (Oudre), slides 78–86

The only place the course material defines detection metrics. Section heading: "Evaluation of event
detection methods".

| Course's name | Formula, as written | Measures | Source | Task |
|---|---|---|---|---|
| **precision** (point-based) | `precision = TP / (TP + FP)` | "where TP is the number of true positive, FP the number of false positive and FN the number of false negative" | `CPDexamples.pdf` sl. 78, "Point-based vs. range-based" | C, A/CPD |
| **recall** (point-based) | `recall = TP / (TP + FN)` | ibid. "These metrics are comprised between 0 and 1" | ibid. | C, A/CPD |
| **recall** (range-based) | `recall = (1/N_R) Σ_{i=1}^{N_R} recall(R_i, P)` | Range-based recall over real intervals R = {R₁,…,R_{N_R}} | `CPDexamples.pdf` sl. 83, "Formulation" | A/CPD |
| **precision** (range-based) | `precision = (1/N_P) Σ_{i=1}^{N_P} precision(R, P_i)` | Range-based precision over predicted intervals P = {P₁,…,P_{N_P}} | ibid. | A/CPD |
| **existence** | `existence(R_i, P) = 1 if Σ_{j=1}^{N_p} \|R_i ∩ P_j\| ≥ 1, 0 elsewhere` | "Catching the existence of the event (even by predicting only a single point)"; "only used for recall" | `CPDexamples.pdf` sl. 84 | A/CPD |
| **size/position** | `size_position(R_i, P) = Σ_{j=1}^{N_p} w(R_i, R_i ∩ P_j)` | "where w(A, B) is an overlap score (between 0 and 1)"; used for precision and recall | ibid. | A/CPD |
| **cardinality** | `cardinality(R_i, P) = 1 if R_i overlaps with at most one P_j, γ(R_i, P) elsewhere` | "where γ is a penalty function" | ibid. | A/CPD |
| **combined recall** | `recall(R_i, P) = α existence(R_i, P) + (1 − α) cardinality(R_i, P) × size_position(R_i, P)` | Weighted combination | `CPDexamples.pdf` sl. 85 | A/CPD |
| **combined precision** | `precision(R, P_i) = cardinality(R, P_i) × size_position(R, P_i)` | | ibid. | A/CPD |
| **Intersection Over Union (IoU)** | `IoU = \|P_i ∩ R_j\| / \|P_i ∪ R_j\|` | "One simpler solution"; "use a threshold value (e.g. 25%, 50%, 75%…) as a detection criteria" | `CPDexamples.pdf` sl. 86, "How to choose the parameters" | A/CPD |

Attribution note: these come from [Tatbul et al., 2018], cited on the slide, via Oudre's deck.

### 2.4 Clustering-quality metric — `Unsupervised models for TS.md`, slide 32

> "**Silhouette Score** / Measures how similar a point is to its own cluster compared to other clusters."
>
> `𝑠 𝑖 = (𝑏 𝑖 − 𝑎 𝑖) / max(𝑎 𝑖 , 𝑏(𝑖))`
>
> "•𝑎 𝑖 : average distance to points in the same cluster
> •𝑏 𝑖 : average distance to points in the nearest other cluster
> Range: +1 → very good clustering / 0 → overlapping clusters / -1 → wrong assignment"

Transcribed from the `.md` with its spacing artefacts intact; read it as s(i) = (b(i) − a(i)) / max(a(i), b(i)).
Carries an explicit course rule:

> "**Important: for time-series, use the same distance metric used in clustering.** If you cluster
> with DTW, evaluate using DTW distance, not regular Euclidean distance."

Task: **Cl**. Implemented in HW3 (`silhouette_score(Xn, km.labels_)`).

### 2.5 Metrics appearing only in homework

| Metric | Where | Note |
|---|---|---|
| **Adjusted Rand Index** | `HW/Assignment 3/…ipynb`: `from sklearn.metrics import silhouette_score, adjusted_rand_score` | Imported and used for clustering evaluation. **Never appears in any lecture.** Cite as homework-only. |

### 2.6 Model-selection criteria (not error metrics)

| Name | Formula, as written | Source | Task |
|---|---|---|---|
| **Akaike Information Criterion (AIC)** | `𝐴𝐼𝐶 = 2𝑘 − 2ln 𝐿` — "𝑘= number of parameters / 𝐿= likelihood"; "**Lower AIC indicates a better model**, which means it has few parameters, but also high log-likelihood" | `Time-Series Forecasting.md`, "Akaike Information Criterion (AIC)" sl. 27 | MS |
| **Bayesian Information Criterion (BIC)** | Rika's slide gives **no formula in text** — "looks very much like AIC. It additionally takes N, the number of samples in the dataset" (formula is an image, not transcribed here) | `Time-Series Forecasting.md` sl. 29 | MS |
| **BIC** (Oudre's) | `BIC = k log N − 2 log L̂` | `CPDexamples.pdf` sl. 46, "Model selection criterion" | MS |
| **AIC** (Oudre's) | `AIC = 2k − 2 log L̂` | ibid. | MS |
| **BIC penalty for L2 CPD** | `β = 4σ² log N` | `CPDexamples.pdf` sl. 47, "Standard criterion" | MS/CPD |
| **AIC penalty for L2 CPD** | `β = 4σ²` | ibid. | MS/CPD |

Rika's AIC and Oudre's AIC agree (`2k − 2ln L` ≡ `2k − 2 log L̂`). Rika's BIC formula was not
recovered from text; use Oudre's, or re-read the slide image before quoting it.

### 2.7 Which metrics were actually implemented in homework

This is the strongest evidence for B8. `HW/Assignment 2/classical_time_series_forecasting_homework.ipynb`:

> `from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error`
>
> ```
> def evaluate(true, pred):
>     return {
>         "MAE":  mean_absolute_error(true, pred),
>         "RMSE": np.sqrt(mean_squared_error(true, pred)),
>         "MAPE": mean_absolute_percentage_error(true, pred),
>     }
> ```

and the question text mandates them:

> "Evaluate each method using: - MAE - RMSE - MAPE" (Question 2 — Baseline Forecasting)

Model choice is made on RMSE: `best_model = pd.DataFrame(results).T["RMSE"].idxmin()`;
`print("Best model by RMSE:", best_model)`.

**For B8, the safest citation is RMSE, MAE or MAPE**: each is defined by formula in
`Time-Series Forecasting.pdf` slide 47/48 *and* required by name in Assignment 2.

---

## 3. Models and algorithms taught

**HW** column marks appearance in a homework notebook — a stronger signal of emphasis than a slide.

### 3.1 Classical / statistical forecasting

| Model | Source | HW |
|---|---|---|
| Simple Moving Average (smoothing) | `Time-Series Forecasting.md` sl. 9: "Simple Moving Average: ➢ Average of previous observations" | HW2 ("Moving average forecast — repeat the average of the last 12 observations") |
| **MA(q)** — Moving Average model | `Time-Series Forecasting.md` sl. 11, "MA(q) Model" | — |
| **AR(p)** — Autoregressive | `Time-Series Forecasting.md` sl. 15, "Autoregressive (AR) Models" | Used as anomaly model in `CPDexamples.pdf` ("AR model with p = 12") |
| **ARMA(p,q)** | `Time-Series Forecasting.md` sl. 17–18, "ARMA Equation" | — |
| **ARIMA(p,d,q)** | sl. 19: "The most widely used classical forecasting model." | **HW2** — `from statsmodels.tsa.arima.model import ARIMA` |
| **SARIMA** | sl. 23: "𝐴𝑅𝐼𝑀𝐴 𝑝 𝑑 𝑞 𝑃 𝐷 𝑄 𝑚"; "m denotes the number of periods in a season" | **HW2** — `ARIMA(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()` |
| **Naive forecast** | Implicit in lecture; explicit in HW | **HW2** — "Naive forecast — repeat the last training value" |
| **Seasonal Naive Forecast** | `Time-Series Forecasting.md` sl. 22 (title slide, image body) | **HW2** — "Seasonal naive forecast — repeat the last observed seasonal cycle" |
| **Simple Exponential Smoothing (SES)** | sl. 31: "𝑠 = 𝛼𝑥 + 1 − 𝛼 𝑠" ; "Higher 𝛼 − More responsive / Lower 𝛼 − Smoother forecasts" | — (class imported in lecture list) |
| **Theta method** | sl. 32: "The Theta model can be understood as simple exponential smoothing (SES) with drift."; "Performed exceptionally well in forecasting competitions" | — |
| **Holt-Winters** (triple exponential smoothing) | sl. 35: "also called triple exponential smoothing because it applies exponential smoothing three times" | **HW2** — `ExponentialSmoothing(train, trend="add", seasonal="add", …)` |
| **ARCH** | sl. 40: "ARIMA models focus on predicting means. ARCH models focus on predicting variance." | **HW2** — diagnostic only, `het_arch(returns, nlags=12)` |
| **GARCH** | sl. 41: "Improves ARCH by modeling: • Previous errors • Previous variances" | — |
| **VAR(p)** | sl. 42: "generalizing the AR model to multivariate time-series" | **HW2 (bonus)** — `VAR(var_train).fit(maxlags=5, ic="aic")` |

### 3.2 Machine learning

| Model | Source | HW |
|---|---|---|
| **kNN** | `ML models for TS.pdf`, "K-Nearest Neighbors (KNN)": "Need to set: ➢ The number of neighbors (k)… ➢ The integration function… ➢ The distance function" | — |
| **kNN + DTW** | ibid., "Dynamic Time Warping (DTW) & KNN": "DTW used as distance function in KNN"; "Historically strong benchmark method" | — |
| **Linear regression** | ibid., "Linear regression" — five named assumptions | — |
| **Ridge regression** | ibid., "Ridge Regression", "Feature importance of ridge regression" | — |
| **Lasso regression** | ibid., "lasso regression" | — |
| **Decision trees** | ibid., "Decision trees": "a decision tree that has M partitions, P1, P2, …, PM" | — |
| **Random forest** | ibid., "Random forest" | — |
| **XGBoost / Gradient Boosting** | ibid., "Gradient Boosting (XGBoost)": "XGBoost can be used for forecasting, by training the model based on past values to predict future values" | — |
| **Silverkite / Greykite** | ibid., "Silverkite Overview": "Forecasting framework from LinkedIn"; "hybrid forecasting framework" | — |
| **Kats** (Meta) | ibid., "Popular Time-Series Libraries": "Kats: forecasting framework from Meta" — named only | — |
| **ROCKET** | `Pre-precessing.md` sl. 64–67: "random convolutional kernels"; "set to 10,000 by default"; "ROCKET doesn't use any hidden layers or non-linearities" | — |
| **Shapelets** | `Pre-precessing.md` sl. 68–70: "discriminative subsequences of a time-series"; "It is independent of the time axis (Shift Invariant)" | — |

Note: none of the ML models appear in any homework notebook. `ML models for TS.pdf` is the deck
with the **weakest** homework backing despite being substantial.

### 3.3 Deep learning — all from `DL for TS.pdf`, none in homework

| Model | Source file | Source fragment (slide heading and/or quote) |
|---|---|---|
| **Feed-forward / fully connected networks** | `DL for TS.pdf` | "Feed-forward networks"; "In the time series forecasting context, an FFN can be used as an encoder as well as a decoder." |
| **Embedding layer** | `DL for TS.pdf` | "Embedding Layer" |
| **RNN** | `DL for TS.pdf` | "Recurrent neural network"; "Many-to-one… Many-to-many"; full PyTorch parameter list (`input_size`, `hidden_size`, `num_layers`, `nonlinearity`, `bias`, `batch_first`, `dropout`, `bidirectional`) |
| **LSTM** | `DL for TS.pdf` | "Long short-term memory (LSTM) networks" — title slide, body is an image |
| **CNN** | `DL for TS.pdf` | "Convolution networks"; "This grid can be 2D (such as an image), 1D (such as a time series)" |
| **Dilated causal CNN** | `DL for TS.pdf` | "Dilated causal convolutional neural network" (3 slides) |
| **Encoder-decoder** | `DL for TS.pdf` | "the encoder consumes the history and retains the information that is required for the decoder to generate the forecast" |
| **Attention / Transformers** | `DL for TS.pdf` | "Attention"; "Forecasting with Transformers" (4 slides) — titles only, bodies are images |

**Zero deep-learning homework.** Treat DL coverage as conceptual, not practised.

### 3.4 Unsupervised

| Method | Source | HW |
|---|---|---|
| **KMeans clustering** | `Unsupervised models for TS.md` sl. 25, "Clustering Time-Series"; `CPDexamples.pdf` "k-Means, spectral clustering" | **HW3** — `KMeans(n_clusters=3, n_init=10, random_state=42)` |
| **Spectral clustering** | `CPDexamples.pdf`, "Other distance-based approaches" — named only | — |
| **PCA** | `EDA.md` sl. 16: "dimensionality reduction techniques such as Principal Component Analysis (PCA)" | **HW3** — `from sklearn.decomposition import PCA` (visualisation) |
| **Euclidean distance** | `Unsupervised models for TS.md` sl. 27: `𝑑 𝑥 𝑦 = ෍( 𝑥 − 𝑦 )` over t=1..T, squared | **HW3** — `def euclidean(a,b): return np.sqrt(np.sum((a-b)**2))` |
| **Manhattan distance** | ibid. sl. 28: `𝑑 𝑥 𝑦 = ෍ ∣ 𝑥 − 𝑦 ∣` | **HW3** — `def manhattan(a,b): return np.sum(np.abs(a-b))` |
| **Correlation distance** | ibid. sl. 29: `𝑑 𝑥 𝑦 = 1 − corr 𝑥 𝑦` | **HW3** — `def correlation(a,b): return 1 - np.corrcoef(a,b)[0,1]` |
| **DTW** | ibid. sl. 30: "The standard DTW complexity is 𝑶 𝑻 𝟐 for two sequences of length 𝑇." | **HW3** — full DP implementation written by hand |
| **STL decomposition** | `EDA continue.md` sl. 11–13 | **HW1b, HW3** — `STL(series, period=288, robust=True)`, `STL(y_context, period=24, robust=True)` |
| **seasonal_decompose** | `EDA continue.md` sl. 9 | **HW1a** — `seasonal_decompose(monthly, model='additive', period=12)` |
| **MSTL** | `EDA continue.md` sl. 20–22 | — |
| **Fourier decomposition** | `EDA continue.md` sl. 14–19 | — |
| **Matrix profile** | `CPDexamples.pdf`: `m[n] = min_{i>n+L or i<n−L} d(x[n : n+L−1], x[i : i+L−1])` | — |

### 3.5 Change-point detection

| Method | Source | HW |
|---|---|---|
| **Binary segmentation** | `Unsupervised models for TS.md` sl. 22: "best split=the split that makes both parts internally consistent"; `CPDexamples.pdf` search methods | **HW3** — hand-implemented `binary_segmentation(x, n_bkps)` with `l2_cost` |
| **Dynamic programming (optimal resolution)** | `CPDexamples.pdf` sl. 30–31: "Complexity of O(KN²)" | — |
| **Sliding window (approximated resolution)** | `CPDexamples.pdf` sl. 34: "consider a sliding window of length 2w" | — |
| **Bottom-up** | `CPDexamples.pdf` sl. 20: "window-based detection, bottom-up methods, or binary segmentation" — named only | — |
| **PELT** | `CPDexamples.pdf` sl. 43: "Pruned Exact Linear Time (PELT) algorithm: under the assumption that regime lengths are randomly drawn from a uniform distribution, the complexity of PELT is O(N)"; "Optimal algorithm: exact solution" | — |
| **Penalized CPD** | `CPDexamples.pdf` sl. 41–42: `(t̂₁,…,t̂_K̂) = argmin Σ_{k=0}^{K} c(x[t_k : t_{k+1}]) + βK` | — |
| Cost: **cML** | `CPDexamples.pdf` sl. 15, "Maximum likelihood estimation": `cML(x[a : b]) = − sup_θ Σ_{n=a+1}^{b} log f(x[n]\|θ)` | — |
| Cost: **cL2** (change in mean) | `CPDexamples.pdf` sl. 16, "Change in mean": `cL2(x[a : b]) = Σ_{n=a+1}^{b} ‖x[n] − µ_{a:b}‖²₂` — "The most popular is indubitably the L2 norm [Page, 1955]" | **HW3** — `l2_cost(x) = np.sum((x - x.mean())**2)`; explanation says "This is the L2 / change-in-mean cost from the lecture" |
| Cost: **cΣ** (mean and variance) | `CPDexamples.pdf` sl. 20, "Change in mean and variance": `cΣ(x[a : b]) = (b − a) log σ²_{a:b} + (1/σ²_{a:b}) Σ_{n=a+1}^{b} ‖x[n] − µ_{a:b}‖²₂` | — |
| Cost: **clinear** (slope/intercept) | `CPDexamples.pdf` sl. 25, "Change in slope and intercept": `clinear(x[a : b]) = min_α Σ_{n=a+1}^{b} ‖x[n] − Σ_{i=1}^{M} α_i β_i[n]‖²₂` — "For slope and intercept, we choose β₁[n] = 1 and β₂[n] = n" | — |

### 3.6 Anomaly / outlier detection

| Method | Source | HW |
|---|---|---|
| **Standard deviation rule** | `EDA continue.md` sl. 25: "𝜇 ± 3𝜎"; "For seasonal time series, apply this rule to the **residuals** after removing seasonality, not to the raw data." | **HW1b** — "5. Standard Deviation Outlier Detection" |
| **IQR** | `EDA continue.md` sl. 26: "𝐼𝑄𝑅 = 𝑄3 − 𝑄1"; outliers outside "𝑄1 − 1.5 ⋅ 𝐼𝑄𝑅" and "𝑄3 + 1.5 ⋅ 𝐼𝑄𝑅" | **HW1b** — "6. IQR Outlier Detection" |
| **Isolation Forest** | `EDA continue.md` sl. 27: "with contamination controlling the expected proportion of anomalies"; "However, **it ignores time order**" | **HW1b, HW3** — `IsolationForest(contamination=0.10, random_state=42)` |
| **ESD / S-ESD** | `EDA continue.md` sl. 28: "ESD is a statistical outlier detection method based on Grubbs's test"; "S-ESD extends this idea by first removing seasonality" | **HW1b** — `import sesd` |
| **Z-score (anomaly)** | `Unsupervised models for TS.pdf` sl. 12 (image): `outlier(µ̂, σ̂) = {x : \|x_i − x̂\| / σ̂ > ϵ}` | **HW3** — "Use a rolling z-score"; "STL residual z-score" |
| **Hampel filter** | `Unsupervised models for TS.md` sl. 13 — title and surrounding text present; the filter itself is not defined in text | — |
| **Mu/sigma (adaptive)** | `CPDexamples.pdf`: `\|x[n] − µ_n\| > λσ_n`; "λ = 1 → 68%, λ = 2 → 95%, λ = 3 → 99.7%" [Roberts, 2000] | — |
| **Median/MAD (adaptive)** | `CPDexamples.pdf`: `\|x[n] − med_n\| > λ mad_n` [Leys et al., 2013] | — |
| **Model-based (trend+seasonality, AR)** | `CPDexamples.pdf`, "Model-based anomaly detection": "1. Choose an adequate model… 2. Compute the prediction/signal reconstruction 3. Anomalies are samples that diverge from the model" | — |
| **Histogram (global statistical)** | `CPDexamples.pdf`, "Example: Histogram" | — |

Anomaly taxonomy, `Unsupervised models for TS.md` sl. 14 — used verbatim as HW3's question structure:

> "**Point Anomalies** — Single unusual observations. **Contextual Anomalies** — Unusual only in
> specific contexts. **Collective Anomalies** — Groups of observations form unusual patterns."

---

## 4. Notation table

Symbols as the course writes them. **Collision** flags where two sources in this same folder use the
symbol differently, or where the course's usage differs from common literature convention.

| Symbol | Course's meaning | Source | Collision risk |
|---|---|---|---|
| `x_t` | The time-series value at time t (Rika's default series variable) | `Time-Series Forecasting.md` MA(q), AR(p), ARMA, VAR | **HIGH.** In the metrics slides, `x_t` is the model **input**, not the series value: "If 𝑓(𝑥ₜ) is a prediction… and the actual target value is 𝑦ₜ". Two meanings inside one deck. |
| `y_t` | The actual target value | `Time-Series Forecasting.pdf` sl. 47; `EDA continue.md` "y_t = trend_t + season_t + noise_t" | Also used as the second series in distance formulas (`d(x,y)`) in `Unsupervised models for TS.md`. |
| `x[n]` | The signal, indexed by sample n | `CPDexamples.pdf` throughout | **HIGH.** Oudre uses `x[n]`; Rika uses `x_t`. Same object, different notation, same folder. |
| `f(x_t)` | The model's prediction for step t | `Time-Series Forecasting.pdf` sl. 47 | Most papers write `ŷ_t` or `x̂_t`. The course does **not** use a hat for predictions in the metrics slides. |
| `e_t` | Forecast error / prediction error / residual: `e_t = y_t − f(x_t)` | ibid. | Sign convention is actual-minus-predicted. Confirm any paper's convention before reusing. |
| `x̂` | The **estimated mean** of the series | `Unsupervised models for TS.pdf` sl. 12: `\|x_i − x̂\| / σ̂ > ϵ` | **HIGHEST.** In nearly all forecasting literature `x̂` means *predicted x*. Here it means *mean of x*. The slide's own prose calls it "𝜇̂" while the formula prints `x̂`. Do not carry a paper's `x̂` into course notation without restating. |
| `ϵ` (epsilon) | (a) Random noise / white noise term in MA, AR, ARMA, VAR; (b) an **outlier threshold** | (a) `Time-Series Forecasting.md` "𝜖 𝑡 = random noise"; (b) `Unsupervised models for TS.md` sl. 12: "𝜖 is a threshold dependent on the confidence interval… often, 2 or 1.96" | **Two meanings, explicitly.** Flagged rather than resolved. |
| `φ` (phi) | (a) "weights given to the past error terms" in **MA(q)**; (b) "autoregressive coefficients whights" in **AR(p)** and in **ARMA** | (a) `Time-Series Forecasting.pdf` sl. 11 (verified on rendered page 9); (b) sl. 15 and sl. 18 (verified on rendered page 16: `x_t = c + ϵ_t + Σφ_i x_{t−i} + Σθ_i ϵ_{t−i}`) | **Internally inconsistent within one deck.** Standard convention reserves φ for AR and θ for MA; the course's MA(q) slide breaks this. Quoted typo "whights" is the source's. **Any report must state which φ it means.** |
| `θ` (theta) | (a) MA coefficients in the ARMA equation; (b) distribution parameter in `cML` (`f(·\|θ)`, `θ ∈ Θ`); (c) the name of the **Theta method** | (a) `Time-Series Forecasting.pdf` sl. 18; (b) `CPDexamples.pdf` sl. 15; (c) `Time-Series Forecasting.md` sl. 32–34 | **Three meanings.** |
| `μ` | (a) "the average (expectation) of 𝑥ₜ (usually assumed to be 0)" in MA(q); (b) population mean in the `𝜇 ± 3𝜎` outlier rule; (c) `µ_{a:b}` = "the empirical mean of the segment x[a : b]"; (d) `µ_n` = local mean in a sliding window | (a) `Time-Series Forecasting.md` sl. 11; (b) `EDA continue.md` sl. 25; (c)(d) `CPDexamples.pdf` | Consistent in spirit (a mean), varying in scope. |
| `σ`, `σ²` | Standard deviation / variance; `σ²` also the normaliser in NMSE | `EDA continue.md` sl. 25; `Time-Series Forecasting.pdf` sl. 48 (`NMSE = MSE/σ²`) | NMSE's `σ²` is not defined on the slide — series variance is the natural reading but is **unverified**. |
| `p` | AR order — "= number of lags" | `Time-Series Forecasting.md` sl. 15, sl. 19 | Also `p` for p-value in HW2 (`lm_pvalue`). |
| `d` | (a) ARIMA differencing order — "The parameter 𝒅 specifies how many times differencing is applied"; (b) a **distance function**: `d(x, y)` | (a) `Time-Series Forecasting.md` sl. 19–20; (b) `Unsupervised models for TS.md` sl. 27–29 | **Two meanings.** |
| `q` | MA order — "= number of lagged errors" | `Time-Series Forecasting.md` sl. 11, 19 | — |
| `P, D, Q` | Seasonal ARIMA orders — "P, D, Q parametrize the autoregressive, integration, and moving average components of the seasonal part" | `Time-Series Forecasting.md` sl. 23 | Uppercase `P` also = the set of **predicted intervals** in `CPDexamples.pdf` sl. 83. **Collision.** |
| `m` | "m denotes the number of periods in a season" | `Time-Series Forecasting.md` sl. 23 | `m[n]` is the **matrix profile** in `CPDexamples.pdf`. |
| `c` | (a) AR/ARMA/VAR constant — "= constant"; (b) the **cost function** `c(·)` in CPD; (c) Theta's "intercept (starting level)" | (a) `Time-Series Forecasting.md` sl. 15; (b) `CPDexamples.pdf` sl. 13: "Cost function c(.) — Measures the homogeneity of the segments"; (c) sl. 33 | **Three meanings.** |
| `N` | (a) Number of points in the error sums (`Σ_{t=1}^{N}`); (b) "N, the number of samples in the dataset" in BIC; (c) signal length in CPD ("Convention : t₀ = 0, t_{K+1} = N"); (d) "The number of Fourier terms, N" | (a) `Time-Series Forecasting.pdf` sl. 47–48; (b) sl. 29; (c) `CPDexamples.pdf` sl. 14; (d) `EDA continue.md` sl. 14 | **(d) collides hard with (a)–(c).** |
| `T` | (a) Series length — "for two sequences of length 𝑇", and `Σ_{t=1}^{T}` in distance formulas; (b) `T` = trend component in Holt-Winters and Theta ("𝑇ₜ : estimated trend at time 𝑡"); (c) `T*` = the set of true change-point times; (d) `T` = "the set of samples that corresponds to unusual phenomenon" (anomalies) | (a) `Unsupervised models for TS.md` sl. 27, 30; (b) `Time-Series Forecasting.md` sl. 33, 35; (c) `CPDexamples.pdf` sl. 12: "T ∗ = (t₁∗, . . . , t_{K∗}∗)"; (d) `CPDexamples.pdf` "Problem 2" | **Four meanings across the folder. The worst overload in this course.** |
| `K`, `K*`, `K̂` | Number of change points (true / estimated) | `CPDexamples.pdf` sl. 12: "Goal: retrieve the number of change-points K ∗ and their times T∗" | `k` lowercase = "number of parameters" in AIC/BIC, and `k` = number of neighbours in kNN, and `n_clusters` in KMeans. |
| `t_k`, `t̂_k` | Change-point times, true and estimated | `CPDexamples.pdf` sl. 13: `(t̂₁,…,t̂_K) = argmin_{(t₁,…,t_K)} Σ_{k=0}^{K} c(x[t_k : t_{k+1}])` | The hat here **does** mean "estimated" — unlike `x̂` in §4 above. Inconsistent hat semantics across the folder. |
| `a : b` | Index range — "a : b = [a, a + 1, . . . , b − 1]" (half-open, right-exclusive) | `CPDexamples.pdf` sl. 14, "Convention" | Explicitly defined; rare and worth stating. |
| `V(T, x)` | Total segmentation cost — `V(T , x) = Σ_{k=0}^{K} c(x[t_k : t_{k+1}])` | `CPDexamples.pdf` sl. 30 | — |
| `β` (beta) | (a) CPD penalty per change point — "Parameter β penalizes the introduction of a new change-point"; (b) `β_i[n]` = "covariate functions" in `clinear` | `CPDexamples.pdf` sl. 42; sl. 25 | **Two meanings in one deck.** |
| `α` (alpha) | (a) SES smoothing parameter — "𝛼 controls responsiveness"; (b) the existence/cardinality weight in range-based recall; (c) `α_i` regression coefficients in `clinear` | (a) `Time-Series Forecasting.md` sl. 31; (b) `CPDexamples.pdf` sl. 85; (c) sl. 25 | **Three meanings.** |
| `λ` (lambda) | (a) Power-transform parameter — "the optimal choice of the parameter 𝜆"; (b) the mu/sigma and med/MAD threshold multiplier | (a) `Pre-precessing.md` sl. 10; (b) `CPDexamples.pdf` | **Two meanings.** |
| `L` | (a) Likelihood — "𝐿= likelihood" in AIC; (b) pattern/window length in the matrix profile — "given a pattern length L" | (a) `Time-Series Forecasting.md` sl. 27; (b) `CPDexamples.pdf` | **Two meanings.** `L̂` = "the maximum value of the likelihood function". |
| `s_t` | SES smoothed level — `𝑠 = 𝛼𝑥 + 1 − 𝛼 𝑠` (read: sₜ = αxₜ + (1−α)s_{t−1}) | `Time-Series Forecasting.md` sl. 31 | `S` uppercase = "Seasonality S with m seasons" (Holt-Winters, sl. 35). `s(i)` = silhouette score. **Three-way.** |
| `w` | (a) Sliding-window half-length — "a sliding window of length 2w"; (b) `w(A, B)` = "an overlap score (between 0 and 1)"; (c) "w is a random sample from a white noise distribution" | (a) `CPDexamples.pdf` sl. 34; (b) sl. 84; (c) `Intro.md` sl. 22 | **Three meanings.** |
| `r` | "a correlation coefficient r" (red noise parameterisation) | `Intro.md` sl. 22 | Collides with R² in `ML models for TS.pdf`. |
| `P` (Fourier) | "The cycle length is denoted by P. For example, in monthly data with yearly seasonality, P = 12." | `EDA continue.md` sl. 14 | Collides with SARIMA's `P` and with CPD's predicted-interval set `P`. **Three meanings for uppercase P.** |
| `a(i)`, `b(i)` | Silhouette's within-cluster and nearest-other-cluster average distances | `Unsupervised models for TS.md` sl. 32 | `a`, `b` also = segment bounds in `c(x[a : b])`. **Collision.** |
| `T_x, T_y, T_z` / `F_x, F_y, F_z` | Torque and force channels in the Robot Execution Failures dataset | `ML models for TS.pdf`, "Meaning of the columns" | Dataset-specific; noted because `T` is already overloaded four ways. |

**Practical consequence for the report.** Four symbols are dangerous enough to need an explicit
restatement wherever they appear: **`T`** (four meanings), **`φ`** (inconsistent between MA and AR
within one deck), **`x̂`** (means *mean*, not *prediction*), and **`x_t`** (series value in the model
slides, model *input* in the metrics slides). State the intended meaning at first use rather than
relying on context.

---

## 5. Preprocessing and evaluation conventions

### 5.1 Train/test splitting

The strongest statement in the course, `ML models for TS.pdf`, "Validation in Time-Series":

> "• **Standard k-fold validation may be misleading** • Time-series data evolves over time
> • **Future data must remain unseen during training** • **Temporal ordering must be preserved**
> • Prevents overly optimistic evaluation"

and "Walk-Forward Validation":

> "• Train on historical observations • Test on future observations • **Move the training window
> forward** • Repeat across multiple folds • Produces realistic performance estimates • Reduces risk
> of overfitting"

> "In terms of training, validation, and test datasets, this means that we'll adjust model parameters
> entirely on training and validation datasets, and we'll benchmark our test based on a set of data
> that's **more advanced in time**"

and the workflow slide: "• **Split data using walk-forward validation**".

**Divergence to be aware of.** The lecture prescribes walk-forward; the homework does not use it.
`HW/Assignment 2` uses a **single chronological holdout**:

> ```
> train = y.iloc[:-24]
> test  = y.iloc[-24:]
> ```

So both are defensible to a grader: walk-forward is the taught standard, single chronological holdout
is the practised one. Requirement B5 permits either ("chronological train/validation/test split,
rolling-origin evaluation, or walk-forward validation"). No shuffled or k-fold split appears anywhere
in the course.

### 5.2 Leakage

`Pre-precessing.md` sl. 51, "Causality and Leakage":

> "Time-series features must only use past and present information. Using future data creates leakage
> and gives overly optimistic results. **A valid preprocessing pipeline must respect the prediction
> time.**"

`EDA.md` sl. 16:

> "**Feature leakage is when a variable unintentionally gives away the target**"

### 5.3 Scaling and transformation

`Pre-precessing.md`, "Scaling" sl. 6:

> "Scaling adjusts the range or distribution of numeric features. It is especially important for
> models sensitive to feature magnitude, such as linear models and distance-based methods."

- **Min-max**: "maps values into a fixed range, usually between 0 and 1 (if 𝑎 = 0, 𝑏 = 1). It
  preserves the relative order of values but changes their scale." (sl. 7)
- **Z-score**: "transforms a feature to have mean 0 and standard deviation 1." (sl. 8)
- **Log**: "compresses large values and reduces skewed distributions… **Always inspect the data before
  and after applying the transformation.**" (sl. 9)
- **Box-Cox**: "When lambda equals 0, Box-Cox is equivalent to a log transformation. **It works only
  with positive values.**" (sl. 11)
- **Yeo-Johnson**: "extends Box-Cox to support zero and negative values." (sl. 12)
- Why these two: "they address one of the most common and challenging issues in modeling and
  forecasting: **variance instability over time (heteroscedasticity) and deviation from normality**.
  While trend and seasonality are typically handled using differencing, power transformations address
  the underlying dispersion structure of the data itself." (sl. 13)
- Normality testing (sl. 18): "Standard scaling and min-max scaling **change the scale but do not
  change the shape of the distribution**. Log transformation can make lognormal data approximately
  normal. Box-Cox with lambda 0 gives the same result as the log transformation."

**No slide states that the scaler must be fitted on train only.** That is standard practice and is
required by B2, but it is **not** stated in this course's material. Do not cite the course for it.

### 5.4 Missing values

`Pre-precessing.md` sl. 21 and `Time-Series Analysis with Python.md` sl. 17 carry the same text:

> "When working with large datasets, missing values **should not be handled automatically by dropping
> rows or filling with the mean**. First, consider the data-generating process. Sometimes what looks
> like missing data actually carries meaning. For example, if a supermarket product has no
> transactions on a day, that usually means zero sales, not missing data, so those gaps should be
> filled with zero. If the missingness follows a pattern, such as data being absent every Sunday,
> handling it depends on the model you plan to use."

Methods taught: forward fill (LOCF), backward fill (NOCB), mean value fill, linear interpolation,
nearest interpolation, spline/polynomial interpolation ("we should always provide order as well"),
seasonal profile imputation, and **seasonal interpolation** (sl. 45):

> "1. Calculate the seasonal profile… 2. Subtract the seasonal profile and apply any of the
> interpolation techniques we saw earlier. 3. Return the seasonal profile to the interpolated series."

Hard constraint, `EDA continue.md` sl. 9:

> "**Important note!! `seasonal_decompose` does not support missing values, so missing data should be
> handled first.**"

Imputation types, sl. 19: "**Unit imputation** replaces missing values with constants such as 0, mean,
or median. **Model-based imputation** predicts missing values using other variables."

### 5.5 Resampling and regular intervals

`Time-Series Analysis with Python.md` sl. 16 distinguishes four operations:

> "**Resampling** - Changes the time frequency. **Shifting** - Moves values to create lags/leads.
> **Rolling window** - Looks at a fixed recent window. **Expanding window** - Looks at all past data
> so far."

`Pre-precessing.md` sl. 37, "Enforcing regular intervals in time series":

> "One of the first things you should check and correct is whether the regularly sampled time series
> data that you have has equal intervals of time. In practice, even regularly sampled time series have
> some samples missing in between… So while working with the data, we will make sure we **enforce
> regular intervals** in the time series."

Data layout convention (`Pre-precessing.md` sl. 34–36): wide / compact / expanded (long). The course
states a preference: "**We are going to use the compact form** because it is easy to work with and
much less resource-hungry" (`Time-Series Analysis with Python.md` sl. 37).

### 5.6 Stationarity

`Intro.md` sl. 30:

> "We call a time series **stationary when the probability distribution remains the same at every
> point in time**. In other words, if you pick different windows in time, the data distribution across
> all those windows should be the same."
>
> "there are two ways the stationarity assumption can be broken… • Change in mean over time • Change
> in variance over time"

Non-stationarity in variance is named: "In statistics, it's called **heteroscedasticity**." (sl. 33)

`Time-Series Forecasting.md` sl. 8, the course's stated **Forecasting Workflow**:

> "1. Test for stationarity 2. **Differencing [if stationarity detected]** 3. Fit method and forecast
> 4. Add back the trend and seasonality"

The bracket text is the source's and reads backwards — differencing is applied when
**non**-stationarity is detected. Quoted as printed; flagged as an apparent error in the slide, not
silently corrected.

Differencing, sl. 20: "Instead of modeling the values themselves, we model the changes between
consecutive values." Transform: `𝑥 − 𝑥` at `𝑡, 𝑡−1` (i.e. x_t − x_{t−1}). "Purpose: •Remove trends
•Stabilize mean •Achieve stationarity".

Tests taught, `Time-Series Forecasting.md` sl. 45, "Statistical Tests":

> "• Augmented Dickey-Fuller (ADF) • KPSS • PACF • ACF • ARCH-LM test • Ljung-Box test /
> Help validate assumptions before modeling."

Only ADF and the ARCH-LM test appear in homework (`adfuller(...)`, `het_arch(returns, nlags=12)`).
KPSS, Ljung-Box: lecture-only.

Full example workflow, sl. 46:

> "1. Load data 2. Resample time series 3. Check stationarity 4. Analyze ACF/PACF 5. Select model
> order 6. Train model 7. Generate forecasts 8. Evaluate performance"

### 5.7 Outliers

`EDA continue.md` sl. 24: "An outlier is an unusual data point that differs strongly from the rest of
the time series. It may result from errors, faulty measurements, or rare events."

Two standing instructions:

> "For seasonal time series, **apply this rule to the residuals after removing seasonality, not to the
> raw data**." (sl. 25, standard deviation rule)
>
> "For time series, it is usually better to **remove trend/seasonality before applying it**." (sl. 26, IQR)

Treatment policy, sl. 29:

> "**Outlier correction should not be done automatically.** Detected outliers should first be
> verified, because they may represent meaningful patterns rather than errors. For small datasets,
> inspect them manually. For many time series, use automated treatment such as replacing outliers with
> a heuristic value or treating them as missing values and imputing them. **Outlier correction is
> optional, especially with modern ML/DL models, so its impact should be tested experimentally.**"

The homework carries this through — `HW/Assignment 1/02_Outlier_Detection…ipynb` Q8:

> "No, not all detected anomalies should be automatically replaced in a real industrial monitoring
> system."

### 5.8 Model selection

`Time-Series Forecasting.md` sl. 26: "**Goal: Find the simplest model that explains the data well.**
Benefits: • Better interpretability • Reduced overfitting • Improved generalization"

Order selection for ARMA, sl. 17: "Look at the **autocorrelation and partial autocorrelation plots**,
where we could see peaks in the correlation for each lag to set p and q."

---

## 6. Baseline candidates for B6

B6 asks for "at least one simple baseline, such as naive forecast, seasonal naive forecast, moving
average, **or a standard classical model when appropriate**". Ranked by strength of evidence.

### Tier 1 — named in a lecture *and* required by name in homework

| Baseline | Lecture source | Homework source |
|---|---|---|
| **Naive forecast** | `Time-Series Forecasting.md` sl. 22 "Seasonal Naive Forecast" section context | HW2 Q2: "**Naive forecast** — repeat the last training value." Implemented: `naive = pd.Series(train.iloc[-1], index=test.index)` |
| **Seasonal naive forecast** | `Time-Series Forecasting.md` sl. 22, dedicated slide "Seasonal Naive Forecast" | HW2 Q2: "**Seasonal naive forecast** — repeat the last observed seasonal cycle." Implemented via `np.resize(last_cycle, len(test))` |
| **Moving average forecast** | `Time-Series Forecasting.md` sl. 9: "Simple Moving Average: ➢ Average of previous observations" | HW2 Q2: "**Moving average forecast** — repeat the average of the last 12 observations." Implemented: `train.iloc[-12:].mean()` |
| **SARIMA** | `Time-Series Forecasting.md` sl. 23–25; sl. 19 calls ARIMA "**The most widely used classical forecasting model**" | HW2 Q3: `ARIMA(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()` |
| **Holt-Winters** (triple exponential smoothing) | `Time-Series Forecasting.md` sl. 35–38 | HW2 Q3: `ExponentialSmoothing(train, trend="add", seasonal="add", …)` |

All three of Naive / Seasonal Naive / Moving Average are named verbatim in B6 *and* implemented in
HW2, so any of them satisfies B6 with a citation the grader can check. **SARIMA and Holt-Winters are
the two strongest candidates for B6's "standard classical model when appropriate"** — both are
lectured with formulas and both are fitted by students in Assignment 2.

### Tier 2 — lectured, not in homework

ARIMA (non-seasonal), ARMA, AR(p), MA(q), SES, Theta method, GARCH. `Time-Series Forecasting.md`
sl. 32 on Theta: "Performed exceptionally well in forecasting competitions. **Key lesson: Simple
methods often outperform complex ones.**" Defensible as a baseline; less directly citable to a
homework requirement.

### Tier 3 — multivariate only

**VAR** — HW2 bonus question, `VAR(var_train).fit(maxlags=5, ic="aic")`. Only appropriate if the task
is multivariate. Note the syllabus promised VECM and Granger causality; neither was delivered (§7).

### Non-forecasting baselines

If the task is anomaly or change-point detection rather than forecasting, the recognisable
"classical" baselines are: **standard-deviation rule on STL residuals**, **IQR**, **Isolation
Forest**, **S-ESD** (all four implemented in HW1b), and **binary segmentation with L2 cost**
(implemented in HW3). For clustering: **KMeans + silhouette** (HW3).

---

## 7. NOT covered

Split by how confident the "not covered" claim is.

### 7.1 Promised in the syllabus, no deck exists

From `Intro.md` sl. 40 "Course Content", with no corresponding lecture PDF:

- **Probabilistic forecasting** — "Probabilistic forecasting (quantiles, **Prophet/BSTS**)" (item 8).
  No quantile forecasting, no prediction intervals, no Prophet, no BSTS anywhere in any deck.
  *(Partial exception: `ML models for TS.pdf` mentions Silverkite "Model residuals to produce
  prediction intervals and anomaly detection" — one clause, no method.)*
- **VECM and Granger causality** — "Multivariate time series and causality (**VAR/VECM, Granger**)"
  (item 11). VAR is covered; VECM and Granger causality appear **nowhere**. `EDA.md` mentions
  "Causal analysis" only as a bullet in a list of problems (`Intro.md` sl. 16).
- **Online / streaming learning and drift detection** (item 12). Nothing.
- **Hierarchical / grouped forecasting, intermittent demand, packaging/APIs, deployment** (item 13).
  Nothing.
- **N-BEATS** — named in item 10 ("Deep learning II: CNN/TCN, **N-BEATS**, Transformers").
  Not in `DL for TS.pdf`.
- **GRU** — named in item 9 ("Deep learning I: RNN/**GRU**/LSTM"). `DL for TS.pdf` covers RNN and
  LSTM only.
- **TCN** as such — item 10 names "CNN/**TCN**". `DL for TS.pdf` covers "Dilated causal convolutional
  neural network", which is the TCN building block, but never uses the term TCN.

### 7.2 Standard time-series topics absent from all 10 decks and 4 notebooks

Verified by reading every deck; each of these has no slide, no heading, and no homework cell:

- **Prediction intervals / uncertainty quantification / conformal prediction.** No coverage of how to
  attach an interval to a forecast.
- **Quantile loss / pinball loss / CRPS.** All course metrics are point-forecast metrics.
- **Cross-learning / global models across many series.** Every model taught is fitted per-series.
- **Foundation models for time series** (TimeGPT, Chronos, Lag-Llama, Moirai and similar).
- **Self-supervised / contrastive representation learning for time series** (TS2Vec, TNC, and
  similar) — despite the final-project brief listing "time-series representation learning" as an
  eligible topic (requirement A1).
- **State-space models and Kalman filtering.** The syllabus mentions "ETS/state-space" (item 3) but
  the delivered exponential-smoothing content (SES, Theta, Holt-Winters) never introduces state-space
  form or the Kalman filter.
- **Bayesian time-series methods** beyond the BIC formula.
- **Spectral analysis / DFT / periodogram / wavelets.** `Unsupervised models for TS.md` sl. 7 mentions
  "wavelet analysis, filtering, and Fourier decomposition" as a one-line list of techniques that "can
  be used", and `CPDexamples.pdf` refers to "DFT" as covered in *Oudre's* Lecture 2 — which is not in
  this folder. Fourier **series** for seasonality is covered (`EDA continue.md`); Fourier **analysis**
  of a signal is not.
- **Time-series classification as a taught pipeline.** ROCKET and shapelets are lectured as feature
  representations, and `ML models for TS.pdf` shows a robot-failure classification pipeline, but no
  classification metric is ever defined for it beyond the precision/recall in Oudre's deck.
- **Hyperparameter tuning methodology.** No grid search, no random search, no Optuna, no nested CV.
  `pmdarima` (auto-ARIMA) is named in a library list only; auto-ARIMA is never demonstrated.
- **Multi-step forecasting strategies** (recursive vs direct vs DirRec) as a named topic.
- **Missing-data mechanisms** (MCAR / MAR / MNAR) as terminology.
- **Statistical significance testing between models** (Diebold-Mariano and similar). Model comparison
  in HW2 is a bare RMSE ranking.
- **Seasonal decomposition of multiple series / hierarchical reconciliation.**
- **Transfer learning / fine-tuning for time series.**
- **Data augmentation for time series.**

### 7.3 Named but never defined

Mentioned in a bullet, with no formula, no worked example and no homework. If the chosen paper leans
on any of these, the report must define it from scratch:

| Topic | Where named | What is missing |
|---|---|---|
| **R²** | `ML models for TS.pdf`, "Forecast Evaluation" | No formula, no definition |
| **"Correlation metrics"** | ibid. | Plural, unnamed, no formula |
| **Hampel filter** | `Unsupervised models for TS.md` sl. 13 | Title slide only; the filter is never specified |
| **Kats** | `ML models for TS.pdf` | One bullet: "forecasting framework from Meta" |
| **TSFresh / SKTime / featuretools / Catch22 / HCTSA** | `Pre-precessing.md` sl. 48; `ML models for TS.pdf` | Named in lists; no usage shown |
| **pmdarima / arch / sktime / darts** | `Time-Series Forecasting.md` sl. 43 | Named in "Popular Forecasting Libraries"; never used |
| **Spectral clustering** | `CPDexamples.pdf` | Named in one clause |
| **Bottom-up CPD search** | `CPDexamples.pdf` sl. 20 | Named alongside window-based and binary segmentation; not developed |
| **LSTM internals** | `DL for TS.pdf` | Title slide; gates and equations are images, never in text |
| **Attention / Transformer mechanics** | `DL for TS.pdf` | 5 slides, all image-only; no equations recoverable from text |

### 7.4 Assumed known from a course we do not have

`CPDexamples.pdf` treats these as already covered in **Oudre's** Lectures 1–4, which are not in this
folder: pattern recognition/extraction, **dictionary learning**, **normalized Euclidean distance**,
**DFT**, wide-sense stationarity, and the **matrix profile** background. The deck says "Reminder :
Matrix profile [Yeh et al., 2016]" — a reminder of something this class was never shown.

---

## 8. Hebrew terms

Per the brief, Hebrew is preserved verbatim with translation. **Scan result: no Hebrew appears in any
lecture deck (`.md` or PDF-extracted) or in any homework notebook.**
(`grep -rlP '[\x{0590}-\x{05FF}]'` over `Lectures/*.md`, `HW/*/*.ipynb` and all PDF text extracts
returned nothing.) The course's own material is entirely in English.

Hebrew appears in exactly two places, both incidental:

| Hebrew, verbatim | Translation | Where |
|---|---|---|
| `ניתוח סדרות עיתיות` | "Time-series analysis" — the course folder name | Directory name. Note: the task brief spelled it `סדרות עתיות`; the folder on disk is `סדרות עיתיות` (with yod). Both spellings occur in Hebrew; the folder's is the one that resolves. |
| `הרצאות` | "Lectures" | `Lectures/gemini-code-1782123087695.py`, in `folder_path = r"C:\...\ניתוח סדרות עיתיות\הרצאות"`. The folder is now named `Lectures`, so the generator script's path no longer resolves. |
| `# יצירת מופע של הכלי` | "# Creating an instance of the tool" | ibid., code comment |
| `print("מתחיל בהמרת הקבצים...")` | "Starting the file conversion…" | ibid. |
| `# מוודאים שזה קובץ ולא תת-תיקייה, ומדלגים על קבצי Markdown קיימים כדי לא להמיר אותם שוב` | "# Verifying this is a file and not a subfolder, and skipping existing Markdown files so as not to convert them again" | ibid. — this comment is why `DL for TS` and `ML models for TS` have no `.md` |
| `# שמירת התוכן לקובץ החדש (utf-8 קריטי כדי שהעברית לא תיהרס)` | "# Saving the content to the new file (utf-8 is critical so the Hebrew isn't destroyed)" | ibid. |
| `print("סיימנו!")` | "We're done!" | ibid. |

---

## 9. Open items and low-confidence entries

Recorded rather than resolved:

1. **NMSE's `σ²` is undefined on the slide.** "series variance" is the natural reading and the
   conventional one, but the course does not say. **Unverified.**
2. **Rika's BIC formula was not recovered.** It is an image on `Time-Series Forecasting.pdf` sl. 29
   and was not rendered. Oudre's `BIC = k log N − 2 log L̂` is available and is the standard form, but
   whether Rika's slide matches it is **unverified**.
3. **`Intro.md` says 4 homework assignments; 3 assignment folders exist** (4 notebooks). Unreconciled.
4. **R² and "correlation metrics"** are safe to cite as *named in class*, not as *defined in class*.
5. **The `.md` files carry no slide boundaries.** Slide numbers in this document come from the trailing
   page numbers embedded in the text, which are the deck's own numbering and skip values (e.g.
   Time-Series Forecasting jumps 3 → 6). Slide number ≠ PDF page number. Where a formula was read from
   a rendered page, both are given.
6. **Image-only content is under-represented throughout.** ~556 images across the 10 PDFs; only the
   metrics tables (Time-Series Forecasting sl. 47–48), the z-score outlier definition (Unsupervised
   sl. 12), MA(q) (sl. 11), and the ARMA equation (sl. 18) were rendered and read. Formula content in
   the other ~550 images is **unread**. In particular, `DL for TS.pdf` (84 images / 31 pages) and
   `EDA.pdf` (76 images / 42 pages) are largely unexamined at the image level.
7. **Whether `Pre-precessing.md`'s garbled markdown tables** (e.g. "| Many classical models | work |
   better | when features | | are |") lost any content, or only mangled the layout of running text, is
   **unverified**. The §0.2 byte-count match with the PDF extract says nothing was dropped, only
   rearranged — but that test cannot see text baked into images.
