# Requirements table — Final Project, Time-Series Analysis (Havana Rika)

Source: **`Final_Project.pdf`** (2 pages) — *corrected 30 Jul; this file previously named a
`Final_Project_Requirements.pdf` that does not exist in the folder, so quotes could not be re-checked
against their stated source.* All quotes are verbatim from the brief.
Built 2026-07-30. **Nothing here is paraphrased.** Where I add interpretation it is marked *[interp]*.

---

## A. Paper-selection requirements

| # | Requirement, quoted | How it will be met | Task owner | Hard? |
|---|---|---|---|---|
| A1 | "each group will select a research paper related to time-series analysis, forecasting, classification, anomaly detection, change-point detection, time-series representation learning, etc." | Selection task scores candidates on the §2 rubric; topic must sit inside one of these families | T1 Paper selection | **YES** — scope gate |
| A2 | "A clear model, algorithm, or architecture that can be implemented." | Selection criterion; disqualify papers whose contribution is a survey/theory result | T1 | **YES** |
| A3 | "A public dataset, or a dataset that can be accessed and documented clearly." | Selection criterion; prefer repo-bundled or generator-produced data | T1 | **YES** |
| A4 | "Experimental results that can be compared to your reconstruction." | Selection criterion: paper must report a number on a named benchmark with a named metric | T1 | **YES** |
| A5 | "Preferably, an official GitHub repository or enough implementation details to reproduce the method." | Selection criterion. Note the brief says **"Preferably"** — soft on paper, treated as near-hard by us per bootstrap §2 | T1 | Soft in brief / hard for us |

## B. Stage 1 — reconstruction

| # | Requirement, quoted | How it will be met | Task owner | Hard? |
|---|---|---|---|---|
| B1 | "Implement the method described in the paper and try to reproduce its main experimental results." | Reconstruction notebook/module | T3 Reconstruction | **YES** |
| B2 | "The reconstruction must respect the temporal structure of the data: future observations must not be used during training, preprocessing, feature construction, or hyperparameter tuning." | Leakage audit as a **separate task with its own checks**, covering all four named stages: training, preprocessing, feature construction, HP tuning. Scaler fitted on train only; CV folds chronological. | T3 + T6 Leakage audit | **YES — PASS/FAIL** |
| B3 | "A short explanation of the original method, including the model architecture, algorithmic steps, loss function, training objective, and key hyperparameters." | Report §1. Five named sub-elements — each gets its own paragraph/table so none is silently dropped | T8 Report | **YES** (5 sub-items) |
| B4 | "A reproducible data pipeline: loading, cleaning, datetime parsing, resampling, missing-value handling, scaling/transformation, and feature construction where relevant." | Single `data.py` / pipeline section covering **all 7 named steps**, each explicitly labelled even if the answer is "not needed, because…" | T2 Data pipeline | **YES** (7 sub-items) |
| B5 | "A valid temporal evaluation protocol, such as chronological train/validation/test split, rolling-origin evaluation, or walk-forward validation." | Protocol fixed once, frozen, and reused verbatim by Stage 2 | T3 | **YES** |
| B6 | "At least one simple baseline, such as naive forecast, seasonal naive forecast, moving average, or a standard classical model when appropriate." | ≥1 baseline (plan for 2: naive/seasonal-naive + one classical) | T4 Baselines | **YES** |
| B7 | "Evaluation using the paper's metric, with an explanation of what the metric measures." | Paper metric implemented + a prose paragraph on what it measures and its failure modes | T5 Metrics | **YES** |
| B8 | "Evaluation using at least one metric studied in class or appropriate for the task." | ≥1 additional metric taken from the course lecture notes (see bootstrap §3.11 — use the course's vocabulary) | T5 | **YES** |
| B9 | "you must provide a serious comparison and explain possible differences, such as limited compute, different preprocessing, shorter training, different random seeds, unavailable hyperparameters, or dataset-version differences." | Report §3 discussion; every named cause explicitly ruled in or out with evidence, not hand-waved | T8 | **YES** |
| B10 | "You do not have to match the exact numbers reported in the paper" | *[interp]* Explicit permission for a gap. Frame per bootstrap §3.9: "failure to match, bounded to what we ran." | T8 | — (permission) |

## C. Stage 2 — improvement

| # | Requirement, quoted | How it will be met | Task owner | Hard? |
|---|---|---|---|---|
| C1 | "Modify the original method in a meaningful way and justify why the change should improve performance, robustness, interpretability, efficiency, or applicability to the time-series task." | Improvement pre-registered (bootstrap §3.1) with a quantitative prediction before it is run | T7 Improvement | **YES** |
| C2 | "The improved method must be evaluated using the same dataset split and the same metrics as the reconstruction." | Split object + metric functions imported from the same frozen module; verified by asserting split hashes match | T7 + T6 | **YES — PASS/FAIL** |

## D. Submission files (5 required)

| # | Requirement, quoted | How it will be met | Task owner | Hard? |
|---|---|---|---|---|
| D1 | "Code for the reconstructed method, including data preprocessing, training, evaluation, and result reproduction." | Deliverable 1 | T3 | **YES** |
| D2 | "Code for the improved method, including the changed architecture/algorithm and the same evaluation protocol." | Deliverable 2 | T7 | **YES** |
| D3 | "The dataset, or a clear download link and instructions if the dataset is too large to submit." | Deliverable 3. **Licensing check first** (bootstrap §3.10): ship data only if the licence permits redistribution; otherwise link + script. | T2 | **YES** |
| D4 | "A PDF report that explains the paper, the reconstruction, the improvement, and the results." | Deliverable 4 — **PDF, not docx** | T8 | **YES** (format) |
| D5 | "A short README file explaining how to run the code, including environment requirements, package versions, and expected outputs." | Deliverable 5. Three named elements: environment requirements, **pinned package versions**, expected outputs. "Short" is stated. | T9 README | **YES** (3 sub-items) |

## E. Implementation requirements

| # | Requirement, quoted | How it will be met | Task owner | Hard? |
|---|---|---|---|---|
| E1 | "All random seeds should be fixed where possible." | Single `set_seed()` covering python/numpy/torch(+cuda)/dataloader workers; documented. Where determinism is impossible, say so explicitly ("where possible"). | T3 | **YES** |
| E2 | "Report the forecast horizon, sampling frequency, input window length, and output window length where relevant, and all additional hyperparameters." | A dedicated hyperparameter table in the report with **all four named quantities as named rows** + full HP list | T8 | **YES** (4 named + all) |
| E3 | "Clearly state whether the task is univariate, multivariate, supervised, or unsupervised, etc." | One explicit sentence early in report §1 | T8 | **YES** |

## F. PDF report structure — mandated sections, in this order

| # | Requirement, quoted | Notes | Hard? |
|---|---|---|---|
| F1 | "Original architecture: describe the model, loss, training objective, and important hyperparameters." | 4 named elements | **YES** |
| F2 | "Paper results: summarize the reported results and metrics." | Must cite the paper's own numbers, traceable to table/page | **YES** |
| F3 | "Reconstruction results: show your results and compare them to the paper." | | **YES** |
| F4 | "Improved architecture: explain what you changed and why." | | **YES** |
| F5 | "Improved results: compare (using a table) paper results, your reconstruction, and your improved model." | **A table is explicitly required**, and it must be **3-way**: paper / reconstruction / improved | **YES — explicit format** |
| F6 | "Discussion: explain what worked, what did not, and what you learned." | "what did not" is required — negative results are graded content, not an embarrassment | **YES** |
| F7 | "References." | | **YES** |

---

## G. Things the brief does NOT say — flagged, not assumed

| Item | Status |
|---|---|
| **Page limit for the PDF report** | **Not stated anywhere in the brief.** Do not assume one exists; also do not assume one does not. → open question Q1 |
| **Deadline** | Not stated in the brief. → Q2 |
| **Grading scheme / point values** | Not present in the brief. Bootstrap §1.3 wants effort ranked by rubric — we have none. → Q3 |
| **List of deductions / common mistakes** | Not present in the brief. → Q3 |
| **Group size** ("each group") | Size not stated; solo permitted? → Q4 |
| **Submission channel and file naming** | Not stated. → Q5 |
| **Report language** (Hebrew / English) | Not stated. → Q6 |
| **Whether the paper choice needs instructor approval** | Not stated. → Q7 |
| **Whether a single notebook may serve as both D1 and D2** | Brief lists them as two items; safest reading is two separate artefacts. → Q8 *[interp]* |

**Count check:** 5 required submission files (D1–D5); 7 mandated report sections (F1–F7); 7 named
pipeline steps (B4); 4 named quantities that must be reported (E2); 4 stages where leakage is
forbidden (B2).
