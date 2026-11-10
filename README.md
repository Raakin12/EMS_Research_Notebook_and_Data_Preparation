# EMS Response-Time Research

Can information available when an emergency call arrives help predict response time—and identify predictions that may be unreliable?

This project uses NYC EMS incident records to compare forecasting models, measure uncertainty and distribution shift, and investigate persistent patterns in prediction errors. It is an **exploratory research study**, not a validated dispatch system or a causal study.

The main artifact is [the executed submission notebook](EMS_Residual_Regimes_HMM_Kalman_Submission.ipynb). Its saved outputs can be read without rerunning the analysis.

## Data and workflow

The source is NYC Open Data's **EMS Incident Dispatch Data**, dataset `76xm-jjuj`. The source URL, acquisition windows, row counts, and file hashes are recorded in [the acquisition manifest](data/acquisition_manifest.json).

- **Historical data:** 913,796 incidents from May 1 through November 17, 2024. After filtering invalid response outcomes and invalid nonmissing ZIP codes, 860,293 incidents remain for modeling.
- **Prediction week:** 30,197 incidents from November 18–24, 2024. These inputs contain incident-start fields without response outcomes.
- **Target:** `incident_response_seconds_qy`, measured in seconds. The notebook checks it against the difference between incident and first-on-scene timestamps.
- **Evaluation:** expanding training windows with July, August, and September validation folds; a later holdout contains 130,345 incidents from October 18 through November 17. Training always precedes validation.

Incoming-call pressure is calculated from **all historical arrivals**, including incidents excluded from outcome modeling. Each count covers `[t − 90 minutes, t)`, excluding simultaneous and future arrivals. The first 90 minutes of the extract have incomplete lookback history.

## Mathematical and statistical approach

| Method | How it is used | What it contributes |
|---|---|---|
| Grouped-mean baseline (M0) | Mean response by initial call type and severity; unseen combinations fall back to the training mean. | Establishes a simple reference that more complex models must improve upon. |
| Linear regression and CatBoost | M1 adds calendar features; M2 adds geography; M3 adds recent call pressure. A separate M_state model adds pressure directly to M1. Categorical fields are encoded for linear models and supplied as categorical features to CatBoost. | Tests whether time, location, congestion, and nonlinear relationships improve future-period predictions. |
| Cyclical features | Month is represented by `sin(2π(month−1)/12)` and `cos(2π(month−1)/12)`, alongside calendar categories. | Represents the annual cycle without treating December and January as far apart. |
| Chronological backtesting | Each model is fitted on earlier incidents and evaluated on a later month. Out-of-fold (OOF) predictions are pooled. | Evaluates temporal generalization without a random train/test split. |
| Error and subgroup analysis | Mean absolute error (MAE), signed error, and the 90th percentile of absolute error; breakdowns by call type, severity, and time. | Separates typical error, systematic bias, and large errors; identifies weak subgroups. |
| Daily bootstrap | 10,000 resamples of whole days, retaining paired incident-level errors within each sampled day. | Produces uncertainty intervals for model improvements while retaining within-day dependence. It does not preserve dependence between successive days. |
| Distribution-shift diagnostics | Categorical total variation distance, `½ Σₖ |pₖ−qₖ|`, and Wasserstein distance for response times standardized using development statistics. | Measures changes in population composition and target distribution that can undermine generalization. |
| Conformalized quantile regression (CQR) | CatBoost estimates the 5th and 95th response-time quantiles; a later calibration period adjusts the interval. | Provides empirical uncertainty intervals whose actual coverage can be tested on a subsequent holdout. |
| Descriptive OLS and residual diagnostics | Severity, weekday, and month effects with HC3 heteroskedasticity-robust uncertainty; daily residual autocorrelation and Ljung–Box tests at lags 1–7. | Describes adjusted associations and checks for remaining temporal structure. The rank-deficient joint call-type/severity specification is explicitly rejected. |
| Gaussian hidden Markov model (HMM) | Two latent states describe daily mean development residuals; state probabilities forecast a next-day residual adjustment. | Explores whether switching residual regimes can explain persistent forecast bias. |
| Kalman local-level model | A latent residual level follows a random walk and is updated after each day's observations. | Explores whether gradual changes in forecast bias can be tracked sequentially. |
| Sample-split residualization | A response model and a logistic held-status model are trained on the earlier period; their residuals are related in the later period. | Estimates the held-response association remaining after adjustment for observed features, without claiming causality. |

The central error measure is:

$$\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i|.$$

For CQR, ordered lower and upper quantile predictions give calibration scores

$$s_i=\max\{\hat q_{0.05}(x_i)-y_i,\;y_i-\hat q_{0.95}(x_i)\}.$$

For nominal 90% coverage, the correction is the calibration score at rank `ceil((n_cal + 1) × 0.90)`. The interval is `[max(0, q_low − correction), q_high + correction]`. Quantile crossing is corrected before calibration. Temporal shift can weaken coverage, so the notebook measures it rather than assuming a guarantee.

For held-status residualization, let `D` indicate a held incident. With earlier-period estimates `m(X) ≈ E[Y|X]` and `p(X) ≈ P(D=1|X)`, the later-period estimate is

$$\hat\theta=\frac{\sum_i(D_i-\hat p(X_i))(Y_i-\hat m(X_i))}{\sum_i(D_i-\hat p(X_i))^2}.$$

This is an adjusted observational association. A separate held-only residual analysis uses a non-held reference model and a daily bootstrap; it estimates a different quantity.

## Main findings

| Result from the saved run | Value |
|---|---:|
| M1 pooled OOF MAE | 587.58 seconds |
| M1 plus call-pressure OOF MAE | 590.45 seconds |
| M0 holdout MAE | 553.84 seconds |
| M1 holdout MAE | 541.42 seconds |
| M1 + HMM exploratory holdout MAE | 537.80 seconds |
| M1 + Kalman exploratory holdout MAE | 541.63 seconds |
| CQR holdout coverage, nominally 90% | 84.45% |
| Adjusted held-status association | 1,480.29 seconds |

M1 has the lowest pooled development error. Adding call pressure does not improve it in this experiment. HMM correction offers a small exploratory gain; Kalman correction does not. Interval undercoverage remains a practical limitation.

The notebook also produces point predictions and intervals for the prediction week. **4,087 incidents (13.53%)** exceed the 85th-percentile interval-width threshold and are flagged for review. Ties explain why the flagged share is below 15%. A risk–coverage curve examines how observed MAE changes as wider intervals are deferred for review.

HMM/Kalman models use in-sample development residuals and were explored after the holdout had been inspected. They assume previous-day outcomes are available before the next day's forecasts. These results require a new prospective test. Future-week accuracy and interval coverage cannot be assessed without outcomes, and this retrospective source does not establish when every field revision became available.

## Reproduce the notebook

Use **Python 3.13**. From this project's directory, create a fresh environment and install dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell instead. If reusing an existing environment that lacks pip, run `python -m ensurepip --upgrade` after activation.

1. **Use the existing data snapshot** to reproduce the saved results. Both CSV files should be present in `data/`. If they are missing, run:

   ```bash
   python prepare_data.py
   ```

   The downloader uses weekly caches and retries interrupted requests. Existing CSVs are retained by default. `python prepare_data.py --refresh` deliberately downloads fresh data and replaces them. The live source can change, so refreshed counts and results may differ; the manifest records file hashes for comparison. No API token is required; `SOCRATA_APP_TOKEN` is optional.

2. **Start Jupyter from the project directory:**

   ```bash
   python -m jupyter lab EMS_Residual_Regimes_HMM_Kalman_Submission.ipynb
   ```

3. **Select the environment's Python kernel, restart it, and run all cells in order.** In VS Code, select the `.venv` interpreter before restarting and running all cells. The verified run took approximately **22 minutes** locally; runtime and memory use depend on hardware, with CatBoost backtests taking most of the time.

4. **Inspect the final validation output.** It checks temporal separation, raw-history pressure counts, timestamp boundaries, model convergence, OLS identifiability, conformal ranks, prediction alignment, interval bounds, and the risk curve. The submitted notebook contains 109 sequentially executed code cells, passing assertions, and no error outputs. Its final timestamp check emits one non-failing NumPy timedelta deprecation warning.

`requirements.txt` specifies version ranges rather than a locked environment. The notebook's `verified_environment` metadata records the exact versions used for the saved run, including Python 3.13.2, CatBoost 1.2.10, and hmmlearn 0.3.3. Preserve the same data snapshot and use those recorded versions when investigating numerical differences.

## Project files

| File | Purpose |
|---|---|
| `EMS_Residual_Regimes_HMM_Kalman_Submission.ipynb` | Analysis, figures, predictions, and validation outputs. |
| `prepare_data.py` | Downloads and assembles the historical and prediction cohorts. |
| `requirements.txt` | Python dependency requirements. |
| `data/incident_history.csv` | Historical incident records with outcomes. |
| `data/incidents_to_predict.csv` | Prediction-week incident-start fields. |
| `data/acquisition_manifest.json` | Source information, date windows, counts, and SHA-256 hashes. |
| `data/_cache/` | Weekly downloads used to resume data acquisition. |

## Downloading the saved data

The two input CSVs are stored using Git LFS. Install Git LFS, then run `git lfs install` and `git lfs pull` from the cloned repository before running the notebook.
