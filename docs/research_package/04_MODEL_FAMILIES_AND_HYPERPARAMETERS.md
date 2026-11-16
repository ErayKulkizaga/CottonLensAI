# 04 · Model families and hyperparameters

Fit counts are from `research/trials.json` (Phase 1 ledger). Phase 2 models are listed after. "Synthetic" fits use generated targets with a known injected signal, or a null, to test the pipeline. They are not market evidence.

## Phase 1

### XGBoost: 14,723 fits
**Where they went**
| Profile | Fits | What it tested |
|---|---|---|
| `cotton-research-v1` (research-v2 round A) | 7,320 market + 6 synthetic | Large nested search on 24 price/cross-asset features, T+1 and T+5, Colab T4 GPU |
| `fx-exploration-v1` | 1,020 | FX features at lags L1/L2/L6 |
| `cftc-exploration-v1` | 1,008 | CFTC positioning |
| `weather-exploration-v1` | 996 | Cotton-belt weather |
| `fas-exploration-v1` | 996 | FAS export sales |
| `nass-exploration-v1` | 984 | NASS crop condition |
| `crop-exploration-v1` | 972 | Other crop prices |
| `wasde-exploration-v1` | 528 | WASDE balance sheet |
| `full-year-v1` | 360 | Full-year baseline across 2016–2023 |
| `oncall-exploration-v2` | 141 | Mill on-call positions |
| `named-weak-signal-control-t5-v1` | 132 synthetic | Can the fixed named recipe learn a weak known effect? |
| `ams-exploration-v1` | 84 | AMS quotes |
| `paired-price-loss-control-v1` | 48 + 6 synthetic | MSE vs MAE loss, everything else fixed |
| `named-history-source-control-t5-v1` | 47 + 1 | Same-contract history features |
| `named-cftc-source-control-t5-v1` | 44 + 4 | 37 CFTC fields, numeric vs mask |
| `named-capacity-control-t5-v1` | 25 + 1 | XGBoost vs Ridge on the same named labels |

**Hyperparameters seen across 387 XGBoost recipes**

| Parameter | Min | Median | Max | Distinct |
|---|---|---|---|---|
| max_depth | 1 | 3 | 8 | 8 |
| eta (learning rate) | 0.0051 | 0.03 | 0.15 | 179 |
| min_child_weight | 1 | 15.9 | 99.96 | 179 |
| alpha (L1) | 0 | ~1e-6 | 9.29 | 119 |
| lambda (L2) | 0.0011 | 1 | 94.96 | 180 |
| subsample | 0.604 | 0.814 | 1.0 | 178 |
| colsample_bytree | 0.603 | 0.815 | 1.0 | 178 |

- **Search.** Round A sampled 128 candidates per horizon on log scales and selected by inner-block price MAE. The top 5 per period were confirmed with 3 seeds.
- **Exploration recipe.** A single fixed "depth 2" recipe was used in every exploration profile, so that only the information changes between arms.
- **Objectives:** `reg:squarederror` (318 recipes) and `reg:absoluteerror` (164). **Targets:** scaled log return (338), price delta (188), raw log (4).
- **Early stopping** on price MAE (after the v1 bug where it used RMSE). Device: CUDA in round A, CPU elsewhere. Seed 42.
- **Explainability.** Native `pred_contribs` TreeSHAP in the app.
- **Best results:**
  - round A T+5 −0.53% (direction 55.3%);
  - market-v1 expanded + availability T+5 +0.76% (direction 57.5%, 5/8 folds);
  - named capacity +7.44% vs Ridge, but −0.49% vs Naive.

  None passed the +5% gate.

### Ridge: 13,730 fits
- **Synthetic.** 3,380 in `weak-signal-control-v1` (37 features; 20 scenarios × matched null/injected seeds).
- **Market.** The linear arm of every exploration (about 830–880 each), availability clock, return path, agri transfer and Texas NASS regional (676 each), full-year (653), WASDE regional (424), contract curve T+1/T+5 (252 + 252), AMS (147), on-call (81), named label control (44).
- **α** ∈ {0.1, 1, 10} over 129 recipes. Most pilots used α = 1, window 1, expanding history, refit every 21 observations, 3×63 inner validation, shrinkage weights {0, .25, .5, .75, 1}.
- **Best:** fundamental joint L1 +0.22% selected (raw −6.64%).

### HAR (heterogeneous autoregressive volatility): 3,048 fits
- **Variants:** classic HAR (daily, weekly and monthly realised variance; `har_1`, `har_5`, `har_22`), log-HAR, and EWMA-relative HAR. The relative variant uses inputs `own_har_k − log(EWMA)`, target `actual / EWMA` and a 4–5 parameter QLIKE log-link learner; a zero coefficient returns EWMA.
- **Fits:**
  - weak-teacher synthetic 1,044;
  - full-year 395;
  - risk level source 348 + 1;
  - signed/unsigned shock 338 + 1;
  - own-level/DXY 338 + 1;
  - AMS named EWMA-HAR 174 + 1;
  - CT EWMA-HAR 169 + 1;
  - recent audit 61;
  - named label/history/loss/EWMA-HAR 22–44 each;
  - pre-2010 replication 31;
  - manual forward 10.
- **Result:** the own-level arm improved QLIKE by +0.0177 (both CIs positive, 7/8 years). It is in a live forward test to Nov 2027.

### GARCH: 707 fits (`full-year-v1`)
GARCH(1,1) from the `arch` package: zero mean, Student-t errors, no rescaling (`ml/src/cottonlens_ml/research/volatility.py`). Used as a symmetric risk baseline beside HAR in the full-year runs. Not better than EWMA.

### ElasticNet: 356 fits (`full-year-v1`)
α ∈ {0.001, 0.01}, l1_ratio 0.5. Also a reference family in round A (24 candidates per horizon). Round A T+1 −0.82%, T+5 −3.58%.

### Quantile regression (conditional median): part of 342 fits (`price-only-median-t1-v1`)
- Linear Quantile regression, q = 0.5, α = 0.01, fit_intercept, 5 price-only features; Ridge as the second arm. 338 market + 4 synthetic.
- **Result:** −0.0825% (0/8 years). STOP.

### EWMA: 23 fits
λ = 0.94 is the risk reference everywhere. `named-ewma-scale-control-t1-v1` fitted a scale correction (22 market + 1 synthetic) with no gain.

### CatBoost: 2 ledger fits, plus the round A and market-v1 GPU searches
- **Ledger:** depth 6, l2_leaf_reg 1, learning_rate 0.1, both synthetic known-signal controls.
- **Market:** in round A, 128 GPU candidates per horizon (T+1 −0.46%, T+5 −1.17%). In the market-v1 ablation, about half of the 20,360 sub-fits (best +0.10% T+5, expanded).
- GPU CatBoost is non-deterministic, so reproduction tolerances were enforced.

### LSTM, TCN, MLP: 1 ledger fit each, plus the v1 LSTM
- **Ledger:** one synthetic known-signal control each, confirming that the sequence pipeline can learn a planted signal. Width 32, learning rate 3e-3, batch 64, dropout 0; window 20 for LSTM and TCN; T+1; 24 features.
- **v1 LSTM** (23 Sep, Colab):
  - 60-step window, two outputs (T+1 and T+5), 4 configurations per horizon and fold, seed 42, early stopping.
  - Result: T+1 −1.09%, T+5 −0.01%; best epoch 1, then overfit.
  - Explained in the app with SHAP GradientExplainer.
  - The original common-cohort LSTM predictions from that run are missing (registered defect DG-008) and were not regenerated.
- **Round C** (MLP/LSTM/TCN, 48 candidates per family, windows 20/60/120) was designed but not run on market data, because round A showed no inner-to-outer transfer.

## Phase 2

### HistGradientBoostingRegressor (scikit-learn 1.6.1): county yield-anomaly models
- **Settings:** max_iter 300, learning_rate 0.04, max_leaf_nodes 15, min_samples_leaf 30, l2_regularization 2.0, random_state 0. The model_dev variant uses max_iter 250. Native NaN handling.
- **Target:** log(production per planted acre / county past mean). **Features:** NDVI anomalies vs the county's own past seasons (n_max, n_last, n_mean, n_int, n_p90max), optionally weather z-scores, Landsat field features, FSA shares.
- **Aggregation:** to state via prior-year planted weights × planted area (March, June or FSA-scaled).
- **Rolling refit** per test year: T05–T07, T11–T13, T28–T29, T31–T32, T34a, and member H in T36–T38.

### RidgeCV (state models T01–T04)
α grid logspace(−1, 4, 30), standardised features.

### Pooled ridge and bias models (USDA corrections)
- **T14–T27** `usda_correction.fit_predict`: ridge λ = 1, unpenalised intercept, features standardised within each fold, at least 10 past rows.
- **T25:** intercept only (pooled past-mean bias). **T40:** state empirical-Bayes shrinkage toward 0, τ² from fold moments. **T41, T43:** Texas-only ridge, λ = 1.

### Histogram CNN (PyTorch, Kaggle GPU), T35
- **Input:** 6 MODIS bands × 22 eight-day slots × 32 fixed bins (1 Oct), or × 14 slots (1 Aug).
- **Architecture:** 3 × [Conv2d 3×3, stride (1,2), BatchNorm, ReLU, Dropout 0.25] with 32/64/128 channels, global average pooling, Linear 128→64, ReLU, Linear 64→1.
- **Training:** Adam lr 1e-3, weight decay 1e-4, batch 64, at most 60 epochs, early stopping patience 10 on validation year Y−1, MSE loss, 5 seeds, per-band z-normalisation from training years.
- 120 trainings in total.
- **Result:** 1 Oct TX 18.4 / US 9.6 MAPE. Seed spread TX 4.2. Zero ensemble weight.

### Rolling simplex ensemble, T36
Weights on a 0.05 simplex (1,771 vectors for 4 members), chosen per year by pooled squared log error on earlier years' out-of-sample member predictions. With fewer than 2 past years, or a member missing, it falls back to the bias correction.

### Revision momentum, T44 and T45 (the result that held)
- **Rule:** ê = b · r/σ_r, with r = log(USDA_m / USDA_{m−1}), b = Σzē / (Σz² + λ), z = r/σ_r, σ_r = RMS(r), λ = 1.
- No intercept; pooled over states; fitted on seasons before Y; at least 10 training rows.
- Frozen 2026 coefficients: Nov b = 0.01011, σ = 0.08305; Dec b = −0.00567, σ = 0.07315.
