# 03 · Phase 1: cotton price and risk forecasting (21 Sep – 7 Nov 2026, Codex)

Machine-readable detail: `data/codex_fit_recipes.csv` (each recipe), `data/codex_profiles.csv` (each profile), `data/codex_experiments.csv` (each scored experiment), `data/docs_catalog.csv` (each document with its result lines).

## 3.1 The question and the target
- **Instrument.** ICE cotton futures ("Cotton No. 2"), mostly the continuous front-month series CT=F as a daily price proxy. Later studies used "named" same-contract series built from USDA AMS daily futures reports.
- **Price target.** The next settlement 1 trading day ahead (T+1) or 5 trading days ahead (T+5), predicted as a scaled log return, a raw log return or a price delta, then mapped back to a price.
- **Baseline.** **Naive** (no change: tomorrow's price = today's). In some risk studies: the drift reference and the historical-majority direction.
- **Price success gates (fixed before results):**
  - MAE gain vs Naive ≥ **+5%**;
  - direction accuracy ≥ 53% (T+1) or 55% (T+5);
  - Naive beaten in at least 3 of 4 folds (later 6 of 8 years).
  - If no model passes, Naive stays the production forecast.
- **Risk target** (from 12 Oct). The next day's squared log return (T+1 variance), scored with **QLIKE** against an EWMA λ = 0.94 reference.
  - Success: both moving-block bootstrap CIs (block 20 and block 60) above zero, at least 6/8 positive years, and a positive top-1% trimmed gain.

## 3.2 Evaluation design used throughout
- **Outer test.** Eight annual blocks, 2016–2023, with 126 eligible origins each in round A and 2,006 origins in the full-year design. 2024 onward was already seen, so it is descriptive only and never used to select.
- **Inner validation.** Three earlier 63-origin blocks (3×63 = 189 origins) before each outer year. Selection uses inner scores only.
- **Label maturity and purges.** Labels must mature (5 sessions for T+5) before a refit can use them. Purges protect inner and fold boundaries.
- **Refit cadence.** 21 observations (some studies 126). Expanding training history; at least 500 mature training rows.
- **Shrinkage selection.** Many studies select a weight w ∈ {0, 0.25, 0.5, 0.75, 1} on the inner blocks: final forecast = Naive + w·(model − Naive). A weight of 0 means "keep Naive", which is why "selected" and "raw" results are both reported.
- **Statistics.** Paired MAE (or QLIKE) differences with moving-block bootstraps (blocks of 20 and 60, 10,000 draws, seed 42), year counts, and trimmed sensitivities.
- **Engineering.**
  - From 5 Oct 2026 (`3d5f7f0`) the Git fit ledger holds hash-keyed receipts (source, data, split, recipe). Earlier September runs and the market-v1 ablation kept their own checkpoint receipts outside it.
  - Completed fits are never retrained (`REPEAT_FIT_RECIPE` check).
  - Results are archived to private GitHub releases. Many were verified by fresh restore and replay; some audits replay saved state with zero new fits, GPU CatBoost uses tolerances, and some old payloads are missing (`research/README.md`).

## 3.3 Studies in order

### A. Application and first models (21–25 Sep)
- **App.** FastAPI backend, Angular frontend (dashboard, Model Lab, replay, sensitivity, "why" pages), ML package, Docker Compose.
- **v1 (Colab, commit `c33d6ae`).** XGBoost vs LSTM, 2010-04-01 to 2026-09-22, 3,905 labelled origins, four 126-observation evaluation blocks.

  | Candidate | T+1 MAE gain vs Naive | T+1 direction | T+5 MAE gain | T+5 direction |
  |---|---|---|---|---|
  | XGBoost | −0.376% | 52.78% | +0.725% | 54.56% |
  | LSTM | −1.094% | 51.98% | −0.011% | 52.78% |

  - **LSTM design:** 60 kept observations × 24 features (not necessarily 60 consecutive sessions); one Keras LSTM layer (32 or 64 units, no internal dropout) → Dropout (0.10 or 0.20) → Dense(32, ReLU) → Dense(2, linear) for T+1 and T+5 log returns. Adam lr 0.001, batch 64, up to 100 epochs, MAE loss on standardised targets, early stopping on `val_loss` with patience 10 and best-weight restore, seed 42. Separate StandardScalers for inputs and targets, fitted on train only.
  - **Search budget:** 8 XGBoost candidates per horizon and 4 two-output LSTM candidates — (32, 0.10), (32, 0.20), (64, 0.10), (64, 0.20) — per fold, in 4 outer folds plus a final train/validation split. In total 104 XGBoost, 20 LSTM and 10 Ridge fits (134).
  - On the final validation split (train to 17 Mar 2022, validation 25 Mar 2022 – 10 Jun 2024), the best 32/0.20 checkpoint was epoch 1: train/validation loss 0.731/0.922, 0.636/0.979 by epoch 11. Fold epochs differ (for example epoch 2 in fold 1), so "epoch 1" describes that split, not every fold.
  - **Decision:** Naive at both horizons.
  - **Bugs found:** percentage-based inner validation withheld about 2 years; XGBoost early stopping optimised RMSE while selection used MAE; interrupted LSTM checkpoints were treated as complete; a missing origin compressed the LSTM context.
- **Corrected v2 (release `v20260924-185501`).** Fixed a 126-observation inner validation, refit before evaluation, separate scalers, price-MAE early stopping for XGBoost (the LSTM still picks its epoch on scaled-target validation MAE). 97 checkpoint receipts: 76 XGBoost, 21 LSTM. Decision: Naive at both horizons.
- **Tabular sprint (`v20260925-074708`, 204 XGBoost checkpoints).** T+1 gain +0.06% (direction 46.6%), T+5 −2.09% (direction 47.8%). Reproduction max difference zero. The development T+5 gain of 0.743% did not transfer. A 21-observation refit failed its pre-declared comparison.

### B. Research engine and the large GPU search (28 Sep – 1 Oct; round A result recorded in the 28 Sep reset)
- **`research-v1`.** Stopped before market search: Keras built a CPU-only `RangeDataset` inside a forced GPU device scope with soft placement off. Fixed in v2 by scoping GPU placement to model construction only and asserting variable placement.
- **`research-v2-tf-placement` (profile `cotton-research-v1`): 7,331 completed fits on a Colab T4 GPU.**
  - **Round A design:** per horizon, 128 CUDA XGBoost and 128 GPU CatBoost candidates, plus 24 Ridge and 24 ElasticNet references. Four target-scale/regularisation diagnostics first. Price MAE drives early stopping and inner selection. Shortlisted candidates are confirmed with 3 seeds.
  - **Ledger content:** 7,320 XGBoost market fits.
    - 257 recipe rows, 179 distinct parameter sets (targets: 119 scaled_log, 134 price_delta, 4 raw_log rows): max_depth 1–8, eta 0.005–0.15, min_child_weight 1–100, alpha 0–9.3, lambda 0.001–95, subsample 0.60–1.0, colsample_bytree 0.60–1.0.
    - Targets: scaled log or price delta. Objectives: `reg:squarederror` or `reg:absoluteerror`. 24 features. Horizons 1 and 5.
    - Roles: early-stopping, inner and outer fits.
  - **Result (outer 2016–2023):**

    | Family | T+1 gain | dir | folds won | T+5 gain | dir | folds won |
    |---|---|---|---|---|---|---|
    | XGBoost | −0.611% | 46.9% | 1/8 | −0.528% | 55.3% | 2/8 |
    | CatBoost | −0.463% | 48.3% | 0/8 | −1.166% | 52.6% | 3/8 |
    | Ridge | −2.135% | 47.4% | 2/8 | −6.340% | 50.0% | 3/8 |
    | ElasticNet | −0.819% | 47.1% | 1/8 | −3.575% | 50.8% | 4/8 |

    All gates failed. Lesson: inner gains did not establish outer superiority.
  - **Planned rounds** (designed in `docs/archive/CONTINUOUS_RESEARCH.md`, not run on market data):
    - B: XGBoost, CatBoost and logistic classification, 64 candidates each;
    - C: MLP, LSTM, TCN, 48 candidates per family, windows 20/60/120;
    - D: expanded causal OHLCV features;
    - E: histories of 3/5/all years and refit cadences 5/21/63/126, with an OOF-weighted ensemble.

    The ledger holds one synthetic known-signal control each of MLP, TCN and LSTM and two of CatBoost (width 32, lr 3e-3, batch 64, dropout 0; window 20 for LSTM/TCN, 1 for MLP). The TCN control's old `passed` label is wrong: its known-signal gain was −23.6%. Round C was parked as a conditional hypothesis after the 28 Sep reset (`docs/archive/RESEARCH_RESET_20260928.md`).
- **`research-market-v1` ablation: 20,360 sub-fits, 128 outer units, 44,085 s (12.2 h) compute, 1,008 outer origins.**
  - Sub-fits: 10,176 XGBoost + 10,176 CatBoost market fits + 8 diagnostics, 16 candidates per unit, 1,008 origins per horizon.
  - Groups: base (24 features), cotton without volume (14: drops volume, DXY/WTI returns and cross-correlations), expanded (39: causal OHLCV/lag/rolling derivatives), expanded with availability (64: adds gap days and 24 `age_*` fields). Families: XGBoost and CatBoost, at T+1 and T+5. Admission needed a paired inner gain of at least 0.5% in at least 5/8 inner periods; no group passed.
  - Best cell: expanded + availability, XGBoost T+5: +0.763% gain, direction 57.5%, 5/8 folds, 95% CI includes 0.
  - **Decision:** no group admitted; do not repeat or widen the same-information search.
- **Legacy loss lineage.** The old XGBoost T+1 selection was rebuilt from 6,648 archived fit receipts: 128 seed-42 candidates × 3 inner blocks, top 5 × 3 seeds. The selected target/loss changed year to year (2016: scaled_log/absolute +0.12%; 2017: scaled_log/squared −2.50%; …). This showed the "best" loss was unstable.

### C. Information sources (2–9 Oct): 1,855 fits per source
Common design, `*-exploration-v1`:
- T+5; Ridge (α = 1) and a fixed depth-2 XGBoost recipe; seed 42; 24 base features (cotton returns 1/5/10/20, momentum 5/20, SMA ratios, volatility 5/20, range, volume change, volume z-score, volatility regime, DXY and WTI returns 1/5/20, cotton–DXY and cotton–WTI 60-day correlations, month sin/cos).
- Each source enters at availability lags L1, L2 and L6: first recorded cotton session after publication, +1 or +5 extra sessions as a stress test.
- Each lag has a **missing** control (only age and missingness indicators) and a **source** arm (the values).
- Result columns: selected gain (raw gain, selected direction).

| Source (profile) | Best group | Selected gain | Raw gain | Note |
|---|---|---|---|---|
| WASDE balance sheet (`wasde-exploration-v1`, 1,162 fits) | missing_L6 | −0.25% | −3.60% | stocks/use, production/use |
| NASS crop condition (`nass-exploration-v1`) | condition_L1 | −0.69% | −4.87% | good/excellent, poor/very poor |
| FAS export sales (`fas-exploration-v1`) | sales_L6 | −0.59% | −2.27% | weekly sales |
| CFTC positioning (`cftc-exploration-v1`) | position_L2 | −0.77% | −1.78% | managed money net/OI |
| FX (`fx-exploration-v1`) | currency_L1 | −0.53% | −2.45% | currency returns |
| Weather (`weather-exploration-v1`) | weather_L1 | −1.01% | −2.28% | belt weather |
| Crop prices (`crop-exploration-v1`) | missing_L1 | −1.02% | −1.94% | corn/soy/wheat prices hurt |
| AMS quotes (`ams-exploration-v1`, 231 fits) | quote_L6 | 0.00% | −0.05% | always shrunk to Naive |
| On-call (`oncall-exploration-v2`, 222 fits) | oncall_L2 | −1.32% | −1.93% | mill on-call positions |

No source passed. Every selected result is near zero because inner validation shrank to Naive; the raw models are worse than Naive.

### D. Mechanism and power checks (5–10 Oct)
- **Availability clock** (676 Ridge fits): T+1 available arm −0.15% vs control −0.33%. 87.5% of the improvement came from one year's weight going to 0. Timing loss is not the main mechanism.
- **Return path** (676 fits, 24–83 features): base + return path −0.04% (T+1). No gain.
- **Agri transfer** (676 fits, 6 features): cotton + corn + soybean +0.18% (T+5), +0.06% (T+1). Noise-level.
- **Fundamental joint pilot** (registry): joint_L1 +0.22% selected, raw −6.64%. The best selected result of Phase 1 is still a near-zero, shrinkage-driven gain.
- **WASDE text pilot:** numeric and text arms −0.21% to −1.37%.
- **WASDE regional T+1** (424 fits, 52 features, 2019–2023, 1,254 origins): selected −0.30% to −0.52%, raw down to −10.2%. Decision `FIXED_RECIPE_BELOW_PRACTICAL_GOAL`.
- **Texas/NASS regional T+1** (676 fits, 37 features: 24 base + report age + national condition + 8 Texas condition/progress gaps): selected −0.03%, raw −3.76%. Same decision.
- **Weak-signal power control** (3,380 synthetic Ridge fits; 20 scenarios; 10 seeds of matched null/injected noise; 40,120 OOS rows): decision `RAW_ONLY_RECOVERS`. The raw model recovers the injected signal and shrinkage tends to hide it, but that does not prove shrinkage caused the market failures, because the real Texas raw model was also worse than Naive.
- **Selection stability:** 160 decisions rebuilt from 1,440 inner fits and 30,240 validation predictions. With one validation block dropped, 59 of 80 injected-signal decisions changed. Selection on three short blocks is unstable.
- **Paired loss (MSE vs MAE)** (48 + 6 fits; 1,008 origins): MAE-trained XGBoost −0.49% vs MSE −1.98%. MAE beats MSE by +1.46% (CIs positive) but still loses to Naive.
- **Price-MAE calibration** (169 scalar corrections, 0 new fits): −0.026%, unchanged.
- **Contract curve T+1 and T+5** (252 + 252 Ridge fits, 28 features, 749 origins): all weights selected 0 at T+1. T+5 numeric D0 −0.55%.

### E. Named (same-contract) studies (10–12 Oct)
- **Why.** CT=F rolls between contracts; a return across a roll is not a tradable return. "Named" series pin the contract at the origin.
- **Label control T+5** (44 Ridge fits): correct contract labels reduce raw error but do not beat Naive.
- **Capacity control T+5** (25 + 1 XGBoost fits, 28 features, 2023, 242 targets): XGBoost +7.44% vs Ridge (positive CIs), but −0.49% vs Naive. The upper bounds of the gain CIs (3.27% / 3.66%) are below the 5% target.
- **History, CFTC (37 fields) and weak-signal (132 synthetic XGBoost) controls:** no practical gain over Naive. The weak teacher showed the recipe can learn a simple useful effect, so the failure is not a learning bug.

### F. Risk and volatility (12 Oct – 4 Nov)
- **Reference.** EWMA λ = 0.94. Models: HAR (daily, weekly, monthly realised variance; log-HAR; EWMA-relative HAR with 4–5 parameters fitted under QLIKE), GARCH (707 fits in `full-year-v1`), EWMA scale.
- **`full-year-v1`** (2,471 fits: GARCH 707, Ridge 653, HAR 395, XGBoost 360, ElasticNet 356): baseline risk and price runs across 2016–2023.
- **CT EWMA-relative HAR** (169 + 1 fits): raw +0.0175 QLIKE but CIs include 0, 4/8 years. INCONCLUSIVE.
- **Named EWMA-HAR** (22 + 1): −0.0198 raw. Not better.
- **Zero-fit screens** (seasonality, asymmetric return sign, CFTC turnover, curvature, AMS curve): all `INCONCLUSIVE` or negative.
- **Risk weak teacher** (1,044 synthetic HAR fits, seeds 42/1701/2703): the HAR + selection pipeline recovers a small known effect.
- **Signed/unsigned shock** (338 + 1): +0.0027 QLIKE, CIs include 0, 4/8 years. INCONCLUSIVE.
- **CT own-level / DXY control** (338 + 1): DXY primary +0.0018, INCONCLUSIVE.
  - **Pre-declared secondary own-level arm: +0.01770 QLIKE, CI20 [+0.0043; +0.0299], CI60 [+0.0047; +0.0296], 7/8 positive years.** This is the one historical risk candidate.
  - 95.9% of the net gain came from 21 large EWMA misses. 169 of 169 models had a negative extra log(EWMA) coefficient.
- **Forward test of the own-level candidate.**
  - Locked cohort; first live origin 30 Oct 2026.
  - Designed as daily automated publishing at 03:05 / 03:20 Turkey time with GitHub release receipts, 252 origins to 2 Nov 2027, at most 31 fits.
  - What happened: v1 published once (31 Oct); a Yahoo revision made it reject the 2 Nov input and miss that origin, breaking its 252/252 rule. v2 (first-observed) published once (4 Nov 2026, 03:20). Both Windows tasks are now **disabled** (checked 16 Nov 2026). No forward QLIKE score exists.
- **Recent-history audit** (61 fits, 711 forecasts, 2024–Oct 2026): +0.00009 QLIKE, CIs include 0. Undetermined.
- **Pre-2010 same-contract replication** (31 fits, 252 origins, Nov 2008 – Nov 2009, rebuilt from USDA PDF/XLS archives): the model weight was selected as 0. Same as EWMA.

### G. Data integrity findings (29 Oct – 6 Nov)
- **AMS annual futures workbooks** carried past contract months (for example 2020-02-26 labelled 2019-07-01). 217 evaluation origins were affected. Annual AMS named-market results were marked METHODOLOGICALLY_COMPROMISED.
- **Reconciliation.** 1,262 raw daily reports checked; 3,320 contract identities corrected in a separate overlay; 110 of 863 checkable windows had different HAR inputs.
- **Yahoo price revision** (30 Oct close 78.13 → 77.83) broke the first forward cohort. A new "first observed" cohort keeps the first accepted price and logs revisions.
- **ICE price-limit announcements:** 36 of 75 verified. The limit signal did not pass its diagnostic.

### H. Final price study and stop (5–6 Nov)
- **Price-only conditional median T+1** (338 market + 4 synthetic fits, 2,006 origins, 5 features).
  - Linear Quantile regression (median) vs Ridge.
  - Primary Quantile selected −0.0825% (0/8 years); Ridge selected −0.157%; raw Quantile −0.119%, raw Ridge −2.78%.
  - Decision **STOP**: Naive stays.
- **Root-cause review (6 Nov).** No single root cause. The common finding is that corrections learned from the tried representations do not carry forward. Stop broad model search; measure contract, price-field and time-coverage gaps first.

### I. Link 1 event study (15 Nov, Claude, bridging to Phase 2; not a Codex-era study)
- 38 USDA Crop Production report days, 2012–2021 (upland series; the preregistration said all-cotton, and the deviation is logged), against 1,164 other days. Prices after 2021-12-31 stay sealed for this chain.
- **H1** (bigger moves on report days): ratio 1.42, p = 0.0038. **Pass.**
- **H2** (USDA's revision sign predicts the move): Spearman −0.23, p = 0.075. **Fail.** The combined gate failed.
- An oracle that knows the revision does not beat naive (−3.8%).
- Without the three |R| ≥ 4% days the ratio is 1.15 (p = 0.16).
- **Reading:** report days move more, but the revision does not predict the direction. Market consensus data would be needed; none was found in the free sources checked. So Phase 2 targeted production accuracy itself.

## 3.4 What Phase 1 established
1. No protocol tried passed the practical price gate at T+1 and T+5: public features gave no usable edge over a no-change forecast. 113 scored experiments have median −0.65% and best +0.22%.
2. The pipeline is unlikely to be the main reason. Synthetic controls (4,597 fits, positive and negative) recovered injected signals in the tested scenarios, and selection instability and shrinkage were measured directly. This is evidence, not a full acquittal.
3. Data identity (which contract, which vintage, when published) matters as much as the model. Several apparent effects traced to contract-roll or source errors.
4. The one surviving lead is risk, not price: a pre-declared secondary own-level HAR arm that passed its historical gates. Its forward test published twice and is now stopped.
