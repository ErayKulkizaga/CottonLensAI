# 05 · Phase 2: US cotton production vs USDA (13 – 16 Nov 2026, Claude)

Full ledger: `docs/CROP_NOWCAST_TRIALS.md`. Machine-readable: `data/claude_phase_trials.csv`, `data/momentum_by_year_2002_2025.csv`, `data/bias_correction_t25_by_year.csv`. Evidence JSONs: `research/evidence/` (named per trial).

## 5.1 The question
Forecast US and Texas upland cotton production (480-lb bales) before or at each USDA monthly Crop Production report (Aug, Sep, Oct, Nov, Dec), and be closer to the NASS final than USDA's own forecast.

- **Metric:** MAPE vs the NASS final, for TX and the US, plus the production-weighted MAPE over 17 states.
- **Benchmark:** USDA's forecast for the same month. Its 2013–2021 error: Aug TX 14.8 / US 7.2; Sep 11.2 / 6.3; Oct 11.0 / 6.1; Nov 8.5 / 4.4; Dec 4.7 / 3.1.

## 5.2 Pipelines built

| Pipeline | Code | What it does |
|---|---|---|
| NASS pull | `research/crop_nowcast/nass_pull.py` | QuickStats county/state/US yield, production, harvested, planted, and monthly forecasts; key from an environment variable; SHA-256 receipts; seal guard for 2022+ |
| June Acreage | `june_acreage.py` | Parses ESMIS June Acreage text reports |
| FSA acreage | `fsa_acreage.py` | Parses FSA crop-acreage workbooks (planted and failed acres by county, irrigated/dryland). Keeps the earliest snapshot per month and checks file save time against report dates |
| Crop condition | `cond_pull.py` | Weekly Crop Progress cotton condition by state |
| Drought | `usdm_pull.py` | US Drought Monitor D2–D4 county shares |
| Weather | `power_pull.py` | NASA POWER daily Tmax, Tmin, precipitation, RH and solar for 264 counties |
| Ginnings | `gin_pull.py` | NASS Cotton Ginnings, two text layouts (2009 vs 2010+) |
| Old forecasts | `cp_pull.py` | USDA Crop Production 2000–2013 text reports: state/US upland forecast (1,000 bales; decimals handled) |
| MODIS NDVI | `county_ndvi.py` | MOD/MYD13Q1 v061 from Planetary Computer. Prior-year CDL cotton mask (class 2), county median/mean/p90 NDVI per composite |
| MODIS histograms | `research/deep/county_hist.py` | 6 bands × 32 fixed bins per county and composite, resumable per tile-year |
| Landsat fields | `landsat_fields.py` | Landsat 5/7/8 C2 L2 on a 120 m grid of prior-year CDL cotton cells; QA clear-pixel mask; per county and cutoff: NDVI p10/p25/p50/p75/p90, failed shares (NDVI < 0.25 / 0.35), last value, observation count |
| CDL source | `cdl_source.py` | Planetary Computer for ≤ 2021, NASS CropScape state GeoTIFFs for 2022+ (downloaded whole, size-checked) |
| Features | `features.py` | County-year features at each cutoff with availability lags (weather −3 days, NDVI composite end + 8 days) |
| Models and scorers | `county_up.py`, `state_model.py`, `usda_correction.py`, `hybrid.py`, `stack.py`, `state_bias.py`, `momentum.py`, `t43_tx_fail.py`, `t42_holdout.py`, `lock_2026.py` | Rolling-origin models, aggregation and scoring |
| Sealed-test driver | `t39_assemble.py`, `t39_run.py` | Merges dev and sealed data without touching dev files; one-shot run |
| CNN | `research/deep/build_tensor.py`, `train_cnn.py`, `score_t35.py` | Tensor builder, GPU trainer (resumable per fold × seed), scorer |

Every puller refuses years ≥ 2022 without `--unseal`. That flag was used only after the sealed-test preregistration was pushed.

## 5.3 Trials (cells: TX / US MAPE, 2013–2021 unless stated)

| Trial | Idea | Key result | Verdict |
|---|---|---|---|
| T01–T04 | State RidgeCV on weather / NDVI | Oct 16.1/11.8 at best | fail |
| T05–T07 | County HGB on NDVI anomalies (March / June area), weather | T06 Oct 12.7/7.7; with FSA area 10.3/6.6 | best standalone; behind USDA (US) |
| T08–T09 | Blend county model with USDA | Oct 10.8 TX | ≈ USDA |
| T10 | Predict the direction of USDA's revision | Sep corr +0.28, hit rate 57% | weak (first hint) |
| T11 | County accuracy vs naive | Oct +7.7% RMSE gain | weak |
| T12–T13 | Landsat 30 m field features | Oct 17.0/7.6; Landsat-only Aug US 6.2 | gate fail, unstable |
| T14–T24 | USDA error correction with FSA, drought, condition, satellite | — | withdrawn (leak + unscaled features) |
| T25 | Pooled past-mean bias correction | Sep 9.5/3.9, Oct 10.4/4.9 | dev win |
| T26–T27 | Correction with standardised signals (ten variants) | no gain over T25 | noise |
| T28–T30 | County FSA, FSA-scaled area, ensembles | T29 Oct 10.6 (claim withdrawn: FSA timing) | fragile |
| T31 | USDA yield × our harvested area | better area (TX Oct 7.2 vs 8.8), worse production | fail |
| T32 | Late-season hybrid vs USDA Nov/Dec | Dec 7.8/4.6 vs 4.7/3.1 | fail |
| T33 | Our gap predicts USDA's Dec revision? | corr −0.04 | no skill |
| T34a | NDVI season-sequence features | mixed | — |
| T35 | GPU histogram CNN | Oct 18.4/9.6 | fail |
| T36 | Rolling simplex ensemble (bias + HGB + CNN + USDA) | Sep 9.4/3.8, Oct 9.9/4.6 | dev pass |
| T37 | Ensemble at Nov/Dec | Dec 8.3/4.6 vs 4.7/3.1 | fail |
| T38 | Ginning pace | Dec 8.1/4.0 vs 4.7/3.1 | fail |
| T39 | **Sealed test 2022–2024** of T36 | Oct 15.0/5.2 vs USDA 20.3/4.2; TX better 3/3 | **fail** |
| T40 | State empirical-Bayes bias | Sep 14.0/5.6 vs 13.5/5.8 (2013–2024) | fail |
| T41 | Texas FSA abandonment gate | Sep 14.1 vs T25 11.4 | fail |
| T42 | **2025 holdout** of T25 | Sep TX −5.3% vs −0.6% | **fail** |
| T43 | Landsat failed-field share for TX | Sep 16.7 vs USDA 15.5; r ≈ 0.8 with abandonment | fail |
| T44 | Revision momentum 2014–2025 | Dec 5.5/3.0 vs 5.9/3.2; Sep fails | primary fail, Dec ok |
| T45 | **Blind 2000–2011** momentum | Dec US 1.7 vs 2.0; TX tie 3–3; Sep/Oct/Nov better | primary fail on tie |

## 5.4 Satellite details
- **MODIS county NDVI.**
  - Composites count only if end date + 8 days ≤ cutoff; a county needs at least 20 valid pixels.
  - Features: n_max, n_last, n_mean, n_int (Σ max(NDVI − 0.2, 0)), n_p90max, monthly means; anomalies vs the county's own earlier seasons.
- **MODIS histograms.** 2,278 composites × 6 bands, 80 tile-years, 5 Kaggle CPU kernels (about 70 min each for 13 years). Fixed bins: NDVI/EVI −0.2..1.0, reflectance 0..0.6.
- **Landsat.**
  - 225 state-years, 28,977 scenes used of 43,485 listed (cloud < 60%, Tier 1), 806,565,561 pixel observations at 49.8M grid points.
  - Scenes count if acquisition + 16 days ≤ cutoff, using the 120 m COG overview.
  - In 2022, 3 unreadable Landsat 7 scenes were skipped and recorded. 2024–2025 have about half the observations, because Landsat 7 was retired and Landsat 9 was left out to keep the spec.
- **CDL.** Prior-year cotton mask only. Planetary Computer covers ≤ 2021; CropScape covers 2022+.

## 5.5 Sealed and blind tests
1. **T39, 2022–2024:** opened once after a pushed preregistration. T36 failed: the US lost to USDA because USDA's bias flipped. In TX, the corrections beat USDA in 3 of 3 years.
2. **T42, 2025:** the last unopened production year. T25 failed (USDA under-forecast 2025).
3. **T45, 2000–2011:** USDA forecasts never pulled before. Parsed from 72 reports; the parser matched QuickStats exactly on 153 of 153 values for 2012–2013. A unit check caught a decimal bug before any score was computed. Momentum Dec primary tied 3–3 in TX; Sep, Oct and Nov beat USDA for TX and the US.

## 5.6 The momentum rule and its full record
- **Rule** (no intercept, pooled, rolling): ê = b · log(USDA_m / USDA_{m−1}) / σ_r, forecast = USDA_m · exp(ê).
- **Record 2002–2025:** MAPE below USDA in 12 of 12 cells. See `data/momentum_by_year_2002_2025.csv` and the table in `01_PROJECT_OVERVIEW.md`.
- **Why it can work:** USDA forecast smoothing, i.e. revisions are positively autocorrelated. This is documented for corn and soybeans (Isengildina, Irwin & Good).
- **Limits:** gains are 3–11%; it does not win every year; the formal pre-declared gates were not passed (T44 needed Sep as well, T45 tied in TX).

## 5.7 Forward lock (2026)
- Locked on 16 Nov 2026, commit `18c3bc7`, in `docs/FORWARD_LOCK_2026.md`.
- From USDA November (12,794,000 bales US; 4,100,000 TX): **US 12,763,510; TX 4,070,552.**
- The December procedure is frozen: b_Dec = −0.00567, σ = 0.07315.
- Scored against the NASS final in May 2027.

## 5.8 Compute and infrastructure
- **Kaggle** (automated from the CLI):
  - 5 MODIS histogram kernels (CPU);
  - 5 sealed-year extraction kernels plus 4 reruns after CropScape remote-read failures;
  - 4 Landsat 2022–2025 kernels plus 1 rerun;
  - 2 GPU CNN runs.

  Units are resumable per tile-year and per fold × seed; finished units are never recomputed.
- **Local:** pinned environment (scikit-learn 1.6.1, NumPy 2.1.3, pandas 2.2.3). A newer environment shifts results by about 0.7 points, so all reported numbers use the pinned one.
- **Git:** every trial is pre-declared, committed and pushed (branch `claude/hls-move3`). Evidence JSONs carry SHA-256 receipts.
