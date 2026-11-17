# 02 · Numbers cheat sheet

Every number below has a source. "Exact" means counted from a file; "approx." means estimated, with the method stated.

## Model fits

| Number | What | Source | Type |
|---|---|---|---|
| **32,934** | Completed model fits in the Phase 1 fit ledger | `research/trials.json` (sum of `completed_fits`) | exact |
| 28,245 | of which on real market data | same, kind `market_fit_receipt` | exact |
| 4,597 | of which synthetic controls (injected signal, null, weak teacher, known signal) | same, kinds `synthetic_*` | exact |
| 92 | of which walk-forward audits and archival replications | same | exact |
| **20,360** | Additional sub-fits in the `research-market-v1` ablation (128 outer units, 12.2 h compute) | `output/reports/research-market-v1-ablation-review-20261001.json`, also `data/codex_market_v1_ablation.csv` | exact; not in the fit ledger |
| 14,723 | XGBoost fits in the ledger: 7,320 market + 6 synthetic in round A, 6,729 in nine source explorations (FX 1,020, CFTC 1,008, weather 996, FAS 996, NASS 984, crop 972, WASDE 528, on-call 141, AMS 84), 668 in other profiles | `research/trials.json` family count | exact |
| 13,901 | Ridge fits (3,380 of them synthetic power controls) | same | exact |
| 3,048 | HAR volatility fits (1,044 synthetic weak-teacher) | same | exact |
| 707 | GARCH fits | same | exact |
| 356 | ElasticNet fits | same | exact |
| 171 | Linear Quantile fits (169 market + 2 synthetic); the Ridge arm of the same study has another 171 | same, profile `price-only-median-t1-v1` | exact |
| 23 | EWMA scale fits | same | exact |
| 2 / 1 / 1 / 1 | CatBoost / MLP / TCN / LSTM fits in the ledger (all synthetic known-signal controls) | same | exact |
| 128 + 128 | XGBoost and CatBoost GPU candidates per horizon in round A design | `docs/archive/CONTINUOUS_RESEARCH.md` | design |
| **120** | GPU CNN trainings in Phase 2 (T35: 5 seeds × 9 folds × 2 cutoffs = 90; T39: 5 × 3 × 2 = 30) | Kaggle outputs, `docs/CROP_NOWCAST_TRIALS.md` | exact |
| 134 / 97 / 204 | Pre-ledger fits, Sep 2026: v1 report (104 XGBoost, 20 LSTM, 10 Ridge); corrected R2 checkpoint receipts (76 XGBoost, 21 LSTM); tabular sprint XGBoost checkpoints | Colab reports and checkpoint receipts (`12_CODEX_ANSWERS.md` Q1–Q2) | exact for those archives; full September total unknown |
| 10,176 + 10,176 | XGBoost and CatBoost market fits inside the 20,360 market-v1 sub-fits (+8 diagnostics) | `research-market-v1/ledger/completed` | exact |
| 1,000+ | Phase 2 rolling refits of gradient boosting, ridge and shrinkage models | 45 trials × ~3 cutoffs × 9–24 test years | approx. |
| 1,771 | Weight vectors evaluated per fold in the T36 simplex grid (4 members, step 0.05) | `research/crop_nowcast/stack.py` | exact (evaluations, not fits) |
| **33,000+** | Conservative headline: 32,934 ledger + 120 CNN (+ 1,000+ Phase 2 refits; + 20,360 ablation sub-fits not counted) | above | exact + approx. |

## Experiments and trials

| Number | What | Source |
|---|---|---|
| 569 | Phase 1 fit-ledger records (recipes) | `research/trials.json` |
| 239 | Phase 1 registry records | `research/registry.json` |
| 133 | Distinct Phase 1 experiments in the registry | same |
| 113 | Phase 1 experiments with a scored MAE gain vs naive | same |
| 41 | Phase 1 research profiles | `data/codex_profiles.csv` |
| 10 | Model families in the Phase 1 ledger | same |
| 487 | Distinct feature names used in Phase 1 | `data/feature_catalog.csv` |
| 387 | XGBoost recipe rows in the ledger (round A: 257 recipe rows, 179 distinct parameter sets) | `data/hyperparameter_ranges.csv`, `data/codex_profiles.csv` |
| 45 | Numbered Phase 2 trials (T01–T45), about 60 variants | `docs/CROP_NOWCAST_TRIALS.md` |
| 178 | 133 + 45 experiments across both phases | above |
| 3 | One-shot tests on held-out years: T39 (2022–2024), T42 (2025), T45 (blind 2000–2011) | ledger |
| 111+ | Private GitHub releases with evidence archives (Phase 1) | `docs/archive/phase2-notes/CLAUDE_HANDOFF.md` |
| 297 | Evidence files with SHA-256 receipts | `data/evidence_index.csv` |
| 127 | Project documents (docs + archive) | `data/docs_catalog.csv` |

## Satellite and data volume (Phase 2)

| Number | What | Source |
|---|---|---|
| **806,565,561** | Landsat pixel NDVI observations over cotton grid points | `landsat_*_*.json` run metadata (225 state-years) |
| 28,977 | Landsat scenes processed (43,485 listed by STAC search) | same |
| 49,837,577 | Cotton grid points summed over state-years (120 m grid, prior-year CDL) | same |
| 2,278 | MODIS MOD13Q1/MYD13Q1 composites in the histogram pipeline (80 tile-years, 2009–2024) | `output/deep/hist_t39/*.npz` |
| 13,668 | MODIS band rasters (2,278 × 6 bands) | same |
| 2,250 | MODIS composites in the county NDVI pipeline | `output/crop-nowcast/ndvi/*.csv` |
| 5 | MODIS tiles (h08v05, h09v05, h09v06, h10v05, h11v05) | panel |
| 263 | Panel counties in 17 states (81 in Texas) | `geo/panel.json` |
| 4,208 | County-year rows in the CNN tensor (3,419 for 2009–2021) | `tensor_t39_oct1.npz` |
| 484 | Official documents parsed: 72 Crop Production, 292 Crop Progress, 65 Cotton Ginnings, 41 FSA, 14 June Acreage | raw folders |
| 42,423 | NASS QuickStats rows (county, state, US) | merged JSON |
| 39,161 | FSA county rows (planted and failed acres, 2011–2024) | `fsa_cotton.csv` |
| 103,622 | Drought Monitor county-weeks | `usdm_county.csv` |
| 264 | Counties with NASA POWER daily weather (1991–2024) | `power*/` |
| 963 / 59 | State / US USDA forecasts parsed from 2000–2011 reports (parser exact 153/153 vs QuickStats) | `output/crop-nowcast-t45/` |
| 20+ | Kaggle runs: 5 MODIS histogram, 9 sealed-year extraction, 5 Landsat 2022–2025, 2 GPU CNN | Kaggle CLI logs |

## Results to quote

| Number | What |
|---|---|
| +0.22% | Best Phase 1 selected MAE gain vs naive (target +5%) |
| −0.65% | Median Phase 1 selected gain (113 experiments) |
| +0.76% | Best ablation cell (XGBoost expanded availability, T+5; direction 57.5%; 5/8 folds) |
| −0.0825% | Price-only median primary (Quantile T+1), decision STOP |
| +0.0177 | Own-level HAR QLIKE gain vs EWMA (both CIs > 0, 7/8 years) |
| 1.42× | Report-day absolute return vs other days (p = 0.004) |
| 12/12 | Report × region cells where the momentum rule's 2002–2025 MAPE is below USDA's |
| 3–11% | Relative MAPE reduction of the momentum rule |
| 3 of 8 | Cotton TX/US cells where the momentum gain is significant at one-sided p < 0.05 (season-level permutation, uncorrected) |
| 1 / 3 | Months passed by the frozen rule on corn and on soybeans in T46 (2 needed): FAIL; pooled p = 0.28 |
| 88 / 128, p = 0.007 | All blind seasons pooled (1987–2011, 23 seasons): rolling momentum closer than USDA (pooling chosen after seeing both eras) |
| 67% (62/93), p = 0.012 | T50: direction of USDA's next revision called on blind cotton 1985–2011, US + TX (TX 79%, states 72%) |
| 57–69% | T50: same rule on corn and soybeans 2000–2025 (US 69% / 67%, states 57%) |
| 43% | T50: same rule on cotton 2012–2025: no longer works |
| 3.8% → 14.9% | Texas 1 Sep harvested-area error, 1995–2011 vs 2012–2025 (T48) |
| ~40% | Cut in USDA's in-season Texas area error by the real-time anchoring model, 2008–2025 (T49, p = 0.0003) |
| 3,258,029 ac | Texas 2026 harvested area, locked (−16.5% vs USDA Nov) |
| 14/7 | US December seasons closer / further than USDA |
| 18.4 / 9.6 | CNN TX / US MAPE at 1 Oct (USDA 11.0 / 6.1) |
| r ≈ 0.8 | Landsat failed-field share vs Texas abandonment |
| +40.5% | USDA's Texas overestimate at 1 Oct 2023 |
| 12,763,510 | Locked 2026 US forecast (bales), commit `18c3bc7` |
