# What we tried, what worked, what did not (as of 16 Nov 2026)

This file records the **effort** behind the cotton-production work, including the approaches that added nothing.

Exact numbers live in `docs/CROP_NOWCAST_TRIALS.md`, evidence files in `research/evidence/`, and the sealed test in `docs/prereg/SEALED_TEST_T39_PREREG_20261115.md`. Every cell is TX / US MAPE (%) unless stated otherwise.

**Goal:** forecast US and Texas upland cotton production better than USDA's monthly Crop Production forecasts.

**Protocol:**
- rolling origin (train on years < Y);
- development years 2013–2021;
- a sealed holdout, 2022–2024, opened once (T39);
- 2025 is still unseen for Aug, Sep, Nov and Dec. USDA published no October 2025 forecast.

## Scale of the work
- **39 numbered trials (T01–T39)**, about 60 variants in all, each logged with its result, including every failure.
- **Before this track:** the Codex era ran **113 price-forecasting experiments**. All failed the 5% daily T+1 MAE target. They are kept in `research/registry.json` and `research/trials.json`.
- **Data pipelines built from public sources only:**
  - NASS QuickStats: county, state and US finals, plus monthly forecasts;
  - June Acreage reports;
  - FSA crop-acreage files, 2011–2024: 30k+ county rows, failed acres;
  - NASS Crop Progress condition;
  - Drought Monitor;
  - NASA POWER weather;
  - NASS Cotton Ginnings reports, 2009–2024, in two text layouts;
  - MODIS NDVI over prior-year CDL cotton pixels;
  - Landsat field features;
  - CropScape CDL for 2022 onward.
- **Cloud compute:**
  - 10 Kaggle CPU extraction kernels (5 MODIS tiles × dev years and the sealed years);
  - 2 Kaggle GPU training runs;
  - a resumable unit-of-work design: one file per tile-year or fold-seed, so nothing finished is ever recomputed after an interruption.

## The GPU histogram CNN (T35, and as a member in T36/T39), recorded in full
- **Idea.** The You et al. (2017) approach: instead of a few NDVI summary numbers, feed a CNN the *full distribution* of cotton-pixel reflectance through the season.
- **Built:**
  - `research/deep/county_hist.py`: per county, 8-day composites, 6 MODIS bands (NDVI, EVI, red, NIR, blue, MIR), 32 fixed bins each, cotton pixels only (prior-year CDL mask). It ran as 5 Kaggle CPU kernels over 2009–2021, about 70 minutes each, producing 65 tile-year files. It ran again for 2022–2024: 15 more files, and a CropScape download fix after remote reads failed.
  - `research/deep/build_tensor.py`: tensors of shape [6 bands × 22 slots × 32 bins] for 1 Oct and [6 × 14 × 32] for 1 Aug. That is 3,419 county-years for dev and 4,208 with the sealed years.
  - `research/deep/train_cnn.py`: a 3-block CNN, 5 seeds, rolling folds with validation year Y−1, trained on a Kaggle GPU. That is 45 (fold, seed) units per cutoff, plus 15 for the sealed years.
- **Result.** Pre-declared rule at 1 Oct: **FAIL**.
  - CNN 18.4 / 9.6, against HGB 10.3 / 6.6 and USDA 11.0 / 6.1.
  - Seeds disagreed by 4 TX points.
  - As an ensemble member (T36), the rolling weights gave it **0 in every 1 Oct year** and 0–0.05 in the sealed years. It got more weight only at 1 Aug, where the ensemble was worse.
- **Why (likely, untested).** About 1,500–2,400 training county-years per fold is too little for a CNN. The literature successes used far more data, or pretraining on other crops.
- **What remains useful:**
  - the extractor and tensors are a ready input for any later model, such as transfer learning from corn or soy;
  - the Kaggle pipeline is proven end to end;
  - the negative result itself: on this data size, distribution-level features do not beat simple NDVI anomalies.

## Everything else, grouped by what it taught us
| Group | Trials | Outcome | Lesson |
|---|---|---|---|
| State models: weather, NDVI | T01–T04 | worse than USDA by a lot | State-level signals are too coarse. |
| County gradient boosting on NDVI anomalies | T05–T07, T11 | best standalone model, still behind USDA | Area knowledge matters more than yield signal. |
| Blends with USDA | T08–T09 | about equal to USDA | |
| Revision-direction prediction | T10, T33 | weak or no skill | USDA's next revision is close to unpredictable from our inputs. |
| Landsat 30 m field features (Move 3) | T12–T13 | gate FAIL | One unstable US-August cell. |
| Correcting USDA's error with FSA, drought, condition, satellite | T14–T27 | first gains; a leak found and fixed in review | **Only the simple past-bias correction (T25) held up.** |
| County FSA failed acres, FSA-scaled area | T28–T29b | some cells beat USDA; timing review withdrew the claims | FSA file timing must be checked. Several Sep/Oct files were saved after USDA's report. |
| Ensembles of our models | T30a/b | below USDA | |
| Hybrid: USDA yield × our harvested area | T31–T32 | better harvested area, worse production | USDA's yield and area are internally consistent; mixing them breaks that. |
| NDVI sequence features | T34a | no gain | |
| Percentage ensemble around the correction | T36 | **dev PASS** (9.9 / 4.6) | Mostly T25 plus 5–10% HGB. |
| Nov/Dec ensemble; ginning pace | T37–T38 | FAIL against USDA December | USDA's December forecast (TX 4.7 / US 3.1) is very hard to beat before release. |
| **Sealed test of T36, 2022–2024** | T39 | **FAIL**: 1 Oct US 5.2 vs USDA 4.2 | Pooled bias correction breaks when USDA's bias reverses (US under-forecast in 2022 and 2024). |
| Texas-only FSA abandonment gate | T41 | **FAIL** against T25 | FSA failed acres do not explain TX error beyond the bias. |
| T25 one-shot holdout, 1 Sep 2025 | T42 | **FAIL**: TX −5.3 vs USDA −0.6; US −8.8 vs −4.3 | USDA under-forecast 2025; the bias reversed a third time (2022, 2024, 2025). |
| State-specific empirical-Bayes bias | T40 | **FAIL** on dev 2013–2024 | Too few years per state (USDA monthly history from 2012). Early folds were unshrunk, later ones shrank to about 0. |

## The one signal that survived everything
**USDA has kept overestimating Texas:**
- dev years, especially 2019–2021;
- every sealed year: +11.1%, +40.5% and +9.2% at 1 Oct in 2022–2024.

Our correction beat USDA in TX in **3 of 3 sealed years**.

After T39 we suspected the pooled application of the bias was the problem. **T40 tested a state-specific correction and failed**, so that diagnosis did not hold up. With about 12 years of USDA monthly history, a state's own bias cannot be estimated reliably, and the pooled T25 remains the better of the two on 2013–2024. The Texas pattern is real in the record but not yet a usable, validated rule.

Interview summary page (private Artifact): https://claude.ai/artifact/8tSPjQucn8pmwcL3dBwVJD

## Update (16 Nov, later): the result that holds up
- **USDA revision momentum** (T44, T45, full record):
  - A fixed, literature-motivated rule fitted only on earlier seasons.
  - Average error is below USDA's at every monthly report, 2002–2025, for TX, the US and production-weighted states (12 of 12 cells).
  - It is 3–11% better and does not win every year. 2002–2011 were tested blind.
  - Formal pre-declared passes: none. T44 needed Sep as well; T45 tied 3–3 in TX at Dec.
- **Landsat failed-field share (T43):** tracks Texas abandonment but missed 2023.
- **RMA crop insurance:** rejected for leakage.
- **2026 forward forecast:** locked in `docs/FORWARD_LOCK_2026.md` (commit 18c3bc7).
