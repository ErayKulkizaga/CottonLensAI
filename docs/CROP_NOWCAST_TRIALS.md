# Crop nowcast trial ledger

**Goal:** beat USDA's monthly forecast of cotton production, for Texas and for the US.
One row per trial. Nothing is deleted. Failed trials stay, so that related ideas can be combined later.

## How to read the numbers

- **Error metric:** mean absolute % error of production, in development years 2013–2021.
- **Protocol:** rolling origin. Each test year uses a model trained only on earlier years.
- **Pinned environment:** sklearn 1.6.1, pandas 2.2.3.
- **Sealed years:** 2022–2025 are not used here.

| Cutoff | USDA, TX / US | Naive base, June area, TX / US | Naive base, March area, TX / US |
|---|---|---|---|
| 1 Aug | **14.8 / 7.2** | 20.8 / 11.9 | 21.0 / 13.0 |
| 1 Sep | **11.2 / 6.3** | 20.8 / 11.9 | 21.0 / 13.0 |
| 1 Oct | **11.0 / 6.1** | 20.8 / 11.9 | 21.0 / 13.0 |

## Trials

Cells give TX / US error. **Bold** marks a cell that beats USDA.

| ID | Trial | Area | 1 Aug | 1 Sep | 1 Oct | Verdict | Evidence |
|---|---|---|---|---|---|---|---|
| T01 | State ridge, weather z-scores | March | 22.1 / 13.0 | 17.6 / 13.8 | 16.1 / 11.8 | worse than USDA | crop-nowcast-dev-ndvi-20261114 |
| T02 | State ridge, weather z + anomalies | March | 24.6 / 11.9 | 17.0 / 12.7 | 16.6 / 10.8 | worse | same |
| T03 | State ridge, MODIS NDVI levels | March | 27.5 / 15.2 | 24.0 / 13.3 | 22.6 / 12.6 | worse than base (overfits) | same |
| T04 | State ridge, weather z + NDVI | March | 24.1 / 15.2 | 21.9 / 14.1 | 20.2 / 13.2 | worse | same |
| T05 | County HGB, NDVI anomaly → state | March | 19.8 / 11.2 | 17.6 / 10.6 | 14.4 / 10.2 | best with March area | same |
| T06 | County HGB, NDVI anomaly → state | June | 18.9 / 9.9 | 15.1 / 7.9 | 12.7 / 7.7 | **best single model**; still above USDA | same |
| T07 | County HGB, weather z (+NDVI) | June | 20.9 / 11.1 | 16.7 / 8.2 | 14.7 / 8.5 | adding weather hurts | same |
| T08 | Blend of T05 and USDA, 50/50 (log-mean) | March | 16.0 / 9.0 | 13.6 / 7.9 | 12.1 / – | worse than USDA | session log 14 Nov |
| T09 | Blend of T05 and USDA, weights from past MSE | March | 15.2 / – | 11.3 / – | **10.8** / – | about equal to USDA (noise) | session log 14 Nov |
| T10 | Predict the direction of USDA's revision | – | – | corr +0.28, hit rate 57% (65% when the gap is above 10%) | corr +0.07 | weak | same |
| T11 | County-level accuracy vs naive base | June | −5.7% | +3.4% | +7.7% | weak at county level | session log 15 Nov |
| T12 | Move 3 primary: NDVI anomaly + Landsat field features | June | 19.0 / 7.4 | 20.6 / 9.0 | 17.0 / 7.6 | gate FAIL (≤16.0 needed) | move3-landsat-dev-20261115 |
| T13 | Landsat field features only (secondary) | June | 14.9 / **6.2** | 21.6 / 8.5 | 22.4 / 10.0 | US Aug beats USDA, but post hoc and unstable | same |

### USDA error correction (T14 onward)

Each trial corrects USDA's own forecast by predicting its error. The model is fitted on all states pooled, using only earlier years. The 1 Sep row uses USDA's September forecast and data up to 1 Sep; the 1 Oct row is the same for October.

> **SUPERSEDED (15 Nov).** Rows T14–T24 below had two defects, found in review:
> 1. **Leak.** USDA's implied abandonment used the *final* planted area, which was not public at the cutoff.
> 2. **Unscaled features.** The ridge penalty was applied to raw units, so small features such as `fsa_gap` were shrunk to almost zero, and T14 acted as a bias correction.
>
> The corrected rerun (June Acreage planted area, features standardised within each fold) is T25–T27, after this table. The claim "FSA beats USDA" in commit b63dd54 is withdrawn.

| ID | Correction signal | 1 Aug | 1 Sep | 1 Oct | Verdict |
|---|---|---|---|---|---|
| T14 | FSA failed-acre share minus USDA's implied abandonment | 15.4 / 8.5 | **9.3 / 3.9** | **10.3 / 4.9** | **beats USDA at Sep and Oct, TX and US** |
| T15 | Bias only (USDA's past mean error) | 15.6 / 8.1 | **10.4 / 5.4** | **10.8 / 5.9** | small gain |
| T16 | Satellite gap: log(T06 / USDA) | 15.1 / 7.2 | **10.7 / 4.9** | 11.0 / **5.0** | gain at US level |
| T17 | Drought Monitor D2–D4 share (last map and season mean) | 15.9 / 8.4 | 11.2 / **5.9** | **10.6** / 6.4 | weak |
| T18 | NASS cotton condition, poor or very poor share | 15.8 / 8.6 | 11.2 / 6.6 | 11.3 / 6.3 | nothing on its own |
| T19 | Satellite + FSA | 15.1 / 7.5 | **10.6 / 4.9** | **10.8 / 5.0** | |
| T20 | FSA + condition | 15.6 / 8.9 | **9.5 / 3.4** | **10.7 / 5.0** | best US result at 1 Sep |
| T21 | FSA + drought | 15.8 / 8.7 | **9.7 / 4.3** | **9.5 / 5.3** | best TX result at 1 Oct |
| T22 | FSA + drought + condition | 15.4 / 9.2 | **9.5 / 3.7** | **9.7 / 5.2** | |
| T23 | All signals | **14.6** / 7.6 | **10.7 / 4.4** | **9.7 / 5.2** | TX at 1 Aug: 14.6 vs 14.8, within noise |
| T24 | Ensemble: equal-weight log mean of T14, T19, T20, T21, T22 | 15.2 / 8.4 | **9.4 / 3.8** | **10.1 / 5.1** | consistent; not one lucky variant |

**Notes on T14–T24**

- About 30 variants were tried at each cutoff, so the best single cell is optimistic. T14, the simplest, beats USDA in all four Sep/Oct cells.
- **TX year by year (T14, 1 Sep):** better than USDA in 4 of 9 years. The gain comes from 2019–2021, when USDA overestimated TX by 12–29%. In 2016–2018 the correction is slightly worse.
- **US year by year (T14, 1 Sep):** better in 6 of 9 years.
- **FSA release-timing assumption (not verified):** the 'as of 1 Sep' file is assumed public before USDA's September report.
  - The stricter test does not need that assumption: our Sep-based T14 (TX 9.3 / US 3.9) also beats USDA's *October* forecast (11.0 / 6.1).
- **Missing data:** the FSA 2012 Oct file has an unknown layout and the 2012 Sep file is damaged on the server.
- FSA failed acres arrive late. In TX the August share was 23% in 2020, against a final abandonment of 54%. The signal is relative, not absolute.

### Corrected rerun (leak fixed, features standardised within each fold)

| ID | Trial | 1 Aug | 1 Sep | 1 Oct | Verdict |
|---|---|---|---|---|---|
| T25 | Bias only, fitted on the same rows as the FSA variant (needs at least 10 past rows; otherwise USDA is left unchanged) | 15.6 / 8.1 | **9.5 / 3.9** | **10.4 / 4.9** | **the consistent gain** |
| T26 | FSA gap (standardised) | 16.2 / 8.2 | **11.1 / 4.6** | **10.6 / 5.1** | no better than T25; FSA coefficient ≈ 0.007, intercept ≈ −0.068 |
| T27 | Ten variants: satellite, FSA, drought, condition and combinations | best 14.7 / 7.7 | mixed: 10.4–12.6 / 4.1–6.4 | mixed: 9.1–12.7 / 4.3–6.4 | no signal beats T25 consistently; T27 cells are noise-level |

**What T25 means**

- USDA's September and October cotton forecasts were biased upward in 2014–2021. A rolling, pooled correction by the mean past error reduces the error.
- **Year by year at 1 Sep:** better in 6 of 9 years for the US and 4 of 9 for TX. The gains come from 2017–2021, when USDA overestimated US production by 4–17%. The correction is worse in 2014 and 2016.
- **Depends on persistence.** The gain holds only if USDA's upward bias persists. If USDA under-forecasts in 2022–2025, the correction will hurt.
- **Not an earlier forecast.** This is a correction to USDA's published number, so it exists only after USDA's report.
- **1 Aug:** still not beaten by any signal (best 14.7 vs 14.8 for TX, noise-level).

### Back to beating USDA: county FSA, FSA area, ensembles (15 Nov, after the chain decision)

All rows use county HGB, rolling 2013–2021, the pinned environment, and 1 Aug / 1 Sep / 1 Oct cutoffs. Each cell gives TX / US error. **Bold** marks a cell that beats USDA.

| ID | Trial | 1 Aug | 1 Sep | 1 Oct | Note |
|---|---|---|---|---|---|
| T28 | NDVI anomaly + county FSA (failed share total, dryland and irrigated; irrigated share; anomaly), June area | 19.6 / **6.4** | 14.3 / 7.0 | 15.3 / 8.7 | Fragile: when rows with any missing FSA value are dropped, US Aug is 7.7 |
| T28b | County FSA only, June area | 21.2 / 8.2 | 17.8 / 8.9 | 17.3 / 8.2 | |
| — | Planted-area check: FSA in-season planted+failed acres × past NASS/FSA ratio, vs June Acreage | | | | Area error TX **1.2%** vs 3.9%; all states **4.0%** vs 9.2% (Aug files); Sep files TX 0.7%, all 2.0% |
| T29 | NDVI anomaly, **FSA-scaled area** | 20.0 / 8.5 | 14.7 / 6.3 | **10.6** / 6.7 | First standalone model (not a USDA correction) to beat USDA in TX at 1 Oct |
| T29b | NDVI anomaly + county FSA, FSA-scaled area | 19.7 / **6.0** | 14.6 / 6.7 | 13.4 / 7.5 | Standalone model beats USDA for the US at 1 Aug |
| T30a | Ensemble: log mean of T06, T13, T28 (June area) | 17.8 / 7.9 | 16.7 / 7.3 | 13.3 / 6.8 | |
| T30b | T30a blended 50/50 (log) with USDA | 14.9 / **6.8** | 13.1 / **6.2** | 11.4 / 6.1 | |

**CORRECTION (review, 15 Nov).** The notes "first standalone model to beat USDA" for T29 and T29b, and the same wording in the message of commit 62506ce, are withdrawn.

- **FSA availability**, checked by comparing each workbook's save time with the USDA report date:
  - **August files:** saved before USDA's August report in every test year 2013–2021. 2012 was saved after. A save time is not proof of publication.
  - **September and October files:** saved **after** the USDA report in 2013, 2014 and 2015, and in Oct 2018. Using them in those years is not a fair comparison, so the T29 result at 1 Oct for TX does not stand.
  - In any case, T29 at 10.6 does not beat the matched bias model T25 (10.4).
- **T29b at 1 Aug, US 6.0 vs 7.2:**
  - It is better than USDA in only 4 of 9 years. The gain comes from 2019 and 2020: model +1.2% and −3.8%, against USDA +13.0% and +24.6%.
  - The production-weighted state error is *worse* than USDA (15.6 vs 14.3), so the US result is cancellation across states.
  - With rows containing missing FSA values dropped, it is 7.2, equal to USDA. Hyperparameter changes give 5.9–6.2.
  - **Verdict:** not robust. It does show a real pattern: the models beat USDA in abandonment years and lose in normal years.
- **Area comparator:** the area check compares FSA with June Acreage. USDA's own in-season planted figures (the AUG/SEP ACREAGE rows) are sparse in our pull, so that comparison is not yet made.

**Selection warning.** About 50 variants have now been tried. The best single cells are optimistic, and none of these methods beats USDA in every cell. Before any sealed test, a single method has to be fixed in advance, and its dev-year numbers taken as an upper bound.

### T31: hybrid of USDA's yield and our harvested area (15 Nov)

**Idea.** USDA's same-month yield per harvested acre, multiplied by our harvested area. Harvested area = FSA-scaled planted area (August file) × county harvested/planted ratio from HGB, using NDVI anomaly + FSA from the August file only.

| Variant | 1 Aug TX / US | 1 Sep | 1 Oct | TX harvested-area error, ours vs USDA |
|---|---|---|---|---|
| T31 NDVI+FSA | 18.6 / 8.4 | 14.9 / 7.5 | 14.2 / 7.1 | Aug 12.0 vs **8.9**; Sep 10.0 vs **8.8**; Oct **7.2** vs 8.8 |
| T31 FSA only | 20.5 / 9.4 | 18.8 / 8.8 | 19.3 / 9.0 | worse |
| T31 NDVI only | 27.3 / 12.5 | 27.7 / 11.7 | 23.0 / 11.0 | much worse |

**Verdict: FAIL.** USDA's own harvested-area forecast is better than ours until October.

- In 2020, the August FSA file showed a 23% failed share, but final abandonment was 54%. Most abandonment happens after August, and nobody sees it in August, neither us nor USDA.
- 2016 is a large miss for every method: USDA −22%, ours −35%. That crop came in far above all forecasts.

### T32 (PRE-DECLARED 15 Nov, before running): late-season hybrid for the Dec 10 report

**Hypothesis.** By 1 Nov and 1 Dec, satellites have seen most abandonment. Our harvested area was already better than USDA's at 1 Oct (TX 7.2 vs 8.8). So the T31 NDVI+FSA specification, *unchanged*, should beat USDA's November and December production forecasts.

**Setup**
- **Cutoffs.** nov1 is compared with USDA's November forecast, dec1 with USDA's December forecast.
- **Satellite.** MODIS composites end in October, so the NDVI input is the same at nov1 and dec1. Weather runs up to each cutoff.
- **FSA.** The August file only, as in T31.
- **Variants.** Only this one: no other features, no tuning.

**Pass rule.** At both cutoffs, mean absolute % error is below USDA's for TX *and* for the US, *and* the production-weighted state error is below USDA's.

**T32 result (run once): FAIL.**

| Cutoff | TX ours / USDA | US ours / USDA | Production-weighted state error, ours / USDA | TX harvested-area error, ours / USDA |
|---|---|---|---|---|
| nov1 | 11.5 / **8.5** | 5.4 / **4.4** | 9.1 / **6.8** | **7.6** / 8.6 |
| dec1 | 7.8 / **4.7** | 4.6 / **3.1** | 6.8 / **4.5** | **7.7** / 8.6 |

- USDA's November and December forecasts are very accurate: 3–5% error at US and TX level by December.
- Our harvested area is slightly better than USDA's, but our production is worse. USDA's yield per harvested acre is internally consistent with its own harvested area, so combining it with our area breaks that consistency.

**T33 (exploratory, not pre-declared):** does our 1 Nov gap to USDA's November forecast, log(our model / USDA Nov), predict USDA's December revision?

- Over 143 state-years, the correlation is **−0.04** and the sign hit rate is **36%**, worse than chance. It is 34% when the gap exceeds 10%.
- The gap does not predict the final figure either: correlation 0.12, sign hit rate 49%.
- Our late model is biased low. Its TX gap was negative in 7 of 9 years.

**Verdict:** no ability to predict USDA's December revision.

### Reference refresh and T34a (15 Nov)

- **T06 reproduced** in the pinned environment: 18.9 at 1 Aug and 12.7 at 1 Oct (June area).
- **Same model with August-file FSA area (`fsa_aug`):** 1 Aug TX 20.0 / US 8.5; **1 Oct TX 10.3** / US 6.6. At 1 Oct, TX beats USDA (11.0) and T25 (10.4), while the US does not (USDA 6.1). This is a new cell and counts as one more variant.
- **T34a (exploratory):** per-16-day composite median and p90 NDVI, each as an anomaly against the county's past mean (season sequence), added to HGB with `fsa_aug` area.

  | Variant | 1 Aug TX / US | 1 Oct TX / US |
  |---|---|---|
  | Sequence only | 18.8 / **6.6** | 15.9 / 7.1 |
  | NDVI + sequence | 18.0 / **6.7** | 14.6 / 7.6 |

  The sequence helps at 1 Aug for the US and hurts at 1 Oct.

### T35 (PRE-DECLARED 15 Nov, before any extraction or training): GPU histogram CNN (You et al. 2017 style)

- **Input.** For each county-year: MOD13Q1 v061 histograms of cotton pixels (prior-year CDL mask, as before).
  - 6 bands: NDVI, EVI, red, NIR, blue and MIR reflectance.
  - 32 fixed bins per band, with ranges set from training years only.
  - One histogram per 16-day composite. A composite is available once its end date plus 8 days falls on or before the cutoff. Missing composites are zero-padded.
  - Tensor shape is [6 bands, T composites, 32 bins].
- **Target.** The same as T06: log(production per planted acre / the county's past mean).
- **Architecture (one, fixed).**
  - 3 conv blocks (Conv2d over composites × bins, with 32, 64 and 128 channels, kernel 3, BatchNorm, ReLU, dropout 0.25, stride 2 on the bin axis);
  - global average pooling;
  - a dense layer of 64 units;
  - 1 output.
  - Adam, learning rate 1e-3, weight decay 1e-4, batch size 64, 60 epochs, MSE loss.
- **Protocol.**
  - Rolling origin over test years 2013–2021.
  - The validation year is the last training year (Y−1). It is used only for early stopping, with patience 10. Inputs are normalised with statistics from training years only.
  - **5 seeds** (0–4). The result is the seed mean of the county predictions.
  - State aggregation, planted area (`fsa_aug`) and scoring use the same local code as T06. Kaggle only writes county predictions per (fold, seed).
- **Primary cutoff: 1 Oct.** 1 Aug is reported as secondary.
- **Pass rule at 1 Oct.** All of the following must hold:
  - TX < 11.0 and US < 6.1 (USDA);
  - TX < 10.4 and US < 4.9 (the T25 bias model);
  - production-weighted state error below USDA's;
  - better than the HGB reference (T06 `fsa_aug`: 10.3 / 6.6) on identical rows, by more than the seed spread (min–max of the 5 seed results).
- **Stop rule.** A second architecture, Prithvi, or any tuning needs the user's explicit word.
- **Seal.** Extraction refuses years 2022 and later.

**T35 result (15 Nov): FAIL on every part of the pass rule.**

Kaggle setup:
- CPU extraction: 5 tiles × 2009–2021, 65 files.
- Training on a Kaggle GPU (CUDA).
- Data: 263 panel counties, 3,419 county-years, 2,753 with a target.
- Scored locally with `score_t35.py`. The HGB reference was re-scored on identical rows and reproduces 10.3 / 6.6.

| Cutoff | CNN seed mean TX / US | HGB same rows | USDA | Seed spread TX / US |
|---|---|---|---|---|
| 1 Oct (primary) | 18.4 / 9.6 | 10.3 / 6.6 | 11.0 / 6.1 | 4.2 / 3.0 |
| 1 Aug (secondary) | 20.7 / 8.9 | 20.0 / 8.5 | 14.8 / 7.2 | 8.1 / 1.6 |

- The CNN only slightly beats the naive past-mean base (21.9 / 10.8) and is far worse than HGB.
- Seeds disagree strongly (TX 16.5–20.7 at 1 Oct). The likely cause is too few samples for a CNN: about 1,500–2,400 training rows per fold.
- **Deviations from the prereg, disclosed:**
  - bins use fixed physical ranges (VI −0.2..1.0, reflectance 0..0.6) instead of ranges fitted on training years. This puts no data into the binning;
  - MOD and MYD composites are interleaved into 8-day slots;
  - composites ending before 7 April are clipped into slot 0, so that slot sums two composites;
  - the last slot is always empty under the lag rule, a constant zero input. About 4.6% of (county-year, slot) cells are empty at 1 Oct, all of them in that slot.
- "Too few samples" is the likely cause but was **not tested**.
- **Provenance:**
  - Kaggle dataset `eraykulkzaga/cottonlens-t35-tensors`, kernel `eraykulkzaga/cottonlens-t35-cnn` (v1);
  - histogram kernels `cottonlens-hist-<tile>` (v1);
  - sha256 `tensor_oct1.npz` b292d358…a4127, `tensor_aug1.npz` 88515ba5…676eb, `train_cnn.py` d31313f4…4ab10;
  - local copies in `CottonLensAI/output/deep/`.
- Evidence: `t35-cnn-oct1-20261115.json`, `t35-cnn-aug1-20261115.json`.
- Per the stop rule, there is no tuning and no second architecture without the user's word.

### T36 (PRE-DECLARED 15 Nov, before running): rolling percentage ensemble around the USDA correction

- **Members, per state-year:**
  - B = T25 (USDA bias-corrected, the same rows as the FSA variant);
  - H = T06 HGB with `fsa_aug` area;
  - N = T35 CNN seed mean (1 Aug and 1 Oct only);
  - U = USDA's raw forecast for the month.
- **Ensemble.** exp(Σ wₖ log memberₖ), with weights wₖ ≥ 0 that sum to 1, on a 0.05 simplex grid.
- **Weights for year Y** minimise the pooled squared log error over states × years 2013…Y−1. They use only the members' out-of-sample predictions from earlier folds.
  - With fewer than 2 past years (2013 and 2014), the ensemble is B.
  - A state with any member missing uses B for that state.
- **Aggregation and scoring** are the same as T25: US = USDA_US × Σ ensemble / Σ USDA over the states. MAPE for TX and US, plus the production-weighted state error.
- **Primary: 1 Oct, members B + H + N + U.** Secondary: 1 Sep (B + H + U) and 1 Aug (B + H + N + U).
- **Pass rule at 1 Oct.** All of the following must hold:
  - beat USDA (11.0 / 6.1) for both TX and US;
  - beat T25 on the same rows, for both TX and US;
  - production-weighted state error below USDA's and below T25's.
- One variant only. No other member sets are reported as results.

**T36 result (15 Nov): PASS at 1 Oct (primary). The margin over T25 is small.**

| Cutoff | T36 TX / US | T25 same rows | USDA | Prod-wtd state: T36 / T25 / USDA | Rule |
|---|---|---|---|---|---|
| 1 Oct (B+H+N+U) | **9.9 / 4.6** | 10.4 / 4.9 | 11.0 / 6.1 | 9.0 / 9.3 / 9.4 | **PASS** |
| 1 Sep (B+H+U) | **9.4 / 3.8** | 9.5 / 3.9 | 11.2 / 6.3 | 9.6 / 9.7 / 10.3 | passes (secondary) |
| 1 Aug (B+H+N+U) | 16.6 / 8.0 | 15.6 / 8.1 | 14.8 / 7.2 | 14.4 / 14.7 / 14.4 | fails |

- **Weights at 1 Oct.** Mostly B: 0.9–0.95 from 2018 on, with H at 0.05–0.10. **The CNN got weight 0 in every year**, so it adds nothing.
- **Years 2014–2016 equal T25.** USDA published no October 2013 forecast (the October 2013 report was cancelled), so 1 Oct is scored on 8 years (2014–2021). That leaves 2014 and 2015 with fewer than 2 past years, so the fallback covered 2015 as well, not only 2013–2014 as the prereg named. 2016 chose B = 1.0.
- **TX, year by year vs T25:** better in 5 of 5 years where they differ (2017–2021), by 0.3–1.7 percentage points. Never worse.
- **US, year by year vs T25 (1 Oct):** better in 4 years and worse in 1 (2018: 4.6 vs 4.3). At 1 Sep: 5 better, 2 worse.
- **Same caveat as T25.** The gain comes from pulling USDA's 2017–2021 overestimates down further, with H acting as a second, independent "lower" signal. If USDA's bias reverses in the sealed years, both B and H weights will hurt.
- **1 Aug** is still not beaten. There, past fits put weight on N and U, and the result is worse than T25.
- Evidence: `t36-stack-20261115.json`; code: `research/crop_nowcast/stack.py`.

### T37 (PRE-DECLARED 15 Nov, before running): the T36 ensemble at Nov/Dec, the 10 Dec target

- **Primary ("before the 10 Dec report").** Uses only information public by 1 Dec. The members are:
  - U = USDA's November forecast;
  - B = U corrected by the pooled past-mean bias of USDA's November forecast (rolling, the T15/T25 idea without FSA rows, since there are no FSA Nov/Dec files);
  - H = T06 HGB at the 1 Dec cutoff with `fsa_aug` area.
- **Weights.** Same T36 rule: a 0.05 simplex, past years only, B when fewer than 2 past years.
- **Benchmark:** USDA's **December** forecast, published 10 Dec. US = USDA_US Nov × Σ ensemble / Σ USDA Nov.
- **Pass rule (primary).** All of the following must hold:
  - TX and US MAPE below USDA's December forecast;
  - production-weighted state error below USDA December's.
- **Secondary (reported, not a claim):**
  - the 1 Nov cutoff (H at nov1) against USDA November;
  - a post-report December correction (U, B and H on USDA December) against USDA December.
- **Prior expectation (written before running).** T32 failed and T33 found no skill on the Nov→Dec revision, so passing is unlikely.

**T37 result (15 Nov): FAIL (primary), as expected. The secondary rows show no gain either.**

| Variant | Ours TX / US | Benchmark TX / US | Raw USDA of the input month | Prod-wtd state, ours / benchmark |
|---|---|---|---|---|
| **Primary:** USDA Nov + HGB dec1 vs USDA Dec | 8.3 / 4.6 | **4.7 / 3.1** | 8.5 / 4.4 | 6.8 / 4.6 |
| Secondary: 1 Nov vs USDA Nov | 8.5 / 4.8 | 8.5 / 4.4 | — | 6.9 / 6.8 |
| Secondary: post-report Dec correction vs USDA Dec | 4.5 / 3.3 | 4.7 / 3.1 | — | 4.7 / 4.6 |

- **Weights go to raw USDA** (U = 0.85–1.0 in 2015–2020). The past-mean bias of USDA's Nov/Dec forecasts is close to zero, so B adds nothing, and HGB gets 0–0.15.
- What USDA learns between November and December (mostly ginning data) is not in any of our inputs. Nothing here beats USDA's 10 Dec number before its release.
- **Implication for the 10 Dec 2026 forward forecast:** we have no validated edge over USDA's December figure. A locked forecast would be USDA November plus a near-zero correction, which is not a claim worth making as "beating USDA".
- Evidence: `t37-late-stack-20261115.json`.

### T38 (PRE-DECLARED 15 Nov, before pulling or running): ginning pace corrects USDA's November forecast

- **Data.** NASS *Cotton Ginnings* reports from ESMIS. For each year 2009–2021, use that year's report "as of November 15", taking all-cotton running bales by state for the current crop as first published.
  - Use it only if the report was released before that year's December Crop Production report.
  - (D) values are treated as missing.
- **Feature.** pace = log(ginned by 15 Nov / USDA November state forecast). Also pace_a = pace minus the state's mean pace over earlier years only.
- **Model.** Rolling pooled ridge on e = log(final / USDA Nov), with feature pace_a, standardised within each fold. This is `usda_correction.fit_predict`, with USDA left unchanged when fewer than 10 past rows exist.
  - Corrected = USDA Nov × exp(ê).
  - US = USDA_US Nov × Σ corrected / Σ USDA Nov (the same as T25).
- **Primary pass rule (the 10 Dec target).** All of the following must hold:
  - TX and US MAPE below USDA's December forecast;
  - production-weighted state error below USDA December's.
- **Secondary (reported, not a claim):**
  - against USDA November (does pace improve the November forecast?);
  - correlation of ê with USDA's actual Nov→Dec revision;
  - T36-style stack of B, H, U and G (this correction) against USDA December.
- One feature set only. No other ginning dates or features are reported as results.

**T38 result (15 Nov): FAIL (primary). Ginning pace improves USDA's November forecast, but not enough to reach USDA's December one.**

Data:
- `gin_pull.py` parsed 13 reports as of 15 November (2009–2021), with 13–17 states each.
- All were released 22–25 Nov, before that year's December Crop Production report.
- 2009 uses an older layout; the parser handles both and tests cover both.

| Variant | Ours TX / US | Benchmark TX / US | Prod-wtd state, ours / benchmark |
|---|---|---|---|
| **Primary:** G (USDA Nov + pace correction) vs USDA Dec | 8.1 / 4.0 | **4.7 / 3.1** | 6.8 / 4.6 |
| Secondary: G vs USDA Nov | **8.1 / 4.0** | 8.5 / 4.4 | 6.8 / 6.8 |
| Secondary: stack of B, H, U, G vs USDA Dec | 7.3 / 3.9 | **4.7 / 3.1** | 6.5 / 4.6 |

- **Improvement over USDA November.** The pace correction moves TX from 8.5 to 8.1 and the US from 4.4 to 4.0. Year by year this is **not robust**:
  - US is better in 4 years and worse in 4, with 1 tie. The mean gain comes from 2019–2021.
  - TX is better in 5 of the 8 years where they differ.
  - The state error is unchanged.
- **The stack uses the ginning signal.** G's weight rises to 0.45–0.65 in 2018–2021, and the stack beats USDA November more (7.3 / 3.9). It is still far from USDA December.
- **Correlation of ê with USDA's actual Nov→Dec revision:** +0.09 over 153 state-years. That is weak.
- **Why it misses.** The large Nov→Dec errors are TX 2016 (−17.7 vs USDA Dec −8.6) and TX 2020 (+30.2 vs +13.8). These are abandonment and late-harvest years, where by 15 November ginning pace does not show what USDA learns by early December.
- **10 Dec implication.** Still no validated edge over USDA's December figure. The one honest pre-report claim we could make is "better than USDA November" (secondary, small, about 9 years).
- Evidence: `t38-ginnings-20261115.json`; data in `output/crop-nowcast/gin/` (raw ESMIS text plus `ginnings.csv`).

### External landscape (research, 15 Nov)

- **Descartes Labs** claimed to beat USDA's August *corn* yield forecast in 2015, by about 1 percentage point. The figures are company-reported and cover one season.
- **No published work** was found that predicts cotton abandonment and compares it with NASS.
- Related work:
  - a Texas High Plains Sentinel-2 field-yield study;
  - commercial county cotton yields from Barchart and EOS;
  - a USDA-ARS 17-state cotton ML project, with no published results yet.
- **CY-Bench** (WUR, open source) is a county yield benchmark for maize and wheat only. It is useful for methods and evaluation, not for data.
- **To check for the GPU track:**
  - the CropNet dataset (Sentinel-2 + HRRR + USDA county yields, including cotton);
  - the pycrop-yield-prediction (You et al. 2017) code.

  Neither has been verified yet.

## Chain: Link 1, report-day price reaction (preregistered; run once on 15 Nov)

Frozen preregistration: `docs/LINK1_EVENT_STUDY_PREREG_20261115.md` (LF sha256 16e2b440…). Code: commit 863f195. Result: `research/evidence/link1-event-study-20261115.json`.

- **Events:** 38 reports, Sep–Dec of 2012–2021. Oct and Nov 2013 are excluded because there was no previous forecast (shutdown).
- **Price:** `cotton` daily close series, 2021 and earlier only.
- **Deviation from the preregistration:** the NASS series used is upland production (about 97% of all cotton), not all-cotton.

| Test | Result | Bar | Pass |
|---|---|---|---|
| H1: mean \|R\| on report days vs other days | **1.42×**, p = 0.0038 | p < 0.025 | **yes** |
| H2: Spearman(revision, return) < 0 | −0.232, p = 0.076 | p < 0.025 | no |
| Oracle gate: R̂ = β·revision vs naive, event-day MAE | −3.8% (32 events) | ≥ +5% | no |
| Sensitivity without the 3 days with \|R\| ≥ 4% (Sep 2017, Sep 2019, Dec 2021) | H1 1.15×, p = 0.16; H2 −0.137 | | no |

**Pre-declared decision:** the price reacts, but the month-to-month revision is the wrong measure of surprise, so Link 3 (consensus) is needed.

- The market appears to anticipate most revisions. In Nov 2018 the revision was −7.4% and the return +0.06%.
- The reaction-size result rests on a few large days.

## Chain: Link 3, analyst consensus archive (time-boxed check, 15 Nov)

- **Rule:** find the pre-report average trade estimate of US production for two past reports, in two independent public sources. Continue only if one source covers most of the 38 events.
- **Sep 2019:** found. A Bloomberg survey expected 21.8 M bales; USDA reported 21.86 M (bloombergquint article).
- **Nov 2018:** not found in any public source.
- Reuters cotton polls appear only scattered in republications (Business Recorder), and most cover exports and stocks rather than production.
- **Result: UNAVAILABLE from public sources.** A paid archive (Bloomberg or Reuters terminal) would be needed. That is the user's call.

## Findings to combine later

1. County NDVI anomaly is the strongest single signal. Its correlation with production per planted acre at 1 Oct is +0.52 (+0.62 in Texas).
2. Satellite captures crop abandonment. On 1 Oct, the per-acre error in TX was about 0 in 2019 and 2020, while USDA's August forecast was +33% and +36%.
3. Errors of the satellite model and USDA correlate at 0.5–0.8. Blending helps only a little (T09).
4. Landsat field features help in August at US level (T13) but degrade later in the season. A likely cause is the varying number of observations; 2012 has only about half the scenes.
5. Area is a large error source. March intentions add about 6% to TX error; the June Acreage report removes part of it.

## Queue (next attempts)

- **Done:** T14 (FSA), T17 (Drought Monitor) and T18 (cotton condition); results are in the table above.
- **Next for 1 Aug:** no signal has beaten USDA yet. Ideas not yet tried:
  - county-level FSA failed share combined with the satellite model;
  - irrigated versus dryland split of FSA failures;
  - planted area from FSA in place of the June Acreage figure.
- **Sealed test candidate:** T25 (bias correction) at 1 Sep and 1 Oct. It needs only USDA's monthly forecasts and final production for 2022–2025.
  - Risk: it wins only if USDA's upward bias persists. It is one shot.
  - It needs the user's explicit yes and a pushed preregistration.
- **Unified goal (approved 15 Nov).** All tracks are now one chain; see [WORK_PLAN_20261115.md](WORK_PLAN_20261115.md). The P items below became Links 1–3.
- **Already tested in the Codex era (P00, not to be repeated).** These used WASDE regional numbers (2019–2023) and Texas condition (2016–2023) as features for *every-day* T+1 price MAE. All stayed below the 5% target, and the weak-signal check showed that shrinkage hides small signals. Report-day behaviour and the revision-to-price sign were never tested. Sources: WASDE_REGIONAL_T1_RESULT, NASS_REGIONAL_T1_RESULT, WEAK_SIGNAL_RESULT, and the two CLOCK_CASE documents.
- **Price track** (second priority; the user's request, relayed on 15 Nov). The idea is "forecast the forecasts": price reacts to the report surprise.
  - **P01** Price after USDA reports. On WASDE and Crop Production days, and the 1–5 days after, does cotton over- or under-react (drift)? Needs only the existing CT history and report dates. Cheapest, so it runs first.
  - **P02** Surprise prediction. Does the gap between our USDA-correction model and the analyst consensus predict USDA's revision and the price move on report day? Needs a pre-report poll archive, not verified to exist.
  - **P03** CFTC speculator crowding as a risk feature, for reversal or big-move risk. In the Codex era it was tested only on price level, never on risk.
  - Context: the cotcast demo of 6 Nov was compared with CTZ26 closes for 9–13 Nov. MAE was 1.26 cents against 0.44 for naive. This is one path and is not evidence (checked by a parallel session).
- Rejected for now:
  - Ginnings: none before 1 Oct.
  - SMAP soil moisture: only from 2015.
  - Sentinel-2: dense only from 2019.
  - RMA cause of loss: revised after the season, so it leaks.
  (Source: check by a parallel session, 15 Nov.)

### T39 (16 Nov): one-shot sealed test of T36 on 2022–2024: **FAIL**

- **1 Oct:** T36 scored TX 15.0 / US 5.2, against T25 13.7 / 5.9 and USDA 20.3 / 4.2.
- **US fails** because USDA under-forecast the US in 2022 and 2024, and the bias correction pushed further down.
- **TX: T25 and T36 beat USDA in 3 of 3 sealed years.** This was not the declared claim; it is a hypothesis for a new forward test.
- Details: `docs/SEALED_TEST_T39_PREREG_20261115.md`; evidence `t39-sealed-test-20261116.json`.
- The seal is now spent for production finals 2022–2024. Prices and returns stay sealed.

### T40 (PRE-DECLARED 16 Nov, before running): state-specific, shrunk USDA bias correction (options A + C)

Motivation (T39): one pooled bias for all states hurt the US total when USDA's bias reversed. Texas's bias persisted.

- **Model.** For state s and year Y, from USDA month-m errors e = log(final / USDA_m) over years 2008…Y−1 of that state:
  - b_s = m_s · τ² / (τ² + σ² / n_s), an empirical-Bayes shrinkage toward **zero** (no pooled bias);
  - m_s = the state's past mean error and n_s its number of past years;
  - σ² = the mean within-state variance;
  - τ² = max(mean(m_s²) − mean(σ² / n_s), 0).
  - Corrected = USDA_m × exp(b_s).
  - US = USDA_US × Σ corrected / Σ USDA (the same as T25).
  - No tuned constants: shrinkage comes from the data in each fold.
- **Development years: 2013–2024** (12 years; 2022–2024 were opened by T39 and are now development data).
  - Cutoffs Aug, Sep, Oct, Nov, Dec, each against USDA's same-month forecast.
  - T25 is reported on the same rows.
- **Dev pass rule at 1 Sep and 1 Oct.** All of the following must hold:
  - TX and US MAPE below USDA;
  - TX and US MAPE below T25;
  - production-weighted state error below both.
- **Holdout: 2025, opened once, only if the dev rule passes.**
  - **Primary 2025 cell: 1 Sep** (there is no USDA October 2025 forecast). It passes if TX and US absolute errors are below USDA September's.
  - 1 Aug, 1 Nov and 1 Dec 2025 are reported as secondaries.
  - One year is a single draw, so it is reported as evidence, not proof.
- **Forward.** If the holdout is not failed, the same model is locked for the 2026 December report (USDA Dec 2026, published 10 Dec). It is scored when 2026 finals are published (May 2027).

**T40 result (16 Nov): FAIL on dev 2013–2024. The 2025 holdout is NOT opened and stays sealed.**

| Cutoff | T40 TX / US | T25 TX / US | USDA TX / US | Prod-wtd state, T40 / T25 / USDA |
|---|---|---|---|---|
| 1 Aug | 19.0 / 8.2 | 17.2 / 8.6 | 17.1 / 8.3 | 15.7 / 15.1 / 14.6 |
| 1 Sep | 14.0 / 5.6 | **11.4 / 4.4** | 13.5 / 5.8 | 11.0 / 10.2 / 10.5 |
| 1 Oct | 13.9 / 5.6 | **11.3 / 5.2** | 13.5 / 5.6 | 10.8 / 10.1 / 9.9 |
| 1 Nov | 10.1 / 4.2 | 8.9 / 4.5 | 10.1 / 4.4 | 7.5 / 7.2 / 7.2 |
| 1 Dec | 6.4 / 3.1 | 5.1 / 3.2 | 5.7 / 3.0 | 5.4 / 4.9 / 4.9 |

- **Why it fails (corrected after review).** USDA's monthly forecasts in our NASS pull start in **2012**, so each state has only 1–12 past years.
  - In the earliest folds (1 Sep 2013, 1 Oct 2013 and 2014) n_s = 1, the within-state variance is undefined, and the model applies the **unshrunk** single-year mean.
  - Example: TX at 1 Oct 2014 was moved by about −18% when USDA was off by +1.2%.
  - In later folds the shrinkage drives TX's bias to −0.00 to −0.05 (log), because TX's year-to-year error (2023: +40%) swamps a short state mean.
  - The verdict stands. Rescoring without the thin folds would be a post-hoc diagnostic, not a result.
- **Side finding on 12 dev years** (2022–2024 now opened): pooled T25 still beats USDA at 1 Sep (TX 11.4 vs 13.5, US 4.4 vs 5.8) and at 1 Oct (11.3 vs 13.5, 5.2 vs 5.6).
  - Its production-weighted state error at 1 Oct is slightly worse (10.1 vs 9.9).
  - This is not a new claim: T25 was chosen on 2013–2021 and the T39 rule judged T36, not T25.
- Evidence: `t40-state-bias-dev-20261116.json`; code: `research/crop_nowcast/state_bias.py`.

### T41 (PRE-DECLARED 16 Nov, before running): cheap gate for option B. Does FSA failed acreage explain Texas's USDA error beyond T25?

- **Data.** TX rows only, at 1 Sep and 1 Oct. Target e = log(final / USDA_m). Feature: `fsa_gap` (FSA failed share minus USDA-implied abandonment, from `UC.build`), standardised within each fold.
- **Model.** Rolling ridge (λ = 1, intercept unpenalised) on TX years < Y. With fewer than 5 past TX rows, it falls back to T25. Years 2013–2024. 2025 is **not** opened.
- **Pass.** At both cutoffs, TX MAPE below T25 and better than T25 in more years than worse. Otherwise option B stops here and is logged.
- **Why this is enough.**
  - 2025/2026 FSA files are not posted, so a full B model could not be scored on 2025 or used for 10 Dec 2026.
  - The pooled versions of these signals (T17, T18, T26, T27, T31) added nothing over T25.

**T41 result (16 Nov): FAIL. Option B stops here.**
- **1 Sep:** TX 14.1 vs T25 11.4 (USDA 13.5). Better than T25 in 2 years, worse in 5.
- **1 Oct:** TX 11.1 vs T25 11.3 (USDA 13.5). Better in 3, worse in 3.
- Texas-only FSA abandonment does not explain USDA's TX error beyond the simple bias. Evidence: `t41-tx-fsa-gate-20261116.json`.

### T42 (PRE-DECLARED 16 Nov, before reading any 2025 value): one-shot 2025 holdout of T25, unchanged

On 16 Nov the user approved opening 2025 ("Evet, 2025'i aç"). 2025 values were pulled with T39 but never printed or read.

- **Model: T25 exactly.**
  - For year Y, the pooled mean of e = log(final / USDA_m) over all states and years < Y.
  - At 1 Aug and 1 Sep, training rows are those with `fsa_gap` (as in T25). At 1 Nov and 1 Dec, all rows (as in T37's B).
  - Corrected = USDA_m × exp(mean e).
  - US = USDA_US × Σ corrected / Σ USDA over the states.
- **Primary: 1 Sep 2025.** USDA published no October 2025 forecast.
  - **Pass:** TX absolute error below USDA September 2025's, **and** US absolute error below USDA September 2025's.
- **Secondaries (reported, not claimed):** 1 Aug, 1 Nov and 1 Dec 2025 (Nov/Dec against USDA's same month), plus the production-weighted state error.
- **Limits.** One year is one draw: a pass is evidence, a loss is informative. 2025 is the last unopened year, and this is its only use for T25.

**T42 result (16 Nov, run once): FAIL.**

| 2025 cell | T25 TX | USDA TX | T25 US | USDA US | Prod-wtd state, T25 / USDA |
|---|---|---|---|---|---|
| **1 Sep (primary)** | −5.3% | **−0.6%** | −8.8% | **−4.3%** | 12.2 / 9.0 |
| 1 Aug | −7.0% | **−2.5%** | −9.2% | **−4.8%** | 12.1 / 10.1 |
| 1 Nov | −1.7% | **+1.3%** | **−1.3%** | +1.8% | 9.1 / 8.7 |
| 1 Dec | −6.5% | **−4.4%** | **+0.7%** | +2.9% | 11.0 / 10.7 |

- USDA *under*-forecast 2025, and the downward correction (log −0.05 at Aug/Sep) made it worse. **USDA's US bias has now reversed in 2022, 2024 and 2025.** The past-mean bias correction does not hold forward.
- All unopened production years are now used. Every model claim from here is in-sample, unless it is tested forward (2026+).
- Evidence: `t42-holdout-2025-20261116.json` / `.log`.

### Data-source screening for new trials (16 Nov, no model run)

- **RMA Cause of Loss (crop insurance), REJECTED for in-season use: leakage.**
  - The files (`colsom_YYYY.zip`, layout "as of 5/28/2026") give the *month the loss occurred*. They give no report or publication date.
  - The files are regenerated as claims settle, often after harvest. Using losses dated ≤ the cutoff month would include claims nobody knew about at the cutoff.
  - Usable only for after-the-fact analysis, or with archived vintages (none found).
- **AMS "Quality of Cotton Classed by Classing Office" (weekly, ESMIS, dated releases), leak-free but SHORT.**
  - The ESMIS archive starts in late 2018: about 7 seasons (2019–2025).
  - Bales classed season-to-date tracks ginnings closely, and the ginning-pace trial (T38) failed against USDA December. It is a weak candidate on its own.

### T43 (PRE-DECLARED 16 Nov): satellite failed-field share corrects USDA's Texas forecast

**Disclosure (written before the trial).** Before writing this I looked at one exploratory table on 2010–2021.
- The TX planted-weighted Move 3 Landsat `l_fail35` (share of prior-year-CDL cotton points whose NDVI stays below 0.35) correlates with:
  - actual TX abandonment: +0.65 to +0.84;
  - USDA's TX abandonment error: −0.43 to −0.75;
  - USDA's TX production error: −0.04 to −0.46 at 1 Oct.
- That look motivated this trial. No model was fitted on it.
- I have also seen USDA's 2022–2025 TX errors (T39, T42). The 2022–2025 test below is out-of-sample for **model fitting**, but not blind to outcomes. It must be read with that in mind.

**Model (one spec).**
- x = TX state `l_fail35` at the cutoff: county values from `landsat_fields.py` (unchanged), weighted by prior-year planted acres over TX panel counties.
- Rolling ridge, TX rows only: e = log(final / USDA_m) on [1, standardised x], λ = 1, intercept unpenalised, trained on years 2010…Y−1. With fewer than 5 rows it falls back to T25.
- Corrected TX = USDA_m × exp(ê).

**Years and cutoffs.**
- Dev: 2013–2021, existing Landsat features.
- Test: 2022–2025. Landsat for TX is extracted with the same script (platforms Landsat 5/7/8 unchanged; prior-year CDL from CropScape for 2022 onward, as in T39).
- 1 Oct is scored on 2022–2024 (there is no Oct 2025 USDA forecast). 1 Sep is scored on 2022–2025. 1 Aug is secondary.

**Pass rule (test years, 1 Sep and 1 Oct each).** TX MAPE below USDA's and below T25's, and better than USDA in more years than worse.

**Scope.** Texas only. A US effect enters only through TX's share of production and is reported, not claimed.

**T43 dev scoring (16 Nov, recorded while the 2022–2025 Landsat extraction runs; the test part is still pending).**

On dev 2013–2021, TX MAPE was worse than USDA and T25 at every cutoff:

| Cutoff | T43 | T25 | USDA | vs USDA, better / worse years |
|---|---|---|---|---|
| 1 Aug | 18.1 | 15.6 | 14.8 | 2 / 7 |
| 1 Sep | 12.1 | 9.5 | 11.2 | 3 / 5 |
| 1 Oct | 13.2 | 10.4 | 11.0 | 1 / 6 |

- The failed-field share tracks abandonment, but a TX-only regression of production error on it, fitted on 5–11 past years, does not improve USDA.
- The pre-declared test (2022–2025) will still be run once and reported as is.

**T43 test result (16 Nov, run once): FAIL.**

| Test, TX MAPE | T43 | T25 | USDA | vs USDA, better / worse |
|---|---|---|---|---|
| 1 Sep, 2022–2025 | 16.7 | 14.3 | 15.5 | 2 / 2 |
| 1 Oct, 2022–2024 | 15.0 | 13.7 | 20.3 | 3 / 0 (but worse than T25) |

- **What the satellite saw.** The TX failed-field share was 0.68 in 2022, a severe drought year with record abandonment, which is correct. But it was only 0.22–0.25 in **2023**, the year USDA missed by +40%.
  - So in 2023 the abandonment that USDA missed was not visible as non-greening fields by 1 Sep or 1 Oct.
  - The regression also over-corrected 2022 at 1 Oct (−10.4% vs T25 +4.1%).
- **Data notes:**
  - 2022 skipped 3 unreadable Landsat 7 scenes (recorded in `landsat_TX_2022.json`).
  - 2024–2025 have about half the observations: Landsat 7 was retired, and Landsat 9 was excluded to keep the declared spec.
- Evidence: `t43-tx-fail-20261116.json` / `.log`.

### T44 (PRE-DECLARED 16 Nov, before running): USDA revision momentum

**Hypothesis (from the literature, not from our data).** USDA crop forecasts are "smoothed": successive revisions are positively correlated (Isengildina, Irwin and Good on USDA corn/soybean forecasts). So the latest revision predicts the direction of the remaining error.

**Model (one spec).**
- For month m with the previous published USDA month p (Sep←Aug, Oct←Sep, Nov←Oct, or Sep when there is no Oct, Dec←Nov):
  - r = log(USDA_m / USDA_p);
  - e = log(final / USDA_m).
- Rolling ridge pooled over states, **no intercept** (so no bias term, which failed in T39 and T42): e = b·z(r), λ = 1, trained on state-years 2012…Y−1. With fewer than 10 rows, USDA is unchanged.
- Corrected = USDA_m × exp(ê). US = USDA_US × Σ corrected / Σ USDA_m.
- Everything used is published before USDA's month-m report: both forecasts are public at m's release.

**Years.** 2014–2025 (12 seasons; Oct has no 2013 or 2025 report).

**Pass rule.**
- **Primary: 1 Dec** (USDA December, the 10 Dec report) **and 1 Sep.**
- At both cutoffs: TX and US MAPE below USDA's; better than USDA in more years than worse (TX and US separately); production-weighted state error below USDA's.
- Oct and Nov are secondary.

**Disclosure.** Every 2014–2025 outcome has been seen in earlier trials, so this is a rolling backtest of a literature-motivated single spec, not a blind holdout. A pass makes it the candidate for a locked forward forecast of the 2026 crop, built from USDA Nov→Dec and scored when 2026 finals are published.

**T44 result (16 Nov, run once): primary FAIL (Sep fails). The December cell passes on its own, narrowly.**

| Cutoff | TX: momentum vs USDA (better / worse years) | US (better / worse) | Prod-wtd state | Rule |
|---|---|---|---|---|
| **1 Dec** | **5.5 vs 5.9** (6 / 4) | **3.0 vs 3.2** (6 / 4) | 5.39 vs 5.42 | passes |
| **1 Sep** | 14.1 vs 13.4 (2 / 8) | 6.5 vs 6.2 (4 / 8) | 11.1 vs 10.5 | fails |
| 1 Oct | 13.5 vs 13.5 (7 / 4) | 5.6 vs 5.6 (4 / 3) | 10.0 vs 9.9 | secondary |
| 1 Nov | 9.4 vs 10.1 (6 / 4) | 4.6 vs 4.4 (3 / 7) | 7.4 vs 7.5 | secondary |

- **December.** USDA's Nov→Dec revision continues in the same direction toward the final (smoothing).
  - The gain is small (0.4 TX, 0.2 US points) and comes mostly from TX 2020 and 2023.
  - 2014, a thin early fold, is worse.
- **September.** The Aug→Sep revision does *not* carry forward; it tends to over-shoot.
- **Status.** The pre-declared primary (Dec **and** Sep) failed. The December cell is a hypothesis with a rolling-backtest edge, not a validated result. It is the natural candidate for a forward test on the 2026 crop.
- Evidence: `t44-momentum-20261116.json` / `.log`.

### T45 (PRE-DECLARED 16 Nov, before parsing any 2000–2011 forecast): blind test in 2000–2011

USDA's monthly cotton forecasts before 2012 were never pulled or looked at in this project. Their errors are unseen, so this is a genuinely blind, time-ordered test.

**Data.**
- Upland cotton production forecasts by state and US: the last column of the "Upland" block of the table "Cotton: Area Harvested, Yield, and Production by Type, State, and United States", in ESMIS Crop Production text reports, Aug–Dec, seasons 2000–2011.
- Finals: NASS QuickStats upland `YEAR` production, pulled with the same `nass_pull.py`.
- Every forecast used is the one in that month's report, which is public at its release.

**Primary.** T44's December rule, unchanged: pooled no-intercept ridge e = b·z(r), r = log(Dec / Nov), λ = 1, ≥ 10 training rows.
- Trained rolling **only on seasons 2000…Y−1** (no information from 2012 onward).
- Scored on 2002–2011.
- Pass rule as in T44's December cell: TX and US MAPE below USDA December's, better in more years than worse (TX and US), production-weighted state error below USDA's.

**Secondaries (reported, not claimed):**
- momentum at Sep, Oct, Nov;
- the pooled past-mean bias (T25 without FSA rows, which do not exist then) at Sep and Oct, rolling within 2000–2011.

Run once, committed as produced.

**T45 result (16 Nov, run once, blind 2002–2011): primary FAIL on a Texas tie. The blind secondaries show momentum working at Sep, Oct and Nov.**

Data: 963 state and 59 US forecasts parsed. Parser validated 153/153 exact against QuickStats for 2012–2013. One parser bug (decimal values read as yield) was caught by a unit check and fixed **before** any score was computed.

| Cutoff (2002–2011) | TX: momentum vs USDA (better / worse) | US (better / worse) | Prod-wtd state | T44 rule |
|---|---|---|---|---|
| **1 Dec (primary)** | 4.3 vs 4.4 (**3 / 3**) | **1.7 vs 2.0** (7 / 2) | 3.1 vs 3.3 | **FAIL** (TX tie) |
| 1 Sep (secondary) | **9.9 vs 12.5** (5 / 3) | **6.8 vs 7.9** (5 / 3) | 9.9 vs 11.3 | would pass |
| 1 Oct (secondary) | **7.8 vs 9.5** (5 / 3) | **4.6 vs 5.9** (7 / 3) | 7.7 vs 8.2 | would pass |
| 1 Nov (secondary) | **4.8 vs 5.2** (6 / 4) | **2.3 vs 2.9** (9 / 1) | 4.8 vs 5.0 | would pass |

- **Pooled past-mean bias (secondary), 2002–2011:**
  - Sep: TX 13.7 vs 12.5; US 8.7 vs 7.9.
  - Oct: TX 10.4 vs 9.5; US 6.7 vs 5.9.
  - **It loses.** USDA's bias does not carry across eras, the same lesson as T39 and T42.
- **Reading.** On data never seen before, the no-intercept revision-momentum rule beat USDA at Sep, Oct and Nov for both TX and the US, and at Dec for the US. Its declared primary (Dec) failed only because TX tied 3–3.
  - In 2014–2025 (T44), momentum helped at Dec but not at Sep.
  - So momentum is the most consistent signal found in this project. It does not depend on the sign of USDA's bias. Its strength varies by month and era.
  - This is still a set of secondaries, not a pre-declared pass.
- Evidence: `t45-blind-2000-2011-20261116.json` / `.log`, `t45-bias-secondary-20261116.json`.

### Full-period record of the momentum rule, 2002–2025 (16 Nov, descriptive, rule unchanged)

- **Setup.** T44's rule, unchanged: no intercept, λ = 1, trained rolling on seasons 2000…Y−1. Run once over all 24 seasons on the merged data: report-parsed 2000–2011, QuickStats 2012–2025.
- **What kind of evidence this is.** 2002–2011 outcomes were blind (T45). 2014–2025 outcomes had been seen, but the rule was never tuned on them. It is a record, not a new pass.

| Report | TX: rule vs USDA (better / worse yrs) | US (better / worse) | Prod-wtd state error |
|---|---|---|---|
| 1 Sep | **11.9** vs 12.9 (10 / 9) | **6.0** vs 6.3 (9 / 10) | **10.5** vs 11.0 |
| 1 Oct | **11.3** vs 12.1 (9 / 9) | **4.9** vs 5.5 (12 / 9) | **9.1** vs 9.3 |
| 1 Nov | **7.5** vs 8.0 (13 / 8) | **3.5** vs 3.6 (12 / 10) | **6.2** vs 6.4 |
| 1 Dec | **5.0** vs 5.3 (9 / 8) | **2.3** vs 2.5 (14 / 7) | **4.3** vs 4.5 |

- **Average error is lower than USDA's at every monthly report**, for TX, the US and the production-weighted states: 12 of 12 cells. The relative reductions are about 3–11%.
- **It does not win every year.** By season count it wins more often than it loses in 6 of 8 TX/US cells; Oct TX is tied and Sep US loses 9–10.
- Evidence: `momentum-full-2002-2025-20261116.json` / `.log`.

### Significance of the cotton momentum record (16 Nov, descriptive, after the fact)
One-sided sign-flip permutation on |USDA error| − |rule error| per season (20,000 draws, seed 42); `research/crop_nowcast/momentum_significance.py`.

| Report | TX: better / worse yrs, mean gain (pts), p | US: better / worse yrs, mean gain (pts), p |
|---|---|---|
| 1 Sep | 10 / 9, +0.96, p 0.14 | 9 / 10, +0.34, p 0.15 |
| 1 Oct | 9 / 9, +0.81, p 0.08 | 12 / 9, +0.61, **p 0.02** |
| 1 Nov | 13 / 8, +0.58, **p 0.03** | 12 / 10, +0.09, p 0.38 |
| 1 Dec | 9 / 8, +0.27, p 0.13 | 14 / 7, +0.20, **p 0.02** |

- 3 of 8 cells reach p < 0.05 one-sided, without multiple-comparison correction. The rest cannot be told apart from chance at this sample size.
- Evidence: `momentum-significance-20261116.json`.

### T46 — the frozen momentum rule on USDA corn and soybeans, 2002–2025 (16 Nov) — **FAIL**
- **Preregistration:** `docs/T46_CORN_SOY_PREREG_20261116.md`, pushed before any value was loaded, with two dated amendments (an "Other States" row with final 0; one fused report column in Sep 2004). Both are data fixes; the rule did not change.
- **Data:** 2000–2011 forecasts parsed from 56 Crop Production reports (parser equal to QuickStats in 441 of 441 overlap cells, 2012–2013); 2012–2025 forecasts and all finals from QuickStats.

| Crop · report | US MAPE rule vs USDA (better / worse yrs, p) | Prod-wtd state error | Cell |
|---|---|---|---|
| Corn · 1 Sep | 2.17 vs 2.20 (13 / 11, p 0.33) | **4.76** vs 4.81 | pass |
| Corn · 1 Oct | 1.57 vs 1.57 (12 / 10, p 0.50) | 3.54 vs **3.48** | fail |
| Corn · 1 Nov | 1.03 vs **1.02** (11 / 13, p 0.56) | **2.72** vs 2.75 | fail |
| Soy · 1 Sep | **3.40** vs 3.60 (14 / 10, p 0.07) | 5.67 vs **5.64** | fail |
| Soy · 1 Oct | 2.03 vs **1.90** (10 / 12, p 0.71) | 3.86 vs **3.85** | fail |
| Soy · 1 Nov | **1.31** vs 1.38 (11 / 13, p 0.13) | **2.72** vs 2.75 | pass |

- **Crop rule (2 of 3 months):** corn 1 of 3, soybeans 1 of 3. **PRIMARY FAIL.**
- **Pooled six cells:** mean gain +0.03 points, year-block p = 0.28. **SECONDARY FAIL.**
- **Reading.** Outside cotton the rule adds nothing: USDA's corn and soybean forecasts are already within 1–3.6% and their revisions carry no usable momentum for this rule. The cotton effect is therefore cotton-specific (large Texas abandonment, a late harvest) or partly luck; it is not a general property of USDA forecasts. This is the result that most limits the cotton claim.
- Evidence: `t46-corn-soy-20261116.json` / `.log`; invalid runs kept as `t46-corn-soy-20261116-invalid.*` and `-run2-fused-2004.log`.
