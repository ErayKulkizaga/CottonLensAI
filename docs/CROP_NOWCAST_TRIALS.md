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
