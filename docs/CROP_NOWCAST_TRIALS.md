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
