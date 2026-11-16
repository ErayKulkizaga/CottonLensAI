# T49 preregistration: is USDA's in-season harvested area anchored against public crop-condition information?

Written 16 Nov 2026, after the Texas discovery in M1 (`docs/research_package/15_USDA_ERROR_MECHANISM.md`). The Oklahoma/Kansas cotton condition series, all winter-wheat series, and the per-state Oklahoma/Kansas cotton area errors have not been downloaded or looked at. (The Oklahoma and Kansas rows exist in the parsed 1995–2011 reports, but entered only pooled statistics.)

## Hypothesis
For a crop with large, drought-driven abandonment, USDA's in-season harvested-area forecast responds too little to the share of the crop rated poor or very poor at the survey date. Therefore the remaining area error e = log(H_final / H_m) is negatively related to that share and predictable from it.

## Settings
| Setting | Role | Seasons | Reports (survey date) | Condition signal |
|---|---|---|---|---|
| **S0 Texas upland cotton** | discovery; real-time rolling demonstration | 2000–2025 | 1 Sep, 1 Oct, 1 Nov, 1 Dec | NASS % poor + very poor, week ending before the 1st |
| **S1 Oklahoma + Kansas upland cotton** (pooled, state-seasons) | **independent test** | 2000–2025 (2000–2011 forecasts from parsed reports, 2012–2025 QuickStats) | same | same, each state's own series |
| **S2 Winter wheat, Kansas + Oklahoma + Texas** (pooled) | **independent test, other crop** | 2012–2025 (QuickStats monthly forecasts) | 1 May, 1 Jun, 1 Jul, 1 Aug | winter-wheat % poor + very poor, week ending before the 1st |

Harvested-area forecasts and finals come from QuickStats (`AREA HARVESTED`, `ACRES`, `YEAR - <MON> FORECAST` and `YEAR`). For S1 in 2000–2011 they come from the T48-parsed reports, with the same 3% production = area × yield gate. Settings whose QuickStats series lack monthly forecasts for some months are scored on the months that exist, and the missing months are listed.

## Model (fixed form, one predictor, per setting and report month)
e_hat = a + b · pvp, fitted by OLS. Adjusted area H_hat = H_m · exp(e_hat).
- **S0:** rolling, with coefficients from seasons < Y only (at least 8 seasons), scored 2008–2025.
- **S1, S2:** leave-one-season-out (all state-rows of the held-out season are excluded from the fit). With 14–26 seasons, rolling fits would be too short; this is stated as a limit.

## Pass rules
For each independent setting (S1, S2):
- **(i) Sign:** the pooled correlation of pvp with e is negative, and its one-sided season-bootstrap 95% upper bound is below 0 (5,000 draws, seed 42, resampling seasons).
- **(ii) Prediction:** the adjusted area has a lower mean absolute log error than USDA's in-season area in at least 3 of 4 report months, and the year-block sign-flip permutation p on |e_USDA| − |e_hat| (pooled over months and states, one sign per season, 20,000 draws, seed 42) is < 0.05.
- **Setting PASS = (i) and (ii).**

**Claims allowed:**
- S1 PASS → "the anchoring generalises to other dryland cotton states".
- S2 PASS → "it generalises to winter wheat".
- Neither → the Texas finding stays a single-state discovery.

S0 is reported as a demonstration only (same pass metrics) and is not counted as confirmation.

## Production (secondary, reported, no pass rule)
For S0, production adjusted by area only (USDA yield kept) is compared with USDA's production. Earlier trials (T31/T32) showed that area and yield errors partly offset, so a better area forecast may not give better production. This is measured, not assumed.

## Forward lock
The S0 model, fitted on 2000–2025, is applied to USDA's 1 Nov 2026 Texas harvested area and the Texas pvp of the last week of October 2026. The result is locked in `research/evidence/t49-forward-lock-2026-*.json` and scored against the NASS final in May 2027. The December procedure is fixed in the same file.

## Data notes (before the single run)
- **First pull:** incomplete. QuickStats throttling (403/429) was treated as "no data". The helper was fixed to stop retrying only on HTTP 400 (no rows), and the pull was repeated before any analysis.
- **Coverage used:**
  - cotton condition: TX and OK 2000–2025, KS 2005–2025 (NASS has no earlier KS series);
  - winter-wheat condition and area: KS, OK and TX, 2012–2025.
  - Winter-wheat June harvested-area forecasts exist for 9 of 14 seasons, so June is scored on those.

## Result (16 Nov 2026, single run)
| Setting | Pooled corr(pvp, area error), one-sided 95% upper bound | Months adjusted < USDA | Mean gain (log pts), year-block p | Verdict |
|---|---|---|---|---|
| S0 Texas cotton, rolling 2008–2025 (demonstration) | −0.70, −0.56 | 4 / 4 | +5.2, p = 0.0003 | pass, not counted as confirmation |
| **S1 Oklahoma + Kansas cotton, LOSO 2000–2025** | **−0.61, −0.49** | **4 / 4** | **+1.7, p = 0.056** | **FAIL** (sign passes; prediction p misses 0.05) |
| **S2 Winter wheat KS/OK/TX, LOSO 2012–2025** | +0.09, +0.35 | 0 / 4 | −0.4, p = 0.97 | **FAIL** |

Texas area error by month (S0, rolling, mean abs. %): USDA vs adjusted.
| Month | USDA | Adjusted |
|---|---|---|
| Sep | 12.8 | 7.9 |
| Oct | 13.2 | 7.6 |
| Nov | 12.6 | 7.0 |
| Dec | 12.6 | 7.3 |

**Reading:**
1. **Texas (discovery state).** A one-predictor model fitted only on past seasons cuts USDA's in-season harvested-area error by about 40% at every report from 2008 to 2025. This is the real-time form of the M1 finding, but Texas was the discovery set, so it is not confirmation.
2. **Oklahoma and Kansas cotton** (independent, never examined state by state):
   - The anchoring signature is clearly present: correlation −0.61 with a one-sided upper bound of −0.49, and the adjusted area is better in all four months.
   - The pre-declared significance gate (p < 0.05) is missed: p = 0.056.
   - By the rule this is a FAIL, and the claim "generalises to other dryland cotton states" is **not** made. The honest summary is "consistent, not yet significant".
3. **Winter wheat:** no anchoring. The correlation is about zero or positive. USDA's wheat harvested area is not anchored against the condition rating in the same way.
   - A plausible reason, not tested: wheat abandonment is decided and surveyed (June Acreage, grazing-out) before most of the forecast season.
   - So the mechanism is specific to cotton, not a general property of USDA's area forecasts.
- Evidence: `research/evidence/t49-anchoring-20261116.json` / `.log`.
