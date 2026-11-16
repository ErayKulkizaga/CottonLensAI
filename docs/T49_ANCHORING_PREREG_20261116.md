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
