# T54 preregistration: is USDA's in-season grain-sorghum harvested area anchored against crop condition?

Written 16 Nov 2026, before any sorghum data was downloaded.

## Why
The anchoring mechanism (M1, T49):
- holds in Texas cotton;
- is consistent but just short of significance in Oklahoma/Kansas cotton (p = 0.056);
- is absent in winter wheat.

Grain sorghum is a dryland summer crop of the same region (Kansas, Texas), with drought-driven abandonment and monthly USDA harvested-area forecasts. If the mechanism is about dryland summer-crop abandonment rather than cotton itself, it should appear here.

## Data
- **QuickStats, SORGHUM, GRAIN:**
  - harvested area (`AREA HARVESTED`, `ACRES`): monthly forecasts `YEAR - <MON> FORECAST` and finals `YEAR`, Kansas and Texas;
  - condition: % poor and % very poor, weekly, same states.
- **Seasons:** every season 2000–2025 with both a forecast and a final.
- **Reports:** every month from Aug to Nov that has forecasts (survey date = the 1st of the month; condition = the last week ending on or before the 1st).
- **Feasibility rule:** if fewer than 10 seasons per state have monthly forecasts, the test is reported as not feasible.

## Model and pass rule (identical to T49 S1)
- **Model:** e = log(H_final / H_m); e_hat = a + b · pvp, by OLS; leave-one-season-out, with all state-rows of the held-out season excluded.
- **PASS = (i) and (ii):**
  - **(i)** pooled corr(pvp, e) < 0, with the one-sided season-bootstrap 95% upper bound < 0;
  - **(ii)** the adjusted area beats USDA in at least 3 of the scored months (or in every month if fewer than 3 are scored), with year-block p < 0.05.
- **Claim if PASS:** "the anchoring of USDA's harvested area generalises to dryland grain sorghum."
