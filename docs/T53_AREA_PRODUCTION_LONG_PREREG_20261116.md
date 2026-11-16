# T53 preregistration: the Texas area correction and production, on the longest available history

Written 16 Nov 2026, before the pre-2000 Texas condition series, the 1985–1994 Texas area forecasts (OCR) or the 1985–1994 area finals were loaded.

## Why
- T51-C: the area-corrected Texas production was better than USDA in 4 of 4 months on 2008–2025, but p = 0.135 with 16–18 seasons.
- QuickStats has a Texas cotton condition series from about 1986.
- The parsed reports give Texas harvested-area forecasts:
  - 1995–2011 from text (T48);
  - 1985–1994 from OCR (T47 stage B), using only rows that pass the production = area × yield gate.

## Design (identical to T51-C except the period)
- **Area error:** e = log(H_final / H_m), Texas, reports Sep–Dec.
- **Signal:** pvp = poor + very poor share of Texas cotton, last week ending on or before the 1st of the report month.
- **Model:** e_hat = a + b · pvp, OLS on all earlier seasons with data (at least 8).
- **Scored:** every season from the first one with 8 earlier seasons through 2025.
- **Adjusted production:** P_hat = P_USDA,m · exp(e_hat), compared with USDA's Texas production forecast.
- **Sources:**
  - production forecasts: the T47 / T45 parsed reports and QuickStats from 2012;
  - finals: QuickStats `YEAR` rows.

## Pass rule
- **PASS:** the mean abs. % production error is below USDA's in at least 3 of 4 months, **and** the pooled year-block p < 0.05 (20,000 draws, seed 42).
- **Also reported:** the same result split into seasons before 2008 (new) and from 2008 (T51), and the area-error results.

## Note stated before running
The anchoring was weaker before 2012 (M1: USDA/final response ratio about 0.9). Adding earlier seasons may therefore dilute the effect rather than strengthen it. The result is reported either way.
