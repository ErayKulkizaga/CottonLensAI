# CottonLensAI — Results Summary (14 Nov 2026)

One page with every result from this round, positive and negative. Each number
links to a committed evidence file. Every result is historical / out-of-sample
on past data. **None of them is an independent forward success yet.**

## 1. Why price forecasting failed (diagnosis)

- The project target (5% MAE gain over the random-walk naive price) needs an
  out-of-sample R² of roughly 10% on daily cotton returns. Published commodity
  return predictability is well below that, so the target is structurally out of reach.
- Across 113 earlier price experiments (Codex era), none reached 5%. The best was +0.22%.
- Conclusion: we stopped forecasting price *levels* and moved to price *risk*,
  where real information exists.

## 2. Confirmed: range information improves volatility forecasts (preregistered)

Branch `claude/range-ewma-preregistration`. The preregistration was committed
and hashed before scoring, the markets were admitted on data quality alone, and
the result was scored once.
Evidence: `research/evidence/range-ewma-result-20261113.json`.

| Test | Result |
|---|---|
| Range-EWMA vs EWMA, next-day QLIKE, 7 admitted futures (CC, HE, KC, KE, SB, ZL, ZW) | pooled gain **+5.96%**, block-bootstrap 95% CI [4.4%, 7.8%] |
| Breadth | **7/7** markets positive; each positive in 19–25 of its 25–26 scored years |

The cotton (CT) figures in section 3 come from the project's own CT history, applied with the same frozen method. CT is not one of the 7 confirmatory markets.

## 3. Risk lab: the full comparison (preregistered, Bonferroni-corrected)

Branch `claude/cotton-risk-lab`. Evidence: `research/evidence/risk-lab-result-20261113.json`.

| Hypothesis | Result |
|---|---|
| H2a range-EWMA vs GARCH(1,1) | **not confirmed** (pooled −5.0%; HE −41%) |
| H2b range-EWMA vs HAR | **not confirmed** (pooled −1.5%, CI includes 0) |
| H3 big-move probability, Brier vs EWMA | **confirmed**, 5/5 markets, Bonferroni |
| H4 80% interval, Winkler score vs EWMA | **confirmed**, 7/7 markets, Bonferroni |

- **Cotton calibration** (3,989 origins): the 80% interval covered 79.2% and the
  90% interval 90.3%.
- **Big-move ranking for cotton:** AUC 0.755. The top-quintile risk days had a
  big move 22.2% of the time, against 3.7% in the bottom quintile (≈6×).
- **Honest reading:** range data makes risk *probabilities* and *intervals*
  better calibrated. It does not beat classic GARCH on variance itself.

## 4. Satellite crop nowcast vs USDA (development years only)

Branch `claude/crop-nowcast`. Evidence: `research/evidence/crop-nowcast-dev-ndvi-20261114.json`.

**Data**
- MODIS 250 m NDVI over 263 cotton counties (90% of US production), 2009–2021:
  65 tile-years and 98,544 county×composite rows.
- Only cotton pixels are used, masked with the prior-year USDA CDL.
- NASA POWER daily weather, 1991–2021.
- NASS county and state statistics.
- USDA June Acreage reports for 2012–2021, parsed from the original releases.

**Leakage rules**
- Prior-year crop mask only.
- Composites are used only after their end date plus 8 days; weather is lagged 3 days.
- Each test year is predicted with a model trained only on earlier years.
- 2022–2025 stay sealed: the code refuses those years without `--unseal`.

**Production error (mean absolute %, 2013–2021), best variant**

| Cutoff | Texas: model / USDA / naive base | US: model / USDA |
|---|---|---|
| 1 Aug | 18.9 / 14.8 / 20.8 | 9.9 / 7.2 |
| 1 Sep | 15.1 / 11.2 | 7.9 / 6.3 |
| 1 Oct | 12.7 / 11.0 | 7.7 / 6.1 |

**What worked**
- The county NDVI anomaly is the strongest single signal. Its correlation with
  production per planted acre is +0.52 on 1 Oct (+0.62 in Texas); the best
  weather variable reaches −0.37.
- Fields abandoned in drought years are captured. On 1 Oct the per-acre error
  for Texas was +3% in 2019 and 0% in 2020. USDA's August forecast was +33% and
  +36% in those years.

**What did not work**
- No cutoff beats USDA.
- Blending the model with USDA does not beat USDA either (10.8% vs 11.0%, within noise).
- Predicting the direction of USDA's revisions is weak: 57% of state-years
  (Texas 6/9), not significant over 9 seasons.
- The predeclared gate (1 Aug Texas error near USDA's 14.8%) was **not met**, so
  no preregistration was written and the sealed years were not opened.
- The best variant was picked after seeing results, so these numbers are
  optimistic. The same variant ranked best at all three cutoffs.

**Why the gap remains:** USDA has ground surveys, and 1-Aug MODIS composites end
around 20 July, before abandonment is visible. Closing the gap would need
field-level, higher-cadence imagery (HLS 30 m). That work was scoped but not started.

## 5. Engineering practices used

- Preregistration with SHA-256 hashes, committed before any scoring.
- Quality-only admission of markets.
- 20- and 60-day common-date block bootstrap, Bonferroni correction for multiple hypotheses.
- Sealed test years enforced in code and by tests.
- The API key is never written to disk; a test checks this.
- Negative results are recorded with the same detail as positive ones.
- A Windows CRLF hash deviation in the risk lab is disclosed in its evidence file.

## 6. What is not claimed

- No forward (post-registration) success yet.
- No price-level forecasting skill.
- No beating USDA.
- The "today" risk card in the risk lab is a demonstration, not externally timestamped.
