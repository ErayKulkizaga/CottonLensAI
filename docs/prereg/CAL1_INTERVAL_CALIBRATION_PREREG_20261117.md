# CAL1 preregistration: real-time calibration of the Texas harvested-area intervals

Written 17 Nov 2026, before any of the three candidate methods below was computed. This is a calibration study, not a hypothesis test: it chooses how the monitor states its uncertainty, and makes no new accuracy claim.

## Why
- T51 reported that in-sample conformal intervals under-cover (80% interval covers 64%).
- The monitor's `model_v1` uses empirical quantiles of the T51 out-of-sample residuals pooled over 2008–2025. Those quantiles use future seasons, so they are calibrated only in hindsight.
- Already computed (17 Nov, before this file): applying the v1 method in real time, with only earlier seasons' residuals (at least 12), over 2011–2025 (58 forecasts), the 80% interval covers **52%** and the 95% interval **79%**. The cause: Texas area errors grew after 2012, so early residuals are too small.

## Data
- `research/evidence/t51-area-production-20261116.json` → `rows`: Texas, 2008–2025, Sep–Dec. For each row: `e` = log(H_final / H_USDA), `eh_anchor` = the rolling anchoring prediction, `pvp` = poor + very poor share.
- Residual r = e − eh_anchor; the interval is eh_anchor ± h, and it covers if |r| ≤ h.
- **Real-time rule:** for season y, every method uses only rows of seasons before y. Scored seasons: those with at least 12 earlier rows (2011–2025, 58 forecasts).

## Candidate methods (q = 0.80 or 0.95)
- **(a) Condition-scaled:**
  - on earlier rows, fit |r| = α + β · pvp by OLS, with the fitted value floored at 0.01;
  - h = k_q · fitted(pvp), with k_q = the q-quantile of |r| / fitted over earlier rows.
- **(b) Recent window:** h = the q-quantile of |r| over the rows of the last 8 earlier seasons.
- **(c) Conservative:** h = 2 × the v1 real-time quantile (the q-quantile of |r| over all earlier rows).

## Selection rule
- **Eligible:** 80% coverage in [0.72, 0.88] and 95% coverage ≥ 0.90, both pooled over the 58 forecasts.
- **Choice:** the eligible method with the smallest mean 80% half-width.
- **If a method is chosen:**
  - it is fitted on all 2008–2025 rows and frozen into `research/monitor/model_v2.json`;
  - `model_v1` and its log records stay unchanged;
  - new log records state which model produced them.
- **If none is eligible:** the monitor keeps v1's numbers and labels them "not calibrated in real time (CAL1)".
- **Also reported:** per-method coverage by month and the mean half-width in % of area.

## Caveats stated before running
- 58 forecasts in 15 seasons: a coverage of 0.80 has a standard error of about 0.05 under independence, and more because months within a season are correlated.
- The choice is made on the same seasons it is scored on. It is a calibration choice to be confirmed by the forward seasons 2026+.
