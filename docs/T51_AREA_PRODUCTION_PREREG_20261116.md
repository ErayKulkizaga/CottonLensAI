# T51 preregistration: strong baselines, prediction intervals, and what the Texas area correction does to production

Written 16 Nov 2026, before any of the quantities below were computed. The Texas area, production and condition series were loaded before (M1, T49). Neither the production effect of the area correction, nor the baselines, intervals or break tests below, have been computed.

## A. Baselines for Texas harvested area (rolling, seasons < Y only, scored 2008–2025, at least 8 training seasons)
Area error e = log(H_final / H_m). Each model predicts e_hat; the adjusted area is H_m · exp(e_hat).
| Model | e_hat |
|---|---|
| USDA | 0 |
| Past-mean bias | mean of e over all past seasons |
| Past-mean bias, last 10 seasons | mean of e over the last 10 past seasons |
| **Anchoring (T49)** | a + b · pvp, OLS on all past seasons |
| Anchoring, last 10 seasons | the same, fitted on the last 10 past seasons |

Reported for each model and month: mean abs. error (%), and a year-block permutation p of anchoring vs each baseline.

## B. Prediction intervals for the Texas area (rolling split-conformal)
- For season Y, the anchoring model's residuals are computed leave-one-season-out within the training seasons.
- The 80% and 95% intervals are e_hat ± the 80th / 95th percentile of their absolute values.
- Reported: empirical coverage over 2008–2025, by month and pooled, and the mean width.
- Target: coverage within ±10 points of nominal.

## C. Production effect (pass rule)
Area-adjusted Texas production: P_hat = P_USDA,m · H_hat / H_m. USDA's yield is kept, so this is the area correction alone.
- **PASS:**
  - mean abs. % error of P_hat < USDA's production error in at least 3 of 4 months (Sep–Dec, 2008–2025, rolling); **and**
  - pooled year-block p < 0.05.
- **Also reported:** the same against the momentum rule's Texas production error on the same seasons.
- **Why this can fail:** area and yield errors partly offset (T31/T32, `15_USDA_ERROR_MECHANISM.md`). If production gets worse, the area correction must not be used for production, and that is reported.

## D. Structural break in the Texas area error
- Regression: e_SEP on pvp_SEP, Texas 2000–2025.
- A Chow F-test at 2012 (the break date suggested by T48).
- A sup-F scan over break years 2005–2019, with the maximising year reported. Its p comes from 5,000 season permutations of the residual order.
- Descriptive. The break date was suggested by data already seen.

## E. Point-in-time audit (automated tests, `research/tests/test_stack.py`)
For every Texas season and month used:
- the condition week used ends on or before the survey reference date minus one day (it is released the following Monday, on or before the 1st);
- the model for season Y is fitted only on seasons < Y;
- no final value of season Y enters its features.

Results: `research/evidence/t51-*.json`.

## Results (16 Nov 2026, single run; seasons 2008–2025)
**A · Baselines, Texas harvested area, mean abs. error %**
| Model | Sep | Oct | Nov | Dec | All | Anchoring better, p |
|---|---|---|---|---|---|---|
| USDA | 11.5 | 11.7 | 11.3 | 11.3 | 11.4 | 0.0005 |
| Past-mean bias | 9.5 | 9.5 | 9.7 | 9.8 | 9.6 | 0.004 |
| Past-mean bias, 10 seasons | 9.7 | 9.2 | 9.8 | 9.8 | 9.6 | 0.008 |
| **Anchoring** | **7.4** | **7.0** | **6.6** | **6.9** | **7.0** | — |
| Anchoring, 10 seasons | 6.6 | 6.9 | 6.6 | 7.1 | 6.8 | 0.61 (no difference) |

The anchoring model beats USDA and both bias baselines. The condition signal adds information beyond a known average bias.

**B · Conformal intervals: under-cover, target missed.**
- The 80% interval covered 64% (half-width 6.8%).
- The 95% interval covered 77% (half-width 11.0%).
- Residuals from earlier seasons understate later errors, consistent with the regime change.
- The intervals must be widened or recalibrated before use. This is recorded as a known gap.

**C · Production effect: FAIL by the pre-declared rule, consistent in direction.**
| Texas production, mean abs. % error | Sep | Oct | Nov | Dec |
|---|---|---|---|---|
| USDA | 14.1 | 14.1 | 9.9 | 6.3 |
| Momentum rule | 14.2 | 13.6 | 8.9 | 5.9 |
| **USDA × area correction** | **11.8** | **10.9** | **7.9** | **5.8** |
- Better than USDA in 4 of 4 months and better than momentum in all 4.
- Year-block p = 0.135 vs USDA and 0.17 vs momentum, so the 0.05 gate is not met. The area correction does not make production worse, despite the area/yield offset, but the production gain is not yet significant on 16–18 seasons.

**D · Structural break**
- Chow F at 2012 = 10.9 (p = 0.0005).
- The sup-F scan peaks at **2019** (F = 12.2), with permutation p = 0.055.
- A break in the relation between condition and Texas area error is strongly supported, but its date is not pinned to 2012. The evidence fits a gradual change or a later break.

**E · Point-in-time audit:** both tests pass. Every condition week used was released by the survey date, and later weeks cannot change the value used. Rolling fits use only past seasons.
