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
