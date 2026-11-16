# 13 · Literature and what this project adds

## Prior work this project builds on
| Work | What it found | How it relates |
|---|---|---|
| Nordhaus (1987), *Review of Economics and Statistics* | Efficient fixed-event forecasts have uncorrelated revisions. Positive correlation is called "smoothing", negative correlation "jumpiness". | The test logic behind the momentum rule. |
| Isengildina-Massa, MacDonald & Xie (2012), "A Comprehensive Evaluation of USDA Cotton Forecasts", *JARE* 37(1): 98–113 | Smoothing is the most widespread and persistent inefficiency in USDA's US cotton estimates, 1985/86–2009/10. | The momentum finding in cotton is a **replication** of this result, not a discovery. |
| Isengildina, MacDonald, Xie & Sharp (2013), "Smoothing in USDA's Commodity Forecasts", NCCC-134 | Revisions are not independent, so forecasts are smoothed. Smoothing coefficients are highly unstable over time. Adjusting for smoothing improved accuracy only for some wheat and cotton forecasts and hurt others. | Our out-of-sample results match this closely (see below). |
| Isengildina, Irwin & Good (2013), *JAAE* 45(1): 95–107 | Corn and soybean yield revisions are correlated, but crop-size information is generally used efficiently. | Consistent with T46: no usable momentum in corn or soybeans. |
| Goyal & Adjemian (2023), "Information rigidities in USDA crop production forecasts", *AJAE* 105(5): 1405–1425 | Correlated revisions fit information-rigidity models (costly or noisy data) better than deliberate smoothing. Remote sensing could make USDA's forecasts more efficient. | This is why the project does not claim USDA "under-reacts" on purpose. The satellite trials (T41, T43) test this suggestion directly for cotton. |
| You et al. (2017), Deep Gaussian Process for crop yield prediction, *AAAI* | Histogram CNN/LSTM on MODIS for county yields. | Method source of the T35 CNN. |

Sources: [JARE 2012 PDF](https://ageconsearch.umn.edu/record/122314/files/Isengildina_%20pg.%2098-113.pdf) · [NCCC-134 2013](https://ideas.repec.org/p/ags/n13413/285801.html) · [JAAE 2013](https://econpapers.repec.org/RePEc:ags:joaaec:143639) · [AJAE 2023](https://econpapers.repec.org/RePEc:wly:ajagec:v:105:y:2023:i:5:p:1405-1425) · [farmdoc daily 2024 update](https://farmdocdaily.illinois.edu/2024/10/revisiting-big-corn-and-soybean-crops-get-bigger.html)

## What this project adds (and what it does not)
**It does not add** a new forecasting idea. Smoothing in USDA cotton forecasts is published, and so is the weak, unstable gain from adjusting for it.

**It adds an independent, leakage-controlled, preregistered out-of-sample test of that idea, with more blind data and more failure modes than the published work:**
1. **Out-of-sample record, 1995–2025.**
   - The rule was trained only on past seasons, and its 2002–2011 and 1995–1999 seasons were never seen before testing (T45, T47 stage A).
   - 2002–2025: lower mean error than USDA in 12 of 12 report × region cells.
   - 1995–1999 with frozen coefficients: 4 of 8 TX/US cells (T47 stage A, FAIL).
2. **The instability is measured directly.** The rolling coefficient shrinks from 0.03–0.08 in the early 2000s to 0.003–0.019 by 2025, and the 1 Sep coefficient turns negative (`robustness-momentum-20261116.json`). This is the same instability the 2013 paper reports, now shown out of sample.
3. **Multiple-comparison honesty.**
   - Per cell, no TX/US cell survives Holm or BH correction at 5%; the smallest BH q is 0.09.
   - Pooled over the 8 cells with one sign per season, p = 0.015, a test defined after the per-cell results were seen.
   - Leave-one-season-out flips the sign in only 1 of 8 cells, so no single year drives the result.
4. **Comparison with a published-style adjustment** (series-level OLS with an intercept, 10-season or expanding window, on the same seasons):
   - The momentum rule is best in only 2 of 8 cells (1 Sep and 1 Oct US).
   - The expanding-window published-style adjustment is better in most Nov/Dec and Texas cells.
   - So the no-intercept, state-pooled rule is not uniformly better than the published approach.
5. **Generalisation test.** Corn and soybeans failed under a pushed preregistration (T46). This matches the 2013 JAAE finding that corn and soybean revisions are largely efficient.
6. **The remote-sensing suggestion, tested.**
   - Landsat field-level greenness tracked Texas abandonment (r ≈ 0.8).
   - It did not anticipate the 2023 write-off at the 1 Sep or 1 Oct cutoffs (T43).
   - County NDVI models were the best non-USDA models, but they did not beat USDA's US total.
7. **Engineering and protocol.**
   - Publication-time rules for every source.
   - Sealed and blind holdouts opened once, after pushed preregistrations.
   - Amendments logged before scoring, and invalid runs kept.
   - Resumable cloud jobs, hash receipts and byte-level parser validation (153/153 and 441/441 exact matches with QuickStats).

## How to say it in one sentence
"I independently re-tested the published finding that USDA smooths its cotton forecasts, using a leakage-controlled, preregistered protocol with blind seasons. The edge is real in the 2002–2025 record but small, shrinking, not significant after multiple-comparison correction, and absent in corn and soybeans. That is exactly what the published instability result predicts."
