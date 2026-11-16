# T52 preregistration: does the predicted direction of USDA's next revision move cotton futures on report day?

Written 16 Nov 2026, before any report-day return was computed with this signal.

## Question
T50 showed that the direction of USDA's next cotton revision is predictable from its last revision: 67% on blind 1985–2011, but only 43% on 2012–2025. If the market does not price this, a lower expected production should push cotton futures up on the report day, and a higher one down.

## Data
- **Prices:** daily `cotton` close (front-month proxy CT=F) from the Phase 1 market history, 2010 onward. The Link 1 seal applies: nothing after 2021-12-31 is loaded.
- **Report days:**
  - 2012–2021 from `research/chain/crop_production_dates.csv`;
  - 2010–2011 from the release dates parsed from the reports (T45 forecasts).
- **Signal, known one report earlier:** at report m−1, the sign of USDA's latest US production revision (m−2 → m−1). The prediction for report m is the same sign. Events: the Oct, Nov and Dec reports of 2010–2021 (the Sep report's signal would need Jul → Aug, which does not exist).
- **Trade direction:** the predicted move is **−sign(signal)**: an expected downward revision means less cotton, so a higher price.
- **Return:** log close-to-close return on the report day (close of the report day vs the previous trading day's close).

## Pass rule
- **PASS:** the mean of −sign(signal) × return over all events is > 0, with one-sided season-block permutation p < 0.05. All events of a season are flipped together, 20,000 draws, seed 42.
- **Also reported:** the hit rate, and the same split for 2010–2011 vs 2012–2021.

## Prior expectation, stated before running
Small sample (about 36 events), and T50 found the signal itself fails in cotton after 2012, so a null is likely. The test is run once and reported either way.

## Result (16 Nov 2026) — FAIL
| | Events | Mean signed return | Hit rate |
|---|---|---|---|
| All, 2010–2021 | 32 | +0.27% | 53% |
| 2010–2011 | 4 | +0.58% | 75% |
| 2012–2021 | 28 | +0.22% | 50% |

- **Pass test:** season-block p = 0.30, so the test **fails**.
- **The signal itself:** in these events it called USDA's actual revision correctly only 47% of the time, matching T50's finding that persistence is gone in cotton after 2012.
- **What this means:** the price data start in 2010, so the market test could only cover the era in which the revision signal no longer works. No market value can be claimed. Testing a market effect in the era where the signal worked (pre-2010) needs older daily futures prices, which this project does not hold.
- **Evidence:** `research/evidence/t52-market-20261116.json`.
