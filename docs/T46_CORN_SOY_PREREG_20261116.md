# T46 preregistration: the frozen revision-momentum rule on corn and soybeans

Written 16 Nov 2026, before any corn or soybean forecast or final value was loaded for analysis. The scoring runs once, after this file is pushed.

## Question
For cotton, USDA's latest monthly revision predicted the direction of its remaining error (`research/crop_nowcast/momentum.py`, T44/T45, full record 2002–2025). If this is a property of how USDA revises, and not a cotton accident, the **same rule, unchanged**, should also beat USDA's corn-for-grain and soybean forecasts.

## Rule (frozen; no setting may change)
`momentum.predict` and `momentum.with_revision`, exactly as committed in `momentum.py` (last changed in `e6be6c1`, the T45 commit; not modified for T46):
- r = log(USDA_m / USDA_prev), where prev is the latest earlier monthly forecast; e = log(final / USDA_m).
- Pooled over states, no-intercept ridge e = b·(r/σ), σ = RMS of r in training, λ = 1.
- Fit only on years < Y, from 2000 (`COTTONLENS_MOMENTUM_FIRST_TRAIN=2000`); at least 10 training rows, otherwise no correction.
- Corrected state forecast = USDA_m · exp(ê). US = USDA_US,m × Σ corrected states / Σ USDA states.
- Fitted separately per crop and per report month. Coefficients are not carried over from cotton (units differ); the procedure is.

## Data
- **Forecasts 2000–2011:** USDA Crop Production text reports (ESMIS), Aug–Nov, tables "Corn for Grain: Area Harvested, Yield, and Production" and "Soybeans for Beans: …". The last number of each state row is the current production forecast (1,000 bushels). Parser `t46_crops.py parse`, unit test in `research/tests/test_stack.py`.
- **Forecasts 2012–2025 and all finals:** NASS QuickStats (`SURVEY`, `PRODUCTION`, `BU`; corn `GRAIN`, soybeans `ALL UTILIZATION PRACTICES`; `ALL PRODUCTION PRACTICES`). The final is the QuickStats annual value (`YEAR`), the same definition as for cotton.
- **Data gate (before scoring):** on the overlap years 2012–2013, parsed report values must equal QuickStats in ≥ 98% of matched cells, with a median ratio of exactly 1, for both crops and both levels. If the gate fails, the parser is fixed and this is recorded as an amendment *before* scoring. Only counts are printed.
- **Scored years:** 2002–2025 (`COTTONLENS_MOMENTUM_YEARS=2002-2025`). **Scored reports:** 1 Sep, 1 Oct, 1 Nov (August has no earlier forecast; there is no December corn/soybean forecast).

## Metrics
Per crop × month:
- US MAPE of the rule vs USDA;
- production-weighted state error (`stack.state_w`) of the rule vs USDA;
- US years better / worse;
- one-sided sign-flip permutation p on |USDA error| − |rule error| (20,000 draws, seed 42).

Across all six cells: a year-block permutation (one sign per year shared by all cells, because the cells of one year are correlated), one-sided p.

## Pass rules
- **Cell pass:** rule US MAPE < USDA US MAPE **and** rule state-weighted error < USDA's.
- **Crop pass:** at least 2 of the 3 months pass.
- **PRIMARY PASS:** corn passes **and** soybeans pass.
- **SECONDARY:** pooled year-block p < 0.05.
- The result is reported in full either way, year by year, in `research/evidence/t46-corn-soy-*.json` and `docs/CROP_NOWCAST_TRIALS.md`.

## Disclosures (before scoring)
- The 2000–2013 Crop Production text files were downloaded for T45 (cotton) and are cached. They contain the corn and soybean tables; those tables were never parsed or analysed before T46.
- While writing the parser on 16 Nov, table layouts were inspected with every digit masked. The front-page headlines of three reports (Sep 2000, Sep 2008, Sep 2010) were also seen; they state the direction of that month's corn/soybean revision, not any final value or error.
- QuickStats `get_counts` was queried for row counts by year (no values) to see that state forecasts exist from 2012 on.
- The cotton result that motivates this test used 2002–2025 cotton data; no corn or soybean data informed the rule.

## Amendment 1 (16 Nov 2026, after an invalid first run, before any valid result)
- **What happened.** The first `score` run produced invalid numbers: infinite errors, NaN state errors and US errors in the millions of percent. Cause: QuickStats state data include an `OT` ("Other States") aggregate row whose annual final is 0 from 2015 on; log(final / forecast) = −∞ contaminated every training fit. The cotton data had no such row.
- **What that run showed.** It printed better/worse year counts and p-values computed from corrupted numbers; they carry no information about the rule and are kept in `research/evidence/t46-corn-soy-20261116-invalid.log`.
- **Fix.** Only the data: state-level rows are limited to the 50 states (`t46_crops.py build`). The rule, years, metrics and pass rules are unchanged. The corrected run is the only scored run.

## Amendment 2 (16 Nov 2026, after the second run)
- **What happened.** The second run (`t46-corn-soy-20261116-run2-fused-2004.log`) was valid in five of six cells. Corn 1 Sep US MAPE was about 3.6 × 10⁸ % for both the rule and USDA. Cause: in the Sep 2004 report the US row's last two columns are printed without a space (`99,999,99999,999,999` with digits masked), and the parser read them as one number.
- **Fix.** `last_value` splits a fused number into two valid comma groupings and keeps the right half closest to current area × current yield (unit test added). A scan of every parsed corn, soybean and cotton row found this single fused row; the T45 cotton data had none.
- **Effect.** Only the corn 1 Sep US numbers can change (states are trained at state level; the US figure only enters the US aggregation). The outcomes of the other five cells were already seen in run 2 and are disclosed here. Corn fails its crop rule in run 2 on 1 Oct and 1 Nov whatever 1 Sep shows, so PRIMARY is FAIL independent of this fix.

## Result (16 Nov 2026) — PRIMARY FAIL, SECONDARY FAIL

| Crop · report | US MAPE rule vs USDA (better / worse yrs, p) | Prod-wtd state error | Cell |
|---|---|---|---|
| Corn · 1 Sep | 2.17 vs 2.20 (13 / 11, p 0.33) | **4.76** vs 4.81 | pass |
| Corn · 1 Oct | 1.57 vs 1.57 (12 / 10, p 0.50) | 3.54 vs **3.48** | fail |
| Corn · 1 Nov | 1.03 vs **1.02** (11 / 13, p 0.56) | **2.72** vs 2.75 | fail |
| Soy · 1 Sep | **3.40** vs 3.60 (14 / 10, p 0.07) | 5.67 vs **5.64** | fail |
| Soy · 1 Oct | 2.03 vs **1.90** (10 / 12, p 0.71) | 3.86 vs **3.85** | fail |
| Soy · 1 Nov | **1.31** vs 1.38 (11 / 13, p 0.13) | **2.72** vs 2.75 | pass |

- Corn passes 1 of 3 months, soybeans 1 of 3; the crop rule needs 2. Pooled mean gain +0.03 points, year-block p = 0.28.
- The frozen cotton rule does not carry over to corn or soybeans. Evidence: `research/evidence/t46-corn-soy-20261116.json`.
