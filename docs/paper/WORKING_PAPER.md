# Where USDA's Cotton Forecasts Miss: Revision Smoothing, Harvested-Area Anchoring, and a Preregistered Out-of-Sample Re-test

**Eray Kulkizaga** · Working paper draft, 16 November 2026 · Code and evidence: `github.com/ErayKulkizaga/CottonLensAI` (branch `claude/hls-move3`, PR #78)

> Draft for review. Every number below is reproduced from committed evidence files; see the reproducibility section. AI coding tools (OpenAI Codex, Anthropic Claude) wrote and ran most of the code under the author's direction; all decisions about scope, unsealing and publication were the author's.

## Abstract
USDA's monthly Crop Production forecasts are the benchmark for the US cotton market. Earlier work found that their revisions are positively correlated ("smoothing"), but that adjusting for it gives weak and unstable gains. We re-test that finding under a leakage-controlled protocol, with tests preregistered in a public repository and with USDA forecasts recovered for seasons never used in development (1985–2011, partly by OCR of scanned reports).

Four results:
1. **Blind out-of-sample gains.** Over all 23 blind seasons, a no-intercept correction estimated only from USDA's own past revisions is closer to the final crop than USDA in 88 of 128 season × report × region comparisons (year-block permutation p = 0.007).
2. **Next-revision direction.** The direction of USDA's next revision is predictable from its last revision 67% of the time on blind cotton seasons (preregistered, p = 0.012), and 57–69% for corn and soybeans. The effect is absent in cotton after 2012.
3. **The source of the error moved.** Decomposing the error into harvested area and yield shows that Texas misses moved from yield (area error about 3.8% at 1 September, 1995–2011) to harvested area (about 14.9%, 2012–2025). USDA's in-season harvested area responds to the publicly reported share of the crop in poor or very poor condition at only about two-thirds of the rate at which final abandonment does.
4. **A real-time area correction.** A one-variable model built only from past seasons cuts USDA's in-season Texas area error from 11.4% to 7.0% (2008–2025), beating past-bias baselines. Independent tests are mixed: consistent but just short of significance in Oklahoma/Kansas cotton (p = 0.056), absent in winter wheat. The resulting production gain is consistent (4 of 4 months) but not significant (p = 0.12–0.19).

We report all failed tests and release the recovered forecast archive.

## 1. Introduction
- **Literature.** Nordhaus (1987) defines efficient revisions; positive revision correlation is "smoothing".
  - Isengildina-Massa, MacDonald & Xie (2012) found smoothing to be the most persistent inefficiency in USDA cotton forecasts.
  - Isengildina-Massa et al. (2013) showed that correcting for it improves accuracy only for some series, and that the coefficients are unstable.
  - Goyal & Adjemian (2023) argue that the cause is information rigidity (costly or noisy data) rather than deliberate smoothing.
- **Contributions:**
  - a public, preregistered, leakage-audited re-test with blind seasons;
  - the decomposition of USDA's cotton error into area and yield over 1995–2025, which locates a regime change in Texas;
  - a specific, testable form of information rigidity (harvested-area anchoring against public condition data);
  - an open archive of recovered monthly forecasts.

## 2. Data
- **USDA forecasts:**
  - 2012–2025: NASS QuickStats.
  - 1995–2011: parsed from Crop Production text reports.
  - 1985–1994: OCR of scanned reports, keeping only rows that satisfy production = area × yield within 3%, with each report's states summing to the US within 2–10%.
  - Validation: parsed values equal QuickStats in 153/153 cotton and 441/441 corn/soybean overlap cells (2012–2013).
- **Finals:** NASS annual estimates.
- **Crop condition:** NASS Crop Progress, weekly share rated poor or very poor; the last week ending before the 1st of the report month.
- **Drought:** US Drought Monitor, county shares weighted by cotton planted area.
- **Recovered archive:** `docs/research_package/data/usda_monthly_forecasts_recovered_1985_2011.csv` (5,088 rows).

## 3. Protocol
- **Rolling fits.** Every model is fitted only on seasons before the one predicted.
- **Preregistration.** Pass rules were written and pushed before the data that tests them was opened (T39–T55). Amendments are dated and limited to data or parsing fixes, and invalid runs are kept.
- **Inference.** Year-block sign-flip permutation (20,000 draws) treats each season as one unit. Holm and Benjamini–Hochberg corrections are reported for multi-cell results.
- **Point-in-time tests** check that every condition value used was released by the survey date.
- **Reproduction.** A single command (`python research/reproduce.py`) reproduces the headline results.

## 4. Results
### 4.1 Revision smoothing, re-tested
- **2002–2025.** The rolling no-intercept correction has a lower mean absolute percentage error than USDA in all 12 report × region cells, but no cell survives Holm/BH correction.
- **Coefficient decay.** The coefficient decays from 0.03–0.08 to 0.003–0.019.
- **Published-style adjustment.** A series-level adjustment with an intercept (Isengildina-Massa et al. 2013) is better in 6 of 8 cells.
- **Blind eras.** Frozen 2000–2025 coefficients fail on 1985–1999. Re-estimated within each blind era, the rule beats USDA in 7 of 8 cells (1987–1999) and, pooled with 2002–2011, in 88 of 128 comparisons (p = 0.007).
- **Disclosure:** the decision to pool the two blind eras was made after both were scored. It uses every blind season.
- **Next-revision direction (T50).** Persistence predicts the sign of USDA's next revision:
  - blind cotton: 66.7% (62/93, p = 0.012); Texas 79%; all states 72%;
  - corn and soybeans: 57–69%;
  - cotton after 2012: 43%.
- **Market test (T52).** A report-day futures test on 2010–2021 is null (p = 0.30). In that era the signal itself no longer works.
- **Market test in the signal era (T55).** On 2000–2009, when the signal called USDA's revision 67% of the time, the report-day futures return in the predicted direction averages +0.47% but is not significant (p = 0.082) and half of it comes from one season (2003).

### 4.2 Where the error comes from
log(final / forecast) = area part + yield part, exactly.
- **Texas.** At 1 Sep the area error is 3.8% in 1995–2011 and 14.9% in 2012–2025; the area share of the error variance is 0.24 and 0.72. USDA's in-season area barely moves between August and December in both periods.
- **United States.** The yield part dominates the US error in both periods.
- **Structural break.** A Chow test at 2012 gives p = 0.0005. The sup-F scan peaks in 2019 (permutation p = 0.055). The change is real, but its date is not pinned.

### 4.3 Anchoring of the harvested area
**Measure.** Final abandonment, 1 − H_final / P, and USDA's in-season abandonment, 1 − H_m / P, are each regressed on the public poor/very-poor share at the survey date.

**Findings, 2012–2025:**
- USDA's slope is 0.65–0.69 of the final's (95% CIs below 1). It was about 0.9 in 2000–2011.
- The correlation between the poor share and USDA's remaining area error is −0.82 to −0.88.
- The frequency of poor conditions did not change between periods; the response of abandonment to them did.

**Real-time correction (T49, T51).** A rolling one-variable model reduces USDA's Texas area error from 11.4% to 7.0% (p = 0.0005) and beats past-bias corrections (9.6%).

**Independent tests:**

| Setting | Result |
|---|---|
| Oklahoma + Kansas cotton | consistent: corr −0.61, 4/4 months better, p = 0.056 |
| Winter wheat | none |
| Grain sorghum (T54) | the same negative relation (corr −0.44), but the correction does not beat USDA's already small area errors (p = 0.70) |

**Why the area gain does not carry over (B2, descriptive).** After 2012 the Texas area and yield parts of USDA's error move in opposite directions: corr −0.57 over Sep–Dec (Dec −0.92), slope of the yield part on the area part −0.42; before 2012 the correlation was +0.21. When USDA overstates harvested area it understates yield per harvested acre, presumably because the fields later written off are the lowest-yielding. Only 36% of the anchoring model's gain on area (5.4 points) survives into production (2.0 points). A production-level correction would have to model both parts; that is left for a separate preregistration.

**Production (T51, T53).** Area-corrected Texas production is better than USDA in 4 of 4 months after 2008 (for example October, 14.1% → 11.3%), but not significantly (p = 0.12). Before 2008 there was little area error to correct.

## 5. What did not work
The project's ledger records 55 numbered trials and many more in an earlier price phase:
- daily price forecasting (about 33,000 model fits, no edge over a no-change forecast);
- satellite county models;
- a histogram CNN on MODIS (data-starved);
- bias corrections that broke when USDA's bias reversed in 2022–2025;
- Landsat abandonment signals that missed 2023;
- corn/soybean production corrections;
- the frozen-coefficient blind tests;
- the market tests (2010–2021 and 2000–2009).

They are reported with the same detail as the successes.

## 6. Forward tests (dated in git)
- **2026 US production:** 12,763,510 bales (USDA Nov 12,794,000).
- **Texas harvested area:** 3,258,029 acres (USDA 3,900,000; 80% interval 2.92–3.64 M).
- **Directions of the 10 Dec 2026 revision:** US down, Texas down.
- **Oklahoma and Kansas harvested area (independent states):** 243,609 and 67,878 acres (USDA 320,000 and 77,000).
- **Monitor.** A public, hash-chained monitor (`research/monitor/`) reproduced these locks from the report text and will log every report.
- **Scoring:** 10 Dec 2026 for the revision directions; May 2027 for the finals.

## 7. Limitations
- Few seasons per test (14–26).
- Texas was the discovery state for the anchoring.
- The blind-era pooling was chosen after seeing the eras.
- The cause of the post-2012 change (FSA failed-acre timing, dryland economics) is untested.
- Intervals: in-sample conformal intervals under-cover (64% for 80%), and so do hindsight out-of-sample quantiles applied in real time (52%). Only a doubled real-time quantile calibrates (CAL1: 83% / 93%), and it is wide (about ±16% of area at 80%); the Texas 2026 80% interval (2.62–4.06 M acres) includes USDA's figure.

## References
- Goyal, R. & Adjemian, M. K. (2023). Information rigidities in USDA crop production forecasts. *American Journal of Agricultural Economics* 105(5), 1405–1425.
- Isengildina-Massa, O., MacDonald, S. & Xie, R. (2012). A comprehensive evaluation of USDA cotton forecasts. *Journal of Agricultural and Resource Economics* 37(1), 98–113.
- Isengildina-Massa, O., MacDonald, S., Xie, R. & Sharp, J. (2013). Smoothing in USDA's commodity forecasts. NCCC-134 Conference, St. Louis.
- Isengildina, O., Irwin, S. H. & Good, D. L. (2013). Do big crops get bigger and small crops get smaller? *Journal of Agricultural and Applied Economics* 45(1), 95–107.
- Nordhaus, W. D. (1987). Forecasting efficiency: concepts and applications. *Review of Economics and Statistics* 69(4), 667–674.
- You, J., Li, X., Low, M., Lobell, D. & Ermon, S. (2017). Deep Gaussian process for crop yield prediction based on remote sensing data. *AAAI*.

## Reproducibility
- `python research/reproduce.py` re-runs the headline results, including the Texas area result (T51), T50 and the pooled blind test; 10/10 matched under the pinned environment. The pooled step re-derives the pool from the committed per-era results.
- The preregistrations are in `docs/T39_…` to `docs/T55_…` and the evidence files in `research/evidence/`.
- The trial ledger is `docs/CROP_NOWCAST_TRIALS.md`.
