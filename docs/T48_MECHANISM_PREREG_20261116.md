# T48 preregistration: does the USDA cotton error mechanism found in 2012–2025 hold in 1995–2011?

Written 16 Nov 2026. Harvested-area and yield finals for 1995–2011 have not been downloaded. The area and yield forecast columns of the 1995–2011 reports have never been parsed or looked at; only their production column was used, in T45 and T47.

## Finding to confirm (from 2012–2025, `research/evidence/error-mechanism-2012-2025-20261116.json`)
Production error splits exactly into a harvested-area part and a yield part.
- **Texas:** the area part carries 72–94% of the error variance from Sep to Nov. The area error does not shrink during the season (Aug 15.1% → Dec 14.7%). The latest production revision correlates with the remaining *area* error (+0.44 to +0.65) and not with the yield error (negative).
- **United States:** from Oct on, the yield part carries most of the variance (58–75%).

## Data
- **Forecasts:** cached Crop Production text reports, 1995–2011, Aug–Dec, upland block. Columns: current harvested area (1,000 acres), current yield (lb/acre) and current production (1,000 bales). Parser `t48_mechanism.rows` uses the T47 pre-clean. A row is kept only if production × 480 / (area × yield) is within 3% of 1.
- **Finals:** QuickStats `YEAR` rows for harvested area, yield and production, state and US.

## Hypotheses (all four must pass for ALL PASS; each is reported separately)
- **H1:** in Texas, the area share of the error variance is > 0.5 in at least 2 of Sep, Oct, Nov.
- **H2:** in Texas, the December mean absolute area error is at least 75% of the August one (USDA does not learn harvested area in-season).
- **H3:** in Texas, corr(revision, area error) > corr(revision, yield error) in at least 3 of Sep, Oct, Nov, Dec.
- **H4:** in the US, the yield share of the error variance is > 0.5 in at least 2 of Oct, Nov, Dec.

These are descriptive hypotheses with 17 seasons each and no p-value gates. They test whether the 2012–2025 pattern recurs in an earlier, separate period. Results go to `research/evidence/t48-mechanism-1995-2011-20261116.json` and are reported in full.

## Amendment 1 (16 Nov 2026, before the finals were downloaded)
- **First parse:** August rows failed the check, because August has no previous-month yield column. The parser now takes, as current yield, the yield column that satisfies the production = area × yield check (scanning from the right).
- **Sep 2010:** this report spells its title "Yield, and, Production". It is normalised before parsing.
- **Result:** 85 of 85 reports parsed, with TX and US rows in every report.
- **Disclosure about T45:** the same typo means the T45 cotton parser (`cp_pull.py`) never read the Sep 2010 report. In T45 and in the 2002–2025 momentum record, Sep 2010 is missing, and the Oct 2010 revision was measured from August. T45 is a one-shot test and is not re-run; this is recorded in `docs/CROP_NOWCAST_TRIALS.md`.

## Result (16 Nov 2026) — ALL PASS: no (H1 FAIL, H2 PASS, H3 FAIL, H4 PASS)

| Texas, 1995–2011 (n = 17) | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|
| Mean abs. area error % | 4.4 | 3.8 | 3.4 | 3.7 | 3.5 |
| Mean abs. yield error % | 13.3 | 10.2 | 7.1 | 4.1 | 3.5 |
| Area share of error variance | 0.21 | 0.24 | 0.28 | 0.46 | 0.56 |
| corr(revision, area) · corr(revision, yield) | – | 0.49 · 0.40 | 0.44 · 0.46 | 0.38 · 0.47 | 0.67 · −0.25 |

| United States, 1995–2011 | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|
| Mean abs. area error % | 2.2 | 2.0 | 1.7 | 1.7 | 1.6 |
| Mean abs. yield error % | 7.2 | 5.7 | 4.3 | 2.0 | 1.5 |
| Yield share of error variance | 0.94 | 0.88 | 0.80 | 0.58 | 0.46 |

- **H1 FAIL.** In 1995–2011 the Texas error was mainly yield, not area: the area share was 0.24–0.46 from Sep to Nov.
- **H2 PASS.** The December area error is 79% of the August one; USDA learns little about harvested area during the season, in both periods.
- **H3 FAIL** (2 of 4). The revision correlates with both parts.
- **H4 PASS.** In the US, yield dominates from Oct to Nov in both periods.

**Reading:**
- The 2012–2025 mechanism is not a fixed property of USDA's cotton forecasts. The Texas harvested-area error grew from about 3.5–4.4% (1995–2011) to about 15% (2012–2025), and the dominant source of the Texas error shifted from yield to area.
- This is a regime change, which matches the shrinking momentum coefficient and the bias flips (T39, T42).
- The data do not identify the cause. Candidates are more frequent dryland abandonment and USDA's growing use of FSA administrative acreage for final harvested area.
- Evidence: `research/evidence/t48-mechanism-1995-2011-20261116.json` / `.log`.
