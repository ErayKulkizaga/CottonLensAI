# 15 · Where USDA's cotton forecast errors come from

Production = harvested area × yield, so USDA's production error splits exactly:

log(final / forecast) = log(area_final / area_forecast) + log(yield_final / yield_forecast)

The check holds in the data (P × 480 / (H × Y) is between 0.999 and 1.001). Code: `research/crop_nowcast/error_mechanism.py` (2012–2025, QuickStats) and `t48_mechanism.py` (1995–2011, parsed reports, preregistered as T48).

## Texas: the error source changed around 2012
| Texas, 1 Sep report | 1995–2011 (17 seasons) | 2012–2025 (14 seasons) |
|---|---|---|
| Mean abs. production error | 12.1% | 11.8% |
| Mean abs. **harvested-area** error | **3.8%** | **14.9%** |
| Mean abs. yield error | 10.2% | 10.6% |
| Area share of error variance | 0.24 | 0.72 |

Texas harvested-area error at 1 Sep, by season (%; negative = USDA's area too high):

| Period | Seasons |
|---|---|
| 1995–2011 | 1995 +0.9 · 1996 0.0 · 1997 −1.9 · 1998 0.0 · 1999 +2.0 · 2000 −18.6 · 2001 −5.7 · 2002 −2.2 · 2003 −3.4 · 2004 −2.8 · 2005 +1.8 · 2006 −2.4 · 2007 0.0 · 2008 −4.5 · 2009 −5.6 · 2010 −0.9 · 2011 −11.6 |
| 2012–2025 | 2012 −25.4 · 2013 −6.3 · 2014 −11.3 · 2015 0.0 · 2016 −1.9 · 2017 −7.0 · 2018 −6.9 · 2019 −15.3 · 2020 −20.1 · 2021 +5.6 · 2022 −22.3 · 2023 −52.5 · 2024 −23.0 · 2025 +10.5 |

**Findings:**
1. **Before 2012, Texas misses were yield misses.** Since 2012 they have been harvested-area misses. The area error roughly quadrupled, and in most recent seasons USDA's in-season harvested area was too high. In other words, abandonment was under-estimated, which is the source of the 2014–2021 over-forecast bias that broke T25/T36 when it reversed.
2. **USDA learns little about harvested area during the season, in both periods.** The December area error is 79% (1995–2011) to 98% (2012–2025) of the August one. The production number is still revised, because area and yield errors offset: in Dec 2012–2025 the Texas area error is 14.7% and the yield error 12.3%, but the production error is only 5.7%.
3. **The United States is a yield story in both periods.** The yield part carries most of the US error variance from August to November (0.58–0.94 in 1995–2011, 0.58–0.75 from Sep in 2012–2025). By December it is close to even in 1995–2011 (0.46) and still 0.65 in 2012–2025.
4. **The momentum effect has no single home.** In 2012–2025 the Texas revision correlated with the remaining *area* error (+0.44 to +0.65) and negatively with the yield error. In 1995–2011 it correlated with both. The preregistered H3 failed.

**What this explains:**
- why a fixed bias correction was a bet on the Texas abandonment regime (T39, T42);
- why the momentum coefficient shrinks and the Sep coefficient turned negative (`13_LITERATURE_AND_CONTRIBUTION.md`);
- why a satellite signal of failed fields is the right target in principle (T43) but must see abandonment before USDA does, which in 2023 it did not at 1 Sep / 1 Oct.

**What the data cannot say:** *why* the Texas area error grew. Candidates are more frequent dryland abandonment in West Texas and USDA's use of FSA administrative acreage for the final harvested area. Testing this needs FSA failed-acre data by season, which exists from about 2008 (T29/T41 used it for 2013–2025).

## The problem worth solving
For Texas since 2012, the useful question is not "what is the yield?" but "**how much planted cotton will not be harvested, and when can that be known?**" That is a narrower, testable problem. It is also where public data (satellite greenness, drought indices, FSA failed acres) could add information that USDA's monthly survey does not yet hold.
