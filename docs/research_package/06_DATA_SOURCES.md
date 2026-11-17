# 06 · Data sources and leakage control

The rule in both phases: a value may enter a forecast dated *t* only if it was public before *t*. Where the true first-publication time could not be proven, the assumption is written down and the result is labelled accordingly.

## Phase 1 (price and risk)

| Source | Used for | Availability rule / notes |
|---|---|---|
| ICE cotton futures continuous CT=F (Yahoo), daily OHLCV, 2010 onward | price targets, returns, volatility, volume features | decision at bar date + 1 day 00:15 UTC. The first observed price is kept and later revisions are logged (a Yahoo revision of the 30 Oct 2026 close broke cohort v1) |
| USDA AMS daily and weekly cotton futures reports | "named" same-contract series | 1,262 daily reports reconciled; past contract months found in the annual workbooks (217 origins affected) |
| USDA AMS archives 2005–2009 (PDF/XLS) | pre-2010 replication | internal dates checked; 1,110 sessions rebuilt |
| DXY (US dollar index), WTI crude | cross-asset returns and 60-day correlations | same-day close, next-day decision |
| USDA WASDE (as-reported vintages) | stocks/use, production/use, balance sheet, text | release-time clock; lags L1/L2/L6 |
| NASS Crop Progress and condition (national and Texas) | condition and progress gaps | report date + 1 day 00:00 UTC (D0), or one more session (D1) |
| USDA FAS export sales | weekly sales | release + first session |
| CFTC Commitments of Traders, cotton 033661 futures-only | managed-money net/OI, turnover | report + 7 days, strict next session |
| ECB and FX rates | currency features | daily |
| Other crop prices (corn, soybeans, wheat) | crop-exploration and agri transfer | daily |
| Cotton on-call | oncall-exploration | weekly |
| NASA POWER weather 2010–2023 | weather-exploration | daily |
| ICE price-limit announcements | limit diagnostic | 36 of 75 verified (32 blocked by HTTP 429) |

## Phase 2 (production vs USDA)

| Source | Volume | Availability rule |
|---|---|---|
| NASS QuickStats (upland cotton; county, state, US; finals and monthly forecasts 2012+) | 42,423 rows | a forecast is public on its release day; county finals for Y−1 before season Y |
| USDA Crop Production text reports 2000–2013 (ESMIS) | 72 reports; 963 state and 59 US forecasts | release date printed in the report |
| Crop Production release dates | 2012–2025 | taken from the ESMIS listing |
| MODIS MOD13Q1 / MYD13Q1 v061, 250 m, 16-day (Planetary Computer) | 2,278 composites, 5 tiles | composite end + 8 days ≤ cutoff |
| Landsat 5/7/8 Collection 2 Level-2 (Planetary Computer) | 28,977 scenes | acquisition + 16 days ≤ cutoff |
| USDA Cropland Data Layer (Planetary Computer ≤ 2021; NASS CropScape 2022+) | cotton mask | year Y−1 only |
| FSA crop acreage data (planted and failed acres) | 41 files, 39,161 rows (2011–2024) | earliest monthly snapshot. FSA "planted" excludes failed acres. Workbook save times checked against USDA report dates: Sep/Oct files of 2013–2015 and Oct 2018 were saved after the report, so those claims were withdrawn |
| NASS Crop Progress (cotton condition) | 292 weekly reports | released the Monday after the week ends |
| US Drought Monitor (county D0–D4) | 103,622 county-weeks | map date + 2 days |
| NASA POWER daily (264 counties) | 1991–2024 | date ≤ cutoff − 3 days |
| June Acreage | 14 reports | released about 30 June |
| NASS Cotton Ginnings | 65 reports | 15 Nov report released 22–25 Nov, before the December forecast. The 1 Dec ginnings report is released on the same day as Crop Production, so it is not usable for that report |
| RMA cause of loss (crop insurance) | screened | **rejected**: records have the month the loss occurred but no publication date, and files are regenerated as claims settle, so they leak future information |
| AMS Quality of Cotton Classed by office | screened | leak-free (dated weekly) but only about 7 seasons (2019+) on ESMIS; not used |

## Seals
- **2022–2025:** sealed until a pushed preregistration (T39). Code guards refuse sealed years without `--unseal`.
- **2025:** kept unseen until T42.
- **2000–2011 USDA forecasts:** never pulled before T45 (blind).
- **Prices and returns after 2021** stay sealed for the price chain.
