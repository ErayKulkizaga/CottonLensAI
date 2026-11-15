# Crop nowcast trial ledger

**Goal:** beat USDA's monthly forecast of cotton production, for Texas and for the US.
One row per trial. Nothing is deleted. Failed trials stay, so that related ideas can be combined later.

## How to read the numbers

- **Error metric:** mean absolute % error of production, in development years 2013–2021.
- **Protocol:** rolling origin. Each test year uses a model trained only on earlier years.
- **Pinned environment:** sklearn 1.6.1, pandas 2.2.3.
- **Sealed years:** 2022–2025 are not used here.

| Cutoff | USDA, TX / US | Naive base, June area, TX / US | Naive base, March area, TX / US |
|---|---|---|---|
| 1 Aug | **14.8 / 7.2** | 20.8 / 11.9 | 21.0 / 13.0 |
| 1 Sep | **11.2 / 6.3** | 20.8 / 11.9 | 21.0 / 13.0 |
| 1 Oct | **11.0 / 6.1** | 20.8 / 11.9 | 21.0 / 13.0 |

## Trials

Cells give TX / US error. **Bold** marks a cell that beats USDA.

| ID | Trial | Area | 1 Aug | 1 Sep | 1 Oct | Verdict | Evidence |
|---|---|---|---|---|---|---|---|
| T01 | State ridge, weather z-scores | March | 22.1 / 13.0 | 17.6 / 13.8 | 16.1 / 11.8 | worse than USDA | crop-nowcast-dev-ndvi-20261114 |
| T02 | State ridge, weather z + anomalies | March | 24.6 / 11.9 | 17.0 / 12.7 | 16.6 / 10.8 | worse | same |
| T03 | State ridge, MODIS NDVI levels | March | 27.5 / 15.2 | 24.0 / 13.3 | 22.6 / 12.6 | worse than base (overfits) | same |
| T04 | State ridge, weather z + NDVI | March | 24.1 / 15.2 | 21.9 / 14.1 | 20.2 / 13.2 | worse | same |
| T05 | County HGB, NDVI anomaly → state | March | 19.8 / 11.2 | 17.6 / 10.6 | 14.4 / 10.2 | best with March area | same |
| T06 | County HGB, NDVI anomaly → state | June | 18.9 / 9.9 | 15.1 / 7.9 | 12.7 / 7.7 | **best single model**; still above USDA | same |
| T07 | County HGB, weather z (+NDVI) | June | 20.9 / 11.1 | 16.7 / 8.2 | 14.7 / 8.5 | adding weather hurts | same |
| T08 | Blend of T05 and USDA, 50/50 (log-mean) | March | 16.0 / 9.0 | 13.6 / 7.9 | 12.1 / – | worse than USDA | session log 14 Nov |
| T09 | Blend of T05 and USDA, weights from past MSE | March | 15.2 / – | 11.3 / – | **10.8** / – | about equal to USDA (noise) | session log 14 Nov |
| T10 | Predict the direction of USDA's revision | – | – | corr +0.28, hit rate 57% (65% when the gap is above 10%) | corr +0.07 | weak | same |
| T11 | County-level accuracy vs naive base | June | −5.7% | +3.4% | +7.7% | weak at county level | session log 15 Nov |
| T12 | Move 3 primary: NDVI anomaly + Landsat field features | June | 19.0 / 7.4 | 20.6 / 9.0 | 17.0 / 7.6 | gate FAIL (≤16.0 needed) | move3-landsat-dev-20261115 |
| T13 | Landsat field features only (secondary) | June | 14.9 / **6.2** | 21.6 / 8.5 | 22.4 / 10.0 | US Aug beats USDA, but post hoc and unstable | same |

## Findings to combine later

1. County NDVI anomaly is the strongest single signal. Its correlation with production per planted acre at 1 Oct is +0.52 (+0.62 in Texas).
2. Satellite captures crop abandonment. On 1 Oct, the per-acre error in TX was about 0 in 2019 and 2020, while USDA's August forecast was +33% and +36%.
3. Errors of the satellite model and USDA correlate at 0.5–0.8. Blending helps only a little (T09).
4. Landsat field features help in August at US level (T13) but degrade later in the season. A likely cause is the varying number of observations; 2012 has only about half the scenes.
5. Area is a large error source. March intentions add about 6% to TX error; the June Acreage report removes part of it.

## Queue (next attempts)

- **T14** FSA monthly crop acreage: planted and failed cotton acres by county, split by irrigation. The August files (as of about 1 Aug) and September files cover 2011–2021. This is a direct measure of abandonment and area.
