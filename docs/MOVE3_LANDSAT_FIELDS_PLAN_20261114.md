# Move 3: field-level Landsat signal for the 1 August Texas nowcast (pre-declared)

Written and committed **before** any Landsat feature is computed.
Development years 2009–2021 only. The sealed years 2022–2025 stay sealed.

## Why this move
At 1 August, the best MODIS model misses Texas production by 18.9% on average
(USDA: 14.8%). Area is no longer the issue: June Acreage is used. The remaining
error is per planted acre.

At 250 m, a MODIS pixel mixes irrigated fields that survive with dryland
fields that fail. Field-scale imagery can count failing fields directly.

## Source and availability (no credentials needed)
- **Imagery:** Landsat Collection 2 Level-2 surface reflectance, from
  Planetary Computer (`landsat-c2-l2`).
  - Satellites: Landsat 5, 7 and 8. Tier-1 scenes only, scene cloud cover < 60%.
  - Clouds are masked with `qa_pixel`: a pixel is kept only if its clear bit is set.
- **Rejected alternatives:**
  - The Planetary Computer HLS archive starts in 2020, too late for a historical model.
  - NASA's own HLS archive needs a login.
  - Sentinel-2 is dense only from 2019, which would change data density across the years.
- **Availability rule:** a scene counts for cutoff C only if acquisition date + **16 days** ≤ C.
  - This is a conservative Level-2 latency.
  - Consequence: at 1 August the last usable scene is from mid-July, the same as MODIS.
  - Any gain must therefore come from spatial resolution, not from timing.
- **Archive vintage:** this is the Collection 2 reprocessing, the same kind of
  vintage limit as MODIS v061.

## Extraction
- **Sample points.** The centers of the prior-year (Y−1) CDL cotton pixels,
  sampled on a 120 m grid inside the 81 Texas panel counties.
- **Read resolution.** Imagery is read from the 120 m COG overview. The QA band
  is checked to make sure its overview is sampled, not averaged (smoke test).
- **NDVI.** For each point and each clear observation:
  NDVI = (NIR − Red) / (NIR + Red), with Collection 2 scale 0.0000275 and offset −0.2.

## County features at each cutoff (aug1, sep1, oct1)
Built from each point's maximum NDVI to date (`pmax`):
- `l_p10`, `l_p25`, `l_p50`, `l_p75`, `l_p90`: percentiles of `pmax` across the
  county's points;
- `l_fail25`, `l_fail35`: share of points with `pmax` < 0.25 and < 0.35 (fields
  that never greened);
- `l_last`: median NDVI of each point's latest clear observation;
- `l_nobs`: median number of clear observations per point.

Each feature also gets a past-only anomaly, `<name>_a`: its value minus the
same county's mean over earlier years.

## Primary variant (the only one that decides the gate)
- **Harness.** Same as before: `research/crop_nowcast/county_up.py`, county
  HistGradientBoosting, June Acreage area, rolling origin, test years 2013–2021.
- **Features.** MODIS NDVI anomalies plus all `l_*` and `l_*_a` features.
- **Secondary, reported but not decisive.** Landsat features alone; sep1 and oct1.

## Gate (Texas, 1 August, mean absolute % production error, 2013–2021)
- **PASS:** TX aug1 MAPE ≤ **16.0%** (within 1.2 points of USDA's 14.8%)
  **and** below the MODIS-only 18.9%.
  - Then write the preregistration, push it, and open 2022–2025 once.
- **FAIL:** record the negative result. The sealed years stay sealed.
- The threshold is fixed here and is not moved after seeing results.
