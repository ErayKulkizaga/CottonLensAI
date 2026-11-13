"""Exploratory satellite vignette: Texas High Plains cotton NDVI, 2009-2021 (pre-test years only).

MODIS MOD13Q1/MYD13Q1 v061 250 m NDVI (Planetary Computer, anonymous) over cotton
pixels from the PREVIOUS year's USDA CDL (class 2), never the same year's map and
never the multi-year frequency/cultivated layers. Composites are dated by
end_datetime. Archive is the 2021 reprocessing, not the real-time vintage.
Illustration only: no yield model, no score, nothing from 2022+.
"""
import json
import sys
from pathlib import Path

import numpy as np

BBOX = [-102.6, 33.0, -101.0, 34.4]          # Lubbock-centred High Plains window
YEARS = range(2009, 2022)
STAC = 'https://planetarycomputer.microsoft.com/api/stac/v1'
COTTON = 2


def main(out_dir):
    import planetary_computer as pc
    import pystac_client
    import rasterio
    from pyproj import Transformer
    from rasterio.windows import from_bounds

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cat = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_MAX_RETRY='3', GDAL_HTTP_RETRY_DELAY='2')
    env.__enter__()

    def modis_items(year):
        items = cat.search(collections=['modis-13Q1-061'], bbox=BBOX, datetime=f'{year}-04-01/{year}-10-31').item_collection()
        return sorted([i for i in items if '.h09v05.' in i.id], key=lambda i: i.properties['end_datetime'])

    # MODIS grid window (fixed for all composites of tile h09v05)
    first = modis_items(2010)[0]
    with rasterio.open(first.assets['250m_16_days_NDVI'].href) as src:
        to_sin = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
        xs, ys = to_sin.transform([BBOX[0], BBOX[2], BBOX[0], BBOX[2]], [BBOX[1], BBOX[1], BBOX[3], BBOX[3]])
        win = from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform).round_offsets().round_lengths()
        wt = src.window_transform(win)
        crs = src.crs
        H, W = int(win.height), int(win.width)
    cols, rows = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    px, py = wt * (cols, rows)
    lon, lat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(px, py)

    def cotton_mask(cdl_year):
        items = cat.search(collections=['usda-cdl'], bbox=BBOX, datetime=f'{cdl_year}-01-01/{cdl_year}-12-31').item_collection()
        items = [i for i in items if i.properties.get('usda_cdl:type') == 'cropland']
        mask = np.zeros((H, W), bool)
        for it in items:
            with rasterio.open(it.assets['cropland'].href) as s:
                tx, ty = Transformer.from_crs('EPSG:4326', s.crs, always_xy=True).transform(lon, lat)
                b = s.bounds
                inside = (tx >= b.left) & (tx < b.right) & (ty > b.bottom) & (ty <= b.top)
                if not inside.any():
                    continue
                w = from_bounds(tx[inside].min(), ty[inside].min(), tx[inside].max(), ty[inside].max(), s.transform)
                w = w.round_offsets().round_lengths()
                scale = 4                                   # ~120 m sampling of the 30 m map
                arr = s.read(1, window=w, out_shape=(max(1, int(w.height) // scale), max(1, int(w.width) // scale)))
                t = s.window_transform(w) * rasterio.Affine.scale(w.width / arr.shape[1], w.height / arr.shape[0])
                c, r = ~t * (tx, ty)
                c, r = np.floor(c).astype(int), np.floor(r).astype(int)
                ok = inside & (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
                mask[ok] |= arr[r[ok], c[ok]] == COTTON
        return mask

    curves, maps = {}, {}
    for year in YEARS:
        mask = cotton_mask(year - 1)
        series = []
        for it in modis_items(year):
            with rasterio.open(it.assets['250m_16_days_NDVI'].href) as s:
                nd = s.read(1, window=win).astype(float)
            nd[(nd < -2000) | (nd > 10000)] = np.nan
            nd /= 10000.0
            end = it.properties['end_datetime'][:10]
            series.append({'end': end, 'median_ndvi_cotton': float(np.nanmedian(nd[mask])), 'id': it.id})
            if end[5:] <= '07-31':
                maps[year] = (end, nd)                      # last composite ending by 31 July
        curves[year] = {'cdl_year': year - 1, 'cotton_pixels': int(mask.sum()), 'composites': series}
        if year in (2010, 2011):
            np.save(out / f'mask_{year}.npy', mask)
        print(year, mask.sum(), len(series), flush=True)
    for y in (2010, 2011):
        end, nd = maps[y]
        np.save(out / f'ndvi_{y}.npy', nd)
        curves[y]['map_composite_end'] = end
    meta = {'bbox': BBOX, 'grid': {'height': H, 'width': W, 'crs': str(crs)}, 'years': list(YEARS),
            'rules': 'CDL year Y-1 cotton class 2; MODIS v061 end_datetime; no 2022+', 'curves': curves}
    (out / 'texas_ndvi.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])
