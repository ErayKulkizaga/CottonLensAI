"""Season time-lapse frames: Texas High Plains cotton NDVI, every MODIS 16-day composite, 2009-2021.

Same rules as satellite_texas.py: MODIS MOD13Q1/MYD13Q1 v061 (Planetary Computer,
anonymous), composites dated by end_datetime, cotton mask from the PREVIOUS year's
USDA CDL class 2 (never same-year, never multi-year layers), nothing from 2022+.
Frames are resampled (nearest) to a lon/lat grid and stored as one grayscale PNG
sprite per year: 0 = not cotton / no data, 1..255 = NDVI 0..1. Also records how
much was processed. Illustration only: no yield model, no score.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

BBOX = [-102.6, 33.0, -101.0, 34.4]
YEARS = range(2009, 2022)
GRID_W, GRID_H = 400, 420
STAC = 'https://planetarycomputer.microsoft.com/api/stac/v1'
COTTON = 2


def main(out_dir):
    import planetary_computer as pc
    import pystac_client
    import rasterio
    from PIL import Image
    from pyproj import Transformer
    from rasterio.windows import from_bounds

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cat = pystac_client.Client.open(STAC, modifier=pc.sign_inplace)
    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_MAX_RETRY='4', GDAL_HTTP_RETRY_DELAY='2')
    env.__enter__()

    def modis_items(year):
        items = cat.search(collections=['modis-13Q1-061'], bbox=BBOX, datetime=f'{year}-04-01/{year}-10-31').item_collection()
        return sorted([i for i in items if '.h09v05.' in i.id], key=lambda i: i.properties['end_datetime'])

    first = modis_items(2010)[0]
    with rasterio.open(first.assets['250m_16_days_NDVI'].href) as src:
        to_sin = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
        xs, ys = to_sin.transform([BBOX[0], BBOX[2], BBOX[0], BBOX[2]], [BBOX[1], BBOX[1], BBOX[3], BBOX[3]])
        win = from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform).round_offsets().round_lengths()
        wt, crs = src.window_transform(win), src.crs
        H, W = int(win.height), int(win.width)
    # window pixel centres -> lon/lat (for CDL sampling)
    cols, rows = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    lon, lat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(*(wt * (cols, rows)))
    # display grid (lon/lat) -> window pixel (nearest)
    glon = BBOX[0] + (np.arange(GRID_W) + 0.5) * (BBOX[2] - BBOX[0]) / GRID_W
    glat = BBOX[3] - (np.arange(GRID_H) + 0.5) * (BBOX[3] - BBOX[1]) / GRID_H
    GL, GT = np.meshgrid(glon, glat)
    sx, sy = to_sin.transform(GL, GT)
    gc, gr = ~wt * (sx, sy)
    gc, gr = np.floor(gc).astype(int), np.floor(gr).astype(int)
    gok = (gr >= 0) & (gr < H) & (gc >= 0) & (gc < W)
    gr, gc = np.clip(gr, 0, H - 1), np.clip(gc, 0, W - 1)

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
                scale = 4
                arr = s.read(1, window=w, out_shape=(max(1, int(w.height) // scale), max(1, int(w.width) // scale)))
                t = s.window_transform(w) * rasterio.Affine.scale(w.width / arr.shape[1], w.height / arr.shape[0])
                c, r = ~t * (tx, ty)
                c, r = np.floor(c).astype(int), np.floor(r).astype(int)
                ok = inside & (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
                mask[ok] |= arr[r[ok], c[ok]] == COTTON
        return mask

    stats = {'composites': 0, 'window_pixels_read': 0, 'cotton_pixel_observations': 0, 'cdl_maps': 0,
             'modis_tile': 'h09v05', 'window_shape': [H, W], 'display_grid': [GRID_H, GRID_W]}
    years = {}
    for year in YEARS:
        mask = cotton_mask(year - 1)
        stats['cdl_maps'] += 1
        frames, meta = [], []
        for it in modis_items(year):
            with rasterio.open(it.assets['250m_16_days_NDVI'].href) as s:
                nd = s.read(1, window=win).astype(float)
            nd[(nd < -2000) | (nd > 10000)] = np.nan
            nd /= 10000.0
            stats['composites'] += 1
            stats['window_pixels_read'] += nd.size
            valid = mask & np.isfinite(nd)
            stats['cotton_pixel_observations'] += int(valid.sum())
            g = nd[gr, gc]
            keep = gok & mask[gr, gc] & np.isfinite(g)
            frame = np.zeros((GRID_H, GRID_W), np.uint8)
            frame[keep] = (1 + np.clip(g[keep], 0, 1) * 254).round().astype(np.uint8)
            frames.append(frame)
            meta.append({'end': it.properties['end_datetime'][:10], 'start': it.properties['start_datetime'][:10],
                         'sensor': it.id[:3], 'median': float(np.nanmedian(nd[valid])) if valid.any() else None})
        Image.fromarray(np.vstack(frames)).save(out / f'frames_{year}.png', optimize=True)
        years[year] = {'cdl_year': year - 1, 'cotton_pixels': int(mask.sum()), 'frames': meta}
        print(year, len(frames), int(mask.sum()), round(time.time() - t0), flush=True)
    stats['seconds'] = round(time.time() - t0)
    stats['years'] = len(years)
    (out / 'frames.json').write_text(json.dumps({'bbox': BBOX, 'grid': [GRID_H, GRID_W], 'stats': stats,
                                                 'years': years}, indent=2), encoding='utf-8')
    print(json.dumps(stats))


if __name__ == '__main__':
    main(sys.argv[1])
