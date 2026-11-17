"""T35 input: per county x 16-day composite histograms of cotton pixels, 6 MOD13Q1 bands (dev years only).

Same masking as county_ndvi.py (prior-year CDL cotton, MODIS v061 from Planetary Computer). Bins are
fixed physical ranges (VI -0.2..1.0, reflectance 0..0.6), so no data enter the binning. Counts are
additive, so counties split across tiles are summed later. One .npz per (tile, year): resumable,
a finished unit is never recomputed. Works the same on a laptop, Kaggle or Colab.
Usage: python county_hist.py panel.json out_dir [first last] [--tile=h09v05] [--workers=8] [--limit=N]
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

STAC = 'https://planetarycomputer.microsoft.com/api/stac/v1'
COTTON, BINS, SEALED_FROM = 2, 32, 2022
BANDS = [('250m_16_days_NDVI', -0.2, 1.0), ('250m_16_days_EVI', -0.2, 1.0),
         ('250m_16_days_red_reflectance', 0.0, 0.6), ('250m_16_days_NIR_reflectance', 0.0, 0.6),
         ('250m_16_days_blue_reflectance', 0.0, 0.6), ('250m_16_days_MIR_reflectance', 0.0, 0.6)]


def histograms(values, county_idx, n_counties):
    """values: [6, n_pixels] scaled band values (NaN = invalid); county_idx: [n_pixels] -> counts [n_counties, 6, BINS]."""
    out = np.zeros((n_counties, len(BANDS), BINS), np.int32)
    for b, (_, lo, hi) in enumerate(BANDS):
        v = values[b]
        ok = np.isfinite(v)
        k = np.clip(((v[ok] - lo) / (hi - lo) * BINS).astype(int), 0, BINS - 1)
        np.add.at(out[:, b, :], (county_idx[ok], k), 1)
    return out


def main(panel_json, out_dir, first=2009, last=2021):
    first, last = int(first), int(last)
    if last >= SEALED_FROM and '--unseal' not in sys.argv:
        raise SystemExit('Sealed years need --unseal after preregistration')
    import planetary_computer as pc

    try:
        from research.crop_nowcast.cdl_source import cdl_hrefs
    except ImportError:  # Kaggle/Colab: shipped flat next to this script
        from cdl_source import cdl_hrefs
    import pystac_client
    import pystac_client.exceptions
    import rasterio
    import rasterio.errors
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.windows import Window, from_bounds

    def arg(name, default):
        return next((a.split('=', 1)[1] for a in sys.argv if a.startswith(f'--{name}=')), default)

    workers, limit = int(arg('workers', 8)), arg('limit', None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel = json.loads(Path(panel_json).read_text())['counties']
    tiles = sorted({t for c in panel for t in c['tiles']})
    if arg('tile', None):
        tiles = [t for t in tiles if t in arg('tile', '').split(',')]
    cat = pystac_client.Client.open(STAC)
    rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_MAX_RETRY='5', GDAL_HTTP_RETRY_DELAY='3').__enter__()

    def retry(fn, tries=6):
        for k in range(tries):
            try:
                return fn()
            except (rasterio.errors.RasterioIOError, OSError, pystac_client.exceptions.APIError):
                if k == tries - 1:
                    raise
                time.sleep(15 * (k + 1))

    def read(href, **kw):
        def go():
            with rasterio.open(pc.sign(href)) as s:
                return s.read(1, **kw)
        return retry(go)

    def search(**kw):
        return retry(lambda: list(cat.search(**kw).item_collection()))

    for tile in tiles:
        members = [c for c in panel if tile in c['tiles']]
        bb = [min(c['bbox'][0] for c in members), min(c['bbox'][1] for c in members),
              max(c['bbox'][2] for c in members), max(c['bbox'][3] for c in members)]
        todo = [y for y in range(first, last + 1) if not (out / f'{tile}_{y}.npz').exists()]
        if not todo:
            continue
        probe = next(i for i in search(collections=['modis-13Q1-061'], bbox=bb, datetime='2015-06-01/2015-07-31')
                     if f'.{tile}.' in i.id)
        with retry(lambda probe=probe: rasterio.open(pc.sign(probe.assets['250m_16_days_NDVI'].href))) as src:
            to_sin = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
            xs, ys = to_sin.transform([bb[0], bb[2], bb[0], bb[2]], [bb[1], bb[1], bb[3], bb[3]])
            win = from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform).round_offsets().round_lengths()
            win = win.intersection(Window(0, 0, src.width, src.height))
            wt, crs, H, W = src.window_transform(win), src.crs, int(win.height), int(win.width)

        def sin_geom(g, to_sin=to_sin):
            def ring(r):
                x, y = to_sin.transform([p[0] for p in r], [p[1] for p in r])
                return list(zip(x, y, strict=True))
            if g['type'] == 'Polygon':
                return {'type': 'Polygon', 'coordinates': [ring(r) for r in g['coordinates']]}
            return {'type': 'MultiPolygon', 'coordinates': [[ring(r) for r in p] for p in g['coordinates']]}

        cid = rasterize([(sin_geom(c['geometry']), i + 1) for i, c in enumerate(members)], out_shape=(H, W), transform=wt,
                        fill=0, dtype='int16')
        sel = np.flatnonzero(cid.ravel())
        rr, cc = np.unravel_index(sel, (H, W))
        lon, lat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(*(wt * (cc + 0.5, rr + 0.5)))
        county_of = cid.ravel()[sel] - 1
        n_pix, n_cty = len(sel), len(members)

        def cotton(cdl_year, bb=bb, lon=lon, lat=lat, n=n_pix, members=members):
            m = np.zeros(n, bool)
            for href in cdl_hrefs(search, bb, cdl_year, [c['fips'][:2] for c in members]):
                def go(href=href):
                    with rasterio.open(pc.sign(href)) as s:
                        tx, ty = Transformer.from_crs('EPSG:4326', s.crs, always_xy=True).transform(lon, lat)
                        b = s.bounds
                        inside = (tx >= b.left) & (tx < b.right) & (ty > b.bottom) & (ty <= b.top)
                        if not inside.any():
                            return None
                        w = from_bounds(tx[inside].min(), ty[inside].min(), tx[inside].max(), ty[inside].max(),
                                        s.transform).round_offsets().round_lengths()
                        arr = s.read(1, window=w, out_shape=(max(1, int(w.height) // 4), max(1, int(w.width) // 4)))
                        t = s.window_transform(w) * rasterio.Affine.scale(w.width / arr.shape[1], w.height / arr.shape[0])
                        c2, r2 = ~t * (tx, ty)
                        c2, r2 = np.floor(c2).astype(int), np.floor(r2).astype(int)
                        ok = inside & (r2 >= 0) & (r2 < arr.shape[0]) & (c2 >= 0) & (c2 < arr.shape[1])
                        hit = np.zeros(len(tx), bool)
                        hit[ok] = arr[r2[ok], c2[ok]] == COTTON
                        return hit
                h = retry(go)
                if h is not None:
                    m |= h
            return m

        for year in todo:
            t0 = time.time()
            mask = cotton(year - 1)
            idx = county_of[mask]
            items = sorted([i for i in search(collections=['modis-13Q1-061'], bbox=bb, datetime=f'{year}-04-01/{year}-10-31')
                            if f'.{tile}.' in i.id], key=lambda i: i.properties['end_datetime'])
            if limit:
                items = items[:int(limit)]

            def composite(it, sel=sel, mask=mask, idx=idx, n=n_cty, win=win):
                vals = []
                for asset, _, _ in BANDS:
                    v = read(it.assets[asset].href, window=win).ravel()[sel][mask].astype(np.float32)
                    v[(v < -2000) | (v > 10000)] = np.nan
                    vals.append(v / 10000.0)
                return histograms(np.stack(vals), idx, n)

            with ThreadPoolExecutor(workers) as ex:
                hists = list(ex.map(composite, items))
            part = out / f'{tile}_{year}.part.npz'
            np.savez_compressed(part, fips=np.array([c['fips'] for c in members]),
                                end=np.array([i.properties['end_datetime'][:10] for i in items]),
                                sensor=np.array([i.id[:3] for i in items]), hist=np.stack(hists).astype(np.int32))
            part.replace(out / f'{tile}_{year}.npz')
            print(tile, year, len(items), int(mask.sum()), round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main(*[a for a in sys.argv[1:] if not a.startswith('--')])
