"""Field-scale Landsat NDVI features for the panel counties (Move 3, dev years only).

Rules (docs/MOVE3_LANDSAT_FIELDS_PLAN_20261114.md):
- points = Y-1 CDL cotton cells on a 120 m (or --grid=30/60) EPSG:5070 grid inside the panel counties;
- Landsat C2 L2 (5/7/8), Tier 1, scene cloud < 60 %, read from the 120 m COG overview,
  qa_pixel clear bit required (the 30 m grid reads full resolution);
- a scene counts for cutoff C only if acquired + LAG_DAYS <= C;
- years >= 2022 need --unseal after preregistration.
Writes landsat_{state}_{year}.csv (one row per county x cutoff), resumable per state-year.
Usage: python landsat_fields.py panel.json out_dir [first last] [--workers=8] [--grid=120] [--states=TX,OK] [--fips=48303,...] [--limit=N smoke test]
"""
import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

STAC = 'https://planetarycomputer.microsoft.com/api/stac/v1'
COTTON = 2
GRID = 120.0
LAG_DAYS = 16
CUTOFFS = {'aug1': '08-01', 'sep1': '09-01', 'oct1': '10-01'}
SEASON = ('04-01', '09-14')  # last scene usable for oct1 is acquired 15 Sep
PLATFORMS = ('landsat-5', 'landsat-7', 'landsat-8')
DEFAULT_YEARS = (2009, 2021)


def county_features(idx, day, ndvi, county_of, fips, year):
    """idx: point index per observation, day: acquisition date (datetime64[D]), ndvi: float.
    Returns rows of per-county features for each cutoff, honouring LAG_DAYS."""
    rows = []
    n_pts = len(county_of)
    for ck, md in CUTOFFS.items():
        lim = np.datetime64(f'{year}-{md}') - np.timedelta64(LAG_DAYS, 'D')
        ok = day <= lim
        i, d, v = idx[ok], day[ok], ndvi[ok]
        pmax = np.full(n_pts, -np.inf)
        np.maximum.at(pmax, i, v)
        nobs = np.bincount(i, minlength=n_pts)
        last = np.full(n_pts, np.nan)
        order = np.lexsort((d, i))  # by point, then date; last occurrence per point wins
        last[i[order]] = v[order]
        seen = nobs > 0
        for k, f in enumerate(fips):
            m = seen & (county_of == k)
            if m.sum() < 20:
                continue
            pm = pmax[m]
            q = np.percentile(pm, [10, 25, 50, 75, 90])
            rows.append({'fips': f, 'year': year, 'cutoff': ck, 'l_pts': int(m.sum()),
                         'l_p10': q[0], 'l_p25': q[1], 'l_p50': q[2], 'l_p75': q[3], 'l_p90': q[4],
                         'l_fail25': float((pm < 0.25).mean()), 'l_fail35': float((pm < 0.35).mean()),
                         'l_last': float(np.median(last[m])), 'l_nobs': float(np.median(nobs[m]))})
    return rows


def main(panel_json, out_dir, first=DEFAULT_YEARS[0], last=DEFAULT_YEARS[1]):
    first, last = int(first), int(last)
    if last >= 2022 and '--unseal' not in sys.argv:
        raise SystemExit('Sealed years need --unseal after preregistration')
    import pandas as pd
    import planetary_computer as pc
    import pystac_client
    import rasterio
    import rasterio.errors
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    def arg(name, default):
        return next((a.split('=', 1)[1] for a in sys.argv if a.startswith(f'--{name}=')), default)

    workers = int(arg('workers', 8))
    grid = float(arg('grid', GRID))
    ovr = {} if grid == 30 else {'overview_level': {60.0: 0, 120.0: 1}[grid]}  # COG overviews are x2, x4
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel = json.loads(Path(panel_json).read_text())['counties']
    only = arg('fips', None)
    if only:
        panel = [c for c in panel if c['fips'] in only.split(',')]
    states = arg('states', ','.join(sorted({c['state'] for c in panel}))).split(',')
    cat = pystac_client.Client.open(STAC)
    rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_MAX_RETRY='5', GDAL_HTTP_RETRY_DELAY='3').__enter__()

    def ropen(href, **kw):
        for k in range(6):  # sign at use; back off on transient 403/IO errors
            try:
                return rasterio.open(pc.sign(href), **kw)
            except rasterio.errors.RasterioIOError:
                if k == 5:
                    raise
                time.sleep(15 * (k + 1))


    def run_state(st, tx):
        fips = [c['fips'] for c in tx]
        # Albers grid over the state's panel counties, aligned to the CDL 30 m grid
        to_alb = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
        lon0, lat0 = min(c['bbox'][0] for c in tx), min(c['bbox'][1] for c in tx)
        lon1, lat1 = max(c['bbox'][2] for c in tx), max(c['bbox'][3] for c in tx)
        xs, ys = to_alb.transform([lon0, lon1, lon0, lon1], [lat0, lat0, lat1, lat1])
        x0, y1 = np.floor(min(xs) / grid) * grid, np.ceil(max(ys) / grid) * grid
        W, H = int(np.ceil((max(xs) - x0) / grid)), int(np.ceil((y1 - min(ys)) / grid))
        gt = from_origin(x0, y1, grid, grid)

        def alb(g):
            def ring(r):
                x, y = to_alb.transform([p[0] for p in r], [p[1] for p in r])
                return list(zip(x, y, strict=True))
            if g['type'] == 'Polygon':
                return {'type': 'Polygon', 'coordinates': [ring(r) for r in g['coordinates']]}
            return {'type': 'MultiPolygon', 'coordinates': [[ring(r) for r in p] for p in g['coordinates']]}

        cid = rasterize([(alb(c['geometry']), i + 1) for i, c in enumerate(tx)], out_shape=(H, W), transform=gt,
                        fill=0, dtype='int16')
        rr, cc = np.nonzero(cid)
        county_all = cid[rr, cc] - 1
        px_all, py_all = gt * (cc + 0.5, rr + 0.5)
        del cid
        print(st, 'grid', H, W, 'county cells', len(px_all), flush=True)

        def cotton_points(cdl_year):
            items = cat.search(collections=['usda-cdl'], bbox=[lon0, lat0, lon1, lat1],
                               datetime=f'{cdl_year}-01-01/{cdl_year}-12-31').item_collection()
            items = [i for i in items if i.properties.get('usda_cdl:type') == 'cropland']
            m = np.zeros(len(px_all), bool)
            for it in items:
                with ropen(it.assets['cropland'].href) as s:
                    tx_, ty_ = Transformer.from_crs('EPSG:5070', s.crs, always_xy=True).transform(px_all, py_all)
                    b = s.bounds
                    inside = (tx_ >= b.left) & (tx_ < b.right) & (ty_ > b.bottom) & (ty_ <= b.top)
                    if not inside.any():
                        continue
                    c2, r2 = ~s.transform * (tx_[inside], ty_[inside])
                    c2, r2 = np.floor(c2).astype(int), np.floor(r2).astype(int)
                    c_lo, r_lo = c2.min(), r2.min()
                    win = Window(c_lo, r_lo, c2.max() - c_lo + 1, r2.max() - r_lo + 1)
                    # one exact 30 m CDL pixel per grid point: decimated nearest read
                    sc = int(grid // 30)
                    arr = s.read(1, window=win, out_shape=(max(1, int(win.height) // sc), max(1, int(win.width) // sc)))
                    fr = np.clip(((r2 - r_lo) * arr.shape[0] / win.height).astype(int), 0, arr.shape[0] - 1)
                    fc = np.clip(((c2 - c_lo) * arr.shape[1] / win.width).astype(int), 0, arr.shape[1] - 1)
                    hit = arr[fr, fc] == COTTON
                    m[np.flatnonzero(inside)[hit]] = True
            return m

        for year in range(first, last + 1):
            dest = out / f'landsat_{st}_{year}.csv'
            if dest.exists():
                continue
            t0 = time.time()
            keep = cotton_points(year - 1)
            px, py, county_of = px_all[keep], py_all[keep], county_all[keep]
            lon, lat = Transformer.from_crs('EPSG:5070', 'EPSG:4326', always_xy=True).transform(px, py)
            items = cat.search(collections=['landsat-c2-l2'], bbox=[lon0, lat0, lon1, lat1],
                               datetime=f'{year}-{SEASON[0]}/{year}-{SEASON[1]}',
                               query={'eo:cloud_cover': {'lt': 60}}).item_collection()
            items = [i for i in items if i.properties.get('platform') in PLATFORMS
                     and i.properties.get('landsat:collection_category') == 'T1']
            limit = next((int(a.split('=', 1)[1]) for a in sys.argv if a.startswith('--limit=')), None)
            if limit:  # smoke tests only; output goes to a separate folder
                items = items[:limit]
            proj = {}

            def scene(it, px=px, py=py, lon=lon, lat=lat, proj=proj):
                b = it.bbox
                sel = np.flatnonzero((lon >= b[0]) & (lon <= b[2]) & (lat >= b[1]) & (lat <= b[3]))
                if not len(sel):
                    return None
                with ropen(it.assets['qa_pixel'].href, **ovr) as q:
                    key = str(q.crs)
                    if key not in proj:
                        proj[key] = Transformer.from_crs('EPSG:5070', q.crs, always_xy=True).transform(px, py)
                    ux, uy = proj[key]
                    c2, r2 = ~q.transform * (ux[sel], uy[sel])
                    c2, r2 = np.floor(c2).astype(int), np.floor(r2).astype(int)
                    ok = (c2 >= 0) & (r2 >= 0) & (c2 < q.width) & (r2 < q.height)
                    sel, c2, r2 = sel[ok], c2[ok], r2[ok]
                    if not len(sel):
                        return None
                    win = Window(c2.min(), r2.min(), c2.max() - c2.min() + 1, r2.max() - r2.min() + 1)
                    qa = q.read(1, window=win)[r2 - r2.min(), c2 - c2.min()]
                clear = ((qa >> 6) & 1).astype(bool) & ~((qa & 1).astype(bool))
                if not clear.any():
                    return None
                band = {}
                for name in ('red', 'nir08'):
                    with ropen(it.assets[name].href, **ovr) as s:
                        band[name] = s.read(1, window=win)[r2 - r2.min(), c2 - c2.min()].astype(float) * 0.0000275 - 0.2
                red, nir = band['red'][clear], band['nir08'][clear]
                with np.errstate(divide='ignore', invalid='ignore'):
                    nd = (nir - red) / (nir + red)
                good = np.isfinite(nd) & (nd > -1) & (nd < 1)
                day = np.datetime64(it.datetime.date())
                return sel[clear][good], np.full(int(good.sum()), day), nd[good].astype(np.float32)

            with ThreadPoolExecutor(workers) as ex:
                res = [r for r in ex.map(scene, items) if r is not None]
            if not res:
                print(st, year, 'no usable scenes', flush=True)
                continue
            idx = np.concatenate([r[0] for r in res])
            day = np.concatenate([r[1] for r in res])
            nd = np.concatenate([r[2] for r in res])
            rows = county_features(idx, day, nd, county_of, fips, year)
            part = dest.with_suffix('.part')
            pd.DataFrame(rows).to_csv(part, index=False)
            part.replace(dest)
            meta = {'state': st, 'year': year, 'grid_m': grid, 'points': len(px), 'scenes_listed': len(items), 'scenes_used': len(res),
                    'observations': len(idx), 'seconds': round(time.time() - t0),
                    'run_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
            (out / f'landsat_{st}_{year}.json').write_text(json.dumps(meta), encoding='utf-8')
            print(meta, flush=True)

    for st in states:
        tx = [c for c in panel if c['state'] == st]
        if tx:
            run_state(st, tx)


if __name__ == '__main__':
    main(*[a for a in sys.argv[1:] if not a.startswith('--')])
