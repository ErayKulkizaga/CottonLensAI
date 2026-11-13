"""County x composite NDVI over cotton pixels for the panel counties (MODIS MOD/MYD13Q1 v061).

Leakage rules (see docs/CROP_NOWCAST_DATA_CONTRACT_20261113.md):
- cotton mask for season Y comes from the USDA CDL of year Y-1 (class 2); the
  multi-year frequency/cultivated layers are never used;
- composites are dated by end_datetime; availability lags are applied later,
  at feature-building time;
- default years stop at 2021; sealed years require --unseal after the
  preregistration is pushed.
Writes one CSV per (tile, year) so the run is resumable. Planetary Computer,
anonymous access. Archive = 2021 reprocessing (vintage limitation).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

STAC = 'https://planetarycomputer.microsoft.com/api/stac/v1'
COTTON = 2
DEFAULT_YEARS = (2009, 2021)


def main(panel_json, out_dir, first=DEFAULT_YEARS[0], last=DEFAULT_YEARS[1]):
    first, last = int(first), int(last)
    if last >= 2022 and '--unseal' not in sys.argv:
        raise SystemExit('Sealed years need --unseal after preregistration')
    import planetary_computer as pc
    import pystac_client
    import rasterio
    import rasterio.errors
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.windows import Window, from_bounds

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel = json.loads(Path(panel_json).read_text())['counties']
    tiles = sorted({t for c in panel for t in c['tiles']})
    only = [a.split('=', 1)[1] for a in sys.argv if a.startswith('--tile=')]
    if only:
        tiles = [t for t in tiles if t in only]
    cat = pystac_client.Client.open(STAC)

    def ropen(href, tries=6):
        # sign at use time (tokens expire on long runs) and back off on transient 403/IO errors
        for k in range(tries):
            try:
                return rasterio.open(pc.sign(href))
            except rasterio.errors.RasterioIOError:
                if k == tries - 1:
                    raise
                time.sleep(15 * (k + 1))

    def rread(href, **kw):
        for k in range(6):
            try:
                with ropen(href) as s:
                    return s.read(1, **kw)
            except rasterio.errors.RasterioIOError:
                if k == 5:
                    raise
                time.sleep(15 * (k + 1))
    rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_MAX_RETRY='5', GDAL_HTTP_RETRY_DELAY='3').__enter__()
    t0 = time.time()

    def modis(tile, bbox, year):
        items = cat.search(collections=['modis-13Q1-061'], bbox=bbox, datetime=f'{year}-04-01/{year}-10-31').item_collection()
        return sorted([i for i in items if f'.{tile}.' in i.id], key=lambda i: i.properties['end_datetime'])

    for tile in tiles:
        members = [c for c in panel if tile in c['tiles']]
        bb = [min(c['bbox'][0] for c in members), min(c['bbox'][1] for c in members),
              max(c['bbox'][2] for c in members), max(c['bbox'][3] for c in members)]
        probe = modis(tile, bb, 2015)[0]
        with ropen(probe.assets['250m_16_days_NDVI'].href) as src:
            to_sin = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
            xs, ys = to_sin.transform([bb[0], bb[2], bb[0], bb[2]], [bb[1], bb[1], bb[3], bb[3]])
            win = from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform).round_offsets().round_lengths()
            win = win.intersection(Window(0, 0, src.width, src.height))
            wt, crs = src.window_transform(win), src.crs
            H, W = int(win.height), int(win.width)

        def to_sin_geom(g, to_sin=to_sin):
            def ring(r):
                x, y = to_sin.transform([p[0] for p in r], [p[1] for p in r])
                return list(zip(x, y, strict=True))
            if g['type'] == 'Polygon':
                return {'type': 'Polygon', 'coordinates': [ring(r) for r in g['coordinates']]}
            return {'type': 'MultiPolygon', 'coordinates': [[ring(r) for r in poly] for poly in g['coordinates']]}

        cid = rasterize([(to_sin_geom(c['geometry']), i + 1) for i, c in enumerate(members)], out_shape=(H, W),
                        transform=wt, fill=0, dtype='int16', all_touched=False)
        sel = np.flatnonzero(cid.ravel())
        if not len(sel):
            continue
        rr, cc = np.unravel_index(sel, (H, W))
        px, py = wt * (cc + 0.5, rr + 0.5)
        lon, lat = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform(px, py)
        county_of = cid.ravel()[sel] - 1

        def cotton(cdl_year, bb=bb, sel=sel, lon=lon, lat=lat):
            items = cat.search(collections=['usda-cdl'], bbox=bb, datetime=f'{cdl_year}-01-01/{cdl_year}-12-31').item_collection()
            items = [i for i in items if i.properties.get('usda_cdl:type') == 'cropland']
            m = np.zeros(len(sel), bool)
            for it in items:
                with ropen(it.assets['cropland'].href) as s:
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
                    c2, r2 = ~t * (tx, ty)
                    c2, r2 = np.floor(c2).astype(int), np.floor(r2).astype(int)
                    ok = inside & (r2 >= 0) & (r2 < arr.shape[0]) & (c2 >= 0) & (c2 < arr.shape[1])
                    m[ok] |= arr[r2[ok], c2[ok]] == COTTON
            return m

        for year in range(first, last + 1):
            dest = out / f'{tile}_{year}.csv'
            if dest.exists():
                continue
            mask = cotton(year - 1)
            lines = ['fips,year,tile,end,start,sensor,n_cotton,n_valid,mean,median,p90']
            groups = county_of[mask]
            for it in modis(tile, bb, year):
                nd = rread(it.assets['250m_16_days_NDVI'].href, window=win).ravel()[sel][mask].astype(float)
                nd[(nd < -2000) | (nd > 10000)] = np.nan
                nd /= 10000.0
                end, start = it.properties['end_datetime'][:10], it.properties['start_datetime'][:10]
                for k, c in enumerate(members):
                    v = nd[groups == k]
                    n = int(len(v))
                    vv = v[np.isfinite(v)]
                    if not n:
                        continue
                    stats = (f'{vv.mean():.4f},{np.median(vv):.4f},{np.percentile(vv, 90):.4f}' if len(vv) else ',,')
                    lines.append(f"{c['fips']},{year},{tile},{end},{start},{it.id[:3]},{n},{len(vv)},{stats}")
            dest.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            print(tile, year, int(mask.sum()), round(time.time() - t0), flush=True)
    print('done', round(time.time() - t0))


if __name__ == '__main__':
    main(*[a for a in sys.argv[1:] if not a.startswith('--')])
