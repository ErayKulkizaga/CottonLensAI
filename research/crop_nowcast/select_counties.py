"""Choose the county panel from development years only and attach boundaries + MODIS tiles.

Rule (fixed before any satellite data is read): rank counties by mean upland
production 2008-2021 (all practices, annual estimate), keep the smallest set that
covers 90% of total production. Only development years are used, so the panel is
not chosen with sealed-year information.
"""
import json
import math
import sys
import zipfile
from pathlib import Path

import pandas as pd

COVERAGE = 0.90
TILE = 1111950.5196666666
X0, Y0 = -20015109.354, 10007554.677
R_SIN = 6371007.181


def nass(path):
    d = pd.DataFrame(json.loads(Path(path).read_text())['data'])
    d = d[(d.reference_period_desc == 'YEAR') & (d.prodn_practice_desc == 'ALL PRODUCTION PRACTICES')
          & d.county_ansi.ne('')].copy()
    d['value'] = pd.to_numeric(d.Value.str.replace(',', ''), errors='coerce')
    d['fips'] = d.state_ansi.str.zfill(2) + d.county_ansi.str.zfill(3)
    return d


def sinus(lon, lat):
    return R_SIN * math.radians(lon) * math.cos(math.radians(lat)), R_SIN * math.radians(lat)


def main(nass_dir, county_zip, out_dir):
    import shapefile
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = nass(Path(nass_dir) / 'county_production.json')
    mean = p.groupby('fips').value.mean().sort_values(ascending=False)
    share = mean.cumsum() / mean.sum()
    keep = share[share.shift(fill_value=0) < COVERAGE].index.tolist()
    names = p.drop_duplicates('fips').set_index('fips')[['state_alpha', 'county_name']]
    zf = zipfile.ZipFile(county_zip)
    stem = 'cb_2020_us_county_500k'
    sf = shapefile.Reader(shp=zf.open(stem + '.shp'), dbf=zf.open(stem + '.dbf'), shx=zf.open(stem + '.shx'))
    fields = [f[0] for f in sf.fields[1:]]
    geoms = {}
    for sr in sf.iterShapeRecords():
        rec = dict(zip(fields, sr.record, strict=True))
        fips = rec['STATEFP'] + rec['COUNTYFP']
        if fips in keep:
            geoms[fips] = sr.shape.__geo_interface__
    missing = [f for f in keep if f not in geoms]
    panel = []
    for f in keep:
        if f in missing:
            continue
        g = geoms[f]
        rings = g['coordinates'] if g['type'] == 'Polygon' else [r for poly in g['coordinates'] for r in poly]
        pts = [pt for ring in rings for pt in ring]
        lon = [q[0] for q in pts]
        lat = [q[1] for q in pts]
        tiles = set()
        for lo, la in pts[:: max(1, len(pts) // 200)] + [(min(lon), min(lat)), (max(lon), max(lat)), (min(lon), max(lat)), (max(lon), min(lat))]:
            x, y = sinus(lo, la)
            tiles.add(f'h{int((x - X0) // TILE):02d}v{int((Y0 - y) // TILE):02d}')
        panel.append({'fips': f, 'state': names.loc[f, 'state_alpha'], 'county': names.loc[f, 'county_name'],
                      'mean_production_bales': float(mean[f]), 'bbox': [min(lon), min(lat), max(lon), max(lat)],
                      'centroid': [sum(lon) / len(lon), sum(lat) / len(lat)], 'tiles': sorted(tiles), 'geometry': g})
    meta = {'rule': f'top counties by mean 2008-2021 upland production covering {COVERAGE:.0%}', 'counties': len(panel),
            'missing_boundaries': missing, 'coverage_share': float(share[keep[-1]]),
            'tiles': sorted({t for c in panel for t in c['tiles']})}
    (out / 'panel.json').write_text(json.dumps({'meta': meta, 'counties': panel}), encoding='utf-8')
    print(json.dumps(meta), pd.Series([c['state'] for c in panel]).value_counts().to_dict())


if __name__ == '__main__':
    main(*sys.argv[1:4])
