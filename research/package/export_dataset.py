"""Export the recovered USDA monthly cotton forecast archive as one open CSV.

USDA's QuickStats has state-level monthly cotton (and corn/soybean) forecasts only from about 2012. This file collects
what the project recovered from USDA Crop Production reports:
  - upland cotton production by state and US, Aug-Dec, 1985-2011 (text reports 1995-2011, OCR 1985-1994, gated);
  - Texas/US upland harvested area and yield for 1995-2011 (rows passing production = area x yield);
  - corn for grain and soybeans production by state and US, Aug-Nov, 2000-2011.
Validation: parsed values equal QuickStats in 153/153 (cotton) and 441/441 (corn/soybeans) overlap cells (2012-2013).
Usage: python research/package/export_dataset.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
O = ROOT / 'output'


def source_from_release(path, release):
    """T45 rows carry no file name; the raw report next to them is named CropProd-MM-DD-YYYY.txt by release date."""
    if not release:
        return ''
    y, m, _ = release.split('-')
    # one report per month; some files carry a '_revision' suffix or a release day that differs by one
    hits = sorted((Path(path).parents[1] / 'cp_raw').glob(f'CropProd-{m}-??-{y}*.txt'))
    return hits[0].name if len(hits) == 1 else ''


def load(path, crop, method, years):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))['data']
    out = []
    for r in rows:
        y = int(r['year'])
        if not years[0] <= y <= years[1]:
            continue
        out.append({'crop': crop, 'year': y, 'report_month': r['reference_period_desc'][7:10], 'release': r.get('release'),
                    'state': r.get('state_alpha', 'US'), 'production': float(r['Value'].replace(',', '')),
                    'unit': 'bales_480lb' if crop == 'cotton_upland' else 'bushels', 'method': method,
                    'source_file': r.get('source') or source_from_release(path, r.get('release'))})
    return out


def main():
    rows = []
    for lvl in ('state', 'us'):
        rows += load(O / f'crop-nowcast-t47b/forecasts/{lvl}_forecasts.json', 'cotton_upland', 'ocr_gated', (1985, 1994))
        rows += load(O / f'crop-nowcast-t47a/forecasts/{lvl}_forecasts.json', 'cotton_upland', 'text_parsed', (1995, 1999))
        rows += load(O / f'crop-nowcast-t45/forecasts/{lvl}_forecasts.json', 'cotton_upland', 'text_parsed', (2000, 2011))
        for crop in ('corn', 'soy'):
            rows += load(O / f'crop-nowcast-t46/{crop}/report_forecasts/{lvl}_forecasts.json',
                         'corn_grain' if crop == 'corn' else 'soybeans', 'text_parsed', (2000, 2011))
    for r in rows:
        if r['state'] == 'US' or not r['state']:
            r['state'] = 'US'
    d = pd.DataFrame(rows).drop_duplicates(['crop', 'year', 'report_month', 'state']).sort_values(['crop', 'year', 'report_month', 'state'])
    ay = pd.read_csv(O / 'crop-nowcast-t48/forecasts_area_yield.csv')
    ay = ay[ay.state.isin(['TX', 'US'])].rename(columns={'month': 'report_month', 'area': 'harvested_acres', 'yield': 'yield_lb_per_acre'})
    d = d.merge(ay[['state', 'year', 'report_month', 'harvested_acres', 'yield_lb_per_acre']].assign(crop='cotton_upland'),
                on=['crop', 'state', 'year', 'report_month'], how='left')
    out = ROOT / 'docs/research_package/data/usda_monthly_forecasts_recovered_1985_2011.csv'
    d.to_csv(out, index=False)
    print(out.name, len(d), d.groupby('crop').year.agg(['min', 'max', 'count']).to_dict(orient='index'))


if __name__ == '__main__':
    main()
