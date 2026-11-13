"""Assemble the Cotton Risk Lab page from committed evidence JSON + local outputs (no fitting)."""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / 'research/evidence'
PKG = ROOT / 'research/risk_lab_package'


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def naive_mape(parquet):
    m = pd.read_parquet(parquet)
    c = m[m.series == 'cotton'].copy()
    c['d'] = pd.to_datetime(c.observed_on)
    c = c.set_index('d').close.sort_index()
    c = c[c.index <= '2026-09-21']
    rows = []
    for h, lab in ((1, '1 seans'), (5, '5 seans'), (21, '~1 ay'), (63, '~3 ay')):
        e = ((c.shift(-h) - c).abs() / c.shift(-h))
        rows.append({'h': lab, 'a': 100 * float(e['2016':'2023'].mean()), 'b': 100 * float(e['2024':'2026'].mean())})
    return rows


def main(out_html, market_parquet, today_dir, sat_dir, png_dir):
    pm = load(PKG / 'predictability_map.json')
    arm_a = load(EV / 'range-ewma-result-20261113.json')['primary_t1']
    lab = load(EV / 'risk-lab-result-20261113.json')
    today = load(Path(today_dir) / 'today.json')
    sat = load(Path(sat_dir) / 'texas_ndvi.json')
    pngs = {y: base64.b64encode((Path(png_dir) / f'ndvi_{y}.png').read_bytes()).decode() for y in (2010, 2011)}
    curves = {}
    for y, c in sat['curves'].items():
        pts = []
        for s in c['composites']:
            d = pd.Timestamp(s['end'])
            if np.isfinite(s['median_ndvi_cotton']):
                pts.append([int(d.dayofyear), round(s['median_ndvi_cotton'], 4)])
        curves[y] = {'pts': sorted(pts), 'peak': max(p[1] for p in pts), 'cdl': c['cdl_year'], 'px': c['cotton_pixels']}
    rows = [{'s': r['source'], 'h': r['horizon'], 'g': r['selected_gain_pct']} for r in pm['rows']
            if r['selected_gain_pct'] is not None]

    def hyp(k):
        r = lab[k]
        return {'mean': r['pooled_mean_gain'], 'ci20': r['ci20_bonferroni'], 'ci60': r['ci60_bonferroni'],
                'pos': r['positive_markets'], 'n': r['markets'], 'pass': r['pass_bonferroni']}

    data = {
        'map': {'rows': rows, 'n': pm['experiments'], 'pos': pm['selected_positive'], 'max': pm['max_selected_gain_pct']},
        'naive': naive_mape(market_parquet),
        'armA': {'mean': arm_a['pooled_mean_gain'], 'ci20': arm_a['ci20'], 'ci60': arm_a['ci60'],
                 'pos': arm_a['positive_markets'], 'n': arm_a['markets_scored']},
        'lab': {k: hyp(k) for k in ('H2a_range_vs_garch_qlike', 'H2b_range_vs_har_qlike',
                                    'H3_bigmove_brier_range_vs_ewma', 'H4_interval80_winkler_range_vs_ewma')},
        'garchPer': lab['H2a_range_vs_garch_qlike']['per_market_mean_gain'],
        'desc': {k: {'c80': v['coverage80_range'], 'c90': v['coverage90_range'], 'auc': v['auc_range'],
                     'aucE': v['auc_ewma'], 'n': v['origins'], 'first': v['first'][:4]}
                 for k, v in lab['descriptive'].items()},
        'today': {k: today[k] for k in ('run_utc', 'origin_session', 'origin_close_cents_lb', 'next_session_sigma_pct',
                                        'big_move_probability', 'big_move_threshold_pct', 'interval80_cents_lb',
                                        'interval90_cents_lb', 'ct_calibration', 'recent', 'ct_reliability')},
        'ndvi': curves,
    }
    html = (PKG / 'page_template.html').read_text(encoding='utf-8')
    html = html.replace('/*__DATA__*/null', json.dumps(data, ensure_ascii=False))
    html = html.replace('__PNG2010__', pngs[2010]).replace('__PNG2011__', pngs[2011])
    Path(out_html).write_text(html, encoding='utf-8')
    print('written', out_html, len(html))


if __name__ == '__main__':
    main(*sys.argv[1:6])
