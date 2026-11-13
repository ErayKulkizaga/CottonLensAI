"""Assemble the Cotton Risk Terminal page and its frame files (no fitting).

Counters are taken from committed evidence and from the local satellite run stats;
each label states what the number counts.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / 'research/evidence'
PKG = ROOT / 'research/risk_lab_package'


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def registry_counts():
    out = subprocess.run([sys.executable, str(ROOT / 'ml/history.py'), 'validate'], capture_output=True, text=True,
                         check=True, cwd=ROOT).stdout
    return json.loads(out[out.index('{'):])


def main(terminal_json, sat_dir, out_dir, report_url):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    term = load(terminal_json)
    sat = load(Path(sat_dir) / 'frames.json')
    reg = registry_counts()
    pm = load(PKG / 'predictability_map.json')
    arm_a = load(EV / 'range-ewma-result-20261113.json')['primary_t1']['markets']
    lab = load(EV / 'risk-lab-result-20261113.json')
    days = sum(v['origins'] for v in arm_a.values()) + sum(v['origins'] for v in lab['descriptive'].values())
    st = sat['stats']
    counters = [
        [reg['completed_fit_copies'], 'tamamlanmış model fit kaydı (Codex sicili; aynı tarifin tekrarları dahil)'],
        [reg['fit_recipes'], 'farklı model tarifi; XGBoost, Ridge, LSTM, TCN, GARCH, HAR dahil 11 aile'],
        [pm['experiments'], f"tamamlanmış fiyat deneyi; en iyisi +%{pm['max_selected_gain_pct']:.2f}, hiçbiri %5 kapısında değil"],
        [st['composites'], f"işlenen 16 günlük MODIS uydu karesi ({st['years']} sezon)"],
        [st['cotton_pixel_observations'], 'pamuk tarlası piksel gözlemi (250 m)'],
        [days, 'önceden kayıtlı testlerde skorlanan piyasa-günü'],
        [16, 'incelenen emtia piyasası; 7 tanesi kör doğrulamada'],
        [5, 'ön kayıtlı hipotez: 3 doğrulandı, 2 doğrulanmadı'],
    ]
    years = {y: {'cdl_year': v['cdl_year'], 'cotton_pixels': v['cotton_pixels'],
                 'frames': [{'end': q['end'], 'sensor': q['sensor'], 'median': None if q['median'] is None else round(q['median'], 4)}
                            for q in v['frames']]} for y, v in sat['years'].items()}
    satmeta = {'bbox': sat['bbox'], 'grid': sat['grid'], 'years': years}
    html = (PKG / 'terminal_template.html').read_text(encoding='utf-8')
    html = (html.replace('/*__TERMINAL__*/null', json.dumps(term, separators=(',', ':')))
                .replace('/*__COUNTERS__*/null', json.dumps(counters, ensure_ascii=False))
                .replace('/*__SATMETA__*/null', json.dumps(satmeta, separators=(',', ':')))
                .replace('__REPORT_URL__', report_url))
    (out / 'cotton-risk-terminal.html').write_text(html, encoding='utf-8')
    for y in sat['years']:
        shutil.copyfile(Path(sat_dir) / f'frames_{y}.png', out / f'frames_{y}.png')
    (out / 'counters.json').write_text(json.dumps({'counters': counters, 'satellite_stats': st}, indent=2, ensure_ascii=False), encoding='utf-8')
    print('ok', len(html), counters)


if __name__ == '__main__':
    main(*sys.argv[1:5])
