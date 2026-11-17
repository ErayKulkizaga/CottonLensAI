"""Year-level significance of the cotton momentum record 2002-2025 (descriptive, added 16 Nov 2026 after the fact).

Per report x region: one-sided sign-flip permutation p on |USDA error| - |rule error| (20,000 draws, seed 42) and a
one-sided sign test on years better vs worse. Input: docs/research_package/data/momentum_by_year_2002_2025.csv.
Usage: python momentum_significance.py out.json
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast.t46_crops import perm_p


def main(out_json):
    d = defaultdict(list)
    for r in csv.DictReader(open(ROOT / 'docs/research_package/data/momentum_by_year_2002_2025.csv', encoding='utf-8')):
        if r['region'] in ('TX', 'US'):
            d[(r['report'], r['region'])].append(abs(float(r['usda_error_pct'])) - abs(float(r['rule_error_pct'])))
    res = {}
    for (m, reg), diff in d.items():
        w, lo = sum(x > 0 for x in diff), sum(x < 0 for x in diff)
        res[f'{m}_{reg}'] = {'n': len(diff), 'better': w, 'worse': lo, 'mean_gain_pts': round(sum(diff) / len(diff), 3),
                             'perm_p_one_sided': perm_p(diff),
                             'sign_p_one_sided': sum(math.comb(w + lo, i) for i in range(w, w + lo + 1)) / 2 ** (w + lo)}
        print(m, reg, res[f'{m}_{reg}'])
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])
