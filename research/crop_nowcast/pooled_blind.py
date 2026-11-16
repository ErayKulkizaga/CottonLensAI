"""All never-seen seasons pooled: the rolling momentum procedure on 1987-1999 (T47 secondary) and 2002-2011 (T45).

Both eras were scored blind with the same rolling procedure (momentum.py, trained only on earlier seasons of the era).
Disclosure: the decision to pool was made on 16 Nov 2026 after both eras' results were seen; pooling ALL blind seasons
is the least selective choice available. Year-block sign-flip test over the 8 TX/US cells (20,000 draws, seed 42).
Usage: python pooled_blind.py out.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'research/evidence'


def main(out_json):
    cells = {}
    for f in ('t47-rolling-1987-1999-20261116.json', 't45-blind-2000-2011-20261116.json'):
        d = json.loads((E / f).read_text())
        for m in ('SEP', 'OCT', 'NOV', 'DEC'):
            for k in ('TX', 'US'):
                for y, v in d[m][f'{k}_by_year'].items():
                    cells.setdefault(f'{m}_{k}', {})[int(y)] = abs(v[1]) - abs(v[0])
    years = sorted(set().union(*[set(v) for v in cells.values()]))
    mat = np.array([[cells[c].get(y, np.nan) for c in cells] for y in years])
    obs = float(np.nanmean(mat))
    rng = np.random.default_rng(42)
    sims = np.array([np.nanmean(mat * rng.choice([-1.0, 1.0], size=(len(years), 1))) for _ in range(20000)])
    per = {c: {'seasons': len(v), 'mean_gain_pts': float(np.mean(list(v.values()))),
               'better': int(sum(x > 0 for x in v.values())), 'worse': int(sum(x < 0 for x in v.values()))} for c, v in cells.items()}
    res = {'seasons': [years[0], years[-1], len(years)], 'mean_gain_pts': obs, 'year_block_p_one_sided': float((sims >= obs - 1e-12).mean()),
           'season_cells_better': int(np.nansum(mat > 0)), 'season_cells_worse': int(np.nansum(mat < 0)), 'cells': per}
    print(json.dumps({k: res[k] for k in res if k != 'cells'}))
    for c, v in per.items():
        print(c, v)
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1])
