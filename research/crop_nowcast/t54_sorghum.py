"""T54 (docs/T54_SORGHUM_PREREG_20261116.md): anchoring of USDA grain-sorghum harvested area, Kansas + Texas.
Usage: python t54_sorghum.py pull root | python t54_sorghum.py run root out.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast.nass_pull import nass_key
from research.crop_nowcast.t49_anchoring import evaluate, pvp_at, qs

STATES = ('KS', 'TX')
MON = {'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11}


def cmd_pull(root):
    key = nass_key()
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    area, cond = [], []
    for st in STATES:
        for y in range(2000, 2026):
            for r in qs(key, commodity_desc='SORGHUM', util_practice_desc='GRAIN', statisticcat_desc='AREA HARVESTED', unit_desc='ACRES',
                        agg_level_desc='STATE', state_alpha=st, year=str(y)):
                if r.get('prodn_practice_desc', 'ALL PRODUCTION PRACTICES') == 'ALL PRODUCTION PRACTICES':
                    area.append({'state': st, 'year': y, 'ref': r['reference_period_desc'], 'v': float(r['Value'].replace(',', ''))})
            for unit in ('PCT POOR', 'PCT VERY POOR'):
                for r in qs(key, commodity_desc='SORGHUM', statisticcat_desc='CONDITION', unit_desc=unit, agg_level_desc='STATE',
                            state_alpha=st, year=str(y)):
                    if r.get('week_ending') and r['Value'].strip().replace('.', '').isdigit():
                        cond.append({'crop': 'SORGHUM', 'state': st, 'year': y, 'unit': unit, 'week_ending': r['week_ending'],
                                     'v': float(r['Value'])})
        print(st, 'area rows', len(area), 'condition rows', len(cond), flush=True)
    pd.DataFrame(area).to_csv(out / 'sorghum_area.csv', index=False)
    pd.DataFrame(cond).to_csv(out / 'sorghum_condition.csv', index=False)


def cmd_run(root, out_json):
    a = pd.read_csv(Path(root) / 'sorghum_area.csv').drop_duplicates(['state', 'year', 'ref'])
    cond = pd.read_csv(Path(root) / 'sorghum_condition.csv')
    fin = a[a.ref == 'YEAR'].set_index(['state', 'year']).v
    f = a[a.ref.str.endswith('FORECAST')].assign(month=lambda x: x.ref.str[7:10])
    f = f[f.month.isin(MON)].rename(columns={'v': 'area'})[['state', 'year', 'month', 'area']]
    f['final'] = [fin.get((s, y)) for s, y in zip(f.state, f.year)]
    f['pvp'] = [pvp_at(cond, 'SORGHUM', s, y, MON[m]) for s, y, m in zip(f.state, f.year, f.month)]
    f = f.dropna()
    per_state = f.groupby('state').year.nunique().to_dict()
    res = {'seasons_per_state': per_state, 'months_present': sorted(f.month.unique().tolist())}
    if min(per_state.values(), default=0) < 10:
        res['feasible'] = False
    else:
        res['feasible'] = True
        months = {m: MON[m] for m in res['months_present']}
        res.update(evaluate(f, months, 'loso'))
    print(json.dumps(res, indent=1, default=float))
    Path(out_json).write_text(json.dumps(res, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    {'pull': cmd_pull, 'run': cmd_run}[sys.argv[1]](*sys.argv[2:])
