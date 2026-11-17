"""T39 one-shot sealed test driver (docs/prereg/SEALED_TEST_T39_PREREG_20261115.md). Run each phase once.

  prep  : assemble dev+sealed inputs, build county features for every cutoff (no scores printed)
  score : T36 primary (1 Oct) + all declared secondaries, scored on 2022-2024 only -> evidence JSON
Usage: python t39_run.py prep|score dev_root t39_root [cnn_root] [out_json]
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('COTTONLENS_LAST_YEAR', '2024')
os.environ.setdefault('COTTONLENS_SCORE_FROM', '2022')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import county_up as C
from research.crop_nowcast import features as F
from research.crop_nowcast import stack as S
from research.crop_nowcast import t39_assemble
from research.crop_nowcast.state_model import summarize

CUTOFFS = ('aug1', 'sep1', 'oct1', 'nov1', 'dec1')


def prep(dev_root, t39_root):
    dev, t = Path(dev_root), Path(t39_root)
    t39_assemble.main(dev, t)
    out = t / 'dev-ndvi'
    out.mkdir(exist_ok=True)
    for ck in CUTOFFS:
        df = F.build(t / 'nass', t / 'power', t / 'ndvi', dev / 'geo/panel.json', ck, dev / 'power_clim')
        df.to_csv(out / f'features_{ck}.csv', index=False)
        print(ck, df.shape, 'years', int(df.year.min()), int(df.year.max()))


def score(dev_root, t39_root, cnn_root, out_json):
    t = Path(t39_root)
    fd, nd, fsa = t / 'dev-ndvi', t / 'nass', t / 'fsa/fsa_cotton.csv'
    res = {'scored_years': [S.SCORE_FROM, int(os.environ['COTTONLENS_LAST_YEAR'])]}
    tmp = Path(out_json).with_suffix('.part.json')
    S.main(fd, nd, fsa, tmp, cnn_root)
    res['T36_and_T25'] = json.loads(tmp.read_text())
    S.main_late(fd, nd, tmp)
    res['T37'] = json.loads(tmp.read_text())
    S.main_gin(fd, nd, t / 'ginnings.csv', tmp)
    res['T38'] = json.loads(tmp.read_text())
    tmp.unlink()
    res['H_alone'] = {}
    for ck in ('aug1', 'sep1', 'oct1'):
        r, us = C.run(C.prepare(fd / f'features_{ck}.csv'), nd, ck, [k + '_a' for k in C.NDVI], 'hgb', area='fsa_aug')
        r, us = r[r.year >= S.SCORE_FROM], us[us.index >= S.SCORE_FROM]
        res['H_alone'][ck] = summarize(r, us, f'{ck} H alone (sealed)')
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    {'prep': prep, 'score': score}[sys.argv[1]](*sys.argv[2:])
