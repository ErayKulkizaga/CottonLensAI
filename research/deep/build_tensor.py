"""Assemble T35 inputs: county-year tensors [6 bands, T slots, 32 bins] from the per-tile histogram files.

Counts from tiles that share a county are summed. A composite is used only if end_date + 8 days <= cutoff
(same rule as the NDVI features). Composites are placed in 8-day slots from 7 April; each (slot, band)
histogram is turned into fractions. Targets and weights come from the same features CSV as T06.
Usage: python build_tensor.py hist_dir features_<ck>.csv <ck> out.npz
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.crop_nowcast import county_up as C
from research.crop_nowcast.features import CUTOFFS, NDVI_LAG_DAYS

START_DOY, SLOT_DAYS = 97, 8


def n_slots(ck):
    last = pd.Timestamp(f'2021-{CUTOFFS[ck]}') - pd.Timedelta(days=NDVI_LAG_DAYS)
    return (last.dayofyear - START_DOY) // SLOT_DAYS + 1


def slot_of(end, ck):
    """Slot index of a composite ending on `end`, or None if it is not available by the cutoff."""
    end = pd.Timestamp(str(end))
    if end + pd.Timedelta(days=NDVI_LAG_DAYS) > pd.Timestamp(f'{end.year}-{CUTOFFS[ck]}'):
        return None
    return int(np.clip((end.dayofyear - START_DOY) // SLOT_DAYS, 0, n_slots(ck) - 1))


def main(hist_dir, features_csv, ck, out_npz):
    T = n_slots(ck)
    acc = defaultdict(lambda: np.zeros((6, T, 32), np.float64))
    for f in sorted(Path(hist_dir).rglob('*.npz')):
        z = np.load(f)
        year = int(f.stem.split('_')[1])
        if year >= 2022 and '--unseal' not in sys.argv:  # T39 prereg pushed
            raise SystemExit(f'sealed year file present: {f.name}')
        for k, end in enumerate(z['end']):
            s = slot_of(end, ck)
            if s is None:
                continue
            for i, fips in enumerate(z['fips']):
                acc[(str(fips), year)][:, s, :] += z['hist'][k, i]
    c = C.prepare(features_csv)
    c = c[c.set_index(['fips', 'year']).index.isin(list(acc))]
    X = np.stack([acc[(f, y)] for f, y in zip(c.fips, c.year, strict=True)])
    tot = X.sum(axis=3, keepdims=True)
    X = np.divide(X, tot, out=np.zeros_like(X), where=tot > 0).astype(np.float32)
    np.savez_compressed(out_npz, X=X, fips=c.fips.to_numpy(), year=c.year.to_numpy(),
                        target=c.target.to_numpy(np.float32), state=c.state.to_numpy())
    print(ck, X.shape, 'targets', int(np.isfinite(c.target).sum()))


if __name__ == '__main__':
    main(*[a for a in sys.argv[1:] if not a.startswith('--')])
