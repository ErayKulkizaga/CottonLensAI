import numpy as np
import pandas as pd

from research.crop_nowcast.stack import ensemble, grid


def test_grid_is_the_simplex():
    g = grid(4)
    assert g.shape == (1771, 4) and np.allclose(g.sum(1), 1) and g.min() >= 0


def test_weights_never_see_the_test_year():
    rng = np.random.default_rng(0)
    rows = [(s, y) for s in 'ABCDE' for y in range(2013, 2022)]
    f = pd.DataFrame(rows, columns=['state', 'year']).assign(final=lambda x: rng.uniform(50, 150, len(x)))
    for k in ('B', 'H', 'U'):
        f[k] = f.final * np.exp(rng.normal(0, 0.1, len(f)))
    w1, w2 = {}, {}
    ensemble(f, ['B', 'H', 'U'], w1)
    g = f.copy()
    g.loc[g.year == 2019, 'final'] *= 3  # changing year-Y outcomes must not move year-Y weights
    ensemble(g, ['B', 'H', 'U'], w2)
    assert w1[2019] == w2[2019] and w1[2014] is None
