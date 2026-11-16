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


def test_gin_parse_both_layouts():
    from research.crop_nowcast.gin_pull import parse

    new = ('Released November 25, 2019, by NASS\n'
           'Running Bales Ginned by Crop - States and United States: November 15, 2016-2019\n'
           'Upland   :\nTexas ......:   (D)   (D)   2,197,400   3,249,800\n'
           'All      :\nTexas ......:   2,613,700   3,140,850   2,200,000   3,254,300\n'
           'Arizona .....:   121,850   109,450   60,050   (D)\n---\n')
    a, r, o = parse(new)
    assert str(a) == '2019-11-15' and str(r) == '2019-11-25' and o == {'TX': 3254300, 'AZ': None}
    old = ('Released November 24, 2009, by NASS\n  As of November 15, Crop Years 2006-2009\n'
           'All Cotton  :\nTX          :    2,322,400   2,635,050   1,485,950   1,868,550\n'
           'NM          :    1/   1/   1/   1/\nUS          :   11,832,450\n')
    a, _, o = parse(old)
    assert str(a) == '2009-11-15' and o == {'TX': 1868550, 'NM': None}


def test_eb_bias_shrinks_toward_zero_and_keeps_consistent_states():
    from research.crop_nowcast.state_bias import eb_bias

    rng = np.random.default_rng(1)
    rows = [('TX', -0.15 + rng.normal(0, 0.03)) for _ in range(12)] + [(s, rng.normal(0, 0.03)) for s in 'ABCD' for _ in range(12)]
    b = eb_bias(pd.DataFrame(rows, columns=['state', 'e']))
    assert -0.16 < b['TX'] < -0.10  # persistent bias kept (slightly shrunk)
    assert all(abs(b[s]) < 0.03 for s in 'ABCD')  # noise states pulled to ~0, not to the pooled mean


def test_cp_parse_takes_production_not_yield():
    from research.crop_nowcast.cp_pull import parse

    old = ('Released December 9, 2005, by NASS\n Cotton:  Area Harvested, Yield, and Production by Type, State,\n'
           'Upland   :\n  MS     : 1,100.0   1,180.0   1,024      854       879      2,346.0    2,160.0\n'
           '  US     :12,000.0 13,000.0   800      830       850     20,000.0   21,500.0\nAmer-Pima:\n')
    new = ('Released August 12, 2010, by NASS\nCotton Area Harvested, Yield, and Production by Type - States\n'
           'Upland           :\nMississippi .....:   300     400     900      950       600      700      800\n'
           'United States ..:  9,000   9,500   800   820   17,000  17,500  18,000\nAmerican Pima    :\n')
    _, o = parse(old)
    assert o == {'MS': 2_160_000, 'US': 21_500_000}
    _, o = parse(new)
    assert o == {'MS': 800_000, 'US': 18_000_000}
