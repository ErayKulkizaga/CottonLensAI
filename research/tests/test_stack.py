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


def test_t46_parse_corn_skips_toc_and_reads_production():
    from research.crop_nowcast.t46_crops import CROPS, parse_table

    old = ('Released September 12, 2000, by NASS\nContents: Corn for Grain:  Area Harvested, Yield, and Production ... 4\n'
           '         Corn for Grain:  Area Harvested, Yield, and Production by State\n'
           '  State :         :\n        :\nIA      :  12,000   12,100   140.0  145.0  150.0  1,700,000  1,815,000\n'
           'US      :  72,000   73,000   136.0  138.0  139.0  9,900,000  10,400,000\n')
    new = ('Released October 12, 2010, by NASS\nSoybeans for Beans Area Harvested, Yield, and Production - States and United States: 2009\n'
           '      State      :         :\nNorth Dakota ....:  4,000   4,100   30.0   32.0   33.5   120,000   137,350\n'
           'United States ...:  76,000  77,000  44.0  44.7  44.4  3,359,000  3,408,000\n')
    rel, o = parse_table(old, CROPS['corn']['title'])
    assert rel.month == 9 and o == {'IA': 1_815_000_000, 'US': 10_400_000_000}
    _, o = parse_table(new, CROPS['soy']['title'])
    assert o == {'ND': 137_350_000, 'US': 3_408_000_000}


def test_t46_splits_fused_columns_by_area_times_yield():
    from research.crop_nowcast.t46_crops import last_value

    # prev production 10,089,222 fused with current 10,917,000; area 73,000 x yield 149.5 = 10,913,500
    assert last_value(['72,000', '73,000', '142.2', '148.0', '149.5', '10,089,22210,917,000']) == 10_917_000
    assert last_value(['1', '2', '3', '4,567']) == 4567


def test_t47_clean_1990s_layouts():
    from research.crop_nowcast.cp_pull import parse
    from research.crop_nowcast.t47_blind_old import clean

    t = ('Released August 12, 1996, by NASS\n Cotton:  Area Harvested, Yield, and Production, by Type, State\n'
         'Upland  :\nFL 2/   :  100.0  110.0  700  710  150.0  160.0  170.0\nTX      : 5,000.0 5,100.0 500 510 4,000.0 4,100.0 4,200.0\n'
         'US      : 5,100.0 5,210.0 505 515 4,150.0 4,260.0 4,370.0\nAm-Pima :\nTX      : 10.0 11.0 800 810 20.0 21.0 22.0\n'
         'US      : 10.0 11.0 800 810 20.0 21.0 22.0\nAll     :\n')
    _, o = parse(clean(t))
    assert o == {'FL': 170_000, 'TX': 4_200_000, 'US': 4_370_000}


def test_t47_ocr_rows_area_times_yield_check():
    from research.crop_nowcast.t47_blind_old import ocr_rows

    t = ('Upland :\n TX  : 5,000.0  4,800.0  480  470  475  5,000.0  4,750.0\n'
         ' AL  :   400.0    410.0  600  610  12,345.6  999.9\n'          # fails the check -> dropped
         ' U.S. : 10.000.0 10,100.0  500  505  10,416.7 10,625.5\nAmer-Pima :\n TX : 1 2 3 4 5 6\n')
    o = ocr_rows(t)
    assert set(o) == {'TX', 'US'} and o['TX'] == (4800.0, 475.0, 4750.0) and o['US'][0] == 10100.0


def test_t49_evaluate_recovers_planted_anchoring():
    import numpy as np
    import pandas as pd

    from research.crop_nowcast.t49_anchoring import evaluate

    rng = np.random.default_rng(0)
    rows = []
    for y in range(2000, 2026):
        for m in ('SEP', 'OCT', 'NOV', 'DEC'):
            pvp = rng.uniform(0.05, 0.7)
            e = -0.5 * pvp + 0.1 + rng.normal(0, 0.02)        # USDA area too high when condition is poor
            rows.append({'state': 'TX', 'year': y, 'month': m, 'area': 1000.0, 'final': 1000.0 * np.exp(e), 'pvp': pvp})
    r = evaluate(pd.DataFrame(rows), {'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}, 'loso')
    assert r['pooled_corr'] < -0.9 and r['pass']
    null = pd.DataFrame(rows).assign(final=lambda x: 1000.0 * np.exp(rng.normal(0, 0.05, len(x))))
    assert not evaluate(null, {'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}, 'loso')['pass']


def test_t50_score_detects_persistence_and_rejects_null():
    import numpy as np
    import pandas as pd

    from research.crop_nowcast.t50_revision_direction import score

    rng = np.random.default_rng(1)
    years = np.repeat(np.arange(1985, 2012), 6)
    sig = pd.DataFrame({'year': years, 'hit': (rng.random(len(years)) < 0.7).astype(int)})
    null = pd.DataFrame({'year': years, 'hit': (rng.random(len(years)) < 0.5).astype(int)})
    assert score(sig)['season_block_p'] < 0.01
    assert score(null)['season_block_p'] > 0.05
