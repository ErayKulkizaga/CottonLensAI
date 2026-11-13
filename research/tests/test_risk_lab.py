"""Synthetic contract tests for the risk-lab preregistration (no market data)."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from research import risk_lab as L
from research.tests.test_range_ewma import synthetic


def test_garch_filter_matches_recursion():
    r = np.random.default_rng(0).standard_normal(50) * 0.01
    s2 = L._garch_filter(r, 1e-6, 0.08, 0.9, 1e-4)
    ref = [1e-4]
    for x in r:
        ref.append(1e-6 + 0.08 * x * x + 0.9 * ref[-1])
    np.testing.assert_allclose(s2, ref, rtol=1e-12)


def test_garch_fit_recovers_persistence():
    rng = np.random.default_rng(3)
    n, w, a, b = 6000, 2e-6, 0.08, 0.90
    r, s2 = np.empty(n), w / (1 - a - b)
    for t in range(n):
        r[t] = math.sqrt(s2) * rng.standard_normal()
        s2 = w + a * r[t] ** 2 + b * s2
    omega, alpha, beta = L.garch_fit(r)
    assert alpha + beta == pytest.approx(a + b, abs=0.03)
    assert alpha == pytest.approx(a, abs=0.04)


@pytest.mark.parametrize('fn', [L.garch_forecasts, L.har_forecasts])
def test_benchmarks_are_past_only(fn):
    f = synthetic(2600, 5)
    base = fn(f)
    cut = 2000
    g = f.copy()
    g.loc[cut + 1:, ['open', 'high', 'low', 'close']] *= np.linspace(1.3, 2.5, len(g) - cut - 1)[:, None]
    other = fn(g)
    pd.testing.assert_series_equal(base.iloc[:cut + 1], other.iloc[:cut + 1])
    assert base.iloc[:L.MIN_FIT_OBS].isna().all()


def test_benchmarks_refit_only_on_january_first():
    f = synthetic(2600, 6)
    r = np.log(f.close / f.close.shift())
    g = L.garch_forecasts(f)
    year = f.date.dt.year.iloc[-1]
    train = r[f.date < pd.Timestamp(f'{year}-01-01')].dropna().to_numpy()
    omega, alpha, beta = L.garch_fit(train)
    s2 = L._garch_filter(r.fillna(0).to_numpy(), omega, alpha, beta, float(np.mean(train ** 2)))
    np.testing.assert_allclose(g.iloc[-1], s2[-1], rtol=1e-10)


def test_exceedance_and_quantiles_match_brute_force():
    rng = np.random.default_rng(1)
    z = np.abs(rng.standard_normal(900))
    z[10] = np.nan
    thr = np.full(900, 1.2)
    p = L._past_only_exceedance(z, thr)
    for i in (L.MIN_Z_HISTORY + 10, 899):
        seen = z[:i][~np.isnan(z[:i])]
        if len(seen) >= L.MIN_Z_HISTORY:
            assert p[i] == pytest.approx((seen > 1.2).mean())
    assert np.isnan(p[:L.MIN_Z_HISTORY]).all()


def test_probabilistic_layer_is_past_only():
    f = synthetic(1800, 7)
    from research import range_ewma as R
    v = R.forecasts(f).candidate
    a = L.probabilistic(f, v)
    cut = 1500
    g = f.copy()
    g.loc[cut + 1:, ['open', 'high', 'low', 'close']] *= 1.7
    b = L.probabilistic(g, R.forecasts(g).candidate)
    cols = ['p_big', 'lo80', 'hi80', 'lo90', 'hi90']
    pd.testing.assert_frame_equal(a.iloc[:cut + 1][cols], b.iloc[:cut + 1][cols])


def test_winkler_and_auc():
    assert L.winkler(np.array([-1.0]), np.array([1.0]), np.array([0.0]), 0.8)[0] == 2.0
    assert L.winkler(np.array([-1.0]), np.array([1.0]), np.array([2.0]), 0.8)[0] == pytest.approx(12.0)
    assert L._auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert L._auc([0.5, 0.5], [0, 1]) == 0.5


def test_score_shapes_and_h3_exclusion():
    frames = {'AA=F': synthetic(2700, 11), 'SB=F': synthetic(2700, 12), 'BB=F': synthetic(2700, 13)}
    res = L.score(frames)
    assert res['H3_bigmove_brier_range_vs_ewma']['markets'] == 2
    assert set(res['H3_reported_SB_KC']) == {'SB=F'}
    for k in ('H2a_range_vs_garch_qlike', 'H2b_range_vs_har_qlike', 'H4_interval80_winkler_range_vs_ewma'):
        assert {'pass_95', 'pass_bonferroni', 'ci20_95', 'ci60_bonferroni'} <= set(res[k])


def test_cli_refuses_wrong_hash(tmp_path):
    p = tmp_path / 'p.md'
    p.write_text('x', encoding='utf-8')
    a = tmp_path / 'a.json'
    a.write_text(json.dumps({'markets': {}}), encoding='utf-8')
    with pytest.raises(SystemExit):
        L.main(['--admission', str(a), '--prereg', str(p), '--expected-prereg-sha256', '0' * 64,
                '--out', str(tmp_path / 'o.json')])
