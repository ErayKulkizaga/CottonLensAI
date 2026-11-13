"""Synthetic contract tests for the frozen range-EWMA preregistration (no market data)."""
import json
import math

import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research import volatility

from research import range_ewma as R


def synthetic(n=1500, seed=0, informative=True):
    """GARCH-like closes with an intraday range that does / does not track today's variance."""
    rng = np.random.default_rng(seed)
    var = np.empty(n)
    var[0] = 1e-4
    ret = np.empty(n)
    for t in range(n):
        if t:
            var[t] = 2e-6 + 0.90 * var[t - 1] + 0.08 * ret[t - 1] ** 2
        ret[t] = math.sqrt(var[t]) * rng.standard_normal()
    close = 70 * np.exp(np.cumsum(ret))
    open_ = np.r_[70, close[:-1]]
    scale = np.sqrt(var) if informative else np.full(n, math.sqrt(var.mean()))
    span = np.abs(rng.standard_normal(n)) * scale * 1.2 + 1e-4
    high = np.maximum(open_, close) * np.exp(span / 2)
    low = np.minimum(open_, close) * np.exp(-span / 2)
    return pd.DataFrame({'date': pd.bdate_range('2005-01-03', periods=n), 'open': open_,
                         'high': high, 'low': low, 'close': close})


def test_baseline_is_unchanged_project_ewma():
    f = synthetic()
    fc = R.forecasts(f)
    hist = volatility.features(pd.DataFrame({'date': f.date, 'cotton_close': f.close}))
    dates = f.date.iloc[300:400]
    np.testing.assert_allclose(fc.ewma.loc[dates].to_numpy(), volatility.ewma(hist, dates, 1), rtol=1e-12)


def test_forecasts_are_past_only():
    f = synthetic()
    base = R.forecasts(f)
    cut = 900
    g = f.copy()
    g.loc[cut + 1:, ['open', 'high', 'low', 'close']] *= np.linspace(1.5, 3, len(g) - cut - 1)[:, None]
    other = R.forecasts(g)
    cols = ['ewma', 'range_ewma', 'candidate', 'k']
    pd.testing.assert_frame_equal(base.iloc[:cut + 1][cols], other.iloc[:cut + 1][cols])


def test_scale_uses_only_earlier_valid_bars():
    f = synthetic()
    fc = R.forecasts(f)
    assert fc.k.iloc[:R.MIN_K_DAYS].isna().all()
    t = 700
    hist = f.iloc[:t]
    r2 = np.log(hist.close / hist.close.shift()) ** 2
    valid = R.valid_bar(hist.set_index('date')).to_numpy() & r2.notna().to_numpy()
    park = np.log(hist.high / hist.low) ** 2 / (4 * math.log(2))
    assert fc.k.iloc[t] == pytest.approx(r2[valid].sum() / park[valid].sum(), rel=1e-12)


def test_invalid_bar_falls_back_to_same_day_squared_return():
    f = synthetic()
    t = 800
    f.loc[t, 'high'] = f.loc[t, 'low']                       # flat bar
    fc = R.forecasts(f)
    r2 = math.log(f.close[t] / f.close[t - 1]) ** 2
    prev = fc.range_ewma.iloc[t - 1]
    assert not fc.valid_bar.iloc[t]
    assert fc.range_ewma.iloc[t] == pytest.approx((1 - R.ALPHA) * prev + R.ALPHA * r2, rel=1e-12)


def test_rejects_unsorted_or_nonpositive_inputs():
    f = synthetic(400)
    with pytest.raises(ValueError):
        R.forecasts(f.iloc[::-1])
    bad = f.copy()
    bad.loc[10, 'close'] = 0
    with pytest.raises(ValueError):
        R.forecasts(bad)


def test_admission_reads_quality_only():
    f = synthetic(2600)
    end = f.date.iloc[-1]
    ok = R.admit(f, end)
    assert ok['admitted'] and ok['valid_fraction'] == 1.0
    g = f.copy()
    g.loc[::3, 'high'] = g.loc[::3, 'low']
    assert 'valid_bar_fraction' in R.admit(g, end)['reasons']
    assert 'too_few_origins' in R.admit(synthetic(1200), end)['reasons']


def test_horizon_targets_and_end_cutoff():
    f = synthetic(600)
    fc = R.forecasts(f)
    r2 = np.log(f.close / f.close.shift()) ** 2
    assert fc.target_5.iloc[400] == pytest.approx(r2.iloc[401:406].sum())
    end = f.date.iloc[500]
    assert R.origins(fc, 5, end).target_date_5.max() <= end
    assert R.origins(fc, 1, end).target_date_1.max() <= end


def test_score_detects_informative_range_and_not_uninformative():
    end = pd.Timestamp('2030-01-01')
    info = R.score({f'm{i}': synthetic(2600, i, True) for i in range(3)}, 1, end)
    assert info['pooled_mean_gain'] > 0
    flat = R.score({f'm{i}': synthetic(2600, 10 + i, False) for i in range(3)}, 1, end)
    assert flat['pooled_mean_gain'] < info['pooled_mean_gain']
    assert set(info) >= {'primary_pass', 'breadth_pass', 'decision', 'ci20', 'ci60'}


def test_bootstrap_is_seeded_and_common_blocks_handle_missing():
    x = np.random.default_rng(1).standard_normal(500)
    assert R.block_ci(x, 20, 500) == R.block_ci(x, 20, 500)
    m = np.c_[x, np.r_[np.full(100, np.nan), x[100:]]]
    lo, hi = R.pooled_ci(m, 20, 500)
    assert lo < np.nanmean(np.nanmean(m, axis=0)) < hi


def test_score_cli_refuses_wrong_prereg_hash(tmp_path):
    prereg = tmp_path / 'p.md'
    prereg.write_text('x', encoding='utf-8')
    adm = tmp_path / 'a.json'
    adm.write_text(json.dumps({'markets': {}}), encoding='utf-8')
    with pytest.raises(SystemExit):
        R.main(['score', '--admission', str(adm), '--prereg', str(prereg),
                '--expected-prereg-sha256', '0' * 64, '--out', str(tmp_path / 'o.json')])
