import numpy as np
import pandas as pd
import pytest
from cottonlens_ml.research.named_weak_signal import (
    SOURCE,
    build,
    calibrate,
    expected,
    teacher,
)


def frame(n=80):
    f = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=n),
            SOURCE: np.linspace(-0.03, 0.03, n),
            "named_eligible": True,
            "named_price": 80.0,
            "named_return": np.resize([-0.04, 0.02, 0.01, -0.01], n),
            "target_return_5": 0.0,
            "named_target_price": 80.0,
            "target_date_5": pd.bdate_range("2020-01-08", periods=n),
            "irrelevant_feature": np.arange(n, dtype=float),
        }
    )
    f.loc[5, SOURCE] = np.nan
    return f


def test_only_initial_train_calibrates_and_exact_expected_strength():
    h = frame()
    dates = h.date.iloc[:40].dt.strftime("%Y-%m-%d").tolist()
    c = calibrate(h, dates)
    later = h.copy()
    later.loc[40:, ["named_price", "named_return", SOURCE]] = 999.0
    assert calibrate(later, dates) == c
    mu = c["beta"] * teacher(h.iloc[:40], c["threshold"])
    metric = expected(h.iloc[:40], mu, mu, c["sigma"])
    assert metric["gain_pct"] == pytest.approx(10.0, abs=1e-10)


def test_same_noise_null_and_injected_missingness_calendar_and_overlap():
    h = frame()
    c = calibrate(h, h.date.iloc[:40].dt.strftime("%Y-%m-%d").tolist())
    null, zero, noise = build(h, c, "null", 703)
    injected, mu, same = build(h, c, "injected", 703)
    np.testing.assert_array_equal(noise, same)
    np.testing.assert_allclose(
        injected.named_return - null.named_return, mu, atol=1e-17
    )
    assert mu[5] == 0.0 and np.all(zero == 0.0)
    np.testing.assert_array_equal(injected[SOURCE].isna(), h[SOURCE].isna())
    pd.testing.assert_series_equal(injected.date, h.date)
    pd.testing.assert_series_equal(injected.target_date_5, h.target_date_5)
    innovations = np.random.default_rng(703).standard_normal(len(h) + 5)
    assert noise[0] == np.sum(innovations[1:6]) / np.sqrt(5.0)
    assert noise[1] == np.sum(innovations[2:7]) / np.sqrt(5.0)
    assert expected(null, np.zeros(len(h)), zero, c["sigma"])["gain_pct"] == 0.0


def test_future_inputs_do_not_change_prefix_features_or_synthetic_labels():
    h = frame()
    c = calibrate(h, h.date.iloc[:40].dt.strftime("%Y-%m-%d").tolist())
    first, mu, noise = build(h, c, "injected", 703)
    altered = h.copy()
    altered.loc[60:, [SOURCE, "named_price"]] = 123.0
    second, _, _ = build(altered, c, "injected", 703)
    pd.testing.assert_frame_equal(first.iloc[:60], second.iloc[:60], check_exact=True)
    extended = pd.concat(
        [
            h,
            frame(10).assign(
                date=pd.bdate_range(h.date.iloc[-1] + pd.Timedelta(days=1), periods=10)
            ),
        ],
        ignore_index=True,
    )
    _, later_mu, later_noise = build(extended, c, "injected", 703)
    np.testing.assert_array_equal(later_mu[: len(h)], mu)
    np.testing.assert_array_equal(later_noise[: len(h)], noise)


@pytest.mark.parametrize("condition,seed", [("unknown", 703), ("injected", 42)])
def test_unregistered_scenarios_rejected(condition, seed):
    h = frame()
    c = calibrate(h, h.date.iloc[:40].dt.strftime("%Y-%m-%d").tolist())
    with pytest.raises(ValueError):
        build(h, c, condition, seed)


def test_bad_calibration_and_uncompressed_calendar_rejected():
    h = frame()
    dates = h.date.iloc[:40].dt.strftime("%Y-%m-%d").tolist()
    bad = h.copy()
    bad.loc[0, "named_eligible"] = False
    with pytest.raises(ValueError):
        calibrate(bad, dates)
    c = calibrate(h, dates)
    with pytest.raises(ValueError):
        build(h.iloc[::-1], c, "null", 703)
