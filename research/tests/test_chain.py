"""Synthetic checks for the Link 1 event-study statistics (no market data)."""
import numpy as np
import pandas as pd

from research.chain import link1_event_study as L


def test_oracle_gate_and_sign_on_a_perfect_inverse_relation():
    years = np.repeat(np.arange(2012, 2022), 4)
    r = np.random.RandomState(0).normal(0, 0.03, len(years))
    e = pd.DataFrame({'year': years, 'r': r, 'R': -0.5 * r})
    g = L.oracle_gate(e)
    assert g['events'] == 32 and g['gain_pct'] > 99
    rho, p = L.h2(e.r.to_numpy(), e.R.to_numpy(), np.random.RandomState(1))
    assert rho == -1.0 and p < 0.001


def test_h1_detects_larger_event_moves_and_not_equal_ones():
    rng = np.random.RandomState(2)
    pool = np.abs(rng.normal(0, 0.01, 2000))
    big = np.abs(rng.normal(0, 0.03, 38))
    same = np.abs(rng.normal(0, 0.01, 38))
    assert L.h1(big, pool, np.random.RandomState(3))[1] < 0.01
    assert L.h1(same, pool, np.random.RandomState(3))[1] > 0.05
