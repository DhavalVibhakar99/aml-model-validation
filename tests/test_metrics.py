"""Metric definitions are where silent mistakes hide, so pin them to toy cases."""
import numpy as np
import pandas as pd
import pytest

from metrics import at_k, psi


def test_at_k_counts_top_of_ranking():
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([.9, .8, .7, .1, .2, .05])
    r = at_k(y, s, 2, n_pos_total=3)
    assert r["tp"] == 1 and r["precision"] == 0.5
    assert r["recall"] == pytest.approx(1 / 3)


def test_recall_denominator_includes_unscoreable_positives():
    # 2 scored positives, but 3 more launderers never showed up in the window:
    # catching both scored ones is 40% recall, not 100%
    y, s = np.array([1, 1, 0]), np.array([.9, .8, .1])
    assert at_k(y, s, 2, n_pos_total=5)["recall"] == pytest.approx(0.4)


def test_ties_break_the_same_way_every_time():
    y, s = np.array([0, 1, 0, 1]), np.zeros(4)
    assert len({at_k(y, s, 2, 2)["tp"] for _ in range(20)}) == 1


def test_psi_is_zero_for_same_distribution_and_grows_with_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=50_000)
    assert psi(a, rng.normal(size=50_000)) < 0.01
    assert psi(a, rng.normal(0.5, 1, 50_000)) > 0.1


def test_psi_handles_spiky_feature():
    # 90% zeros: decile edges collapse, which must not crash or return NaN
    a = np.r_[np.zeros(900), np.arange(100)]
    assert np.isfinite(psi(a, a[::-1]))
    assert psi(np.ones(10), np.ones(10)) == 0.0


def test_prior_shift_keeps_ranking_and_moves_mean():
    from validate import prior_shift
    p = np.array([0.001, 0.01, 0.2, 0.9])
    adj = prior_shift(p, pi_from=0.01, pi_to=0.005)
    assert (np.argsort(adj) == np.argsort(p)).all()
    assert (adj < p).all()
    # no change in prior -> no change in probability
    assert prior_shift(p, 0.01, 0.01) == pytest.approx(p)


def test_rules_score_orders_by_rule_count_then_flow():
    from features import FEATURES
    from train import rules_score
    d = pd.DataFrame(0.0, index=range(3), columns=FEATURES)
    d.loc[0, ["in_uniq", "out_uniq"]] = 5          # R2 + R3
    d.loc[1, "in_uniq"] = 5                          # R2 only
    d.loc[1, "in_usd"] = 1e9                         # huge flow can't beat an extra rule
    d.loc[2, "in_usd"] = 10                          # no rules
    s = rules_score(d)
    assert s[0] > s[1] > s[2]
