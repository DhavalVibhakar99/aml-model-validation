"""Tiny transaction sets where the right answer can be worked out by hand."""
import pytest

from conftest import run, txn


def test_fan_in_counts_distinct_senders(con):
    # three different people pay A, one of them twice: 4 txns but 3 counterparties
    rows = [txn("B", "A", 1, 100), txn("C", "A", 2, 100),
            txn("D", "A", 3, 100), txn("D", "A", 4, 100)]
    f = run(con, rows)
    assert f.loc["A", "in_count"] == 4
    assert f.loc["A", "in_uniq"] == 3
    assert f.loc["A", "out_count"] == 0
    assert f.loc["D", "out_uniq"] == 1


def test_pass_through_windows_and_gap(con):
    # A gets 1000, forwards 900 two hours later and the last 100 after 30h.
    # 24h window catches only the 900; 48h catches both.
    rows = [txn("B", "A", 0, 1000), txn("A", "C", 2, 900), txn("A", "D", 30, 100)]
    f = run(con, rows)
    assert f.loc["A", "pt_24h"] == pytest.approx(0.9)
    assert f.loc["A", "pt_48h"] == pytest.approx(1.0)
    assert f.loc["A", "med_gap_h"] == pytest.approx(16.0)   # median of 2h and 30h
    assert f.loc["A", "out_share"] == pytest.approx(0.5)     # 1000 in, 1000 out


def test_gap_uses_most_recent_incoming(con):
    # two deposits, then one send: the gap should be measured from the later one
    rows = [txn("B", "A", 0, 50), txn("C", "A", 10, 50), txn("A", "D", 11, 100)]
    assert run(con, rows).loc["A", "med_gap_h"] == pytest.approx(1.0)


def test_sending_before_receiving_is_not_pass_through(con):
    # money left before anything arrived, so it can't have "passed through"
    rows = [txn("A", "C", 0, 500), txn("B", "A", 5, 500)]
    f = run(con, rows)
    assert f.loc["A", "pt_24h"] == 0
    assert f.loc["A", "med_gap_h"] == pytest.approx(96.0)   # no gap -> whole window


def test_pass_through_is_capped(con):
    rows = [txn("B", "A", 0, 1), txn("A", "C", 1, 1_000_000)]
    assert run(con, rows).loc["A", "pt_24h"] == 10.0


def test_self_transfers_excluded_but_counted_after_seeding(con):
    # A->A twice: once during seeding (ignored), once after. Neither should
    # count as a counterparty or as in/out flow.
    rows = [txn("A", "A", 0, 5000), txn("A", "A", 30, 5000), txn("A", "B", 31, 10)]
    f = run(con, rows, seed_end=txn("x", "x", 24, 0)["ts"])
    assert f.loc["A", "self_count"] == 1
    assert f.loc["A", "out_count"] == 1
    assert f.loc["A", "in_count"] == 0
    assert f.loc["A", "out_usd"] == 10


def test_seeding_only_account_is_not_scored(con):
    rows = [txn("A", "A", 0, 5000), txn("B", "C", 1, 10)]
    f = run(con, rows, seed_end=txn("x", "x", 24, 0)["ts"])
    assert "A" not in f.index
    assert {"B", "C"} <= set(f.index)


def test_currency_converted_to_usd(con):
    rows = [txn("A", "B", 1, 100, pay="Euro")]
    assert run(con, rows).loc["B", "in_usd"] == pytest.approx(120.0)


def test_unknown_currency_fails_loudly(con):
    with pytest.raises(ValueError, match="no USD rate"):
        run(con, [txn("A", "B", 1, 100, pay="Doubloon")])


def test_cross_currency_share(con):
    rows = [txn("A", "B", 1, 100, pay="Euro", recv="US Dollar"), txn("A", "C", 2, 100)]
    assert run(con, rows).loc["A", "xccy_share"] == pytest.approx(0.5)


def test_format_mix_and_structuring(con):
    rows = [txn("A", "B", 1, 9_500, fmt="Cash"), txn("A", "B", 2, 9_900, fmt="Cash"),
            txn("A", "B", 3, 10_000, fmt="Wire"), txn("A", "B", 4, 100, fmt="ACH")]
    f = run(con, rows)
    assert f.loc["A", "fmt_cash"] == pytest.approx(0.5)
    assert f.loc["A", "fmt_wire"] == pytest.approx(0.25)
    assert f.loc["A", "near_10k_count"] == 2   # 10,000 itself is not "under"


def test_burstiness_extremes(con):
    # perfectly regular: every 5h -> sd 0 -> B = -1
    regular = [txn("A", f"B{i}", 5 * i, 10) for i in range(5)]
    assert run(con, regular).loc["A", "burstiness"] == pytest.approx(-1.0)
    # four within 20 minutes, then one ~2 days later. Gaps (min): 6, 6, 6, 2982
    # -> mean 750, sd 1288.6 -> B = 538.6 / 2038.6 = 0.264
    bursty = [txn("A", f"B{i}", 0.1 * i, 10) for i in range(4)] + [txn("A", "Z", 50, 10)]
    f = run(con, bursty)
    assert f.loc["A", "burstiness"] == pytest.approx(0.2642, abs=1e-3)
    assert f.loc["A", "max_txns_1h"] == 4
