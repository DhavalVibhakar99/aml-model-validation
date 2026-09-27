"""The rule that matters most: nothing at or after the cutoff can touch a feature."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from conftest import T0, run, txn

CUTOFF = T0 + pd.Timedelta(days=2)
START = T0 + pd.Timedelta(hours=1)

HISTORY = [txn("B", "A", 2, 1000), txn("A", "C", 3, 900, fmt="Cash"),
           txn("A", "A", 30, 50), txn("D", "A", 20, 9_500)]


def test_rows_at_or_after_cutoff_change_nothing(con):
    before = run(con, HISTORY, start=START, cutoff=CUTOFF)
    # the future: huge amounts, new counterparties, one exactly on the cutoff
    # minute, laundering flags everywhere
    future = [txn("A", "Z", 48, 1e9, launder=1),            # ts == cutoff
              txn("Y", "A", 48.5, 1e9, launder=1),
              txn("A", "A", 60, 1, launder=1)]
    after = run(con, HISTORY + future, start=START, cutoff=CUTOFF)
    assert_frame_equal(before, after)


def test_rows_before_start_change_nothing(con):
    before = run(con, HISTORY, start=START, cutoff=CUTOFF)
    past = [txn("A", "Q", 0, 1e6), txn("Q", "A", 0.5, 1e6)]
    after = run(con, HISTORY + past, start=START, cutoff=CUTOFF)
    assert_frame_equal(before, after)


def test_label_column_is_not_used(con):
    # shuffle is_laundering completely: if any feature moves, a label leaked in
    rows = pd.DataFrame(HISTORY)
    a = run(con, rows.to_dict("records"), start=START, cutoff=CUTOFF)
    rows["is_laundering"] = np.random.default_rng(0).integers(0, 2, len(rows))
    b = run(con, rows.to_dict("records"), start=START, cutoff=CUTOFF)
    assert_frame_equal(a, b)


def test_windows_do_not_overlap():
    from split import WINDOWS
    for name, w in WINDOWS.items():
        assert w["start"] < w["cutoff"] < w["label_end"], name
    tr, te = WINDOWS["train"], WINDOWS["test"]
    # everything training saw (its features AND labels) ends before the test
    # labels begin. Test *features* may reuse days that were train labels -
    # that's just what "history" means at scoring time.
    assert tr["label_end"] <= te["cutoff"]
    # same lengths, so a count over 4 days means the same thing in both
    assert tr["cutoff"] - tr["start"] == te["cutoff"] - te["start"]
    assert tr["label_end"] - tr["cutoff"] == te["label_end"] - te["cutoff"]


def test_labels_only_come_from_label_window(con):
    from split import make_labels
    rows = pd.DataFrame([
        txn("A", "B", 10, 1, launder=1),    # feature window: must NOT label anyone
        txn("C", "D", 49, 1, launder=1),    # label window
        txn("E", "F", 80, 1, launder=1),    # after label window
    ])
    con.register("txns", rows)
    labels = make_labels(con, "txns", CUTOFF, CUTOFF + pd.Timedelta(days=1))
    assert set(labels) == {"C", "D"}
