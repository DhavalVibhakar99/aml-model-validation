import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

T0 = pd.Timestamp("2022-09-01 00:00")
FX = pd.DataFrame({"currency": ["US Dollar", "Euro"], "usd": [1.0, 1.2]})


def txn(src, dst, hours, amount, fmt="ACH", pay="US Dollar", recv=None, launder=0):
    """One row in the same shape ingest.py writes. `hours` is an offset from T0,
    which keeps the tests readable: 'A pays B 2h in' instead of timestamps."""
    return {
        "ts": T0 + pd.Timedelta(hours=hours), "src_acct": src, "dst_acct": dst,
        "amount_paid": amount, "payment_currency": pay,
        "amount_received": amount, "receiving_currency": recv or pay,
        "payment_format": fmt, "is_laundering": launder,
    }


@pytest.fixture
def con():
    return duckdb.connect()


def run(con, rows, start=T0, cutoff=T0 + pd.Timedelta(days=4), seed_end=T0):
    from features import build_features
    con.register("txns", pd.DataFrame(rows))
    return build_features(con, "txns", start, cutoff, seed_end, FX).set_index("acct")
