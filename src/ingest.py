"""
ingest.py - turn IBM's raw AML transaction CSVs into clean, typed Parquet.

Why bother with Parquet? HI-Small is ~5M rows. Every later script (features,
training, validation) would otherwise re-parse that CSV from scratch, which gets
old fast. Parquet is columnar and typed, so a feature script that only needs
4 columns reads only those 4 columns.

Usage:
    python src/ingest.py --dataset HI-Small
    python src/ingest.py --dataset LI-Small
"""
import argparse
from pathlib import Path

import duckdb

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

# The raw header has TWO columns literally named "Account" (sender, receiver).
# pandas quietly renames the second one "Account.1" - the kind of thing that
# bites you three scripts later. So we name every column ourselves, by position.
COLUMNS = {
    "ts_raw": "VARCHAR",
    "from_bank": "BIGINT",
    "from_account": "VARCHAR",
    "to_bank": "BIGINT",
    "to_account": "VARCHAR",
    "amount_received": "DOUBLE",
    "receiving_currency": "VARCHAR",
    "amount_paid": "DOUBLE",
    "payment_currency": "VARCHAR",
    "payment_format": "VARCHAR",
    "is_laundering": "INTEGER",
}


def ingest(dataset: str) -> Path:
    src = RAW_DIR / f"{dataset}_Trans.csv"
    dst = OUT_DIR / f"{dataset.lower().replace('-', '_')}_transactions.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found - did the Kaggle download/unzip finish?")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT
                -- a stable id per transaction; we'll need it to join labels back
                -- onto predictions later without guessing
                row_number() OVER () AS txn_id,
                -- try_strptime returns NULL instead of crashing on a weird row,
                -- then sanity_check() below refuses to pass if any NULLs exist.
                -- Fail loudly, but only after we can see how many rows are bad.
                try_strptime(ts_raw, '%Y/%m/%d %H:%M') AS ts,
                -- account numbers are only unique *within* a bank. The real
                -- identity is bank + account; skip this and we'd silently merge
                -- unrelated customers who happen to share an account string.
                from_bank::VARCHAR || '_' || from_account AS src_acct,
                to_bank::VARCHAR   || '_' || to_account   AS dst_acct,
                from_bank,
                to_bank,
                amount_paid,
                payment_currency,
                amount_received,
                receiving_currency,
                payment_format,
                is_laundering
            FROM read_csv('{src}', header = true, columns = {COLUMNS})
        ) TO '{dst}' (FORMAT PARQUET)
    """)
    return dst


def sanity_check(path: Path) -> None:
    """Numbers to eyeball before trusting anything downstream."""
    con = duckdb.connect()
    n, n_bad, n_launder, t_min, t_max, cross_ccy = con.execute(f"""
        SELECT
            count(*),
            count(*) FILTER (WHERE ts IS NULL),
            sum(is_laundering),
            min(ts),
            max(ts),
            count(*) FILTER (WHERE payment_currency <> receiving_currency)
        FROM '{path}'
    """).fetchone()

    # every account that ever sends OR receives - a mule might only receive
    # in this window, so counting senders alone would undercount
    n_accounts = con.execute(f"""
        SELECT count(DISTINCT acct) FROM (
            SELECT src_acct AS acct FROM '{path}'
            UNION ALL
            SELECT dst_acct FROM '{path}'
        )
    """).fetchone()[0]

    print(f"\n{path.name}")
    print(f"  transactions      : {n:,}")
    print(f"  accounts          : {n_accounts:,}")
    print(f"  laundering txns   : {n_launder:,}  (1 in {n // max(n_launder, 1):,})")
    print(f"  time range        : {t_min}  ->  {t_max}")
    print(f"  cross-currency    : {cross_ccy:,}  ({cross_ccy / n:.2%})")

    # amounts feed every in/out and pass-through feature, so a NULL or a zero
    # here would quietly turn into a divide-by-zero or a fake "0% pass-through"
    n_bad_amt, n_self = con.execute(f"""
        SELECT
            count(*) FILTER (WHERE amount_paid IS NULL OR amount_received IS NULL
                             OR amount_paid <= 0 OR amount_received <= 0),
            count(*) FILTER (WHERE src_acct = dst_acct)
        FROM '{path}'
    """).fetchone()
    print(f"  null/non-pos amts : {n_bad_amt:,}")
    print(f"  self-transfers    : {n_self:,}  ({n_self / n:.2%})")

    # the split design depends on what each day looks like - a sparse tail
    # where most rows are laundering would be a generator artifact, not behavior
    daily = con.execute(f"""
        SELECT ts::DATE, count(*), sum(is_laundering)
        FROM '{path}' GROUP BY 1 ORDER BY 1
    """).fetchall()
    print("  daily volume      :")
    for day, cnt, bad in daily:
        print(f"    {day}  {cnt:>9,} txns  {bad:>4} laundering  ({bad / cnt:.2%})")

    # the checks that should stop the pipeline cold
    if n_bad:
        raise ValueError(f"{n_bad:,} rows had unparseable timestamps - inspect before moving on")
    if n_bad_amt:
        raise ValueError(f"{n_bad_amt:,} rows had missing or non-positive amounts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IBM AML CSV -> Parquet")
    parser.add_argument("--dataset", choices=["HI-Small", "LI-Small"], required=True)
    args = parser.parse_args()

    out = ingest(args.dataset)
    sanity_check(out)
