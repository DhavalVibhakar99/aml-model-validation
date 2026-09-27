"""
features.py - account-level behavior features from a window of transactions.

Everything here is computed from rows with start <= ts < cutoff and nothing
else. That's the whole leakage story in one line, so the filter lives in
exactly one place (the `base` CTE) and tests/test_leakage.py hammers it.

The function takes a DuckDB connection plus a table name rather than a file
path, so the tests can register a tiny hand-built DataFrame under the same
name and run the exact same SQL the real pipeline runs.
"""
import duckdb
import pandas as pd

# Reinvestment never shows up between two different accounts (it's always a
# self-transfer), so after we drop self-transfers its share would be all zeros
FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Wire"]

FEATURES = [
    "in_count", "out_count", "in_uniq", "out_uniq", "in_usd", "out_usd",
    "out_share", "pt_24h", "pt_48h", "med_gap_h", "xccy_share",
    *[f"fmt_{f.lower().replace(' ', '_')}" for f in FORMATS],
    "near_10k_count", "burstiness", "max_txns_1h", "self_count",
]

# pass-through ratios blow up when an account received almost nothing, so they
# get capped. 10x is already "sent out way more than came in".
PT_CAP = 10.0

# structuring = keeping amounts just under a reporting threshold. $10k is the
# US CTR line; $8k-$10k is the usual band a TM rule would watch.
NEAR_10K = (8_000, 10_000)


def build_features(con: duckdb.DuckDBPyConnection, table: str, start, cutoff,
                   seed_end, fx: pd.DataFrame) -> pd.DataFrame:
    """One row per account active in [start, cutoff). `seed_end`: self-transfers
    before this are simulation seeding and are ignored (see DECISIONS #2)."""
    con.register("fx_rates", fx)
    fmt_cols = ",\n".join(
        f"avg((fmt = '{f}')::INT) AS fmt_{f.lower().replace(' ', '_')}" for f in FORMATS
    )
    window_h = (pd.Timestamp(cutoff) - pd.Timestamp(start)).total_seconds() / 3600

    df = con.execute(f"""
        WITH base AS (
            -- the only place transactions enter. Everything downstream reads this.
            -- LEFT JOIN on purpose: an unknown currency becomes a NULL we can catch
            -- below, instead of an inner join quietly dropping the row
            SELECT t.*, t.amount_paid * fx.usd AS usd
            FROM {table} t
            LEFT JOIN fx_rates fx ON t.payment_currency = fx.currency
            WHERE t.ts >= $start AND t.ts < $cutoff
        ),
        -- each real transfer becomes two legs: an 'out' for the sender and an 'in'
        -- for the receiver. Self-transfers are excluded here, they'd pad fan-in AND
        -- fan-out and push the flow balance toward 50/50 for no reason
        legs AS (
            SELECT src_acct AS acct, 'out' AS dir, dst_acct AS cp, ts, usd,
                   payment_format AS fmt, payment_currency <> receiving_currency AS xccy
            FROM base WHERE src_acct <> dst_acct
            UNION ALL
            SELECT dst_acct, 'in', src_acct, ts, usd,
                   payment_format, payment_currency <> receiving_currency
            FROM base WHERE src_acct <> dst_acct
        ),
        agg AS (
            SELECT acct,
                count(*) FILTER (WHERE dir = 'in')  AS in_count,
                count(*) FILTER (WHERE dir = 'out') AS out_count,
                count(DISTINCT cp) FILTER (WHERE dir = 'in')  AS in_uniq,
                count(DISTINCT cp) FILTER (WHERE dir = 'out') AS out_uniq,
                coalesce(sum(usd) FILTER (WHERE dir = 'in'), 0)  AS in_usd,
                coalesce(sum(usd) FILTER (WHERE dir = 'out'), 0) AS out_usd,
                avg(xccy::INT) AS xccy_share,
                {fmt_cols},
                count(*) FILTER (WHERE usd >= {NEAR_10K[0]} AND usd < {NEAR_10K[1]}) AS near_10k_count
            FROM legs GROUP BY acct
        ),
        -- for every outgoing payment, find the most recent incoming one at or
        -- before it. The gap tells us how fast money moved through the account.
        -- Minute-level timestamps mean a same-minute in/out counts as gap 0.
        gaps AS (
            SELECT o.acct, o.usd, date_diff('minute', i.ts, o.ts) / 60.0 AS gap_h
            FROM (SELECT acct, ts, usd FROM legs WHERE dir = 'out') o
            ASOF JOIN (SELECT acct, ts FROM legs WHERE dir = 'in') i
              ON o.acct = i.acct AND o.ts >= i.ts
        ),
        pt AS (
            SELECT acct,
                coalesce(sum(usd) FILTER (WHERE gap_h <= 24), 0) AS out_usd_24h,
                coalesce(sum(usd) FILTER (WHERE gap_h <= 48), 0) AS out_usd_48h,
                median(gap_h) AS med_gap_h
            FROM gaps GROUP BY acct
        ),
        -- burstiness (Goh & Barabasi 2008): B = (sd - mean) / (sd + mean) of the
        -- time between an account's transactions. -1 = clockwork, 0 = random,
        -- towards 1 = long quiet spells then a flurry. Needs at least 2 gaps.
        inter AS (
            SELECT acct, date_diff('minute', lag(ts) OVER (PARTITION BY acct ORDER BY ts), ts) AS dt
            FROM legs
        ),
        burst AS (
            SELECT acct,
                CASE WHEN count(dt) < 2 THEN 0.0
                     WHEN stddev_pop(dt) + avg(dt) = 0 THEN 1.0  -- everything in one minute
                     ELSE (stddev_pop(dt) - avg(dt)) / (stddev_pop(dt) + avg(dt)) END AS burstiness
            FROM inter GROUP BY acct
        ),
        hourly AS (
            SELECT acct, max(n) AS max_txns_1h
            FROM (SELECT acct, count(*) AS n FROM legs GROUP BY acct, date_trunc('hour', ts))
            GROUP BY acct
        ),
        selfs AS (
            SELECT src_acct AS acct, count(*) AS self_count
            FROM base WHERE src_acct = dst_acct AND ts >= $seed_end
            GROUP BY acct
        ),
        -- who gets scored: anyone with a real transfer, or a self-transfer that
        -- isn't day-1 seeding. Seeding-only accounts did nothing in this window.
        pop AS (SELECT acct FROM agg UNION SELECT acct FROM selfs)
        SELECT pop.acct,
            coalesce(in_count, 0) AS in_count, coalesce(out_count, 0) AS out_count,
            coalesce(in_uniq, 0) AS in_uniq, coalesce(out_uniq, 0) AS out_uniq,
            coalesce(in_usd, 0) AS in_usd, coalesce(out_usd, 0) AS out_usd,
            -- share of the account's total flow that went out: 0.5 means as much
            -- left as arrived, which is what a pass-through account looks like.
            -- Bounded, so it's defined even when nothing came in.
            CASE WHEN coalesce(in_usd, 0) + coalesce(out_usd, 0) = 0 THEN 0
                 ELSE out_usd / (coalesce(in_usd, 0) + out_usd) END AS out_share,
            CASE WHEN coalesce(in_usd, 0) = 0 THEN 0
                 ELSE least(coalesce(out_usd_24h, 0) / in_usd, {PT_CAP}) END AS pt_24h,
            CASE WHEN coalesce(in_usd, 0) = 0 THEN 0
                 ELSE least(coalesce(out_usd_48h, 0) / in_usd, {PT_CAP}) END AS pt_48h,
            -- never sent anything after receiving: call the gap "the whole window"
            coalesce(med_gap_h, {window_h}) AS med_gap_h,
            coalesce(xccy_share, 0) AS xccy_share,
            {", ".join(f"coalesce(fmt_{f.lower().replace(' ', '_')}, 0) AS fmt_{f.lower().replace(' ', '_')}" for f in FORMATS)},
            coalesce(near_10k_count, 0) AS near_10k_count,
            coalesce(burstiness, 0) AS burstiness,
            coalesce(max_txns_1h, 0) AS max_txns_1h,
            coalesce(self_count, 0) AS self_count,
            -- carried along only so we can check for unconverted currencies
            (SELECT count(*) FROM base WHERE usd IS NULL) AS _n_missing_fx
        FROM pop
        LEFT JOIN agg USING (acct) LEFT JOIN pt USING (acct)
        LEFT JOIN burst USING (acct) LEFT JOIN hourly USING (acct)
        LEFT JOIN selfs USING (acct)
        ORDER BY pop.acct
    """, {"start": pd.Timestamp(start), "cutoff": pd.Timestamp(cutoff),
          "seed_end": pd.Timestamp(seed_end)}).df()

    if len(df) and df["_n_missing_fx"].iloc[0] > 0:
        raise ValueError(f"{df['_n_missing_fx'].iloc[0]} txns in a currency with no USD rate")
    return df.drop(columns="_n_missing_fx")[["acct", *FEATURES]]


def estimate_fx(con: duckdb.DuckDBPyConnection, table: str, start, end) -> pd.DataFrame:
    """USD value of one unit of each currency, read off cross-currency transfers.

    The simulator uses fixed rates, so the median implied rate is stable. Both
    directions (X->USD and USD->X) count, which matters for thin currencies like
    Bitcoin. Only [start, end) is used, so the rates come from the training
    feature window and nothing later.
    """
    fx = con.execute(f"""
        WITH r AS (
            SELECT payment_currency AS currency, amount_received / amount_paid AS usd
            FROM {table}
            WHERE receiving_currency = 'US Dollar' AND payment_currency <> 'US Dollar'
              AND ts >= $start AND ts < $end
            UNION ALL
            SELECT receiving_currency, amount_paid / amount_received
            FROM {table}
            WHERE payment_currency = 'US Dollar' AND receiving_currency <> 'US Dollar'
              AND ts >= $start AND ts < $end
        )
        SELECT currency, median(usd) AS usd, count(*) AS n_obs FROM r GROUP BY currency
        UNION ALL SELECT 'US Dollar', 1.0, NULL
        ORDER BY currency
    """, {"start": pd.Timestamp(start), "end": pd.Timestamp(end)}).df()
    return fx
