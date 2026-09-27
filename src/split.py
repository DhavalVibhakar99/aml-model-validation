"""
split.py - rolling time windows -> one modelling table per (dataset, window).

Each window is 4 days of transactions [start, cutoff), and carries two labels:
    label       laundering inside [start, cutoff)      - detection, the primary
    label_next  laundering in [cutoff, next_end)       - forecasting, secondary
Train (Sept 1-4) and test (Sept 5-8) don't overlap. DECISIONS #5 and #6 explain
why detection is primary and why this isn't the split first proposed.

Usage:
    python src/split.py              # both datasets
    python src/split.py --dataset HI-Small
"""
import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from features import build_features, estimate_fx

PROCESSED = Path("data/processed")

# Sept 11 onward is a sparse tail that's ~60% laundering - the generator wrapping
# up patterns it already started (DECISIONS #1). Nothing at or after this is used.
DATA_END = pd.Timestamp("2022-09-11")

# day-1 self-transfers are balance seeding, not behaviour (DECISIONS #2)
SEED_END = pd.Timestamp("2022-09-02")

WINDOWS = {
    #            features + label [start, cutoff)     label_next [cutoff, next_end)
    "train": {"start": pd.Timestamp("2022-09-01"), "cutoff": pd.Timestamp("2022-09-05"),
              "next_end": pd.Timestamp("2022-09-07")},
    "test":  {"start": pd.Timestamp("2022-09-05"), "cutoff": pd.Timestamp("2022-09-09"),
              "next_end": pd.Timestamp("2022-09-11")},
}


def make_labels(con, table: str, label_start, label_end) -> set:
    """Accounts that sent or received >= 1 laundering txn in [label_start, label_end)."""
    rows = con.execute(f"""
        SELECT src_acct FROM {table} WHERE is_laundering = 1 AND ts >= $a AND ts < $b
        UNION
        SELECT dst_acct FROM {table} WHERE is_laundering = 1 AND ts >= $a AND ts < $b
    """, {"a": pd.Timestamp(label_start), "b": pd.Timestamp(label_end)}).fetchall()
    return {r[0] for r in rows}


def build_window(con, dataset_key: str, name: str, fx: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    w = WINDOWS[name]
    assert w["next_end"] <= DATA_END, "label window runs into the trimmed tail"
    feats = build_features(con, "txns", w["start"], w["cutoff"], SEED_END, fx)
    scored = set(feats["acct"])
    stats = {"accounts": len(feats)}
    for col, (a, b) in {"label": (w["start"], w["cutoff"]),
                        "label_next": (w["cutoff"], w["next_end"])}.items():
        positives = make_labels(con, "txns", a, b)
        feats[col] = feats["acct"].isin(positives).astype(int)
        # positives we can't score because they did nothing in the feature window.
        # A real TM system would miss these too - they're a blind spot, not an
        # error, so they stay in the recall denominator (metrics.at_k). For
        # detection this is ~0 by construction (only a laundering *self*-transfer
        # on day 1 could do it); for forecasting it's a big chunk.
        stats[col] = {"positives": int(feats[col].sum()),
                      "positives_no_history": len(positives - scored),
                      "prevalence": float(feats[col].mean())}
    return feats, stats


def run(dataset: str) -> dict:
    key = dataset.lower().replace("-", "_")
    con = duckdb.connect()
    con.execute(f"CREATE VIEW txns AS SELECT * FROM '{PROCESSED / f'{key}_transactions.parquet'}'")

    # FX rates come from the train feature window only, and get reused for test
    tr = WINDOWS["train"]
    fx = estimate_fx(con, "txns", tr["start"], tr["cutoff"])

    summary = {}
    for name in WINDOWS:
        df, stats = build_window(con, key, name, fx)
        df.to_parquet(PROCESSED / f"{key}_{name}.parquet", index=False)
        summary[name] = stats
        print(f"{dataset:9s} {name:5s}  accounts {stats['accounts']:>7,}  " + "  ".join(
            f"{c}: {stats[c]['positives']:,} pos ({stats[c]['prevalence']:.3%}), "
            f"{stats[c]['positives_no_history']:,} unscoreable" for c in ("label", "label_next")))
    (PROCESSED / f"{key}_split_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["HI-Small", "LI-Small"])
    args = parser.parse_args()
    for ds in [args.dataset] if args.dataset else ["HI-Small", "LI-Small"]:
        run(ds)
