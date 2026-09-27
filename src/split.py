"""
split.py - rolling time windows -> one modelling table per (dataset, window).

Each window is: features from [start, cutoff), label from [cutoff, label_end).
Train and test use the same lengths (4 days of history, 2 days of label) and
don't overlap at all - see DECISIONS #5 for why this isn't the 1-4/5-6 vs
3-8/9-10 split first proposed.

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
    #            features [start, cutoff)        label [cutoff, label_end)
    "train": {"start": pd.Timestamp("2022-09-01"), "cutoff": pd.Timestamp("2022-09-05"),
              "label_end": pd.Timestamp("2022-09-07")},
    "test":  {"start": pd.Timestamp("2022-09-05"), "cutoff": pd.Timestamp("2022-09-09"),
              "label_end": pd.Timestamp("2022-09-11")},
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
    assert w["label_end"] <= DATA_END, "label window runs into the trimmed tail"
    feats = build_features(con, "txns", w["start"], w["cutoff"], SEED_END, fx)
    positives = make_labels(con, "txns", w["cutoff"], w["label_end"])
    feats["label"] = feats["acct"].isin(positives).astype(int)

    # positives we can't score because they had no history in the feature window.
    # A real TM system would miss these too - they're a blind spot, not an error,
    # and they belong in the recall denominator (see metrics.py / DECISIONS #5)
    n_unseen = len(positives - set(feats["acct"]))
    stats = {
        "accounts": len(feats), "positives": int(feats["label"].sum()),
        "positives_no_history": n_unseen,
        "prevalence": float(feats["label"].mean()),
    }
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
        print(f"{dataset:9s} {name:5s}  accounts {stats['accounts']:>7,}  "
              f"positives {stats['positives']:>5,}  ({stats['prevalence']:.3%})  "
              f"no-history positives {stats['positives_no_history']:,}")
    (PROCESSED / f"{key}_split_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["HI-Small", "LI-Small"])
    args = parser.parse_args()
    for ds in [args.dataset] if args.dataset else ["HI-Small", "LI-Small"]:
        run(ds)
