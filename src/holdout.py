"""
holdout.py - score the final holdout (Sept 7-10, HI-Small) ONCE with the
frozen artifacts from train.py. See DECISIONS #11 for the design, which was
written down before this ever ran.

It loads the saved models rather than refitting, records their hashes and the
git sha, and refuses to run again if reports/holdout.json already exists. The
point of a holdout is that you only get to look once - a script that happily
reruns makes it too easy to "just check one more thing".

Usage: python src/holdout.py
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from features import FEATURES, build_features, estimate_fx
from metrics import KS, at_k, psi, summary
from split import DATA_END, PROCESSED, SEED_END, WINDOWS, make_labels
from train import ARTIFACTS, rules_score

OUT = "reports/holdout.json"
START, CUTOFF = pd.Timestamp("2022-09-07"), DATA_END
# the two days no decision has ever seen; Sept 7-8 were part of the test window
FRESH_START = pd.Timestamp("2022-09-09")
FROZEN = ["lightgbm.txt", "logreg.joblib", "rule_thresholds.json"]


def sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = Path(OUT)
    if out.exists():
        sys.exit(f"{OUT} exists - the holdout has already been scored. Not rerunning.")
    # the holdout must be the same length as the windows the model learned on
    assert CUTOFF - START == WINDOWS["train"]["cutoff"] - WINDOWS["train"]["start"]

    con = duckdb.connect()
    con.execute(f"CREATE VIEW txns AS SELECT * FROM '{PROCESSED / 'hi_small_transactions.parquet'}'")
    tr = WINDOWS["train"]
    fx = estimate_fx(con, "txns", tr["start"], tr["cutoff"])
    df = build_features(con, "txns", START, CUTOFF, SEED_END, fx)

    pos = make_labels(con, "txns", START, CUTOFF)
    seen_days = make_labels(con, "txns", START, FRESH_START)          # Sept 7-8
    fresh = make_labels(con, "txns", FRESH_START, CUTOFF) - seen_days  # only Sept 9-10
    y = df.acct.isin(pos).to_numpy().astype(int)
    y_fresh = df.acct.isin(fresh).to_numpy().astype(int)
    stale = df.acct.isin(seen_days).to_numpy()
    n_pos = len(pos)  # includes any positive we couldn't score, for recall

    th = json.loads((ARTIFACTS / "rule_thresholds.json").read_text())
    booster = lgb.Booster(model_file=str(ARTIFACTS / "lightgbm.txt"))
    logreg = joblib.load(ARTIFACTS / "logreg.joblib")
    scores = {
        "rules": rules_score(df, th["rules"]),
        "rules_untuned": rules_score(df, th["rules_untuned"]),
        "logreg": logreg.predict_proba(df[FEATURES])[:, 1],
        "lightgbm": booster.predict(df[FEATURES]),
    }

    # score drift vs the train window, with the same frozen model
    train_tbl = pd.read_parquet(PROCESSED / "hi_small_train.parquet")
    psi_gbm = psi(booster.predict(train_tbl[FEATURES]), scores["lightgbm"])

    results = {}
    for name, s in scores.items():
        r = summary(y, s, n_pos, probabilistic=not name.startswith("rules"))
        # fresh positives: same global ranking and budget, but only count
        # launderers whose labels nobody has seen before
        order = np.argsort(-s, kind="stable")
        for k in KS:
            r[f"fresh_recall_at_{k}"] = float(y_fresh[order[:k]].sum() / len(fresh))
        # PR-AUC on the fresh labels, leaving out accounts already known from
        # Sept 7-8 (they're neither clean negatives nor fresh positives)
        keep = ~stale
        r["fresh_pr_auc"] = float(average_precision_score(y_fresh[keep], s[keep]))
        results[name] = r

    record = {
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(),
        "artifact_sha256": {f: sha(ARTIFACTS / f) for f in FROZEN},
        "window": {"start": str(START), "cutoff": str(CUTOFF), "fresh_from": str(FRESH_START)},
        "population": {"accounts": len(df), "positives": int(y.sum()), "positives_total": n_pos,
                       "prevalence": float(y.mean()), "fresh_positives": len(fresh),
                       "fresh_scored": int(y_fresh.sum()), "stale_accounts": int(stale.sum())},
        "psi_lightgbm_score_vs_train": psi_gbm,
        "metrics": results,
    }
    out.write_text(json.dumps(record, indent=2))

    print(f"holdout Sept 7-10: {len(df):,} accounts, {y.sum():,} positives ({y.mean():.2%}), "
          f"{len(fresh):,} fresh (Sept 9-10 only)")
    for name, r in results.items():
        print(f"  {name:14s} PR-AUC {r['pr_auc']:.3f}  P@500 {r['precision_at_500']:.2f}  "
              f"R@500 {r['recall_at_500']:.3f}  R@1000 {r['recall_at_1000']:.3f}  | fresh: "
              f"PR-AUC {r['fresh_pr_auc']:.3f}  R@500 {r['fresh_recall_at_500']:.3f}  "
              f"R@1000 {r['fresh_recall_at_1000']:.3f}")
    print(f"  LightGBM score PSI vs train window: {psi_gbm:.3f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
