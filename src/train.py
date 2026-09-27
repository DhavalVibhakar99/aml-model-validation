"""
train.py - fit the three candidates on the HI-Small train window (Sept 1-4)
and score every window validation needs.

    (a) rules    - a transaction-monitoring rule set, like a bank would run
    (b) logreg   - logistic regression, the "explainable" challenger
    (c) lightgbm - gradient boosting, the "accurate" challenger

Plus three supporting fits that exist only to answer validation questions:
    lightgbm_no_format   - same, minus payment-format shares. How much of the
                           result is "the simulator launders over ACH"?
    lightgbm_forecast    - trained on label_next. The originally proposed
                           framing, kept to show why it isn't primary (DECISIONS #6)
    lightgbm_li_native   - trained on LI-Small's own train window. Reference
                           point for the stress test.

Writes:
    artifacts/scores.parquet      one row per (model, dataset, window, account)
    artifacts/lightgbm_*.txt      the fitted boosters
    artifacts/logreg_coefs.json   so the LR is inspectable without pickles
    reports/runs/<run_id>.json    params + headline metrics for this run

Usage: python src/train.py
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from features import FEATURES
from metrics import summary
from split import PROCESSED, WINDOWS

ARTIFACTS = Path("artifacts")
RUNS = Path("reports/runs")
SEED = 42

# counts and dollar amounts span 0 to billions; LR needs them squashed or the
# one $30bn account dominates the fit. Shares, ratios and burstiness are
# already on a small scale and stay as they are.
HEAVY = ["in_count", "out_count", "in_uniq", "out_uniq", "in_usd", "out_usd",
         "med_gap_h", "near_10k_count", "max_txns_1h", "self_count", "pt_24h", "pt_48h"]

# Fixed, fairly conservative settings - no tuning. Tuning needs a validation
# window, and with 10 days of data the only candidate is the test window,
# which would make the test result optimistic. min_child_samples is high
# because there are only ~1,250 positives; small leaves would memorise them.
LGBM_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
                   min_child_samples=200, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
                   n_jobs=4, verbose=-1)
LR_PARAMS = dict(C=1.0, max_iter=2000)

# Rule thresholds: typology-style, set by looking at the HI train window only.
# No class weights anywhere - they'd buy nothing for ranking and would wreck
# calibration (DECISIONS #6).
RULES = {
    "R1_rapid_movement": lambda d: (d.pt_24h >= 0.9) & (d.in_usd >= 5_000),
    "R2_fan_in":         lambda d: d.in_uniq >= 5,
    "R3_fan_out":        lambda d: d.out_uniq >= 5,
    "R4_structuring":    lambda d: d.near_10k_count >= 2,
    "R5_velocity":       lambda d: d.max_txns_1h >= 5,
    "R6_high_risk_channel": lambda d: ((d.fmt_cash + d.fmt_bitcoin) >= 0.5)
                                      & (d.in_count + d.out_count >= 3),
}


def rule_flags(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: rule(d) for name, rule in RULES.items()}).astype(int)


def rules_score(d: pd.DataFrame) -> np.ndarray:
    # alert priority = number of rules hit; ties go to the bigger dollar flow,
    # the way an alert queue is usually sorted. The rank term is < 1 so it can
    # never outrank an extra rule.
    flow_rank = (d.in_usd + d.out_usd).rank(pct=True, method="first").to_numpy() * 0.999
    return rule_flags(d).sum(axis=1).to_numpy() + flow_rank


def log_heavy(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X[HEAVY] = np.log1p(X[HEAVY])
    return X


def fit_logreg(X, y):
    return make_pipeline(FunctionTransformer(log_heavy), StandardScaler(),
                         LogisticRegression(**LR_PARAMS)).fit(X, y)


def fit_lgbm(X, y):
    return lgb.LGBMClassifier(**LGBM_PARAMS).fit(X, y)


FORMAT_COLS = [f for f in FEATURES if f.startswith("fmt_")]
NO_FORMAT = [f for f in FEATURES if f not in FORMAT_COLS]


def load(key: str, window: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{key}_{window}.parquet")


def n_positives(key: str, window: str, target: str) -> int:
    """All positives for recall, including ones with no history to score."""
    s = json.loads((PROCESSED / f"{key}_split_summary.json").read_text())[window][target]
    return s["positives"] + s["positives_no_history"]


# (model name, target column, columns it sees, datasets it's scored on)
EVAL_PLAN = [
    ("rules", "label", FEATURES, ("hi_small", "li_small")),
    ("logreg", "label", FEATURES, ("hi_small", "li_small")),
    ("lightgbm", "label", FEATURES, ("hi_small", "li_small")),
    ("lightgbm_no_format", "label", NO_FORMAT, ("hi_small",)),
    ("lightgbm_forecast", "label_next", FEATURES, ("hi_small",)),
    ("lightgbm_li_native", "label", FEATURES, ("li_small",)),
]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    data = {(k, w): load(k, w) for k in ("hi_small", "li_small") for w in WINDOWS}
    hi, li = data[("hi_small", "train")], data[("li_small", "train")]
    models = {
        "logreg": fit_logreg(hi[FEATURES], hi["label"]),
        "lightgbm": fit_lgbm(hi[FEATURES], hi["label"]),
        "lightgbm_no_format": fit_lgbm(hi[NO_FORMAT], hi["label"]),
        "lightgbm_forecast": fit_lgbm(hi[FEATURES], hi["label_next"]),
        "lightgbm_li_native": fit_lgbm(li[FEATURES], li["label"]),
    }

    rows = []
    for name, target, cols, datasets in EVAL_PLAN:
        for (key, window), df in data.items():
            if key not in datasets:
                continue
            s = rules_score(df) if name == "rules" else models[name].predict_proba(df[cols])[:, 1]
            rows.append(pd.DataFrame({"model": name, "target": target, "dataset": key,
                                      "window": window, "acct": df["acct"], "score": s,
                                      "label": df[target]}))
    scores = pd.concat(rows, ignore_index=True)
    scores.to_parquet(ARTIFACTS / "scores.parquet", index=False)

    for name in ("lightgbm", "lightgbm_no_format", "lightgbm_forecast", "lightgbm_li_native"):
        models[name].booster_.save_model(ARTIFACTS / f"{name}.txt")
    lr = models["logreg"][-1]
    (ARTIFACTS / "logreg_coefs.json").write_text(json.dumps(
        {"intercept": float(lr.intercept_[0]),
         "coefs": dict(zip(FEATURES, map(float, lr.coef_[0])))}, indent=2))

    # headline metrics for the run log. validate.py recomputes everything
    # (with intervals) from scores.parquet; this is just the quick look.
    metrics = {}
    test = scores[scores.window == "test"]
    for (name, key, target), g in test.groupby(["model", "dataset", "target"]):
        metrics[f"{key}/{name}"] = summary(g.label, g.score, n_positives(key, "test", target),
                                           probabilistic=name != "rules")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = {"run_id": run_id, "git_sha": git_sha(), "train_on": "hi_small/train",
           "windows": {w: {k: str(v) for k, v in d.items()} for w, d in WINDOWS.items()},
           "features": FEATURES, "rules": list(RULES),
           "params": {"lightgbm": LGBM_PARAMS, "logreg": LR_PARAMS}, "metrics": metrics}
    (RUNS / f"{run_id}.json").write_text(json.dumps(run, indent=2))

    for name, m in metrics.items():
        print(f"{name:30s} PR-AUC {m['pr_auc']:.3f}  "
              + "  ".join(f"P@{k} {m[f'precision_at_{k}']:.2f} R@{k} {m[f'recall_at_{k}']:.3f}"
                          for k in (100, 500, 1000)))
    print(f"run log -> {RUNS / f'{run_id}.json'}")


if __name__ == "__main__":
    main()
