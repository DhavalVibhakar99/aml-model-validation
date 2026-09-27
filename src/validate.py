"""
validate.py - every number and figure in the validation report, from scratch.

Reads artifacts/scores.parquet (from train.py) plus the split tables, writes:
    reports/figures/*.png
    reports/metrics.json              every number, machine-readable
    reports/report_values.json        the formatted values the report uses
    reports/validation_report.md      rendered from reports/templates/

The report is a template on purpose: if a number in the prose could drift from
the code that produced it, eventually it will. Here it can't - rerun this
script and the report is rebuilt.

Usage: python src/validate.py
"""
import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from features import FEATURES  # noqa: E402
from metrics import KS, at_k, bootstrap_ci, psi, summary  # noqa: E402
from split import DATA_END, PROCESSED, SEED_END, WINDOWS  # noqa: E402
from train import ARTIFACTS, RULES, NO_FORMAT, load, n_positives, rule_flags  # noqa: E402

REPORTS = Path("reports")
FIGS = REPORTS / "figures"
TEMPLATE = REPORTS / "templates" / "validation_report.md"

# fixed colour per model, never per rank (first three slots of the reference
# palette - the only three that stay distinguishable for colour-blind readers
# when every pair can be on screen at once)
COLOR = {"lightgbm": "#2a78d6", "rules": "#eb6834", "logreg": "#1baf7a"}
NAME = {"lightgbm": "LightGBM", "rules": "Rules baseline (tuned)", "logreg": "Logistic regression",
        "rules_untuned": "Rules baseline (hand-set)"}
TAG = {"lightgbm": "gbm", "logreg": "lr", "rules": "rules", "rules_untuned": "rulesu"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4df", "#fcfcfb"
MODELS = ["rules", "logreg", "lightgbm"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
        "lines.linewidth": 2, "legend.frameon": False, "figure.dpi": 150,
    })


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def get(scores, model, dataset="hi_small", window="test"):
    g = scores[(scores.model == model) & (scores.dataset == dataset) & (scores.window == window)]
    return g.label.to_numpy(), g.score.to_numpy()


# ---------------------------------------------------------------- data

def data_section(v):
    rows = []
    for key, label in (("hi_small", "HI-Small"), ("li_small", "LI-Small")):
        s = json.loads((PROCESSED / f"{key}_split_summary.json").read_text())
        for w in WINDOWS:
            d, f = s[w]["label"], s[w]["label_next"]
            rows.append({"Dataset": label, "Window": w,
                         "Dates": f"Sept {WINDOWS[w]['start'].day}-{WINDOWS[w]['cutoff'].day - 1}",
                         "Accounts scored": f"{s[w]['accounts']:,}",
                         "Positives (detection)": f"{d['positives']:,}",
                         "Prevalence": pct(d["prevalence"], 2),
                         "Positives (forecast)": f"{f['positives'] + f['positives_no_history']:,}",
                         "of which no history": f"{f['positives_no_history']:,}"})
            v[f"{key}_{w}_prev"] = pct(d["prevalence"], 2)
            v[f"{key}_{w}_pos"] = f"{d['positives']:,}"
            v[f"{key}_{w}_accts"] = f"{s[w]['accounts']:,}"
            v[f"{key}_{w}_next_pos"] = f"{f['positives'] + f['positives_no_history']:,}"
            v[f"{key}_{w}_next_unseen"] = f"{f['positives_no_history']:,}"
    v["table_data"] = md_table(pd.DataFrame(rows))
    v["n_features"] = str(len(FEATURES))
    data_quirks(v)


def data_quirks(v):
    """The numbers behind the two trimming decisions (DECISIONS #1, #2), HI-Small."""
    con = duckdb.connect()
    con.execute(f"CREATE VIEW t AS SELECT * FROM '{PROCESSED / 'hi_small_transactions.parquet'}'")
    end = {"end": DATA_END}
    tail_share = con.execute("SELECT avg(is_laundering) FROM t WHERE ts >= $end", end).fetchone()[0]
    daily = con.execute("""SELECT avg(is_laundering) FROM t WHERE ts < $end GROUP BY ts::DATE""",
                        end).df().iloc[:, 0]
    # tail laundering that touches an account already laundering before the tail:
    # the "generator finishing what it started" evidence
    persist = con.execute("""
        WITH early AS (SELECT src_acct AS a FROM t WHERE is_laundering = 1 AND ts < $end
                       UNION SELECT dst_acct FROM t WHERE is_laundering = 1 AND ts < $end)
        SELECT avg((src_acct IN (SELECT a FROM early) OR dst_acct IN (SELECT a FROM early))::INT)
        FROM t WHERE is_laundering = 1 AND ts >= $end
    """, end).fetchone()[0]
    seed_self, other_self = con.execute("""
        SELECT avg((src_acct = dst_acct)::INT) FILTER (WHERE ts < $seed),
               avg((src_acct = dst_acct)::INT) FILTER (WHERE ts >= $seed AND ts < $end)
        FROM t
    """, {**end, "seed": SEED_END}).fetchone()
    v.update(tail_launder_share=pct(tail_share, 0), pre_daily_min=pct(daily.min(), 2),
             pre_daily_max=pct(daily.max(), 2), tail_persist=pct(persist, 0),
             seed_self_share=pct(seed_self, 0), other_self_share=pct(other_self, 1))


# ---------------------------------------------------------------- outcomes

def outcomes_section(scores, v, m):
    n_pos = n_positives("hi_small", "test", "label")
    # lift is measured against the *tuned* rules - the strongest version of the
    # baseline we could build from the train window (DECISIONS #9)
    rules = {k: at_k(*get(scores, "rules"), k, n_pos)["recall"] for k in KS}
    untuned = {k: at_k(*get(scores, "rules_untuned"), k, n_pos)["recall"] for k in KS}
    rows = []
    for model in ["rules_untuned", *MODELS]:
        y, s = get(scores, model)
        r = summary(y, s, n_pos, probabilistic=not model.startswith("rules"))
        lo, hi = bootstrap_ci(y, s, n_pos, lambda a, b, n: average_precision_score(a, b))
        rlo, rhi = bootstrap_ci(y, s, n_pos, lambda a, b, n: at_k(a, b, 500, n)["recall"])
        r.update(pr_auc_ci=[lo, hi], recall_at_500_ci=[rlo, rhi])
        m[f"hi_test/{model}"] = r
        row = {"Model": NAME[model], "PR-AUC [95% CI]": f"{r['pr_auc']:.3f} [{lo:.3f}-{hi:.3f}]"}
        for k in KS:
            row[f"P@{k}"] = pct(r[f"precision_at_{k}"], 0)
            row[f"R@{k}"] = pct(r[f"recall_at_{k}"])
        row["Lift vs tuned rules (R@500)"] = ("-" if model.startswith("rules")
                                        else f"{r['recall_at_500'] / rules[500]:.1f}x")
        rows.append(row)
        tag = TAG[model]
        v[f"{tag}_prauc"] = f"{r['pr_auc']:.3f}"
        v[f"{tag}_prauc_ci"] = f"{lo:.3f}-{hi:.3f}"
        for k in KS:
            v[f"{tag}_p{k}"] = pct(r[f"precision_at_{k}"], 0)
            v[f"{tag}_r{k}"] = pct(r[f"recall_at_{k}"])
            v[f"{tag}_tp{k}"] = f"{at_k(y, s, k, n_pos)['tp']:,}"
        v[f"{tag}_r500_ci"] = f"{pct(rlo)}-{pct(rhi)}"
    v["table_outcomes"] = md_table(pd.DataFrame(rows))
    for k in KS:
        v[f"lift_r{k}"] = f"{m['hi_test/lightgbm'][f'recall_at_{k}'] / rules[k]:.0f}"
        v[f"lift_untuned_r{k}"] = f"{m['hi_test/lightgbm'][f'recall_at_{k}'] / untuned[k]:.0f}"
        v[f"max_r{k}"] = pct(min(k, n_pos) / n_pos)
    v["hi_test_npos"] = f"{n_pos:,}"
    v["base_rate_test"] = pct(n_pos / len(get(scores, "rules")[0]), 2)

    # overfitting check: in-sample vs out-of-time
    for model in ("logreg", "lightgbm"):
        y, s = get(scores, model, window="train")
        tag = {"lightgbm": "gbm", "logreg": "lr"}[model]
        v[f"{tag}_prauc_train"] = f"{average_precision_score(y, s):.3f}"

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for model in MODELS:
        y, s = get(scores, model)
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=COLOR[model], label=f"{NAME[model]} (AP {average_precision_score(y, s):.3f})")
    ax.axhline(n_pos / len(y), color=INK2, lw=1, ls="--")
    ax.text(0.5, n_pos / len(y) + 0.02, "base rate", color=INK2, ha="center", fontsize=8)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
           title="Precision-recall, HI-Small test (Sept 5-8)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ks = np.arange(1, 3001)
    for model in MODELS:
        y, s = get(scores, model)
        caught = np.cumsum(y[np.argsort(-s, kind="stable")])[: len(ks)]
        ax.plot(ks, caught / n_pos, color=COLOR[model])
        ax.text(3050, caught[-1] / n_pos, NAME[model], color=INK, fontsize=8, va="center")
    ax.plot(ks, np.minimum(ks, n_pos) / n_pos, color=INK2, lw=1, ls="--")
    ax.text(1150, 1500 / n_pos, "perfect ranking", color=INK2, fontsize=8, ha="right")
    for k in KS:
        ax.axvline(k, color=GRID, lw=1, zorder=0)
    ax.set(xlabel="Alert budget K (accounts reviewed)", ylabel="Recall",
           xlim=(0, 3000), ylim=(0, 1), title="Recall at alert budget")
    fig.tight_layout()
    fig.subplots_adjust(right=0.86)
    fig.savefig(FIGS / "outcomes.png")
    plt.close(fig)


# ---------------------------------------------------------------- seen vs unseen

def segment_section(scores, v, m):
    """Split the test population by whether the account was active in the train
    window. The models see no account ids, so they can't memorise anyone - but
    'seen' accounts might still be easier (familiar behaviour, repeat
    launderers), and a bank's new-customer book is exactly where you'd worry.
    The ranking and the 500-alert budget stay global; we just ask where the
    alerts and the misses land."""
    tr, te = load("hi_small", "train"), load("hi_small", "test")
    seen = te.acct.isin(set(tr.acct)).to_numpy()
    repeat = te.acct.isin(set(tr.acct[tr.label == 1])).to_numpy()
    rows = []
    for model in MODELS:
        g = scores[(scores.model == model) & (scores.dataset == "hi_small") & (scores.window == "test")]
        # scores.parquet rows are in the same account order as the split table
        assert (g.acct.to_numpy() == te.acct.to_numpy()).all()
        y, s = g.label.to_numpy(), g.score.to_numpy()
        top = np.zeros(len(y), bool)
        top[np.argsort(-s, kind="stable")[:500]] = True
        for name, mask in (("seen in train", seen), ("new in test", ~seen)):
            n_pos = int(y[mask].sum())
            r = {"accounts": int(mask.sum()), "positives": n_pos, "prevalence": float(y[mask].mean()),
                 "pr_auc": float(average_precision_score(y[mask], s[mask])),
                 "alerts_in_top500": int((top & mask).sum()),
                 "precision_in_top500": float(y[top & mask].mean()) if (top & mask).any() else 0.0,
                 "recall_at_500": float(y[top & mask].sum() / n_pos)}
            m[f"segment/{model}/{name}"] = r
            rows.append({"Model": NAME[model], "Segment": name, "Accounts": f"{r['accounts']:,}",
                         "Positives": f"{n_pos:,} ({pct(r['prevalence'], 2)})",
                         "PR-AUC": f"{r['pr_auc']:.3f}",
                         "Alerts in top 500": f"{r['alerts_in_top500']:,}",
                         "Precision of those": pct(r["precision_in_top500"], 0),
                         "Recall@500": pct(r["recall_at_500"])})
            tag = f"seg_{'gbm' if model == 'lightgbm' else model}_{'seen' if mask is seen else 'new'}"
            v[f"{tag}_prauc"] = f"{r['pr_auc']:.3f}"
            v[f"{tag}_r500"] = pct(r["recall_at_500"])
            v[f"{tag}_alerts"] = f"{r['alerts_in_top500']:,}"
            v[f"{tag}_prec"] = pct(r["precision_in_top500"], 0)
    v["table_segments"] = md_table(pd.DataFrame(rows))
    y = te.label.to_numpy()
    v.update(seg_seen_accts=f"{seen.sum():,}", seg_new_accts=f"{(~seen).sum():,}",
             seg_seen_share=pct(seen.mean(), 0),
             seg_seen_pos=f"{int(y[seen].sum()):,}", seg_new_pos=f"{int(y[~seen].sum()):,}",
             seg_seen_prev=pct(y[seen].mean(), 2), seg_new_prev=pct(y[~seen].mean(), 2),
             seg_repeat=f"{int((repeat & (y == 1)).sum()):,}",
             seg_repeat_share=pct((repeat & (y == 1)).sum() / y.sum()))


# ---------------------------------------------------------------- calibration

def calibration_section(scores, v, m):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([1e-4, 1], [1e-4, 1], color=INK2, lw=1, ls="--")
    ax.text(0.3, 0.12, "perfect calibration", color=INK2, fontsize=8, ha="right")
    for model in ("logreg", "lightgbm"):
        y, s = get(scores, model)
        # quantile bins: nearly every account scores near 0, so equal-width bins
        # would put 99% of accounts in the first bin and tell us nothing
        frac, mean_pred = calibration_curve(y, s, n_bins=10, strategy="quantile")
        keep = frac > 0
        ax.plot(mean_pred[keep], frac[keep], marker="o", ms=4, color=COLOR[model], label=NAME[model])
        tag = {"lightgbm": "gbm", "logreg": "lr"}[model]
        v[f"{tag}_brier"] = f"{brier_score_loss(y, s):.5f}"
        v[f"{tag}_mean_pred"] = pct(s.mean(), 2)
        m[f"calibration/{model}"] = {"brier": brier_score_loss(y, s), "mean_pred": float(s.mean())}
    y = get(scores, "lightgbm")[0]
    # the "no skill" reference: predict the test base rate for everyone
    v["brier_ref"] = f"{brier_score_loss(y, np.full(len(y), y.mean())):.5f}"
    # log axes: almost all the action is below 1%, which a linear plot squashes
    # into the corner. Points above the diagonal = the model under-predicts.
    ax.set(xscale="log", yscale="log", xlabel="Mean predicted probability (decile)",
           ylabel="Observed laundering rate", title="Reliability, HI-Small test",
           xlim=(1e-4, 1), ylim=(1e-4, 1))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "calibration.png")
    plt.close(fig)


# ---------------------------------------------------------------- stability

def stability_section(scores, v, m):
    tr, te = load("hi_small", "train"), load("hi_small", "test")
    booster = lgb.Booster(model_file=str(ARTIFACTS / "lightgbm.txt"))
    gain = pd.Series(booster.feature_importance("gain"), booster.feature_name())
    top = gain.sort_values(ascending=False).index[:8].tolist()

    rows = []
    for model in ("logreg", "lightgbm"):
        val = psi(get(scores, model, window="train")[1], get(scores, model, window="test")[1])
        rows.append({"Variable": f"{NAME[model]} score", "PSI": f"{val:.3f}", "Status": psi_band(val)})
        v[f"{'gbm' if model == 'lightgbm' else 'lr'}_psi"] = f"{val:.3f}"
        m[f"psi/score/{model}"] = val
    feats = {}
    for f in top:
        val = psi(tr[f], te[f])
        feats[f] = val
        rows.append({"Variable": f"`{f}`", "PSI": f"{val:.3f}", "Status": psi_band(val)})
        m[f"psi/feature/{f}"] = val
    v["table_psi"] = md_table(pd.DataFrame(rows))
    worst = max(feats, key=feats.get)
    v["psi_worst_feat"], v["psi_worst_val"] = worst, f"{feats[worst]:.3f}"
    v["n_psi_red"] = str(sum(val > 0.25 for val in feats.values()))
    v["psi_top_n"] = str(len(top))
    return gain


def psi_band(x):
    return "stable" if x < 0.1 else ("monitor" if x < 0.25 else "shifted")


# ---------------------------------------------------------------- conceptual soundness

def soundness_section(scores, gain, v, m):
    share = (gain / gain.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(share.index, share.values, color=COLOR["lightgbm"], height=0.6)
    for i, val in enumerate(share.values):
        ax.text(val + 0.003, i, pct(val), va="center", fontsize=7, color=INK2)
    ax.set(xlabel="Share of total split gain", title="LightGBM feature importance (gain)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGS / "feature_importance.png")
    plt.close(fig)
    top3 = share.sort_values(ascending=False)
    v["imp_1"], v["imp_1_share"] = top3.index[0], pct(top3.iloc[0])
    v["imp_2"], v["imp_2_share"] = top3.index[1], pct(top3.iloc[1])
    v["imp_3"], v["imp_3_share"] = top3.index[2], pct(top3.iloc[2])
    v["imp_fmt_share"] = pct(share[[f for f in FEATURES if f.startswith("fmt_")]].sum())
    m["importance_gain_share"] = share.sort_values(ascending=False).to_dict()

    coefs = json.loads((ARTIFACTS / "logreg_coefs.json").read_text())["coefs"]
    c = pd.Series(coefs).sort_values(key=abs, ascending=False).head(8)
    v["table_lr"] = md_table(pd.DataFrame({
        "Feature (standardised)": [f"`{k}`" for k in c.index],
        "Coefficient": [f"{x:+.2f}" for x in c.values],
        "Odds ratio per 1 sd": [f"{np.exp(x):.2f}" for x in c.values]}))

    # ablation: the "is this just the simulator's ACH habit?" question
    n_pos = n_positives("hi_small", "test", "label")
    y, s = get(scores, "lightgbm_no_format")
    r = summary(y, s, n_pos)
    m["hi_test/lightgbm_no_format"] = r
    v["nofmt_prauc"] = f"{r['pr_auc']:.3f}"
    v["nofmt_p500"], v["nofmt_r1000"] = pct(r["precision_at_500"], 0), pct(r["recall_at_1000"])
    v["nofmt_drop"] = pct(1 - r["pr_auc"] / m["hi_test/lightgbm"]["pr_auc"], 0)
    v["nofmt_lift"] = f"{r['recall_at_500'] / m['hi_test/rules']['recall_at_500']:.0f}"
    v["n_nofmt_features"] = str(len(NO_FORMAT))

    # the rules one at a time, hand-set vs tuned: how noisy is each, and what
    # did tuning do to it?
    te = load("hi_small", "test")
    th = json.loads((ARTIFACTS / "rule_thresholds.json").read_text())
    hand, tuned = rule_flags(te, th["rules_untuned"]), rule_flags(te, th["rules"])

    def fmt_th(p):
        return "off" if p is None else ", ".join(f"{x:,}" if isinstance(x, int) else f"{x:g}" for x in p)

    def cell(flag):
        hit = flag == 1
        return f"{hit.sum():,}", pct(te.label[hit].mean()) if hit.any() else "-"

    rows = []
    for rule in RULES:
        (ha, hp), (ta, tp) = cell(hand[rule]), cell(tuned[rule])
        rows.append({"Rule": rule, "Hand-set threshold": fmt_th(th["rules_untuned"][rule]),
                     "Alerts": ha, "Precision": hp,
                     "Tuned threshold": fmt_th(th["rules"][rule]), "Alerts ": ta, "Precision ": tp})
    (ha, hp), (ta, tp) = cell((hand.sum(axis=1) > 0).astype(int)), cell((tuned.sum(axis=1) > 0).astype(int))
    rows.append({"Rule": "**any rule**", "Hand-set threshold": "", "Alerts": ha, "Precision": hp,
                 "Tuned threshold": "", "Alerts ": ta, "Precision ": tp})
    v["table_rules"] = md_table(pd.DataFrame(rows))
    anyhit = hand.sum(axis=1) > 0
    v["rules_any_alerts"] = f"{anyhit.sum():,}"
    v["rules_any_prec"] = pct(te.label[anyhit].mean())
    v["rules_any_recall"] = pct(te.label[anyhit].sum() / n_pos)
    v["rules_n_on"] = str(sum(p is not None for p in th["rules"].values()))
    v["rules_tuned_desc"] = " and ".join(
        f"`{r}` at {fmt_th(p)}" for r, p in th["rules"].items() if p is not None)
    run = json.loads(sorted((REPORTS / "runs").glob("*.json"))[-1].read_text())
    hist = run["rule_tuning"]["history"]
    v["rules_train_obj_start"], v["rules_train_obj_end"] = f"{hist[0][1]:.3f}", f"{hist[-1][1]:.3f}"


# ---------------------------------------------------------------- stress test

def prior_shift(p, pi_from, pi_to):
    """Rescale probabilities for a new base rate (Saerens et al., 2002).
    Bayes: odds_new = odds_old * (prior odds ratio). Assumes the behaviour
    given the label is unchanged and only the mix changed."""
    r = (pi_to / (1 - pi_to)) / (pi_from / (1 - pi_from))
    return r * p / (r * p + (1 - p))


def stress_section(scores, v, m):
    n_pos = n_positives("li_small", "test", "label")
    rows = []
    for model, label in (("lightgbm", "LightGBM trained on HI"),
                         ("lightgbm_li_native", "LightGBM trained on LI (reference)"),
                         ("logreg", "Logistic regression trained on HI"),
                         ("rules", "Rules baseline (tuned on HI)")):
        y, s = get(scores, model, "li_small")
        r = summary(y, s, n_pos, probabilistic=model != "rules")
        m[f"li_test/{model}"] = r
        rows.append({"Model": label, "PR-AUC": f"{r['pr_auc']:.3f}",
                     **{f"P@{k}": pct(r[f"precision_at_{k}"], 0) for k in KS},
                     **{f"R@{k}": pct(r[f"recall_at_{k}"]) for k in KS}})
        tag = {"lightgbm": "st_gbm", "lightgbm_li_native": "st_native",
               "logreg": "st_lr", "rules": "st_rules"}[model]
        v[f"{tag}_prauc"] = f"{r['pr_auc']:.3f}"
        for k in KS:
            v[f"{tag}_p{k}"] = pct(r[f"precision_at_{k}"], 0)
            v[f"{tag}_r{k}"] = pct(r[f"recall_at_{k}"])
    v["table_stress"] = md_table(pd.DataFrame(rows))
    v["li_test_npos"] = f"{n_pos:,}"

    # calibration under the shift, and the textbook fix. Priors come from each
    # dataset's *train* window - the only rates you'd know at deployment time.
    s_hi = json.loads((PROCESSED / "hi_small_split_summary.json").read_text())
    s_li = json.loads((PROCESSED / "li_small_split_summary.json").read_text())
    pi_hi, pi_li = s_hi["train"]["label"]["prevalence"], s_li["train"]["label"]["prevalence"]
    y, s = get(scores, "lightgbm", "li_small")
    adj = prior_shift(s, pi_hi, pi_li)
    v.update(st_obs=pct(y.mean(), 2), st_mean_pred=pct(s.mean(), 2), st_mean_adj=pct(adj.mean(), 2),
             st_brier=f"{brier_score_loss(y, s):.5f}", st_brier_adj=f"{brier_score_loss(y, adj):.5f}",
             pi_hi=pct(pi_hi, 2), pi_li=pct(pi_li, 2), prior_ratio=f"{pi_hi / pi_li:.2f}")
    m["stress/prior_correction"] = {"pi_hi_train": pi_hi, "pi_li_train": pi_li,
                                    "brier_raw": brier_score_loss(y, s),
                                    "brier_adjusted": brier_score_loss(y, adj)}
    # ranking is untouched by a monotone rescale - worth proving, not asserting
    same = np.array_equal(np.argsort(-s, kind="stable")[:1000], np.argsort(-adj, kind="stable")[:1000])
    v["st_rank_unchanged"] = "identical" if same else "different"

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ks = np.arange(1, 3001)
    for model, color, label in (("lightgbm", COLOR["lightgbm"], "Trained on HI"),
                                ("lightgbm_li_native", COLOR["rules"], "Trained on LI")):
        yy, ss = get(scores, model, "li_small")
        caught = np.cumsum(yy[np.argsort(-ss, kind="stable")])[: len(ks)]
        ax.plot(ks, caught / ks, color=color, label=label)
    ax.axhline(y.mean(), color=INK2, lw=1, ls="--")
    ax.text(2950, y.mean() + 0.02, "LI base rate", color=INK2, ha="right", fontsize=8)
    ax.set(xlabel="Alert budget K", ylabel="Precision in top K", xlim=(0, 3000), ylim=(0, 1.02),
           title="Stress test: LightGBM on LI-Small test (Sept 5-8)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "stress_test.png")
    plt.close(fig)


# ---------------------------------------------------------------- forecast framing

def forecast_section(scores, v, m):
    n_pos = n_positives("hi_small", "test", "label_next")
    y, s = get(scores, "lightgbm_forecast")
    r = summary(y, s, n_pos)
    m["hi_test/lightgbm_forecast"] = r
    ytr, str_ = get(scores, "lightgbm_forecast", window="train")
    v.update(fc_prauc=f"{r['pr_auc']:.3f}", fc_prauc_train=f"{average_precision_score(ytr, str_):.3f}",
             fc_p500=pct(r["precision_at_500"], 0), fc_r1000=pct(r["recall_at_1000"]),
             fc_npos=f"{n_pos:,}")
    te = load("hi_small", "test")
    both = int(((te.label == 1) & (te.label_next == 1)).sum())
    v["fc_persist"] = f"{both:,}"
    v["fc_persist_share"] = pct(both / n_pos)
    s_hi = json.loads((PROCESSED / "hi_small_split_summary.json").read_text())
    unseen = s_hi["test"]["label_next"]["positives_no_history"]
    v["fc_unseen"], v["fc_unseen_share"] = f"{unseen:,}", pct(unseen / n_pos)


# ---------------------------------------------------------------- final holdout

def holdout_section(v, m):
    """Reads reports/holdout.json - written once by holdout.py, never recomputed
    here. Also checks the artifacts on disk are still the ones it scored with."""
    import hashlib
    h = json.loads((REPORTS / "holdout.json").read_text())
    now = {f: hashlib.sha256((ARTIFACTS / f).read_bytes()).hexdigest() for f in h["artifact_sha256"]}
    v["ho_artifacts_match"] = ("match" if now == h["artifact_sha256"]
                               else "**do NOT match** (models were retrained after the holdout)")
    pop, hm = h["population"], h["metrics"]
    test = {"rules": m["hi_test/rules"], "logreg": m["hi_test/logreg"], "lightgbm": m["hi_test/lightgbm"],
            "rules_untuned": m["hi_test/rules_untuned"]}
    rows = []
    for model in ("rules_untuned", "rules", "logreg", "lightgbm"):
        r, t = hm[model], test[model]
        rows.append({"Model": NAME[model],
                     "PR-AUC (test → holdout)": f"{t['pr_auc']:.3f} → {r['pr_auc']:.3f}",
                     "P@500": f"{pct(t['precision_at_500'], 0)} → {pct(r['precision_at_500'], 0)}",
                     "R@500": f"{pct(t['recall_at_500'])} → {pct(r['recall_at_500'])}",
                     "R@1000": f"{pct(t['recall_at_1000'])} → {pct(r['recall_at_1000'])}",
                     "Fresh PR-AUC": f"{r['fresh_pr_auc']:.3f}",
                     "Fresh R@500": pct(r["fresh_recall_at_500"]),
                     "Fresh R@1000": pct(r["fresh_recall_at_1000"])})
    v["table_holdout"] = md_table(pd.DataFrame(rows))
    g, ru = hm["lightgbm"], hm["rules"]
    v.update(ho_sha=h["git_sha"], ho_when=h["scored_at"], ho_accts=f"{pop['accounts']:,}",
             ho_pos=f"{pop['positives']:,}", ho_prev=pct(pop["prevalence"], 2),
             ho_fresh=f"{pop['fresh_positives']:,}", ho_stale=f"{pop['stale_accounts']:,}",
             ho_gbm_prauc=f"{g['pr_auc']:.3f}", ho_gbm_p500=pct(g["precision_at_500"], 0),
             ho_gbm_r500=pct(g["recall_at_500"]), ho_gbm_r1000=pct(g["recall_at_1000"]),
             ho_gbm_fresh_prauc=f"{g['fresh_pr_auc']:.3f}",
             ho_gbm_fresh_r500=pct(g["fresh_recall_at_500"]),
             ho_gbm_fresh_r1000=pct(g["fresh_recall_at_1000"]),
             ho_rules_p500=pct(ru["precision_at_500"], 0), ho_rules_r500=pct(ru["recall_at_500"]),
             ho_lift_r500=f"{g['recall_at_500'] / ru['recall_at_500']:.0f}",
             ho_fresh_lift_r500=f"{g['fresh_recall_at_500'] / ru['fresh_recall_at_500']:.0f}",
             ho_max_r500=pct(500 / pop["positives_total"]),
             ho_fresh_max_r500=pct(500 / pop["fresh_positives"]),
             ho_psi=f"{h['psi_lightgbm_score_vs_train']:.3f}")
    m["holdout"] = h


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    style()
    scores = pd.read_parquet(ARTIFACTS / "scores.parquet")
    v, m = {}, {}
    data_section(v)
    outcomes_section(scores, v, m)
    segment_section(scores, v, m)
    calibration_section(scores, v, m)
    gain = stability_section(scores, v, m)
    soundness_section(scores, gain, v, m)
    stress_section(scores, v, m)
    forecast_section(scores, v, m)
    holdout_section(v, m)
    runs = sorted((REPORTS / "runs").glob("*.json"))
    v["run_id"] = json.loads(runs[-1].read_text())["run_id"] if runs else "n/a"

    (REPORTS / "metrics.json").write_text(json.dumps(m, indent=2, default=float))
    # the exact strings that went into the report, so README numbers can be checked
    (REPORTS / "report_values.json").write_text(json.dumps(
        {k: val for k, val in v.items() if not k.startswith("table_")}, indent=2))
    report = TEMPLATE.read_text().format_map(v)
    (REPORTS / "validation_report.md").write_text(report)
    print(f"wrote {REPORTS / 'validation_report.md'} ({len(v)} values), figures in {FIGS}")


if __name__ == "__main__":
    main()
