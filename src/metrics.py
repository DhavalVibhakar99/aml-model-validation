"""
metrics.py - the only metrics this project reports. No accuracy: at a 0.3%
base rate, "flag nobody" scores 99.7% and catches zero launderers.
"""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

KS = (100, 500, 1000)


def rank(scores: np.ndarray) -> np.ndarray:
    # stable sort so ties break the same way on every run (rules produce ties)
    return np.argsort(-np.asarray(scores), kind="stable")


def at_k(y: np.ndarray, scores: np.ndarray, k: int, n_pos_total: int) -> dict:
    """Precision and recall if investigators can only review the top-k accounts.

    n_pos_total includes positives with no history in the feature window. We
    can't score them, but they're still launderers we missed, so they stay in
    the recall denominator (DECISIONS #5).
    """
    tp = int(np.asarray(y)[rank(scores)[:k]].sum())
    return {"k": k, "tp": tp, "precision": tp / k, "recall": tp / n_pos_total}


def summary(y, scores, n_pos_total: int, probabilistic: bool = True) -> dict:
    y, scores = np.asarray(y), np.asarray(scores)
    out = {
        "pr_auc": float(average_precision_score(y, scores)),
        "base_rate": float(y.mean()),
        **{f"{m}_at_{r['k']}": r[m] for r in (at_k(y, scores, k, n_pos_total) for k in KS)
           for m in ("precision", "recall")},
    }
    # a rule count isn't a probability, so a Brier score for it would be nonsense
    if probabilistic:
        out["brier"] = float(brier_score_loss(y, scores))
    return out


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index, bins = deciles of the *expected* sample.

    Rule of thumb from credit risk: < 0.1 stable, 0.1-0.25 keep an eye on it,
    > 0.25 the population has moved. Features with a big spike (e.g. lots of
    zeros) collapse into fewer bins, which is the honest thing to do - fake
    bins inside a spike would just split identical values at random.
    """
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:        # constant feature: nothing to compare
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    # empty bins would make log() blow up; a tiny floor keeps it finite
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def bootstrap_ci(y, scores, n_pos_total: int, fn, n: int = 200, seed: int = 0):
    """95% interval by resampling accounts. With ~600 catchable positives,
    a recall@500 of 0.30 can easily wobble by a few points, and a model risk
    reviewer would ask how much."""
    rng = np.random.default_rng(seed)
    y, scores = np.asarray(y), np.asarray(scores)
    # scale the no-history positives with each resample so recall stays comparable
    unseen = n_pos_total - y.sum()
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        stats.append(fn(y[idx], scores[idx], y[idx].sum() + unseen))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
