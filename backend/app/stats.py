"""Confidence intervals and paired comparisons for the voice results."""

from __future__ import annotations

import numpy as np

from .train import _binary_metrics, auroc

BOOTSTRAP_ROUNDS = 1000
SEED = 7


def speaker_bootstrap(truth, prob, speakers, threshold: float, rounds: int = BOOTSTRAP_ROUNDS) -> dict:
    """95% intervals for balanced accuracy and AUROC, resampling whole test speakers.

    Clips from one speaker are not independent, so speakers (not clips) are
    drawn with replacement.
    """
    truth = np.asarray(truth).astype(np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    speakers = np.asarray(speakers)
    groups = [np.flatnonzero(speakers == s) for s in np.unique(speakers)]
    rng = np.random.default_rng(SEED)
    bal, area = [], []
    for _ in range(rounds):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        y, p = truth[idx], prob[idx]
        if y.min() == y.max():
            continue
        bal.append(_binary_metrics(y, (p >= threshold).astype(np.int64))["balanced_accuracy"])
        area.append(auroc(y, p))

    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        lo, hi = np.percentile(values, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    return {"balanced_accuracy_ci95": interval(bal), "auroc_ci95": interval(area), "rounds": len(bal), "unit": "speaker"}


def paired(a: list[float], b: list[float]) -> dict:
    """Paired t-test and Wilcoxon signed-rank test over the same folds (a minus b)."""
    from scipy import stats

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    diff = a - b
    out = {"folds": len(diff), "mean_difference": round(float(diff.mean()), 4)}
    if len(diff) < 2 or np.allclose(diff, 0):
        return {**out, "t_test_p": None, "wilcoxon_p": None}
    out["t_test_p"] = round(float(stats.ttest_rel(a, b).pvalue), 4)
    out["wilcoxon_p"] = round(float(stats.wilcoxon(a, b).pvalue), 4)
    return out
