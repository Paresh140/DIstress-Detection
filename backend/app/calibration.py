"""Temperature scaling, decision threshold and calibration error.

Everything here is fitted on validation speakers only, then applied once to the
test speakers. A model hands over raw logits: one per clip for a binary model,
or one row per clip for the four-class head (neutral, sad, angry, fearful),
where P(distress) is everything except neutral.
"""

from __future__ import annotations

import numpy as np
import torch

from .train import _binary_metrics, always_distress_f1, auroc

ECE_BINS = 10
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)


def distress_probability(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = torch.as_tensor(np.asarray(logits, dtype=np.float64)) / temperature
    if z.ndim == 1:
        return torch.sigmoid(z).numpy()
    return (1.0 - torch.softmax(z, dim=1)[:, 0]).numpy()


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    """Temperature that minimizes validation NLL (Guo et al., 2017).

    Binary logits take 0/1 targets; four-class logits take class indices.
    """
    z = torch.as_tensor(np.asarray(logits, dtype=np.float64))
    y = torch.as_tensor(np.asarray(targets))
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def nll():
        opt.zero_grad()
        scaled = z / log_t.exp()
        if z.ndim == 1:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled, y.double())
        else:
            loss = torch.nn.functional.cross_entropy(scaled, y.long())
        loss.backward()
        return loss

    opt.step(nll)
    return float(np.clip(log_t.exp().item(), 0.05, 20.0))


def ece(prob: np.ndarray, truth: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error of P(distress), equal-width bins."""
    prob = np.asarray(prob, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            total += mask.mean() * abs(prob[mask].mean() - truth[mask].mean())
    return float(total)


def tune_threshold(prob: np.ndarray, truth: np.ndarray) -> float:
    """Threshold with the best validation balanced accuracy; ties go to the one nearest 0.5."""
    truth = np.asarray(truth).astype(np.int64)
    best, best_score = 0.5, -1.0
    for t in sorted(THRESHOLDS, key=lambda t: abs(t - 0.5)):
        score = _binary_metrics(truth, (prob >= t).astype(np.int64))["balanced_accuracy"]
        if score > best_score + 1e-9:
            best, best_score = float(t), score
    return best


def metrics_at(truth: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    truth = np.asarray(truth).astype(np.int64)
    out = _binary_metrics(truth, (prob >= threshold).astype(np.int64))
    out["auroc"] = auroc(truth, prob)
    out["always_distress_f1"] = always_distress_f1(truth)
    out["beats_always_distress"] = bool(out["f1"] > out["always_distress_f1"])
    return out


def evaluate(val_logits, val_targets, test_logits, test_targets) -> dict:
    """Score the test set twice: raw at 0.5, and calibrated with a validation-tuned threshold.

    Targets are 0/1 for binary logits and class indices (0 = non-distress) for
    four-class logits. The calibrated block is the proposal's protocol; the raw
    block keeps the numbers comparable with earlier runs.
    """
    val_y = (np.asarray(val_targets) > 0).astype(np.int64)
    test_y = (np.asarray(test_targets) > 0).astype(np.int64)
    temperature = fit_temperature(val_logits, val_targets)
    raw_test = distress_probability(test_logits)
    cal_val = distress_probability(val_logits, temperature)
    cal_test = distress_probability(test_logits, temperature)
    threshold = tune_threshold(cal_val, val_y)
    raw = metrics_at(test_y, raw_test, 0.5)
    raw["ece"] = ece(raw_test, test_y)
    calibrated = metrics_at(test_y, cal_test, threshold)
    calibrated["ece"] = ece(cal_test, test_y)
    return {
        "temperature": round(temperature, 4),
        "threshold": threshold,
        "raw": raw,
        "calibrated": calibrated,
        "test_probability": cal_test,
    }
