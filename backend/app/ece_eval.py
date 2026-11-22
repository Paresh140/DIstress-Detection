"""ECE sensitivity to the bin count, and a reliability diagram (audit C13).

Expected calibration error depends on how P(distress) is binned. This
recomputes it at 5, 10 and 15 equal-width bins (and with equal-mass bins) from
saved per-clip probabilities:
  live head   artifacts/voice_report.json, RAVDESS actors 21-24 (calibrated)
  cv folds    artifacts/cv_report.json, binary-mean folds, when the folds carry
              test_probability (written by app.cv from this version on)

Usage, from backend/:
  python -m app.ece_eval

Writes artifacts/ece_report.json and artifacts/reliability.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import ece

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "backend" / "artifacts"
BINS = (5, 10, 15)


def ece_equal_mass(prob: np.ndarray, truth: np.ndarray, bins: int) -> float:
    order = np.argsort(prob)
    total = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            total += len(chunk) / len(prob) * abs(prob[chunk].mean() - truth[chunk].mean())
    return float(total)


def reliability(prob: np.ndarray, truth: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = which == b
        if mask.any():
            rows.append({"bin": [round(edges[b], 2), round(edges[b + 1], 2)], "clips": int(mask.sum()), "mean_probability": round(float(prob[mask].mean()), 4), "distress_rate": round(float(truth[mask].mean()), 4)})
    return rows


def block(prob: np.ndarray, truth: np.ndarray) -> dict:
    return {
        "clips": len(prob),
        "ece_equal_width": {str(b): round(ece(prob, truth, b), 4) for b in BINS},
        "ece_equal_mass": {str(b): round(ece_equal_mass(prob, truth, b), 4) for b in BINS},
        "reliability_10_bins": reliability(prob, truth),
    }


def sources() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    path = ARTIFACTS / "voice_report.json"
    if path.is_file():
        preds = json.loads(path.read_text())["predictions"]
        out["live head, RAVDESS 21-24"] = (np.array([p["probability"] for p in preds]), np.array([p["truth"] == "distress" for p in preds], dtype=float))
    path = ARTIFACTS / "cv_report.json"
    if path.is_file():
        model = json.loads(path.read_text()).get("models", {}).get("wavlm/binary-mean", {})
        folds = [f for f in model.get("folds", []) if "test_probability" in f]
        if folds:
            out["binary-mean, RAVDESS 6-fold CV (pooled)"] = (
                np.concatenate([f["test_probability"] for f in folds]),
                np.concatenate([f["test_labels"] for f in folds]).astype(float),
            )
    return out


def run() -> dict:
    found = sources()
    if not found:
        raise SystemExit("No saved probabilities: run app.voice or app.cv first")
    report = {name: block(*data) for name, data in found.items()}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "ece_report.json").write_text(json.dumps(report, indent=2))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=150)
        ax.plot([0, 1], [0, 1], color="#9aa0a6", lw=1, ls="--", label="perfect calibration")
        for (name, rows), color in zip(((n, r["reliability_10_bins"]) for n, r in report.items()), ("#1f6feb", "#d97706")):
            ax.plot([r["mean_probability"] for r in rows], [r["distress_rate"] for r in rows], marker="o", color=color, lw=1.5,
                    label=f"{name} (ECE10 {report[name]['ece_equal_width']['10']:.3f})")
        ax.set_xlabel("mean predicted P(distress) in bin")
        ax.set_ylabel("observed share distressed")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper left", frameon=False)
        ax.set_title("Reliability, 10 equal-width bins", fontsize=9)
        fig.tight_layout()
        fig.savefig(ARTIFACTS / "reliability.png")
    except ImportError:
        print("matplotlib not installed; no figure")
    for name, r in report.items():
        print(f"{name} ({r['clips']} clips)")
        print("  equal width " + "  ".join(f"{b} bins {v:.3f}" for b, v in r["ece_equal_width"].items()))
        print("  equal mass  " + "  ".join(f"{b} bins {v:.3f}" for b, v in r["ece_equal_mass"].items()))
    return report


if __name__ == "__main__":
    run()
