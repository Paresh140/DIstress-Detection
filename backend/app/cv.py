"""Leave-speakers-out cross-validation on all 24 RAVDESS actors.

Six folds of four actors. In each fold, four actors are the test set, the
next four actors choose the checkpoint (or the regularization strength), and
the other sixteen train. Every actor is a test speaker exactly once, and the
folds are reported as mean ± standard deviation.

Two models run on the same folds:
  cnn      the DistressNet audio branch (log-mel CNN)
  egemaps  eGeMAPS v02 functionals (88 features, openSMILE) + logistic regression

Usage, from backend/:
  python -m app.cv                 # both models
  python -m app.cv --only cnn
  python -m app.cv --only egemaps
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .audio import file_to_logmel
from .dataset import ALL_ACTORS, discover
from .train import EPOCHS, SEED, _binary_metrics, always_distress_f1, auroc, fit_audio, score_audio

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "ravdess"
ARTIFACTS = ROOT / "backend" / "artifacts"
FOLD_SIZE = 4
METRICS = ("balanced_accuracy", "macro_f1", "auroc", "f1", "specificity", "recall", "always_distress_f1")
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def folds(actors: list[int] = sorted(ALL_ACTORS)) -> list[dict[str, set[int]]]:
    """Consecutive blocks of four actors, so each block is two male and two female."""
    blocks = [set(actors[i : i + FOLD_SIZE]) for i in range(0, len(actors), FOLD_SIZE)]
    out = []
    for k, test in enumerate(blocks):
        val = blocks[(k + 1) % len(blocks)]
        train = set(actors) - test - val
        out.append({"train": train, "val": val, "test": test})
    return out


def _scored(truth: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(np.int64)
    metrics = _binary_metrics(truth.astype(np.int64), pred)
    metrics["auroc"] = auroc(truth, prob)
    metrics["always_distress_f1"] = always_distress_f1(truth)
    return metrics


def _summary(fold_rows: list[dict]) -> dict:
    out = {}
    for key in METRICS:
        values = [row[key] for row in fold_rows if row.get(key) is not None]
        if values:
            out[key] = {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)}
    return out


def _public(metrics: dict) -> dict:
    keep = (*METRICS, "accuracy", "precision", "tp", "tn", "fp", "fn")
    return {key: (round(value, 4) if isinstance(value, float) else value) for key, value in metrics.items() if key in keep}


def run_cnn(clips: list, plan: list[dict], epochs: int) -> dict:
    cache: dict[str, torch.Tensor] = {}

    def rows(actors: set[int]) -> list[dict]:
        out = []
        for clip in clips:
            if clip.actor in actors:
                key = str(clip.path)
                if key not in cache:
                    cache[key] = file_to_logmel(key)
                out.append({"mel": cache[key], "label": clip.label, "clip": clip})
        return out

    per_fold = []
    for k, split in enumerate(plan, start=1):
        print(f"\ncnn fold {k}: test actors {sorted(split['test'])}  val {sorted(split['val'])}")
        test_rows = rows(split["test"])
        fitted = fit_audio(rows(split["train"]), rows(split["val"]), epochs=epochs)
        scored = score_audio(fitted["state"], test_rows)
        scored["always_distress_f1"] = always_distress_f1(np.array([row["label"] for row in test_rows]))
        per_fold.append(
            {
                "fold": k,
                "test_actors": sorted(split["test"]),
                "val_actors": sorted(split["val"]),
                "selected_epoch": fitted["epoch"],
                "val_balanced_accuracy": round(fitted["val_balanced_accuracy"], 4),
                "test_clips": len(test_rows),
                **_public(scored),
            }
        )
    return {"model": "DistressNet audio branch (log-mel CNN)", "epochs": epochs, "folds": per_fold, "summary": _summary(per_fold)}


def _egemaps(paths: list[str], cache_file: Path) -> np.ndarray:
    import opensmile

    cached: dict[str, list[float]] = json.loads(cache_file.read_text()) if cache_file.is_file() else {}
    missing = [path for path in paths if path not in cached]
    if missing:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        for i, path in enumerate(missing, start=1):
            cached[path] = smile.process_file(path).iloc[0].astype(float).tolist()
            if i % 200 == 0:
                print(f"  eGeMAPS {i}/{len(missing)}")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cached))
    return np.array([cached[path] for path in paths], dtype=np.float64)


def run_egemaps(clips: list, plan: list[dict]) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    paths = [str(clip.path) for clip in clips]
    features = _egemaps(paths, ARTIFACTS / "egemaps_cache.json")
    labels = np.array([clip.label for clip in clips])
    actors = np.array([clip.actor for clip in clips])

    def model(c: float):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=5000),
        )

    per_fold = []
    for k, split in enumerate(plan, start=1):
        train = np.isin(actors, list(split["train"]))
        val = np.isin(actors, list(split["val"]))
        test = np.isin(actors, list(split["test"]))
        best_c, best_score = C_GRID[0], -1.0
        for c in C_GRID:
            fitted = model(c).fit(features[train], labels[train])
            score = _scored(labels[val], fitted.predict_proba(features[val])[:, 1])["balanced_accuracy"]
            if score > best_score:
                best_c, best_score = c, score
        fitted = model(best_c).fit(features[train], labels[train])
        scored = _scored(labels[test], fitted.predict_proba(features[test])[:, 1])
        print(f"egemaps fold {k}: C {best_c}  val bal {best_score:.3f}  test bal {scored['balanced_accuracy']:.3f}")
        per_fold.append(
            {
                "fold": k,
                "test_actors": sorted(split["test"]),
                "val_actors": sorted(split["val"]),
                "selected_c": best_c,
                "val_balanced_accuracy": round(best_score, 4),
                "test_clips": int(test.sum()),
                **_public(scored),
            }
        )
    return {
        "model": "eGeMAPS v02 functionals + logistic regression (balanced class weights)",
        "folds": per_fold,
        "summary": _summary(per_fold),
    }


def run(only: str | None = None, epochs: int = EPOCHS) -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    clips = discover(DATA_ROOT)
    found = sorted({clip.actor for clip in clips})
    if found != sorted(ALL_ACTORS):
        raise SystemExit(f"Cross-validation needs RAVDESS actors 01–24 under {DATA_ROOT}; found {found}")
    plan = folds()
    report = {
        "dataset": "RAVDESS speech audio, actors 01–24",
        "seed": SEED,
        "label": {
            "distress": ["sad", "angry", "fearful"],
            "non_distress": ["neutral", "calm"],
            "excluded": ["happy", "surprised", "disgust", "song"],
        },
        "protocol": (
            "Six folds of four actors. Each fold tests on four actors, chooses the epoch or C on the next four "
            "by balanced accuracy, and trains on the remaining sixteen. Threshold 0.5. "
            "Values are mean and standard deviation over the six test folds."
        ),
        "folds": [{key: sorted(value) for key, value in split.items()} for split in plan],
        "models": {},
    }
    if only in (None, "cnn"):
        report["models"]["cnn"] = run_cnn(clips, plan, epochs)
    if only in (None, "egemaps"):
        report["models"]["egemaps"] = run_egemaps(clips, plan)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / "cv_report.json"
    if only is not None and out.is_file():
        previous = json.loads(out.read_text())
        report["models"] = {**previous.get("models", {}), **report["models"]}
    out.write_text(json.dumps(report, indent=2))

    print("\nmodel     " + "  ".join(f"{key:>20}" for key in METRICS))
    for name, result in report["models"].items():
        cells = []
        for key in METRICS:
            stat = result["summary"].get(key)
            cells.append(f"{stat['mean']:.3f} ± {stat['std']:.3f}".rjust(20) if stat else "-".rjust(20))
        print(f"{name:<9} " + "  ".join(cells))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=("cnn", "egemaps"))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    run(args.only, args.epochs)
