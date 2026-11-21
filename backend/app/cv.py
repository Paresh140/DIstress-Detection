"""Leave-speakers-out cross-validation within one corpus.

RAVDESS: six folds of four actors (all 24). In each fold, four actors are the
test set, the next four choose the checkpoint, the regularization strength, the
calibration temperature and the decision threshold, and the other sixteen
train. CREMA-D: five folds over its 91 speakers (shuffled once with the seed);
nine speakers from the next fold validate and the rest train. Every speaker is
a test speaker exactly once, and folds are reported as mean ± standard
deviation.

Each fold is scored twice: raw (threshold 0.5, no calibration, comparable with
the first runs) and calibrated (temperature scaling and a threshold chosen on
the validation speakers, the proposal's protocol). Expected calibration error
is reported for both. Models are compared on identical folds with a paired
t-test and a Wilcoxon signed-rank test.

Three models run on the same folds:
  cnn      the DistressNet audio branch (log-mel CNN)
  egemaps  eGeMAPS v02 functionals (88 features, openSMILE) + logistic regression
  wavlm    frozen WavLM-base-plus, learned layer weights, attentive statistics
           pooling, four-class head (app/speech_encoder.py)

Usage, from backend/:
  python -m app.cv                                  # RAVDESS, all three models
  python -m app.cv --corpus crema-d                 # CREMA-D, five folds
  python -m app.cv --only wavlm
  python -m app.cv --only wavlm --head binary-mean  # the first WavLM head, reported as wavlm/binary-mean
  python -m app.cv --only wavlm --encoder facebook/wav2vec2-base
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch

from . import calibration
from .audio import file_to_logmel
from .corpora import discover
from .speech_encoder import DEFAULT_ENCODER, DEFAULT_HEAD, HEAD_EPOCHS, HEADS, cache_path, embed_clips, fit_head, predict_logits, targets
from .stats import paired
from .train import EPOCHS, SEED, audio_logits, fit_audio

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "backend" / "artifacts"
CORPORA = {
    "ravdess": {"name": "RAVDESS", "root": ROOT / "data" / "ravdess", "folds": 6, "val": 4, "shuffle": False, "report": "cv_report.json"},
    "crema-d": {"name": "CREMA-D", "root": ROOT / "data" / "crema-d", "folds": 5, "val": 9, "shuffle": True, "report": "cv_report_crema-d.json"},
}
METRICS = ("balanced_accuracy", "macro_f1", "auroc", "f1", "specificity", "recall", "always_distress_f1", "ece")
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def folds(speakers: list[str], n_folds: int, n_val: int, shuffle: bool) -> list[dict[str, set[str]]]:
    """Contiguous blocks of speakers. Unshuffled RAVDESS blocks are two male and two female actors."""
    order = sorted(speakers)
    if shuffle:
        random.Random(SEED).shuffle(order)
    blocks = [set(block.tolist()) for block in np.array_split(np.array(order), n_folds)]
    out = []
    for k, test in enumerate(blocks):
        val = set(sorted(blocks[(k + 1) % len(blocks)])[:n_val])
        out.append({"train": set(order) - test - val, "val": val, "test": test})
    return out


def take(clips: list, speakers: set[str]) -> list:
    return [clip for clip in clips if clip.speaker in speakers]


def labels(clips: list) -> np.ndarray:
    return np.array([clip.label for clip in clips], dtype=np.int64)


# One split per call: each returns the choice made on validation speakers plus
# raw validation and test logits, which calibration.evaluate then scores.


def split_cnn(train: list, val: list, test: list, cache: dict[str, torch.Tensor], epochs: int) -> dict:
    def rows(clips: list) -> list[dict]:
        out = []
        for clip in clips:
            key = str(clip.path)
            if key not in cache:
                cache[key] = file_to_logmel(key)
            out.append({"mel": cache[key], "label": clip.label, "clip": clip})
        return out

    val_rows = rows(val)
    fitted = fit_audio(rows(train), val_rows, epochs=epochs)
    return {
        "selected": {"selected_epoch": fitted["epoch"], "val_balanced_accuracy": round(fitted["val_balanced_accuracy"], 4)},
        "val_logits": audio_logits(fitted["state"], val_rows),
        "val_targets": labels(val),
        "test_logits": audio_logits(fitted["state"], rows(test)),
    }


def egemaps_features(paths: list[str], cache_file: Path) -> np.ndarray:
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


_egemaps = egemaps_features  # older name, kept for scripts that import it


def _stack(clips: list, features: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([features[str(clip.path)] for clip in clips])


def split_egemaps(train: list, val: list, test: list, features: dict[str, np.ndarray]) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    xt, xv, xs = (_stack(c, features) for c in (train, val, test))
    yt, yv = labels(train), labels(val)

    def model(c: float):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))

    best_c, best_score = C_GRID[0], -1.0
    for c in C_GRID:
        score = calibration.metrics_at(yv, model(c).fit(xt, yt).predict_proba(xv)[:, 1], 0.5)["balanced_accuracy"]
        if score > best_score:
            best_c, best_score = c, score
    fitted = model(best_c).fit(xt, yt)
    return {
        "selected": {"selected_c": best_c, "val_balanced_accuracy": round(best_score, 4)},
        "val_logits": fitted.decision_function(xv),
        "val_targets": yv,
        "test_logits": fitted.decision_function(xs),
    }


def split_ssl(train: list, val: list, test: list, features: dict[str, np.ndarray], head: str = DEFAULT_HEAD, epochs: int = HEAD_EPOCHS) -> dict:
    fitted = fit_head(_stack(train, features), targets(train, head), _stack(val, features), targets(val, head), epochs=epochs, head=head)
    return {
        "selected": {
            "selected_epoch": fitted["epoch"],
            "val_balanced_accuracy": round(fitted["val_balanced_accuracy"], 4),
            "layer_weights": fitted["layer_weights"],
        },
        "val_logits": predict_logits(fitted, _stack(val, features)),
        "val_targets": targets(val, head),
        "test_logits": predict_logits(fitted, _stack(test, features)),
        "fitted": fitted,
    }


def score_split(out: dict, test: list) -> dict:
    """Calibrate on validation and score the test speakers once."""
    scored = calibration.evaluate(out["val_logits"], out["val_targets"], out["test_logits"], labels(test))
    scored["selected"] = out["selected"]
    return scored


def _round(metrics: dict) -> dict:
    keep = (*METRICS, "accuracy", "precision", "beats_always_distress", "tp", "tn", "fp", "fn")
    return {key: (round(value, 4) if isinstance(value, float) else value) for key, value in metrics.items() if key in keep and value is not None}


def _fold_row(k: int, split: dict, test: list, scored: dict) -> dict:
    return {
        "fold": k,
        "test_speakers": sorted(split["test"]),
        "val_speakers": sorted(split["val"]),
        **scored["selected"],
        "test_clips": len(test),
        **_round(scored["raw"]),
        "temperature": scored["temperature"],
        "threshold": scored["threshold"],
        "calibrated": _round(scored["calibrated"]),
    }


def _summary(rows: list[dict]) -> dict:
    out = {}
    for key in METRICS:
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            out[key] = {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)}
    out["beats_always_distress_folds"] = f"{sum(bool(row.get('beats_always_distress')) for row in rows)}/{len(rows)}"
    return out


def _result(model: str, fold_rows: list[dict], **extra) -> dict:
    return {
        "model": model,
        **extra,
        "folds": fold_rows,
        "summary": _summary(fold_rows),
        "summary_calibrated": _summary([row["calibrated"] for row in fold_rows]),
    }


def run_cnn(clips: list, plan: list[dict], epochs: int) -> dict:
    cache: dict[str, torch.Tensor] = {}
    rows = []
    for k, split in enumerate(plan, start=1):
        print(f"\ncnn fold {k}: test {sorted(split['test'])}  val {sorted(split['val'])}")
        test = take(clips, split["test"])
        scored = score_split(split_cnn(take(clips, split["train"]), take(clips, split["val"]), test, cache, epochs), test)
        rows.append(_fold_row(k, split, test, scored))
    return _result("DistressNet audio branch (log-mel CNN)", rows, epochs=epochs)


def run_egemaps(clips: list, plan: list[dict]) -> dict:
    paths = [str(clip.path) for clip in clips]
    features = dict(zip(paths, egemaps_features(paths, ARTIFACTS / "egemaps_cache.json")))
    rows = []
    for k, split in enumerate(plan, start=1):
        test = take(clips, split["test"])
        scored = score_split(split_egemaps(take(clips, split["train"]), take(clips, split["val"]), test, features), test)
        print(f"egemaps fold {k}: C {scored['selected']['selected_c']}  test bal {scored['raw']['balanced_accuracy']:.3f}  calibrated {scored['calibrated']['balanced_accuracy']:.3f}")
        rows.append(_fold_row(k, split, test, scored))
    return _result("eGeMAPS v02 functionals + logistic regression (balanced class weights)", rows)


def encoder_key(encoder: str) -> str:
    return "wavlm" if "wavlm" in encoder.lower() else "wav2vec2"


def model_key(encoder: str, head: str = DEFAULT_HEAD) -> str:
    return encoder_key(encoder) + ("" if head == DEFAULT_HEAD else f"/{head}")


def run_ssl(clips: list, plan: list[dict], encoder: str, head: str = DEFAULT_HEAD, epochs: int = HEAD_EPOCHS) -> dict:
    paths = [str(clip.path) for clip in clips]
    features = dict(zip(paths, embed_clips(paths, encoder, cache_path(ARTIFACTS, encoder))))
    name = model_key(encoder, head)
    rows = []
    for k, split in enumerate(plan, start=1):
        test = take(clips, split["test"])
        scored = score_split(split_ssl(take(clips, split["train"]), take(clips, split["val"]), test, features, head, epochs), test)
        top = int(np.argmax(scored["selected"]["layer_weights"]))
        print(
            f"{name} fold {k}: epoch {scored['selected']['selected_epoch']}  test bal {scored['raw']['balanced_accuracy']:.3f}  "
            f"calibrated {scored['calibrated']['balanced_accuracy']:.3f}  T {scored['temperature']:.2f}  heaviest layer {top}"
        )
        rows.append(_fold_row(k, split, test, scored))
    pool = HEADS[head]["pool"]
    classes = HEADS[head]["classes"]
    return _result(
        f"{encoder} (frozen), learned layer weights, {pool} pooling, {classes}-class head",
        rows,
        encoder=encoder,
        head=head,
        epochs=epochs,
    )


def paired_tests(models: dict) -> list[dict]:
    """Every pair of models on the same folds, on raw and calibrated balanced accuracy."""
    out = []
    for a, b in itertools.combinations(sorted(models), 2):
        fa, fb = models[a]["folds"], models[b]["folds"]
        if any("calibrated" not in f for f in fa + fb):
            continue  # a result from before calibration was added
        if [f["test_speakers"] for f in fa] != [f["test_speakers"] for f in fb]:
            continue
        for block in ("raw", "calibrated"):
            pick = (lambda f: f["balanced_accuracy"]) if block == "raw" else (lambda f: f["calibrated"]["balanced_accuracy"])
            out.append({"a": a, "b": b, "scores": block, "metric": "balanced_accuracy", **paired([pick(f) for f in fa], [pick(f) for f in fb])})
    return out


def run(
    only: str | None = None,
    epochs: int = EPOCHS,
    encoder: str = DEFAULT_ENCODER,
    corpus: str = "ravdess",
    head: str = DEFAULT_HEAD,
) -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    spec = CORPORA[corpus]
    clips = discover(spec["root"], spec["name"])
    speakers = sorted({clip.speaker for clip in clips})
    if corpus == "ravdess" and speakers != [f"{a:02d}" for a in range(1, 25)]:
        raise SystemExit(f"Cross-validation needs RAVDESS actors 01–24 under {spec['root']}; found {speakers}")
    if not clips:
        raise SystemExit(f"No {spec['name']} clips under {spec['root']}")
    plan = folds(speakers, spec["folds"], spec["val"], spec["shuffle"])
    report = {
        "dataset": f"{spec['name']} speech audio, {len(speakers)} speakers, {len(clips)} clips",
        "seed": SEED,
        "label": {
            "distress": ["sad", "angry", "fearful"],
            "non_distress": ["neutral", "calm"],
            "excluded": ["happy", "surprised", "disgust", "song"],
        },
        "protocol": (
            f"{spec['folds']} folds by speaker. Each fold tests on one block of speakers, uses {spec['val']} speakers "
            "from the next block to choose the epoch or C, the calibration temperature and the decision threshold, "
            "and trains on the rest. 'summary' is raw (threshold 0.5, uncalibrated); 'summary_calibrated' uses the "
            "validation temperature and threshold. Values are mean and standard deviation over the test folds."
        ),
        "folds": [{key: sorted(value) for key, value in split.items()} for split in plan],
        "models": {},
    }
    if only in (None, "cnn"):
        report["models"]["cnn"] = run_cnn(clips, plan, epochs)
    if only in (None, "egemaps"):
        report["models"]["egemaps"] = run_egemaps(clips, plan)
    if only in (None, "wavlm"):
        report["models"][model_key(encoder, head)] = run_ssl(clips, plan, encoder, head)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / spec["report"]
    if only is not None and out.is_file():
        previous = json.loads(out.read_text())
        report["models"] = {**previous.get("models", {}), **report["models"]}
    report["paired_tests"] = paired_tests(report["models"])
    out.write_text(json.dumps(report, indent=2))

    for title, key in (("raw, threshold 0.5", "summary"), ("calibrated, validation threshold", "summary_calibrated")):
        print(f"\n{spec['name']}, {title}")
        print("model".ljust(22) + "".join(f"{m:>20}" for m in METRICS))
        for name, result in report["models"].items():
            cells = []
            for metric in METRICS:
                stat = result.get(key, {}).get(metric)
                cells.append(f"{stat['mean']:.3f} ± {stat['std']:.3f}".rjust(20) if stat else "-".rjust(20))
            print(f"{name:<22}" + "".join(cells))
    for test in report["paired_tests"]:
        if test["scores"] == "calibrated":
            print(f"{test['a']} vs {test['b']}: diff {test['mean_difference']:+.3f}  t p={test['t_test_p']}  wilcoxon p={test['wilcoxon_p']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", choices=sorted(CORPORA), default="ravdess")
    parser.add_argument("--only", choices=("cnn", "egemaps", "wavlm"))
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="CNN epochs")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, help="Hugging Face id of the pretrained speech encoder")
    parser.add_argument("--head", choices=sorted(HEADS), default=DEFAULT_HEAD, help="WavLM head")
    args = parser.parse_args()
    run(args.only, args.epochs, args.encoder, args.corpus, args.head)
