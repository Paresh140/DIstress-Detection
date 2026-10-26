"""Cross-corpus audio study, plus a controlled text-disagreement check.

Each voice model runs the same four train/test settings on the same speaker
splits: within RAVDESS, within CREMA-D, and across corpora in both directions.
  cnn      the DistressNet audio branch (log-mel CNN)
  egemaps  eGeMAPS v02 functionals + logistic regression, C chosen on validation speakers
  wavlm    frozen WavLM-base-plus, mean-pooled layers, small trained head (app/speech_encoder.py)

Usage, from backend/:
  python -m app.study                          # all three models
  python -m app.study --models wavlm egemaps
  python -m app.study --models wavlm --encoder facebook/wav2vec2-base   # reported as wav2vec2
  python -m app.study --models egemaps wavlm --norm speaker              # reported as wavlm+speaker

--norm standardizes the eGeMAPS and WavLM features of each corpus (corpus) or
each speaker (speaker) with that group's own mean and std before the model sees
them. It uses no labels, only unlabeled audio from the same group, and removes
the offset between recording setups that otherwise shifts every score across
corpora. The CNN is not affected.

The network is audio only. Checkpoint selection uses speakers held out of
training, and the reported test speakers are touched once. Text is not a
trained branch: RAVDESS repeats two sentences and CREMA-D repeats twelve
neutral sentences, so a text model on those transcripts cannot learn emotion.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .audio import file_to_logmel
from .corpora import discover, sentence_coverage
from .dataset import TEST_ACTORS
from .dataset import discover as discover_demo
from .judge import decide
from .language import read_language
from .model import DistressNet
from .cv import C_GRID, _egemaps, _scored, encoder_key
from .speech_encoder import DEFAULT_ENCODER, embed_clips, fit_head, predict
from .train import device, fit_audio, score_audio

ROOT = Path(__file__).resolve().parents[2]
RAVDESS_ROOT = ROOT / "data" / "ravdess"
CREMA_ROOT = ROOT / "data" / "crema-d"
ARTIFACTS = ROOT / "backend" / "artifacts"
WEIGHTS = ARTIFACTS / "audio_branch.pt"
SEED = 7
EPOCHS = 8

DISTRESS_LINE = "I am not feeling well today"
CALM_LINE = "I feel fine"


def _speaker_split(clips: list, test_ratio: float = 0.2, val_ratio: float = 0.15) -> dict[str, set[str]]:
    speakers = sorted({clip.speaker for clip in clips})
    rng = random.Random(SEED)
    rng.shuffle(speakers)
    n_test = max(1, round(len(speakers) * test_ratio))
    n_val = max(1, round((len(speakers) - n_test) * val_ratio))
    return {
        "test": set(speakers[:n_test]),
        "val": set(speakers[n_test : n_test + n_val]),
        "train": set(speakers[n_test + n_val :]),
    }


def _rows(clips: list, cache: dict[str, torch.Tensor]) -> list[dict]:
    rows = []
    for clip in clips:
        key = str(clip.path)
        if key not in cache:
            cache[key] = file_to_logmel(key)
        rows.append({"mel": cache[key], "label": clip.label, "clip": clip})
    return rows


def _take(clips: list, speakers: set[str]) -> list:
    return [clip for clip in clips if clip.speaker in speakers]


def _result(name: str, train_clips: list, val_clips: list, test_clips: list, selected: dict, scored: dict) -> dict:
    return {
        "name": name,
        "train_corpus": sorted({clip.corpus for clip in train_clips}),
        "test_corpus": sorted({clip.corpus for clip in test_clips}),
        "train_speakers": sorted({clip.speaker for clip in train_clips}),
        "val_speakers": sorted({clip.speaker for clip in val_clips}),
        "test_speakers": sorted({clip.speaker for clip in test_clips}),
        "train_clips": len(train_clips),
        "val_clips": len(val_clips),
        "test_clips": len(test_clips),
        **selected,
        "test_loss": round(scored["loss"], 4) if scored.get("loss") is not None else None,
        "test_accuracy": round(scored["accuracy"], 4),
        "test_balanced_accuracy": round(scored["balanced_accuracy"], 4),
        "test_auroc": round(scored["auroc"], 4) if scored.get("auroc") is not None else None,
        "test_precision": round(scored["precision"], 4),
        "test_recall": round(scored["recall"], 4),
        "test_specificity": round(scored["specificity"], 4),
        "test_f1": round(scored["f1"], 4),
        "majority_f1": _majority_f1(scored),
        "confusion": {"tn": scored["tn"], "fp": scored["fp"], "fn": scored["fn"], "tp": scored["tp"]},
    }


def _run(name: str, train_clips: list, val_clips: list, test_clips: list, cache: dict) -> dict:
    print(f"\ncnn {name}: train {len(train_clips)}  val {len(val_clips)}  test {len(test_clips)}")
    fitted = fit_audio(_rows(train_clips, cache), _rows(val_clips, cache), epochs=EPOCHS)
    scored = score_audio(fitted["state"], _rows(test_clips, cache))
    selected = {"selected_epoch": fitted["epoch"], "val_balanced_accuracy": round(fitted["val_balanced_accuracy"], 4)}
    return _result(name, train_clips, val_clips, test_clips, selected, scored)


def _arrays(clips: list, features: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return np.stack([features[str(clip.path)] for clip in clips]), np.array([clip.label for clip in clips])


def _normalize(features: dict[str, np.ndarray], clips: list, how: str) -> dict[str, np.ndarray]:
    if how == "none":
        return features
    groups: dict[tuple, list[str]] = {}
    for clip in clips:
        key = (clip.corpus,) if how == "corpus" else (clip.corpus, clip.speaker)
        groups.setdefault(key, []).append(str(clip.path))
    out = {}
    for paths in groups.values():
        x = np.stack([features[path] for path in paths])
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
        out.update(zip(paths, x))
    return out


def _run_wavlm(name: str, train_clips: list, val_clips: list, test_clips: list, features: dict) -> dict:
    (xt, yt), (xv, yv), (xs, ys) = (_arrays(c, features) for c in (train_clips, val_clips, test_clips))
    fitted = fit_head(xt, yt, xv, yv)
    scored = _scored(ys, predict(fitted, xs))
    selected = {
        "selected_epoch": fitted["epoch"],
        "val_balanced_accuracy": round(fitted["val_balanced_accuracy"], 4),
        "layer_weights": fitted["layer_weights"],
    }
    print(f"{name}: val bal {fitted['val_balanced_accuracy']:.3f}  test bal {scored['balanced_accuracy']:.3f}")
    return _result(name, train_clips, val_clips, test_clips, selected, scored)


def _run_egemaps(name: str, train_clips: list, val_clips: list, test_clips: list, features: dict) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    (xt, yt), (xv, yv), (xs, ys) = (_arrays(c, features) for c in (train_clips, val_clips, test_clips))

    def model(c: float):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=5000))

    best_c, best_score = C_GRID[0], -1.0
    for c in C_GRID:
        score = _scored(yv, model(c).fit(xt, yt).predict_proba(xv)[:, 1])["balanced_accuracy"]
        if score > best_score:
            best_c, best_score = c, score
    scored = _scored(ys, model(best_c).fit(xt, yt).predict_proba(xs)[:, 1])
    print(f"{name}: C {best_c}  val bal {best_score:.3f}  test bal {scored['balanced_accuracy']:.3f}")
    selected = {"selected_c": best_c, "val_balanced_accuracy": round(best_score, 4)}
    return _result(name, train_clips, val_clips, test_clips, selected, scored)


def _majority_f1(scored: dict) -> float:
    positive = scored["tp"] + scored["fn"]
    negative = scored["tn"] + scored["fp"]
    total = max(positive + negative, 1)
    if positive >= negative:
        precision = positive / total
        return round(2 * precision / (precision + 1), 4) if precision else 0.0
    return 0.0


def _public(run: dict) -> dict:
    hidden = {"train_speakers", "val_speakers", "test_speakers"}
    shown = {key: value for key, value in run.items() if key not in hidden}
    shown["train_speaker_count"] = len(run["train_speakers"])
    shown["val_speaker_count"] = len(run["val_speakers"])
    shown["test_speaker_count"] = len(run["test_speakers"])
    return shown


def _disagreement(ravdess_demo: list) -> dict:
    """Pair real held-out voices with text we assign, including conflicts."""
    if not WEIGHTS.exists():
        return {"ready": False, "note": "The Monitor audio weights are not saved, so the disagreement check was skipped."}
    held = [clip for clip in ravdess_demo if clip.actor in TEST_ACTORS]
    distress = [clip for clip in held if clip.label == 1][:16]
    calm = [clip for clip in held if clip.label == 0][:16]
    dev = device()
    model = DistressNet().to(dev)
    state = torch.load(WEIGHTS, map_location=dev, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    cells = {
        "acted distress, words distress": [],
        "acted distress, words calm": [],
        "acted calm, words distress": [],
        "acted calm, words calm": [],
    }
    with torch.no_grad():
        for clip, voice_name in [(item, "distress") for item in distress] + [(item, "calm") for item in calm]:
            mel = file_to_logmel(str(clip.path)).unsqueeze(0).unsqueeze(0).to(dev)
            probability = float(torch.sigmoid(model.forward_audio(mel)).squeeze().cpu())
            heard = "distress" if probability >= 0.5 else "calm"
            for line, word_name in ((DISTRESS_LINE, "distress"), (CALM_LINE, "calm")):
                language = read_language(line, allow_model=False)
                verdict = decide(probability, language, line)
                cells[f"acted {voice_name}, words {word_name}"].append(
                    {
                        "file": clip.path.name,
                        "acted": voice_name,
                        "heard": heard,
                        "voice_probability": round(probability, 4),
                        "decision": verdict["decision"],
                    }
                )
    summary = []
    for name, rows in cells.items():
        summary.append(
            {
                "case": name,
                "clips": len(rows),
                "heard_distress": sum(row["heard"] == "distress" for row in rows),
                "judge_distress": sum(row["decision"] == "distress" for row in rows),
                "judge_calm": sum(row["decision"] == "calm" for row in rows),
            }
        )
    return {
        "ready": True,
        "model": "Monitor audio branch, RAVDESS test actors 21–24",
        "distress_line": DISTRESS_LINE,
        "calm_line": CALM_LINE,
        "rule": (
            "The judge is a fixed rule, not a network trained on these transcripts. "
            "Distress in the words is kept even when the voice is calm. "
            "Distress in the voice is kept even when the words are calm."
        ),
        "cells": summary,
    }


def _both_labels(clips: list) -> tuple[int, int]:
    coverage = sentence_coverage(clips)
    shared = 0
    for emotions in coverage.values():
        names = set(emotions)
        if "neutral" in names and names & {"sad", "angry", "fearful"}:
            shared += 1
    return len(coverage), shared


def _text_report(ravdess: list, crema: list) -> dict:
    rav_total, rav_shared = _both_labels(ravdess)
    crema_total, crema_shared = _both_labels(crema)
    return {
        "ravdess_sentences": sentence_coverage(ravdess),
        "crema_sentences": sentence_coverage(crema),
        "ravdess_sentence_count": rav_total,
        "crema_sentence_count": crema_total,
        "finding": (
            f"RAVDESS contributes {rav_total} fixed sentences, and {rav_shared} of them are spoken as both neutral and distress. "
            f"CREMA-D contributes {crema_total} fixed sentences, and {crema_shared} of them are spoken as both neutral and distress. "
            "The words do not carry the emotion label, so no text branch was trained on these transcripts. "
            "At use time the assistant reads the words separately and the judge combines that reading with the voice score."
        ),
    }


def run(models: tuple[str, ...] = ("cnn", "egemaps", "wavlm"), encoder: str = DEFAULT_ENCODER, norm: str = "none") -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    ravdess = discover(RAVDESS_ROOT, "RAVDESS")
    crema = discover(CREMA_ROOT, "CREMA-D")
    if not ravdess:
        raise SystemExit(f"No shared-label RAVDESS clips under {RAVDESS_ROOT}")
    if not crema:
        raise SystemExit(f"No shared-label CREMA-D clips under {CREMA_ROOT}")

    rav_split = _speaker_split(ravdess)
    crema_split = _speaker_split(crema)
    rav_val_ids = rav_split["val"]
    crema_val_ids = crema_split["val"]

    # The same four settings, on the same speakers, for every model.
    settings = [
        ("RAVDESS speakers held out", _take(ravdess, rav_split["train"]), _take(ravdess, rav_val_ids), _take(ravdess, rav_split["test"])),
        ("CREMA-D speakers held out", _take(crema, crema_split["train"]), _take(crema, crema_val_ids), _take(crema, crema_split["test"])),
        ("Train RAVDESS, test CREMA-D", _take(ravdess, rav_split["train"] | rav_split["test"]), _take(ravdess, rav_val_ids), crema),
        ("Train CREMA-D, test RAVDESS", _take(crema, crema_split["train"] | crema_split["test"]), _take(crema, crema_val_ids), ravdess),
    ]
    clips = ravdess + crema
    paths = [str(clip.path) for clip in clips]
    suffix = "" if norm == "none" else f"+{norm}"
    results: dict[str, list[dict]] = {}
    if "cnn" in models:
        cache: dict[str, torch.Tensor] = {}
        results["cnn"] = [_run(*setting, cache) for setting in settings]
    if "egemaps" in models:
        print(f"\negemaps{suffix}")
        features = _normalize(dict(zip(paths, _egemaps(paths, ARTIFACTS / "egemaps_cache.json"))), clips, norm)
        results[f"egemaps{suffix}"] = [_run_egemaps(*setting, features) for setting in settings]
    if "wavlm" in models:
        name = encoder_key(encoder) + suffix
        print(f"\n{name}")
        cache_file = ARTIFACTS / "ssl_cache" / f"{encoder.replace('/', '__')}.npz"
        features = _normalize(dict(zip(paths, embed_clips(paths, encoder, cache_file))), clips, norm)
        results[name] = [_run_wavlm(*setting, features) for setting in settings]

    out = ARTIFACTS / "cross_corpus_report.json"
    previous = json.loads(out.read_text()) if out.is_file() else {}
    by_model = {**previous.get("models", {}), **{name: [_public(r) for r in rows] for name, rows in results.items()}}
    report = {
        "ready": True,
        "seed": SEED,
        "epochs": EPOCHS,
        "selection": (
            "The epoch (or C for eGeMAPS) is chosen by balanced accuracy on validation speakers from the training corpus, "
            "so neutral and distress count equally. Test speakers are scored once. "
            "Distress F1 is also shown, next to the F1 of a classifier that always predicts the larger class."
        ),
        "iemocap": (
            "IEMOCAP is not in this study. The files require a signed license from USC SAIL and were not downloaded. "
            "The third rotation, training on IEMOCAP plus one corpus and testing on the other, is the planned extension once that license is in hand."
        ),
        "label": {
            "distress": ["sad", "angry", "fearful"],
            "non_distress": ["neutral", "calm (RAVDESS only)"],
            "excluded": [
                "happy",
                "disgust",
                "surprise",
                "frustration (IEMOCAP only, corpus absent)",
            ],
        },
        "text": _text_report(ravdess, crema),
        # "runs" stays the CNN, which the studio and chat read; "models" holds every model.
        "runs": by_model.get("cnn", previous.get("runs", [])),
        "models": by_model,
        "disagreement": _disagreement(discover_demo(RAVDESS_ROOT)),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    def cell(row: dict | None, key: str) -> str:
        if not row or row.get(key) is None:
            return "-".rjust(16)
        if key == "test_balanced_accuracy":
            return f"{row[key]:.3f} ({row['test_specificity']:.2f})".rjust(16)
        return f"{row[key]:.3f}".rjust(16)

    for key, title in (("test_balanced_accuracy", "balanced accuracy (specificity)"), ("test_auroc", "AUROC")):
        print(f"\n{title:<30}" + "".join(f"{name:>16}" for name in by_model))
        for i, (setting, *_rest) in enumerate(settings):
            print(f"{setting:<30}" + "".join(cell(rows[i] if i < len(rows) else None, key) for rows in by_model.values()))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=("cnn", "egemaps", "wavlm"), default=["cnn", "egemaps", "wavlm"])
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, help="Hugging Face id of the pretrained speech encoder")
    parser.add_argument("--norm", choices=("none", "corpus", "speaker"), default="none", help="feature normalization for eGeMAPS and WavLM")
    args = parser.parse_args()
    run(tuple(args.models), args.encoder, args.norm)
