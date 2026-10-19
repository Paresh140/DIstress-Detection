"""Cross-corpus audio study, plus a controlled text-disagreement check.

The network is audio only. Checkpoint selection uses speakers held out of
training, and the reported test speakers are touched once. Text is not a
trained branch: RAVDESS repeats two sentences and CREMA-D repeats twelve
neutral sentences, so a text model on those transcripts cannot learn emotion.
"""

from __future__ import annotations

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


def _run(name: str, train_clips: list, val_clips: list, test_clips: list, cache: dict) -> dict:
    print(f"\n{name}: train {len(train_clips)}  val {len(val_clips)}  test {len(test_clips)}")
    fitted = fit_audio(_rows(train_clips, cache), _rows(val_clips, cache), epochs=EPOCHS)
    scored = score_audio(fitted["state"], _rows(test_clips, cache))
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
        "selected_epoch": fitted["epoch"],
        "val_balanced_accuracy": round(fitted["val_balanced_accuracy"], 4),
        "test_loss": round(scored["loss"], 4),
        "test_accuracy": round(scored["accuracy"], 4),
        "test_balanced_accuracy": round(scored["balanced_accuracy"], 4),
        "test_precision": round(scored["precision"], 4),
        "test_recall": round(scored["recall"], 4),
        "test_specificity": round(scored["specificity"], 4),
        "test_f1": round(scored["f1"], 4),
        "majority_f1": _majority_f1(scored),
        "confusion": {"tn": scored["tn"], "fp": scored["fp"], "fn": scored["fn"], "tp": scored["tp"]},
    }


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


def run() -> dict:
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

    cache: dict[str, torch.Tensor] = {}
    runs = [
        _run(
            "RAVDESS speakers held out",
            _take(ravdess, rav_split["train"]),
            _take(ravdess, rav_split["val"]),
            _take(ravdess, rav_split["test"]),
            cache,
        ),
        _run(
            "CREMA-D speakers held out",
            _take(crema, crema_split["train"]),
            _take(crema, crema_split["val"]),
            _take(crema, crema_split["test"]),
            cache,
        ),
        _run(
            "Train RAVDESS, test CREMA-D",
            _take(ravdess, rav_split["train"] | rav_split["test"]),
            _take(ravdess, rav_val_ids),
            crema,
            cache,
        ),
        _run(
            "Train CREMA-D, test RAVDESS",
            _take(crema, crema_split["train"] | crema_split["test"]),
            _take(crema, crema_val_ids),
            ravdess,
            cache,
        ),
    ]
    report = {
        "ready": True,
        "seed": SEED,
        "epochs": EPOCHS,
        "selection": (
            "The epoch is chosen by balanced accuracy on validation speakers from the training corpus, "
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
        "runs": [_public(run) for run in runs],
        "disagreement": _disagreement(discover_demo(RAVDESS_ROOT)),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "cross_corpus_report.json").write_text(json.dumps(report, indent=2))
    for run in report["runs"]:
        print(f"{run['name']}: test F1 {run['test_f1']:.3f}  acc {run['test_accuracy']:.3f}")
    return report


if __name__ == "__main__":
    run()
