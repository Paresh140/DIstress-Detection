"""The WavLM voice model the studio uses live.

Training (from backend/, GPU recommended for the first embedding pass):
  python -m app.voice

Trains the binary mean-pooled head (the best head in speaker cross-validation
and across corpora; --head four-attentive gives the proposal's four-class head)
on RAVDESS actors 01–18, plus CREMA-D
when data/crema-d/ exists (all but nine speakers). RAVDESS actors 19–20 and
those nine CREMA-D speakers choose the epoch, the temperature and the
threshold. RAVDESS actors 21–24 are scored once and the result is saved with
the head in artifacts/voice_head.pt, with voice_report.json beside it (every
epoch and every test clip). Copy those next to the studio and the head
replaces the CNN on Monitor; without it the CNN is used, as before.
--progress appends one JSON line per epoch as it finishes.

At run time the encoder weights come from Hugging Face (about 380 MB, cached
after the first load).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from . import calibration
from . import enrollment as enroll
from .corpora import discover
from .dataset import TEST_ACTORS, TRAIN_ACTORS, VAL_ACTORS
from .speech_encoder import (
    CLASSES,
    DEFAULT_ENCODER,
    DEFAULT_HEAD,
    HEADS,
    LayerMixHead,
    Standardizer,
    cache_path,
    embed_clips,
    _distress_prob,
    fit_head,
    load_16k,
    load_encoder,
    pooled_layers,
    predict_logits,
    targets,
)
from .train import SEED, _binary_metrics, device

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "backend" / "artifacts"
HEAD_FILE = "voice_head.pt"
REPORT_FILE = "voice_report.json"
CREMA_VAL_SPEAKERS = 9
LIVE_HEAD = "binary-mean"


def _ids(actors: set[int]) -> set[str]:
    return {f"{actor:02d}" for actor in actors}


def _emit(progress: Path | None, payload: dict) -> None:
    if progress is None:
        return
    with progress.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def train(
    encoder: str = DEFAULT_ENCODER,
    head: str = LIVE_HEAD,
    out: Path = ARTIFACTS / HEAD_FILE,
    progress: Path | None = None,
) -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    ravdess = discover(ROOT / "data" / "ravdess", "RAVDESS")
    crema = discover(ROOT / "data" / "crema-d", "CREMA-D")
    if not ravdess:
        raise SystemExit("RAVDESS clips are needed under data/ravdess/")
    crema_speakers = sorted({clip.speaker for clip in crema})
    random.Random(SEED).shuffle(crema_speakers)
    crema_val = set(crema_speakers[:CREMA_VAL_SPEAKERS])

    train_clips = [c for c in ravdess if c.speaker in _ids(TRAIN_ACTORS)] + [c for c in crema if c.speaker not in crema_val]
    val_clips = [c for c in ravdess if c.speaker in _ids(VAL_ACTORS)] + [c for c in crema if c.speaker in crema_val]
    test_clips = [c for c in ravdess if c.speaker in _ids(TEST_ACTORS)]
    clips = train_clips + val_clips + test_clips
    paths = [str(c.path) for c in clips]
    features = dict(zip(paths, embed_clips(paths, encoder, cache_path(ARTIFACTS, encoder))))

    def stack(group: list) -> np.ndarray:
        return np.stack([features[str(c.path)] for c in group])

    test_x = stack(test_clips)
    test_y = np.array([c.label for c in test_clips])
    history: list[dict] = []

    def log_epoch(row: dict, model: LayerMixHead, scale: Standardizer) -> None:
        # Test speakers are only drawn here; the epoch is still chosen on validation speakers.
        prob = _distress_prob(torch.from_numpy(predict_logits({"head": model, "scale": scale}, test_x))).numpy()
        metrics = _binary_metrics(test_y, (prob >= 0.5).astype(np.int64))
        row = {
            **{k: round(float(v), 4) for k, v in row.items() if k != "epoch"},
            "epoch": row["epoch"],
            "test_f1": round(metrics["f1"], 4),
            "test_balanced_accuracy": round(metrics["balanced_accuracy"], 4),
        }
        history.append(row)
        _emit(progress, {"type": "epoch", **row})

    fitted = fit_head(
        stack(train_clips), targets(train_clips, head), stack(val_clips), targets(val_clips, head), head=head, on_epoch=log_epoch
    )
    val_logits = predict_logits(fitted, stack(val_clips))
    test_logits = predict_logits(fitted, test_x)
    scored = calibration.evaluate(val_logits, targets(val_clips, head), test_logits, test_y)
    cal = scored["calibrated"]
    summary = {
        "encoder": encoder,
        "head": head,
        "train_clips": len(train_clips),
        "val_clips": len(val_clips),
        "test": "RAVDESS actors 21–24",
        "test_clips": len(test_clips),
        "corpora": sorted({c.corpus for c in train_clips}),
        "selected_epoch": fitted["epoch"],
        "temperature": scored["temperature"],
        "threshold": scored["threshold"],
        "test_balanced_accuracy": round(cal["balanced_accuracy"], 4),
        "test_auroc": round(cal["auroc"], 4) if cal.get("auroc") is not None else None,
        "test_specificity": round(cal["specificity"], 4),
        "test_ece": round(cal["ece"], 4),
        "layer_weights": fitted["layer_weights"],
    }
    enrolled = _train_enrolled(features, train_clips, val_clips, test_clips, head)
    if enrolled is not None:
        summary["enrolled"] = enrolled["summary"]
    model: LayerMixHead = fitted["head"]
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": {k: v.cpu() for k, v in model.state_dict().items()},
            "n_layers": int(features[paths[0]].shape[0]),
            "dim": int(features[paths[0]].shape[-1]),
            "mean": fitted["scale"].mean,
            "std": fitted["scale"].std,
            "summary": summary,
            **({"enrolled": enrolled["saved"]} if enrolled is not None else {}),
        },
        out,
    )
    predictions = [
        {
            "file": clip.path.name,
            "speaker": clip.speaker,
            "emotion": clip.emotion,
            "truth": "distress" if clip.label else "calm",
            "probability": round(float(p), 4),
            "correct": bool((p >= scored["threshold"]) == bool(clip.label)),
        }
        for clip, p in zip(test_clips, scored["test_probability"])
    ]
    report = {"summary": summary, "epochs": len(history), "history": history, "predictions": predictions}
    (out.parent / REPORT_FILE).write_text(json.dumps(report, indent=2))
    _emit(progress, {"type": "done", "summary": summary})
    print(json.dumps(summary, indent=2))
    return summary


def _train_enrolled(features: dict, train_clips: list, val_clips: list, test_clips: list, head: str, k: int = enroll.ENROLL_CLIPS) -> dict | None:
    """A second head on calm-enrolled features: each speaker z-scored by k of their own calm clips.

    The enrollment clips are left out of training, validation and test, so no
    clip is both a reference and a scored example. This is the head the studio
    uses once a person has recorded their calm reading.
    """
    clips = train_clips + val_clips + test_clips
    chosen = enroll.pick_enrollment(clips, k)
    if not chosen:
        return None
    shifted = enroll.enrolled_features(features, clips, chosen)
    tr, va, te = (enroll.scorable(group, chosen) for group in (train_clips, val_clips, test_clips))

    def stack(group: list) -> np.ndarray:
        return np.stack([shifted[str(c.path)] for c in group])

    fitted = fit_head(stack(tr), targets(tr, head), stack(va), targets(va, head), head=head)
    test_y = np.array([c.label for c in te])
    scored = calibration.evaluate(predict_logits(fitted, stack(va)), targets(va, head), predict_logits(fitted, stack(te)), test_y)
    cal = scored["calibrated"]
    summary = {
        "enroll_clips": k,
        "normalization": "per speaker z-score over k calm/neutral clips (layers x channels), enrollment clips excluded from every split",
        "train_clips": len(tr),
        "val_clips": len(va),
        "test_clips": len(te),
        "selected_epoch": fitted["epoch"],
        "temperature": scored["temperature"],
        "threshold": scored["threshold"],
        "test_balanced_accuracy": round(cal["balanced_accuracy"], 4),
        "test_auroc": round(cal["auroc"], 4) if cal.get("auroc") is not None else None,
        "test_specificity": round(cal["specificity"], 4),
        "test_ece": round(cal["ece"], 4),
    }
    print("enrolled head:", json.dumps(summary))
    model: LayerMixHead = fitted["head"]
    saved = {
        "state": {key: v.cpu() for key, v in model.state_dict().items()},
        "mean": fitted["scale"].mean,
        "std": fitted["scale"].std,
        "temperature": float(scored["temperature"]),
        "threshold": float(scored["threshold"]),
        "enroll_clips": k,
    }
    return {"summary": summary, "saved": saved}


class VoiceModel:
    """Loads the encoder and the saved head once and scores one clip at a time."""

    def __init__(self, path: Path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self.summary = saved["summary"]
        spec = HEADS[self.summary["head"]]
        self.dev = device()
        self.head = LayerMixHead(saved["n_layers"], saved["dim"], classes=spec["classes"], pool=spec["pool"])
        self.head.load_state_dict(saved["state"])
        self.head.to(self.dev).eval()
        self.scale = Standardizer.__new__(Standardizer)
        self.scale.mean, self.scale.std = saved["mean"], saved["std"]
        self.enrolled = None
        if "enrolled" in saved:
            e = saved["enrolled"]
            enrolled_head = LayerMixHead(saved["n_layers"], saved["dim"], classes=spec["classes"], pool=spec["pool"])
            enrolled_head.load_state_dict(e["state"])
            enrolled_head.to(self.dev).eval()
            enrolled_scale = Standardizer.__new__(Standardizer)
            enrolled_scale.mean, enrolled_scale.std = e["mean"], e["std"]
            self.enrolled = {"head": enrolled_head, "scale": enrolled_scale, "temperature": float(e["temperature"]), "threshold": float(e["threshold"])}
        self.extractor, self.encoder = load_encoder(self.summary["encoder"])
        self.encoder.to(self.dev).eval()
        self.mtime = path.stat().st_mtime

    @torch.no_grad()
    def features(self, path: str) -> np.ndarray:
        """Binned WavLM layer features of one clip, [layers, BINS, dim]."""
        return pooled_layers(load_16k(path), self.extractor, self.encoder, self.dev)

    @torch.no_grad()
    def score(self, path: str, calm: dict | None = None) -> dict:
        """P(distress) for one clip. With calm stats and an enrolled head, the clip is z-scored against the person's calm reading first."""
        return self.score_features(self.features(path), calm)

    @torch.no_grad()
    def score_features(self, feats: np.ndarray, calm: dict | None = None) -> dict:
        use_enrolled = calm is not None and self.enrolled is not None
        if use_enrolled:
            head, scale = self.enrolled["head"], self.enrolled["scale"]
            temperature, threshold = self.enrolled["temperature"], self.enrolled["threshold"]
            feats = enroll.apply(feats, calm)
        else:
            head, scale = self.head, self.scale
            temperature, threshold = float(self.summary["temperature"]), float(self.summary["threshold"])
        x = scale(feats[None], self.dev)
        logits = head(x)
        probability = float(calibration.distress_probability(logits.cpu().numpy(), temperature)[0])
        out = {
            "probability": probability,
            "threshold": threshold,
            "enrolled": use_enrolled,
            "model": f"{self.summary['encoder']} + {self.summary['head']} head" + (" (calm-enrolled)" if use_enrolled else ""),
        }
        if logits.ndim == 2:
            classes = torch.softmax(logits[0] / temperature, dim=0).cpu().tolist()
            out["classes"] = {name: round(p, 4) for name, p in zip(CLASSES, classes)}
        if head.pool == "attentive":
            mixed = (head.layer_weights()[None, :, None, None] * x).sum(dim=1)
            out["time_weights"] = [round(w, 4) for w in head.time_weights(mixed)[0].cpu().tolist()]
        out["layer_weights"] = [round(float(w), 4) for w in head.layer_weights().cpu()]
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--head", choices=sorted(HEADS), default=LIVE_HEAD)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / HEAD_FILE, help="where voice_head.pt goes; voice_report.json is written beside it")
    parser.add_argument("--progress", type=Path, help="append one JSON line per epoch to this file")
    args = parser.parse_args()
    train(args.encoder, args.head, args.out, args.progress)
