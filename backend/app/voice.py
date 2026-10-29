"""The WavLM voice model the studio uses live.

Training (from backend/, GPU recommended for the first embedding pass):
  python -m app.voice

Trains the binary mean-pooled head (the best head in speaker cross-validation
and across corpora; --head four-attentive gives the proposal's four-class head)
on RAVDESS actors 01–18, plus CREMA-D
when data/crema-d/ exists (all but nine speakers). RAVDESS actors 19–20 and
those nine CREMA-D speakers choose the epoch, the temperature and the
threshold. RAVDESS actors 21–24 are scored once and the result is saved with
the head in artifacts/voice_head.pt. A second head is trained the same way on
features shifted by each speaker's calm enrollment (two neutral clips); the
studio uses it once the person has recorded their enrollment. Copy that file next to the studio and it
replaces the CNN on Monitor; without it the CNN is used, as before.

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
from .corpora import discover
from .dataset import TEST_ACTORS, TRAIN_ACTORS, VAL_ACTORS
from .enrollment import ENROLL_CLIPS, drop, enrollment_paths, neutral_shift, reference
from .speech_encoder import (
    CLASSES,
    DEFAULT_ENCODER,
    DEFAULT_HEAD,
    HEADS,
    LayerMixHead,
    Standardizer,
    cache_path,
    embed_clips,
    fit_head,
    load_16k,
    load_encoder,
    pooled_layers,
    predict_logits,
    targets,
)
from .train import SEED, device

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "backend" / "artifacts"
HEAD_FILE = "voice_head.pt"
CREMA_VAL_SPEAKERS = 9
LIVE_HEAD = "binary-mean"


def _ids(actors: set[int]) -> set[str]:
    return {f"{actor:02d}" for actor in actors}


def train(encoder: str = DEFAULT_ENCODER, head: str = LIVE_HEAD, out: Path = ARTIFACTS / HEAD_FILE) -> dict:
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

    plain = _fit(train_clips, val_clips, test_clips, features, encoder, head)
    # The same head on features shifted by each speaker's neutral enrollment (app/enrollment.py).
    enroll = enrollment_paths(clips)
    shifted = neutral_shift(features, clips, enroll)
    enrolled = _fit(*(drop(group, enroll, shifted) for group in (train_clips, val_clips, test_clips)), shifted, encoder, head)
    enrolled["summary"]["enrollment"] = f"mean of {ENROLL_CLIPS} neutral clips per speaker, left out of every split"
    out.parent.mkdir(parents=True, exist_ok=True)
    first = features[paths[0]]
    torch.save(
        {
            **plain,  # the plain head at the top level, as older files had it
            "n_layers": int(first.shape[0]),
            "dim": int(first.shape[-1]),
            "enrolled": enrolled,
            "enroll_clips": ENROLL_CLIPS,
        },
        out,
    )
    summary = {"plain": plain["summary"], "enrolled": enrolled["summary"]}
    print(json.dumps(summary, indent=2))
    return summary


def _fit(train_clips: list, val_clips: list, test_clips: list, features: dict, encoder: str, head: str) -> dict:
    def stack(group: list) -> np.ndarray:
        return np.stack([features[str(c.path)] for c in group])

    fitted = fit_head(stack(train_clips), targets(train_clips, head), stack(val_clips), targets(val_clips, head), head=head)
    val_logits = predict_logits(fitted, stack(val_clips))
    test_logits = predict_logits(fitted, stack(test_clips))
    scored = calibration.evaluate(val_logits, targets(val_clips, head), test_logits, np.array([c.label for c in test_clips]))
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
    model: LayerMixHead = fitted["head"]
    return {
        "state": {k: v.cpu() for k, v in model.state_dict().items()},
        "mean": fitted["scale"].mean,
        "std": fitted["scale"].std,
        "summary": summary,
    }


class _Head:
    def __init__(self, saved: dict, n_layers: int, dim: int, dev: torch.device):
        self.summary = saved["summary"]
        spec = HEADS[self.summary["head"]]
        self.model = LayerMixHead(n_layers, dim, classes=spec["classes"], pool=spec["pool"])
        self.model.load_state_dict(saved["state"])
        self.model.to(dev).eval()
        self.scale = Standardizer.__new__(Standardizer)
        self.scale.mean, self.scale.std = saved["mean"], saved["std"]


class VoiceModel:
    """Loads the encoder and the saved heads once and scores one clip at a time.

    With enough enrollment clips from the current speaker, the enrolled head
    scores features shifted by that speaker's calm reference; otherwise the
    plain head is used.
    """

    def __init__(self, path: Path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self.dev = device()
        self.plain = _Head(saved, saved["n_layers"], saved["dim"], self.dev)
        self.enrolled = _Head(saved["enrolled"], saved["n_layers"], saved["dim"], self.dev) if "enrolled" in saved else None
        self.enroll_clips = int(saved.get("enroll_clips", ENROLL_CLIPS))
        self.summary = self.plain.summary
        self.extractor, self.encoder = load_encoder(self.summary["encoder"])
        self.encoder.to(self.dev).eval()
        self.mtime = path.stat().st_mtime

    def features(self, path: str) -> np.ndarray:
        return pooled_layers(load_16k(path), self.extractor, self.encoder, self.dev)

    @torch.no_grad()
    def score(self, path: str, enrollment: list[np.ndarray] | None = None) -> dict:
        feats = self.features(path)
        use_enrolled = self.enrolled is not None and len(enrollment or []) >= self.enroll_clips
        head = self.enrolled if use_enrolled else self.plain
        if use_enrolled:
            feats = feats - reference(np.stack(enrollment))
        x = head.scale(feats[None], self.dev)
        logits = head.model(x)
        temperature = float(head.summary["temperature"])
        probability = float(calibration.distress_probability(logits.cpu().numpy(), temperature)[0])
        out = {
            "probability": probability,
            "threshold": float(head.summary["threshold"]),
            "model": f"{head.summary['encoder']} + {head.summary['head']} head" + (", calm enrollment" if use_enrolled else ""),
            "enrolled": use_enrolled,
        }
        if logits.ndim == 2:
            classes = torch.softmax(logits[0] / temperature, dim=0).cpu().tolist()
            out["classes"] = {name: round(p, 4) for name, p in zip(CLASSES, classes)}
        if head.model.pool == "attentive":
            mixed = (head.model.layer_weights()[None, :, None, None] * x).sum(dim=1)
            out["time_weights"] = [round(w, 4) for w in head.model.time_weights(mixed)[0].cpu().tolist()]
        out["layer_weights"] = [round(float(w), 4) for w in head.model.layer_weights().cpu()]
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--head", choices=sorted(HEADS), default=LIVE_HEAD)
    args = parser.parse_args()
    train(args.encoder, args.head)
