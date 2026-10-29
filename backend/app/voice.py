"""The WavLM voice model the studio uses live.

Training (from backend/, GPU recommended for the first embedding pass):
  python -m app.voice

Trains the binary mean-pooled head (the best head in speaker cross-validation
and across corpora; --head four-attentive gives the proposal's four-class head)
on RAVDESS actors 01–18, plus CREMA-D
when data/crema-d/ exists (all but nine speakers). RAVDESS actors 19–20 and
those nine CREMA-D speakers choose the epoch, the temperature and the
threshold. RAVDESS actors 21–24 are scored once and the result is saved with
the head in artifacts/voice_head.pt. Copy that file next to the studio and it
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
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": {k: v.cpu() for k, v in model.state_dict().items()},
            "n_layers": int(features[paths[0]].shape[0]),
            "dim": int(features[paths[0]].shape[-1]),
            "mean": fitted["scale"].mean,
            "std": fitted["scale"].std,
            "summary": summary,
        },
        out,
    )
    print(json.dumps(summary, indent=2))
    return summary


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
        self.extractor, self.encoder = load_encoder(self.summary["encoder"])
        self.encoder.to(self.dev).eval()
        self.mtime = path.stat().st_mtime

    @torch.no_grad()
    def score(self, path: str) -> dict:
        feats = pooled_layers(load_16k(path), self.extractor, self.encoder, self.dev)[None]
        x = self.scale(feats, self.dev)
        logits = self.head(x)
        temperature = float(self.summary["temperature"])
        probability = float(calibration.distress_probability(logits.cpu().numpy(), temperature)[0])
        out = {
            "probability": probability,
            "threshold": float(self.summary["threshold"]),
            "model": f"{self.summary['encoder']} + {self.summary['head']} head",
        }
        if logits.ndim == 2:
            classes = torch.softmax(logits[0] / temperature, dim=0).cpu().tolist()
            out["classes"] = {name: round(p, 4) for name, p in zip(CLASSES, classes)}
        if self.head.pool == "attentive":
            mixed = (self.head.layer_weights()[None, :, None, None] * x).sum(dim=1)
            out["time_weights"] = [round(w, 4) for w in self.head.time_weights(mixed)[0].cpu().tolist()]
        out["layer_weights"] = [round(float(w), 4) for w in self.head.layer_weights().cpu()]
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--head", choices=sorted(HEADS), default=LIVE_HEAD)
    args = parser.parse_args()
    train(args.encoder, args.head)
