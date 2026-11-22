"""Calm enrollment: z-score a speaker's voice against a few of their own calm clips.

Why: the live voice head was trained on studio recordings. A new microphone,
room and voice is a new corpus, and the cross-corpus study showed what that
does without normalization (RAVDESS to CREMA-D 0.52 balanced accuracy,
specificity 0.06: nearly everything is called distress). Standardizing each
speaker with their own statistics fixed most of that (0.75 / 0.64), but it
needs audio of every emotion from the speaker. A live user can only give a
short calm reading, so this module does the same standardization with calm
clips only, the same way in training, in evaluation and live:

  stats = per layer and channel mean and std over K calm clips (and their time bins)
  x'    = (x - mean) / max(std, STD_FLOOR)

The enrolled head in voice_head.pt is trained on features standardized this
way, per training speaker, so live features standardized against the user's
own calm reading land where the head expects them.

The person reads fixed neutral sentences (CALM_SCRIPT, taken from RAVDESS and
CREMA-D). Stats are cached on disk per browser session (EnrollmentStore), so
they survive restarts and every request from that session uses them.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import numpy as np

ENROLL_CLIPS = 5  # calm clips per speaker, in training, evaluation and live
MIN_LIVE_CLIPS = 3  # the studio accepts an enrollment from this many readings
STD_FLOOR = 1e-2  # few clips can give a near-zero std on a channel; never divide by less than this
CALM_EMOTIONS = ("neutral", "calm")

# What the person reads, in a calm voice. The fixed RAVDESS and CREMA-D sentences.
CALM_SCRIPT = (
    "Kids are talking by the door.",
    "Dogs are sitting by the door.",
    "I'm on my way to the meeting.",
    "I wonder what this is about.",
    "The airplane is almost full.",
    "Maybe tomorrow it will be cold.",
    "I would like a new alarm clock.",
    "I think I have a doctor's appointment.",
)


def calm_stats(feats: np.ndarray) -> dict[str, np.ndarray]:
    """Mean and std per layer and channel over clips and time bins: each [layers, 1, dim] (or [dim] for flat features)."""
    x = np.asarray(feats, dtype=np.float32)
    axes = (0, 2) if x.ndim == 4 else (0,)
    mean = x.mean(axis=axes, keepdims=True)[0]
    std = np.maximum(x.std(axis=axes, keepdims=True)[0], STD_FLOOR)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply(feats: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Standardize one clip [layers, BINS, dim] or a batch [clips, layers, BINS, dim] with calm stats."""
    x = np.asarray(feats, dtype=np.float32)
    return (x - stats["mean"]) / stats["std"]


def pick_enrollment(clips: list, k: int = ENROLL_CLIPS, seed: int = 7) -> dict[tuple, list[str]]:
    """k calm or neutral clips per (corpus, speaker), drawn with a fixed seed. Speakers with fewer than k are skipped."""
    by_speaker: dict[tuple, list[str]] = {}
    for clip in clips:
        if clip.emotion in CALM_EMOTIONS:
            by_speaker.setdefault((clip.corpus, clip.speaker), []).append(str(clip.path))
    rng = random.Random(seed)
    out = {}
    for key in sorted(by_speaker):
        paths = sorted(by_speaker[key])
        if len(paths) >= k:
            out[key] = rng.sample(paths, k)
    return out


def enrolled_features(features: dict[str, np.ndarray], clips: list, enrollment: dict[tuple, list[str]]) -> dict[str, np.ndarray]:
    """Every clip of an enrolled speaker, standardized with that speaker's calm stats. Others are left out."""
    stats = {key: calm_stats(np.stack([features[r] for r in refs])) for key, refs in enrollment.items()}
    out = {}
    for clip in clips:
        speaker = (clip.corpus, clip.speaker)
        if speaker in stats:
            key = str(clip.path)
            out[key] = apply(features[key], stats[speaker]).astype(np.float16)
    return out


def scorable(clips: list, enrollment: dict[tuple, list[str]]) -> list:
    """Clips that can be scored: from an enrolled speaker and not one of that speaker's enrollment clips."""
    used = {path for paths in enrollment.values() for path in paths}
    return [c for c in clips if (c.corpus, c.speaker) in enrollment and str(c.path) not in used]


SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


class EnrollmentStore:
    """Calm readings per browser session, on disk under root/<session>.npz, so they persist across requests and restarts."""

    def __init__(self, root: Path):
        self.root = root

    def _path(self, session: str) -> Path:
        if not SESSION_ID.match(session or ""):
            raise ValueError("bad session id")
        return self.root / f"{session}.npz"

    def clips(self, session: str) -> np.ndarray | None:
        path = self._path(session)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            return data["feats"]

    def add(self, session: str, feats: np.ndarray, keep: int = 8) -> int:
        """Add one calm reading's features; keeps the most recent `keep`. Returns how many are stored."""
        old = self.clips(session)
        stacked = feats[None].astype(np.float16) if old is None else np.concatenate([old, feats[None].astype(np.float16)])[-keep:]
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(session)
        tmp = path.with_suffix(".part.npz")
        np.savez(tmp, feats=stacked)
        tmp.replace(path)
        (self.root / f"{session}.json").write_text(json.dumps({"clips": len(stacked), "updated": time.time()}))
        return len(stacked)

    def stats(self, session: str, minimum: int = MIN_LIVE_CLIPS) -> dict[str, np.ndarray] | None:
        feats = self.clips(session) if session else None
        if feats is None or len(feats) < minimum:
            return None
        return calm_stats(feats)

    def count(self, session: str) -> int:
        feats = self.clips(session) if session else None
        return 0 if feats is None else len(feats)

    def clear(self, session: str) -> None:
        for suffix in (".npz", ".json"):
            (self.root / f"{session}{suffix}").unlink(missing_ok=True)
