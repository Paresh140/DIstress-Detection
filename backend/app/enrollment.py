"""Neutral-speech enrollment: normalize each speaker against a few of their own calm clips.

Speaker normalization was the biggest cross-corpus gain, but it needs many
clips from the speaker, of every emotion. A live user only gives a short calm
reading. So each speaker's features are shifted by the mean of K of their own
neutral clips (the enrollment), which is exactly what the studio can collect.

In evaluation the enrollment clips are taken out of every split, so no clip is
both a reference and a scored example. Using the neutral label to pick them
mirrors the live setting, where the person is asked to read calmly.
"""

from __future__ import annotations

import numpy as np

ENROLL_CLIPS = 2


def enrollment_paths(clips: list, k: int = ENROLL_CLIPS) -> set[str]:
    """The first k neutral clips of every speaker (sorted by path, so the choice is fixed)."""
    by_speaker: dict[tuple, list[str]] = {}
    for clip in clips:
        if clip.emotion == "neutral":
            by_speaker.setdefault((clip.corpus, clip.speaker), []).append(str(clip.path))
    return {path for paths in by_speaker.values() for path in sorted(paths)[:k]}


def reference(feats: np.ndarray) -> np.ndarray:
    """Mean over enrollment clips (and time bins), one vector per layer: [layers, 1, dim] or [features]."""
    x = np.asarray(feats, dtype=np.float32)
    return x.mean(axis=(0, 2), keepdims=True)[0] if x.ndim == 4 else x.mean(axis=0)


def neutral_shift(features: dict[str, np.ndarray], clips: list, enroll: set[str]) -> dict[str, np.ndarray]:
    """Subtract each speaker's enrollment mean from all of that speaker's clips."""
    groups: dict[tuple, list] = {}
    for clip in clips:
        groups.setdefault((clip.corpus, clip.speaker), []).append(clip)
    out = {}
    for members in groups.values():
        refs = [features[str(c.path)] for c in members if str(c.path) in enroll]
        if not refs:
            continue  # a speaker with no neutral clip cannot be enrolled and is left out
        ref = reference(np.stack(refs))
        for c in members:
            key = str(c.path)
            out[key] = (features[key].astype(np.float32) - ref).astype(features[key].dtype)
    return out


def drop(clips: list, enroll: set[str], features: dict[str, np.ndarray]) -> list:
    """Clips that can be scored: not an enrollment clip, and from an enrolled speaker."""
    return [c for c in clips if str(c.path) not in enroll and str(c.path) in features]
