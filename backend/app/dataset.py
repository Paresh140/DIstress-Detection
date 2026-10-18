"""RAVDESS speech files and the distress label used in this project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# RAVDESS emotion codes.
NEUTRAL = 1
CALM = 2
HAPPY = 3
SAD = 4
ANGRY = 5
FEARFUL = 6
DISGUST = 7
SURPRISED = 8

DISTRESS_EMOTIONS = {SAD, ANGRY, FEARFUL}
CALM_EMOTIONS = {NEUTRAL, CALM}
EMOTION_NAMES = {
    NEUTRAL: "neutral",
    CALM: "calm",
    HAPPY: "happy",
    SAD: "sad",
    ANGRY: "angry",
    FEARFUL: "fearful",
    DISGUST: "disgust",
    SURPRISED: "surprised",
}

TRAIN_ACTORS = set(range(1, 7))
TEST_ACTORS = {7, 8}


@dataclass(frozen=True)
class Clip:
    path: Path
    actor: int
    emotion: int
    intensity: int
    label: int  # 1 distress, 0 calm

    @property
    def emotion_name(self) -> str:
        return EMOTION_NAMES[self.emotion]

    @property
    def split(self) -> str:
        if self.actor in TRAIN_ACTORS:
            return "train"
        if self.actor in TEST_ACTORS:
            return "test"
        return "unused"


def parse_clip(path: Path) -> Clip | None:
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    try:
        modality, channel, emotion, intensity, _statement, _rep, actor = (int(p) for p in parts)
    except ValueError:
        return None
    if modality != 3 or channel != 1:
        return None
    if emotion in DISTRESS_EMOTIONS:
        label = 1
    elif emotion in CALM_EMOTIONS:
        label = 0
    else:
        return None
    return Clip(path=path, actor=actor, emotion=emotion, intensity=intensity, label=label)


def discover(root: Path) -> list[Clip]:
    clips = []
    for path in sorted(root.rglob("*.wav")):
        clip = parse_clip(path)
        if clip is not None:
            clips.append(clip)
    return clips
