"""Shared distress labels for RAVDESS and CREMA-D.

IEMOCAP is not loaded. It is released under a signed USC SAIL license and is
not in this workspace. The audio study therefore uses the planned fallback:
RAVDESS and CREMA-D, in both directions.

Distress is sad, angry or fearful. Non-distress is neutral or calm, the same
map the Monitor demo and the Colab run use. Calm only occurs in RAVDESS.
Happy, disgust, and surprise are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

RAVDESS_SENTENCES = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}

CREMA_SENTENCES = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

# Names shared by both corpora.
DISTRESS = {"sad", "angry", "fearful"}
NON_DISTRESS = {"neutral", "calm"}

RAVDESS_EMOTION = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}
CREMA_EMOTION = {
    "NEU": "neutral",
    "CAL": "calm",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fearful",
    "DIS": "disgust",
    "SUR": "surprised",
}


@dataclass(frozen=True)
class Example:
    path: Path
    corpus: str
    speaker: str
    emotion: str
    sentence: str
    label: int  # 1 distress, 0 non-distress


def _label_for(emotion: str) -> int | None:
    if emotion in DISTRESS:
        return 1
    if emotion in NON_DISTRESS:
        return 0
    return None


def parse_ravdess(path: Path) -> Example | None:
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    try:
        modality, channel, emotion_code, _intensity, statement, _rep, actor = (int(part) for part in parts)
    except ValueError:
        return None
    if modality != 3 or channel != 1:
        return None
    emotion = RAVDESS_EMOTION.get(emotion_code)
    label = _label_for(emotion or "")
    if emotion is None or label is None:
        return None
    sentence = RAVDESS_SENTENCES.get(f"{statement:02d}", f"statement-{statement:02d}")
    return Example(path, "RAVDESS", f"{actor:02d}", emotion, sentence, label)


def parse_crema(path: Path) -> Example | None:
    parts = path.stem.split("_")
    if len(parts) < 4:
        return None
    speaker, sentence_code, emotion_code = parts[0], parts[1], parts[2]
    emotion = CREMA_EMOTION.get(emotion_code)
    label = _label_for(emotion or "")
    if emotion is None or label is None:
        return None
    sentence = CREMA_SENTENCES.get(sentence_code, sentence_code)
    return Example(path, "CREMA-D", speaker, emotion, sentence, label)


def discover(root: Path, corpus: str) -> list[Example]:
    parser = parse_ravdess if corpus == "RAVDESS" else parse_crema
    clips = []
    for path in sorted(root.rglob("*.wav")):
        clip = parser(path)
        if clip is not None:
            clips.append(clip)
    return clips


def sentence_coverage(clips: list[Example]) -> dict:
    """Show that each fixed sentence appears under both labels."""
    by_sentence: dict[str, set[str]] = {}
    for clip in clips:
        by_sentence.setdefault(clip.sentence, set()).add(clip.emotion)
    return {
        sentence: sorted(emotions)
        for sentence, emotions in sorted(by_sentence.items())
    }
