from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.enrollment import drop, enrollment_paths, neutral_shift, reference


@dataclass
class Clip:
    path: Path
    corpus: str
    speaker: str
    emotion: str


def _clips():
    clips = []
    for speaker in ("01", "02"):
        for i, emotion in enumerate(("neutral", "neutral", "neutral", "angry", "sad")):
            clips.append(Clip(Path(f"/{speaker}/{i}.wav"), "RAVDESS", speaker, emotion))
    clips.append(Clip(Path("/03/0.wav"), "RAVDESS", "03", "angry"))  # never neutral
    return clips


def test_enrollment_takes_first_two_neutral_clips_per_speaker():
    enroll = enrollment_paths(_clips())
    assert enroll == {"/01/0.wav", "/01/1.wav", "/02/0.wav", "/02/1.wav"}


def test_reference_shapes():
    assert reference(np.ones((2, 13, 16, 8))).shape == (13, 1, 8)
    assert reference(np.ones((2, 88))).shape == (88,)


def test_shift_subtracts_speaker_reference_and_drop_leaves_scorable_clips():
    clips = _clips()
    features = {str(c.path): np.full((3, 4, 5), float(c.speaker), dtype=np.float16) for c in clips}
    features["/01/3.wav"] = np.full((3, 4, 5), 5.0, dtype=np.float16)
    enroll = enrollment_paths(clips)
    shifted = neutral_shift(features, clips, enroll)
    assert "/03/0.wav" not in shifted
    assert np.allclose(shifted["/01/3.wav"], 4.0) and shifted["/01/3.wav"].dtype == np.float16
    assert np.allclose(shifted["/02/4.wav"], 0.0)
    kept = {str(c.path) for c in drop(clips, enroll, shifted)}
    assert kept == {"/01/2.wav", "/01/3.wav", "/01/4.wav", "/02/2.wav", "/02/3.wav", "/02/4.wav"}
