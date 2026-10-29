"""Speech to text with faster-whisper, so one clip gives the voice reading and the words.

The model loads once, on the CPU with int8 weights. WHISPER_MODEL picks the
size (default "base"; "small" is more accurate and about three times slower).
If faster-whisper is not installed, transcribe() returns None and the studio
falls back to the transcript the browser typed or the person entered.
"""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _model():
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    return WhisperModel(os.environ.get("WHISPER_MODEL", "base"), device="cpu", compute_type="int8")


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def transcribe(path: str) -> dict | None:
    model = _model()
    if model is None:
        return None
    segments, info = model.transcribe(path, language="en", beam_size=1, vad_filter=True)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return {"text": text, "model": f"whisper-{os.environ.get('WHISPER_MODEL', 'base')}", "duration": round(float(info.duration), 2)}
