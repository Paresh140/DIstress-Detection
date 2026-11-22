"""Time one end-to-end reading on the live stack (practical deployment gap).

Stages, in the order the studio runs them for one spoken turn:
  asr          faster-whisper transcription (WHISPER_MODEL, int8, CPU)
  voice        WavLM-base-plus features + head (artifacts/voice_head.pt)
  reader       words reader (language model when available, else phrase list)
  crisis       crisis screen (keyword floor + risk model when available)
  judge        the band rule
Each stage is timed separately over --runs repeats after one warm-up, on one
clip (--clip, default: a RAVDESS test clip if present, else 4 s of noise).
Stages that cannot run on this machine are reported as skipped.

Usage, from backend/:
  python -m app.latency_eval
  python -m app.latency_eval --clip some.wav --runs 10

Writes artifacts/latency_report.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import torch

from . import llm
from .judge import decide
from .language import read_language
from .safety import screen

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "backend" / "artifacts"
TEXT = "I haven't really been sleeping and everything feels like too much lately."


def _noise_clip(seconds: float = 4.0) -> Path:
    path = Path(tempfile.gettempdir()) / "latency_noise.wav"
    y = (np.random.default_rng(0).normal(0, 0.05, int(16000 * seconds)) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(y.tobytes())
    return path


def _time(fn, runs: int) -> dict:
    fn()  # warm-up (model loads, caches)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return {"median_s": round(float(np.median(times)), 4), "p90_s": round(float(np.percentile(times, 90)), 4), "runs": runs}


def run(clip: Path | None = None, runs: int = 5, out: str = "latency_report.json") -> dict:
    if clip is None:
        found = sorted((ROOT / "data" / "ravdess").rglob("03-01-04-*-21.wav"))
        clip = found[0] if found else _noise_clip()
    stages: dict[str, dict] = {}

    from . import transcribe as stt

    if stt.available():
        stages["asr"] = _time(lambda: stt.transcribe(str(clip)), runs)
    else:
        stages["asr"] = {"skipped": "faster-whisper not installed"}

    head = ARTIFACTS / "voice_head.pt"
    try:
        from .voice import VoiceModel

        model = VoiceModel(head)
        stages["voice"] = _time(lambda: model.score(str(clip)), runs)
        stages["voice"]["device"] = str(model.dev)
    except Exception as exc:  # noqa: BLE001
        stages["voice"] = {"skipped": f"voice model could not load here: {type(exc).__name__}: {exc}"[:200]}

    kind = llm.available() or "none"
    stages["reader"] = {**_time(lambda: read_language(TEXT), runs), "reader": kind if kind != "none" else "phrase list"}
    stages["crisis"] = {**_time(lambda: screen(TEXT), runs), "screen": "keyword floor + " + (kind if kind != "none" else "phrase list")}
    language = read_language(TEXT)
    risk = screen(TEXT)
    stages["judge"] = _time(lambda: decide(0.6, language, TEXT, voice_threshold=0.5, risk=risk), runs)

    timed = [s["median_s"] for s in stages.values() if "median_s" in s]
    report = {
        "machine": f"{platform.system()} {platform.machine()}, {platform.processor() or 'cpu'}",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "clip": clip.name,
        "stages": stages,
        "total_median_s": round(sum(timed), 4),
        "complete": all("median_s" in s for s in stages.values()),
        "note": "Stages run one after another in the studio, so the total is the sum of the medians. ASR and the voice model can run in parallel; they do not today.",
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / out).write_text(json.dumps(report, indent=2))
    for name, s in stages.items():
        print(f"  {name:<8} " + (f"{s['median_s'] * 1000:8.1f} ms median" if "median_s" in s else f"skipped: {s['skipped']}"))
    print(f"  total    {report['total_median_s'] * 1000:8.1f} ms" + ("" if report["complete"] else " (incomplete: some stages skipped)"))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", default="latency_report.json", help="report name under artifacts/")
    args = parser.parse_args()
    run(args.clip, args.runs, args.out)
