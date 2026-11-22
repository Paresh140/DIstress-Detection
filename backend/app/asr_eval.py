"""Word error rate of the live speech-to-text on acted speech (audit C8).

RAVDESS and CREMA-D read fixed sentences, so every clip has a ground-truth
transcript (app/corpora.py). This transcribes a balanced sample of calm and
distressed clips with the live settings (faster-whisper, WHISPER_MODEL, default
base, int8 on the CPU) and reports WER overall, calm vs distressed, and per
emotion. The crisis screen and the words reader run on this text, so a higher
WER on distressed speech would matter more than the overall number.

Normalization before scoring: lower case, punctuation removed, apostrophes
dropped ("it's" -> "its"), numbers 0-20 spelled out ("11" -> "eleven").

Usage, from backend/:
  python -m app.asr_eval                  # 25 calm + 25 distressed clips per corpus present
  python -m app.asr_eval --per-class 40

Writes artifacts/asr_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np

from .corpora import discover

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "backend" / "artifacts"
SEED = 7
NUMBERS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()


def normalize(text: str) -> list[str]:
    text = text.lower().replace("’", "'").replace("'", "")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    words = []
    for word in text.split():
        if word.isdigit() and int(word) <= 20:
            words.append(NUMBERS[int(word)])
        else:
            words.append(word)
    return words


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, start=1):
            cur = min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
            prev, row[j] = row[j], cur
    return row[-1]


def wer(pairs: list[tuple[str, str]]) -> dict:
    errors = words = exact = 0
    for ref, hyp in pairs:
        r, h = normalize(ref), normalize(hyp)
        e = edit_distance(r, h)
        errors += e
        words += len(r)
        exact += e == 0
    return {"wer": round(errors / max(words, 1), 4), "errors": errors, "reference_words": words, "clips": len(pairs), "exact_match": round(exact / max(len(pairs), 1), 4)}


def sample(clips: list, per_class: int, seed: int = SEED) -> list:
    rng = random.Random(seed)
    calm = [c for c in clips if c.label == 0]
    distress = [c for c in clips if c.label == 1]
    return rng.sample(calm, min(per_class, len(calm))) + rng.sample(distress, min(per_class, len(distress)))


def run(per_class: int = 25) -> dict:
    from . import transcribe as stt

    if not stt.available():
        raise SystemExit("faster-whisper is not installed (pip install faster-whisper)")
    corpora = {name: discover(ROOT / "data" / folder, name) for name, folder in (("RAVDESS", "ravdess"), ("CREMA-D", "crema-d"))}
    rows = []
    for name, clips in corpora.items():
        if not clips:
            continue
        for clip in sample(clips, per_class):
            start = time.perf_counter()
            out = stt.transcribe(str(clip.path)) or {"text": ""}
            seconds = time.perf_counter() - start
            rows.append(
                {
                    "corpus": name,
                    "file": clip.path.name,
                    "emotion": clip.emotion,
                    "truth": "distress" if clip.label else "calm",
                    "reference": clip.sentence,
                    "hypothesis": out["text"],
                    "seconds": round(seconds, 3),
                }
            )
            print(f"  [{name} {clip.emotion:<8}] {clip.sentence!r} -> {out['text']!r}")
    if not rows:
        raise SystemExit("No RAVDESS or CREMA-D clips under data/")

    def pick(test) -> list[tuple[str, str]]:
        return [(r["reference"], r["hypothesis"]) for r in rows if test(r)]

    report = {
        "model": f"faster-whisper {os.environ.get('WHISPER_MODEL', 'base')}, int8, CPU, beam 1, VAD on",
        "normalization": "lower case, punctuation removed, apostrophes dropped, numbers 0-20 spelled out",
        "overall": wer(pick(lambda r: True)),
        "calm": wer(pick(lambda r: r["truth"] == "calm")),
        "distress": wer(pick(lambda r: r["truth"] == "distress")),
        "by_corpus": {
            name: {
                "overall": wer(pick(lambda r, n=name: r["corpus"] == n)),
                "calm": wer(pick(lambda r, n=name: r["corpus"] == n and r["truth"] == "calm")),
                "distress": wer(pick(lambda r, n=name: r["corpus"] == n and r["truth"] == "distress")),
            }
            for name in sorted({r["corpus"] for r in rows})
        },
        "by_emotion": {e: wer(pick(lambda r, e=e: r["emotion"] == e)) for e in sorted({r["emotion"] for r in rows})},
        "seconds_per_clip": {"median": round(float(np.median([r["seconds"] for r in rows])), 3), "max": round(max(r["seconds"] for r in rows), 3)},
        "clips": rows,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "asr_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n{report['model']}")
    for key in ("overall", "calm", "distress"):
        print(f"  {key:<9} WER {report[key]['wer']:.3f}  exact {report[key]['exact_match']:.2f}  ({report[key]['clips']} clips)")
    for emotion, block in report["by_emotion"].items():
        print(f"  {emotion:<9} WER {block['wer']:.3f}  ({block['clips']} clips)")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=25, help="calm and distressed clips per corpus")
    run(parser.parse_args().per_class)
