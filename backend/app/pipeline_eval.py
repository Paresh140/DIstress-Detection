"""End-to-end judge experiments: the mismatch scenario and the joint false-alarm rate.

1. Mismatch (audit C6, paper Section 6). Real voice readings are paired with
   written sentences in four cells:
     calm text + calm voice, calm text + distressed voice,
     distressed text + calm voice, distressed text + distressed voice
   Each pair goes through the live decision path: voice band -> words reader
   -> crisis screen -> judge (decide, the max rule). The report has the cell x
   band table, how often the one distressed modality raised the band on its own
   when the other was calm, and whether the final band was ever below the
   higher of the two side bands (the non-canceling rule).

2. Joint false alarms (paper Section 6). The 55 harmless safety messages
   (venting, sarcasm, false triggers) are paired with calm voice clips and put
   through the same path. The share that end at check-in or support is the
   joint false-alarm rate, next to words alone and voice alone, which shows how
   much the max rule adds.

Voice readings: the calibrated P(distress) the live head gave each held-out
RAVDESS test clip (actors 21-24), read from artifacts/voice_report.json,
with the head's validation threshold. These are in-corpus studio clips, the
best case for the voice; a live microphone is harder (see enrollment_report).

Words readers: phrases (the live fallback, computed here), roberta (offline
reference, probabilities from reader_report.json for the 200 sentences;
computed when transformers is installed for the safety messages), llm (the
live reader; from reader_report.json or computed when a model is available).
The crisis screen is the keyword floor plus the risk model when one is
available, otherwise the indirect phrase list.

Usage, from backend/:
  python -m app.pipeline_eval                    # phrases (+ llm/roberta where available)
  python -m app.pipeline_eval --readers phrases llm --pairs 50

Writes artifacts/pipeline_report.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import llm
from .judge import BANDS, decide, voice_band, words_band
from .language import _llm_read, rule_read
from .safety import screen

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "backend" / "artifacts"
EVAL = ROOT / "backend" / "eval"
HARMLESS = ("venting", "sarcasm", "false_trigger")
CELLS = (
    ("calm text", "calm voice"),
    ("calm text", "distressed voice"),
    ("distressed text", "calm voice"),
    ("distressed text", "distressed voice"),
)
OUTCOMES = (*BANDS, "crisis")
SEED = 7


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def voice_readings(path: Path = ARTIFACTS / "voice_report.json") -> tuple[list[dict], float, str]:
    report = json.loads(path.read_text())
    return report["predictions"], float(report["summary"]["threshold"]), f"{report['summary']['encoder']} + {report['summary']['head']}, {report['summary']['test']}"


class Reader:
    """P(distress) for a text from one words reader, using saved probabilities where they exist."""

    def __init__(self, name: str, saved: dict[str, float] | None = None):
        self.name = name
        self.saved = saved or {}
        self.failed = 0

    def prepare(self, texts: list[str]) -> bool:
        """Make sure every text can be read. RoBERTa is run once in a batch; False when it cannot run here."""
        if self.name != "roberta":
            return True
        missing = sorted({t for t in texts if t not in self.saved})
        if not missing:
            return True
        try:
            from .reader_eval import roberta

            self.saved.update(zip(missing, roberta(missing)))
        except ImportError:
            return False
        return True

    def __call__(self, text: str) -> float | None:
        if text in self.saved:
            return self.saved[text]
        if self.name == "phrases":
            return float(rule_read(text)["probability"])
        if self.name == "llm":
            scored = _llm_read(text)
            if scored is None:
                self.failed += 1
                scored = rule_read(text)
            return float(scored["probability"])
        raise ValueError(self.name)


def available_readers(requested: tuple[str, ...]) -> dict[str, Reader]:
    saved = {}
    path = ARTIFACTS / "reader_report.json"
    if path.is_file():
        for name, block in json.loads(path.read_text()).get("readers", {}).items():
            saved[name] = {row["text"]: float(row["probability"]) for row in block.get("predictions", [])}
    out = {}
    for name in requested:
        if name == "llm" and not llm.available() and "llm" not in saved:
            print("llm: no language model and no saved llm readings, skipped")
            continue
        out[name] = Reader(name, saved.get(name) if name != "phrases" else None)
    return out


def judge_pair(voice_p: float | None, threshold: float, text: str, reader: Reader, risk: dict) -> dict:
    words_p = reader(text)
    verdict = decide(voice_p, {"probability": words_p}, text, voice_threshold=threshold, risk=risk)
    vb, wb = voice_band(voice_p, threshold), words_band(words_p)
    side_max = max((b for b in (vb, wb) if b), key=BANDS.index, default="calm")
    final = verdict["band"]
    held = final == "crisis" or BANDS.index(final) >= BANDS.index(side_max)
    return {"voice_p": voice_p, "words_p": round(words_p, 4) if words_p is not None else None, "voice_band": vb, "words_band": wb, "band": final, "risk": risk["risk"], "non_canceling_held": held}


def _risk_cache(texts: list[str], allow_model: bool) -> dict[str, dict]:
    return {t: screen(t, allow_model=allow_model) for t in sorted(set(texts))}


def mismatch(sentences: list[dict], voices: list[dict], threshold: float, readers: dict[str, Reader], pairs: int, allow_model: bool) -> dict:
    rng = random.Random(SEED)
    calm_text = [s["text"] for s in sentences if s["label"] == 0]
    distress_text = [s["text"] for s in sentences if s["label"] == 1]
    calm_voice = [v for v in voices if v["truth"] == "calm"]
    distress_voice = [v for v in voices if v["truth"] == "distress"]
    plan = []
    for text_kind, voice_kind in CELLS:
        texts = calm_text if text_kind == "calm text" else distress_text
        clips = calm_voice if voice_kind == "calm voice" else distress_voice
        for t, v in zip(rng.sample(texts, min(pairs, len(texts))), rng.sample(clips, min(pairs, len(clips)))):
            plan.append((f"{text_kind} + {voice_kind}", t, v))
    risks = _risk_cache([t for _, t, _ in plan], allow_model)
    out = {}
    for name, reader in readers.items():
        if not reader.prepare([t for _, t, _ in plan]):
            print(f"{name}: cannot run here (transformers missing) and no saved readings; skipped in the mismatch table")
            continue
        rows = []
        for cell, text, clip in plan:
            row = judge_pair(clip["probability"], threshold, text, reader, risks[text])
            rows.append({"cell": cell, "text": text, "clip": clip["file"], "emotion": clip["emotion"], **row})
        table = {}
        for text_kind, voice_kind in CELLS:
            cell = f"{text_kind} + {voice_kind}"
            picked = [r for r in rows if r["cell"] == cell]
            table[cell] = {band: sum(r["band"] == band for r in picked) for band in OUTCOMES} | {"pairs": len(picked)}

        def raised(cell: str, side: str) -> dict:
            picked = [r for r in rows if r["cell"] == cell]
            by_side = sum(r[side] in ("check-in", "support") for r in picked)
            final = sum(r["band"] != "calm" for r in picked)
            screen_only = sum(r["risk"] != "none" and r[side] not in ("check-in", "support") for r in picked)
            return {"pairs": len(picked), "raised_by_that_modality": by_side, "raised_by_crisis_screen_only": screen_only, "final_not_calm": final}

        out[name] = {
            "table": table,
            "voice_only_distress": raised("calm text + distressed voice", "voice_band"),
            "words_only_distress": raised("distressed text + calm voice", "words_band"),
            "calm_calm_false_alarms": sum(r["band"] != "calm" for r in rows if r["cell"] == "calm text + calm voice"),
            "non_canceling_violations": sum(not r["non_canceling_held"] for r in rows),
            "pairs": rows,
        }
        if reader.failed:
            out[name]["llm_failed_calls"] = reader.failed
    return out


def false_alarms(safety: list[dict], voices: list[dict], threshold: float, readers: dict[str, Reader], allow_model: bool) -> dict:
    rng = random.Random(SEED + 1)
    harmless = [r for r in safety if r["category"] in HARMLESS]
    calm_voice = [v for v in voices if v["truth"] == "calm"]
    risks = _risk_cache([r["text"] for r in harmless], allow_model)
    out = {}
    for name, reader in readers.items():
        if not reader.prepare([r["text"] for r in harmless]):
            print(f"{name}: cannot run here (transformers missing); skipped for the harmless messages")
            continue
        rows = []
        for msg in harmless:
            clip = rng.choice(calm_voice)
            joint = judge_pair(clip["probability"], threshold, msg["text"], reader, risks[msg["text"]])
            words = judge_pair(None, threshold, msg["text"], reader, risks[msg["text"]])
            rows.append({"text": msg["text"], "category": msg["category"], "clip": clip["file"], "joint": joint["band"], "words_only": words["band"], "voice_band": joint["voice_band"], "words_band": joint["words_band"], "risk": joint["risk"]})

        def rate(key: str, picked: list[dict]) -> dict:
            hits = sum(r[key] != "calm" for r in picked)
            return {"flagged": hits, "messages": len(picked), "rate": round(hits / max(len(picked), 1), 4), "bands": {b: sum(r[key] == b for r in picked) for b in OUTCOMES}}

        voice_alone = sum((r["voice_band"] or "calm") != "calm" for r in rows)
        out[name] = {
            "joint": rate("joint", rows),
            "words_only": rate("words_only", rows),
            "voice_only": {"flagged": voice_alone, "messages": len(rows), "rate": round(voice_alone / max(len(rows), 1), 4)},
            "by_category": {cat: {"joint": rate("joint", [r for r in rows if r["category"] == cat]), "words_only": rate("words_only", [r for r in rows if r["category"] == cat])} for cat in HARMLESS},
            "messages": rows,
        }
    return out


def run(readers: tuple[str, ...] = ("phrases", "roberta", "llm"), pairs: int = 50) -> dict:
    voices, threshold, voice_model = voice_readings()
    chosen = available_readers(readers)
    allow_model = bool(llm.available())
    report = {
        "voice": {"model": voice_model, "threshold": threshold, "clips": len(voices), "source": "artifacts/voice_report.json predictions (calibrated, held-out actors)"},
        "crisis_screen": "keyword floor + " + (f"risk model ({llm.available()})" if allow_model else "indirect phrase list (no model available)"),
        "readers": list(chosen),
        "mismatch": mismatch(_jsonl(EVAL / "reader_sentences.jsonl"), voices, threshold, chosen, pairs, allow_model),
        "false_alarms": false_alarms(_jsonl(EVAL / "safety_set.jsonl"), voices, threshold, chosen, allow_model),
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "pipeline_report.json").write_text(json.dumps(report, indent=2))
    print(f"voice: {voice_model}, threshold {threshold}; crisis screen: {report['crisis_screen']}")
    for name in chosen:
        if name not in report["mismatch"]:
            continue
        m = report["mismatch"][name]
        print(f"\n[{name}] mismatch: pair type x final band")
        print("".ljust(38) + "".join(f"{b:>10}" for b in OUTCOMES))
        for cell, counts in m["table"].items():
            print(cell.ljust(38) + "".join(f"{counts[b]:>10}" for b in OUTCOMES))
        v, w = m["voice_only_distress"], m["words_only_distress"]
        print(
            f"  voice alone raised the band: {v['raised_by_that_modality']}/{v['pairs']}   words alone raised it: {w['raised_by_that_modality']}/{w['pairs']}"
            f" (+{w['raised_by_crisis_screen_only']} by the crisis screen's concern label)"
        )
        print(f"  calm+calm false alarms: {m['calm_calm_false_alarms']}   non-canceling violations: {m['non_canceling_violations']}")
        f = report["false_alarms"].get(name)
        if f is None:
            continue
        print(f"  harmless messages flagged check-in or above: joint {f['joint']['flagged']}/{f['joint']['messages']}  words only {f['words_only']['flagged']}  voice only {f['voice_only']['flagged']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--readers", nargs="+", choices=("phrases", "roberta", "llm"), default=["phrases", "roberta", "llm"])
    parser.add_argument("--pairs", type=int, default=50, help="pairs per cell")
    args = parser.parse_args()
    run(tuple(args.readers), args.pairs)
