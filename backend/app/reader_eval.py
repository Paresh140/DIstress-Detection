"""Compare the words readers on 200 written, labeled sentences.

backend/eval/reader_sentences.jsonl has 100 calm and 100 distressed sentences
in eight groups of 25: plainly calm, neutral statements, calm sentences with
emotional words in them ("I cried at the wedding because I was so happy"),
mildly negative but calm, plainly distressed, indirect distress, distress
described through the body, and understated distress.

Readers:
  phrases  the written phrase list (language.rule_read)
  llm      the language-model reader the studio uses (needs OPENAI_API_KEY or Ollama)
  roberta  SamLowe/roberta-base-go_emotions; P(distress) is the summed
           probability of the negative GoEmotions labels

Usage, from backend/:
  python -m app.reader_eval
  python -m app.reader_eval --readers phrases roberta

Writes artifacts/reader_report.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import llm
from .calibration import metrics_at
from .language import _llm_read, rule_read

ROOT = Path(__file__).resolve().parents[2]
SET = ROOT / "backend" / "eval" / "reader_sentences.jsonl"
ARTIFACTS = ROOT / "backend" / "artifacts"
ROBERTA = "SamLowe/roberta-base-go_emotions"
NEGATIVE = (
    "sadness",
    "grief",
    "fear",
    "nervousness",
    "anger",
    "annoyance",
    "disappointment",
    "remorse",
    "disgust",
    "embarrassment",
)


def phrases(texts: list[str]) -> list[float]:
    return [float(rule_read(text)["probability"]) for text in texts]


def language_model(texts: list[str]) -> list[float] | None:
    if not llm.available():
        return None
    out = []
    for i, text in enumerate(texts, start=1):
        scored = _llm_read(text)
        out.append(float(scored["probability"]) if scored else 0.5)
        if i % 50 == 0:
            print(f"  llm {i}/{len(texts)}")
    return out


def roberta(texts: list[str]) -> list[float]:
    from transformers import pipeline

    classify = pipeline("text-classification", model=ROBERTA, top_k=None)
    out = []
    for scores in classify(texts, batch_size=16, truncation=True):
        by_label = {item["label"]: item["score"] for item in scores}
        out.append(float(min(1.0, sum(by_label.get(label, 0.0) for label in NEGATIVE))))
    return out


READERS = {"phrases": phrases, "llm": language_model, "roberta": roberta}


def run(readers: tuple[str, ...] = ("phrases", "llm", "roberta")) -> dict:
    rows = [json.loads(line) for line in SET.read_text().splitlines() if line.strip()]
    texts = [row["text"] for row in rows]
    truth = np.array([row["label"] for row in rows])
    categories = sorted({row["category"] for row in rows})
    report: dict = {"sentences": len(rows), "threshold": 0.5, "readers": {}}
    for name in readers:
        print(f"\n{name}")
        probs = READERS[name](texts)
        if probs is None:
            print("  skipped: no language model is available")
            continue
        prob = np.array(probs)
        overall = metrics_at(truth, prob, 0.5)
        per_category = {}
        for cat in categories:
            mask = np.array([row["category"] == cat for row in rows])
            per_category[cat] = round(float(((prob[mask] >= 0.5).astype(int) == truth[mask]).mean()), 4)
        report["readers"][name] = {
            "balanced_accuracy": round(overall["balanced_accuracy"], 4),
            "macro_f1": round(overall["macro_f1"], 4),
            "auroc": round(overall["auroc"], 4) if overall["auroc"] is not None else None,
            "recall": round(overall["recall"], 4),
            "specificity": round(overall["specificity"], 4),
            "accuracy_by_category": per_category,
            "predictions": [{"text": t, "label": int(y), "probability": round(float(p), 4)} for t, y, p in zip(texts, truth, prob)],
        }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "reader_report.json").write_text(json.dumps(report, indent=2))

    names = list(report["readers"])
    print("\n" + "".ljust(22) + "".join(f"{n:>10}" for n in names))
    for key in ("balanced_accuracy", "macro_f1", "auroc", "recall", "specificity"):
        print(key.ljust(22) + "".join(f"{report['readers'][n][key] or 0:>10.3f}" for n in names))
    print("accuracy by group")
    for cat in categories:
        print(f"  {cat:<20}" + "".join(f"{report['readers'][n]['accuracy_by_category'][cat]:>10.2f}" for n in names))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--readers", nargs="+", choices=sorted(READERS), default=["phrases", "llm", "roberta"])
    args = parser.parse_args()
    run(tuple(args.readers))
