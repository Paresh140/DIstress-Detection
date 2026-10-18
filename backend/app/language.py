"""Read distress from what was said.

The audio network never sees these words. This reader is the second opinion.
A language model scores the sentence when one is available. Otherwise a written
phrase list does, so a line like "I am not feeling well today" still counts.
"""

from __future__ import annotations

from .llm import complete, parse_json

DISTRESS_PHRASES = (
    "not feeling well",
    "not feeling good",
    "don't feel well",
    "do not feel well",
    "dont feel well",
    "i am not well",
    "i'm not well",
    "im not well",
    "not well",
    "feeling unwell",
    "feel unwell",
    "feel sick",
    "feeling sick",
    "not okay",
    "not ok",
    "i'm hurting",
    "i am hurting",
    "in pain",
    "it hurts",
    "can't breathe",
    "cannot breathe",
    "help me",
    "scared",
    "afraid",
    "anxious",
    "panicking",
    "panic attack",
    "depressed",
    "hopeless",
    "want to die",
    "kill myself",
    "hurt myself",
)

CALM_PHRASES = (
    "feeling well",
    "i feel fine",
    "i'm fine",
    "i am fine",
    "i'm okay",
    "i am okay",
    "i feel okay",
    "all good",
)


def rule_read(transcript: str) -> dict:
    text = " ".join(transcript.lower().split())
    hit = next((phrase for phrase in DISTRESS_PHRASES if phrase in text), "")
    if hit:
        return {
            "label": "distress",
            "probability": 0.9,
            "source": "phrase list",
            "evidence": hit,
            "note": f"The words include “{hit}”.",
        }
    calm = next((phrase for phrase in CALM_PHRASES if phrase in text), "")
    if calm:
        return {
            "label": "calm",
            "probability": 0.12,
            "source": "phrase list",
            "evidence": calm,
            "note": f"The words include “{calm}”.",
        }
    if text:
        return {
            "label": "calm",
            "probability": 0.2,
            "source": "phrase list",
            "evidence": "",
            "note": "No distress phrase was found in the words.",
        }
    return {
        "label": None,
        "probability": None,
        "source": "none",
        "evidence": "",
        "note": "No words were captured.",
    }


def _llm_read(transcript: str) -> dict | None:
    raw = complete(
        [
            {
                "role": "system",
                "content": (
                    "You score whether a person's own words describe distress. "
                    "Distress includes feeling unwell, sick, scared, in pain, or asking for help. "
                    "A soft or calm tone is irrelevant because you only see text. "
                    "Return JSON with keys label (distress or calm), probability (0 to 1, "
                    "higher means more distress), and note (one sentence). No advice."
                ),
            },
            {"role": "user", "content": transcript},
        ],
        json_mode=True,
    )
    parsed = parse_json(raw or "")
    if not parsed:
        return None
    label = "distress" if str(parsed.get("label", "")).lower() == "distress" else "calm"
    try:
        probability = float(parsed.get("probability", 0.5))
    except (TypeError, ValueError):
        probability = 0.5
    probability = min(1.0, max(0.0, probability))
    if label == "calm":
        probability = min(probability, 0.45)
    return {
        "label": label,
        "probability": round(probability, 4),
        "source": "language model",
        "evidence": transcript.strip(),
        "note": str(parsed.get("note", "")).strip() or "The language model scored the words.",
    }


def read_language(transcript: str, *, allow_model: bool = True) -> dict:
    text = transcript.strip()
    if not text:
        return rule_read("")
    if allow_model:
        scored = _llm_read(text)
        if scored is not None:
            return scored
        fallback = rule_read(text)
        if fallback.get("label"):
            fallback["note"] = fallback["note"] + " The language model was unavailable, so the phrase list was used."
        return fallback
    return rule_read(text)
