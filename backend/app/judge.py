"""Combine the voice network and the language reading.

The live judge is a language model that sees both opinions. A written floor
sits under it: clear distress in either opinion is never flipped to calm.
The cross-corpus check keeps using decide(), the written floor alone.
"""

from __future__ import annotations

import re

from .llm import complete, parse_json

CRISIS_CARE = (
    "I'm glad you said that out loud. I can't help in a crisis. "
    "In the US, call or text 988. If you are elsewhere, contact local emergency services or someone who is with you."
)


def is_crisis(text: str) -> bool:
    lowered = text.lower()
    return any(
        re.search(rf"\b{re.escape(phrase)}\b", lowered)
        for phrase in ("suicide", "kill myself", "want to die", "end my life", "hurt myself")
    )


def decide(voice_probability: float | None, language: dict, transcript: str) -> dict:
    words = transcript.strip()
    lang_probability = language.get("probability")
    voice_distress = voice_probability is not None and voice_probability >= 0.5
    words_distress = lang_probability is not None and lang_probability >= 0.6

    if not words or lang_probability is None:
        probability = voice_probability if voice_probability is not None else 0.5
        decision = "distress" if probability >= 0.5 else "calm"
        return {
            "decision": decision,
            "confidence": round(probability if decision == "distress" else 1 - probability, 4),
            "reason": "Only the voice network ran. No words were available.",
            "ask": False,
            "question": "",
        }

    if words_distress and voice_probability is None:
        decision = "distress"
        reason = "No voice clip was attached. The words mark distress, so the judge follows the words."
    elif words_distress and not voice_distress:
        decision = "distress"
        reason = (
            "The neural network heard a calm voice. The language reading still marks distress "
            "because of what was said. Speaking softly does not cancel that."
        )
    elif voice_distress and words_distress:
        decision = "distress"
        reason = "The voice network and the words agree. Both point to distress."
    elif voice_distress and not words_distress:
        decision = "distress"
        reason = (
            "The voice network heard distress, and the words did not clearly say so. "
            "The judge keeps that voice result and asks, instead of waving it off."
        )
    else:
        decision = "calm"
        if voice_probability is None:
            reason = "No voice clip was attached, and the words do not describe distress."
        else:
            reason = "The voice sounds calm, and the words do not describe distress."

    ask = decision == "distress"
    question = ""
    if ask:
        soft = (
            "Your voice sounded calm, but the words matter. "
            if words_distress and voice_probability is not None and voice_probability < 0.5
            else ""
        )
        question = f'I caught “{words}”. {soft}What is going on?'

    strength = max(voice_probability or 0, lang_probability or 0)
    confidence = strength if decision == "distress" else 1 - strength
    return {
        "decision": decision,
        "confidence": round(confidence, 4),
        "reason": reason,
        "ask": ask,
        "question": question,
        "source": "policy",
    }


def adjudicate(voice_probability: float | None, language: dict, transcript: str) -> dict:
    words = transcript.strip()
    if is_crisis(words):
        return {
            "decision": "distress",
            "confidence": 1.0,
            "reason": "The words describe a crisis. The assistant stops and points to help.",
            "ask": True,
            "question": CRISIS_CARE,
            "source": "safety",
        }

    floor = decide(voice_probability, language, transcript)
    if not words or language.get("probability") is None:
        floor["source"] = "voice network"
        return floor

    voice_label = "not run"
    if voice_probability is not None:
        voice_label = "distress" if voice_probability >= 0.5 else "calm"
    raw = complete(
        [
            {
                "role": "system",
                "content": (
                    "You are the judge between two opinions. "
                    "A neural network scored the voice. A language model scored the words. "
                    "If either opinion is distress, your decision must be distress. "
                    "A calm voice does not cancel distress in the words. "
                    "Distress in the voice is kept even when the words sound calm. "
                    "If you decide distress, ask what is going on, in one sentence, so a person can say what is wrong. "
                    "Do not diagnose, name a disorder, or give medical advice. "
                    'Return JSON with keys decision ("distress" or "calm"), confidence (0 to 1), '
                    "reason (one or two sentences that name both opinions), and question (empty when calm)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Transcript: {words}\n"
                    f"Voice network: {voice_label}"
                    + (f", probability {voice_probability:.2f}" if voice_probability is not None else "")
                    + f"\nLanguage model: {language.get('label')}, probability {language.get('probability')}. "
                    f"{language.get('note', '')}"
                ),
            },
        ],
        json_mode=True,
    )
    parsed = parse_json(raw or "")
    if not parsed:
        return floor

    decision = "distress" if str(parsed.get("decision", "")).lower() == "distress" else "calm"
    try:
        confidence = float(parsed.get("confidence", floor["confidence"]))
    except (TypeError, ValueError):
        confidence = floor["confidence"]
    confidence = min(1.0, max(0.0, confidence))
    if floor["decision"] == "distress":
        decision = "distress"
        confidence = max(confidence, float(floor["confidence"]))
    reason = str(parsed.get("reason", "")).strip() or floor["reason"]
    ask = decision == "distress"
    question = ""
    if ask:
        question = str(parsed.get("question", "")).strip() or floor["question"]
        if is_crisis(question):
            question = floor["question"] or f"I caught “{words}”. What is going on?"
    return {
        "decision": decision,
        "confidence": round(confidence, 4),
        "reason": reason,
        "ask": ask,
        "question": question,
        "source": "judge model",
    }
