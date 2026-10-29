"""Combine the voice reading and the words reading into one of three bands.

  calm      the conversation continues normally
  check-in  the reading is uncertain, so the assistant asks one gentle question
  support   the reading is clearly distressed, so the full support flow starts

Each side is banded on its own calibrated probability, and the judge takes the
higher band. That is the floor the prototype already had, now with a middle
band: distress from either side is never overruled to calm, and an uncertain
reading from either side is never dropped. The voice side is banded around the
threshold chosen on validation speakers, so a clip at that threshold or above
is support and one just under it is a check-in.

The crisis screen runs before the bands. High risk bypasses the judge and the
assistant gives the 988 referral. Concern starts the support flow with a
safety question.
"""

from __future__ import annotations

from .safety import RISK_ORDER, screen

CRISIS_CARE = (
    "I'm really glad you told me. I'm not able to help with this on my own, and you deserve support from a person right now. "
    "In the US, call or text 988 to reach the Suicide and Crisis Lifeline, any time. "
    "If you are somewhere else, contact local emergency services, or reach out to someone near you."
)

BANDS = ("calm", "check-in", "support")
VOICE_CHECK_MARGIN = 0.15  # below the voice threshold by up to this much is a check-in
WORDS_SUPPORT_AT = 0.6
WORDS_CHECK_AT = 0.45


def is_crisis(text: str) -> bool:
    """Keyword floor only, without a model call."""
    return screen(text, allow_model=False)["risk"] == "high"


def voice_band(probability: float | None, threshold: float = 0.5) -> str | None:
    if probability is None:
        return None
    if probability >= threshold:
        return "support"
    if probability >= threshold - VOICE_CHECK_MARGIN:
        return "check-in"
    return "calm"


def words_band(probability: float | None) -> str | None:
    if probability is None:
        return None
    if probability >= WORDS_SUPPORT_AT:
        return "support"
    if probability >= WORDS_CHECK_AT:
        return "check-in"
    return "calm"


def _higher(*bands: str | None) -> str:
    present = [band for band in bands if band]
    return max(present, key=BANDS.index) if present else "calm"


QUESTIONS = {
    "check-in": "How are you doing right now, honestly?",
    "support": "It sounds like things might be hard right now. Do you want to tell me what's going on?",
    "concern": (
        "I want to check on something, because what you said matters to me. "
        "Are you thinking about hurting yourself or not wanting to be alive?"
    ),
}


def decide(
    voice_probability: float | None,
    language: dict,
    transcript: str,
    *,
    voice_threshold: float = 0.5,
    risk: dict | None = None,
) -> dict:
    """The written rule. No model call except the optional risk screen passed in."""
    words = transcript.strip()
    lang_probability = language.get("probability") if words else None
    vb = voice_band(voice_probability, voice_threshold)
    wb = words_band(lang_probability)
    band = _higher(vb, wb)
    level = (risk or {}).get("risk", "none")

    if level == "high":
        return {
            "band": "crisis",
            "decision": "distress",
            "confidence": 1.0,
            "reason": "The words describe a crisis, so the judge is bypassed and the assistant points to help.",
            "ask": True,
            "question": CRISIS_CARE,
            "risk": level,
            "voice_band": vb,
            "words_band": wb,
            "source": "safety",
        }

    if level == "concern":
        band = "support"
    if vb is None and wb is None:
        reason = "Nothing was read: no clip and no words."
    elif vb is None:
        reason = f"Only the words were read; they fall in the {wb} band."
    elif wb is None:
        reason = f"Only the voice was read; it falls in the {vb} band."
    elif vb == wb:
        reason = f"The voice and the words agree on {vb}."
    else:
        reason = f"The voice reads {vb} and the words read {wb}; the judge keeps the higher band, {band}."
    if level == "concern":
        reason += " The crisis screen found possible indirect risk, so the assistant asks about safety."

    strength = max(p for p in (voice_probability, lang_probability, 0.0) if p is not None)
    ask = band != "calm"
    question = QUESTIONS["concern"] if level == "concern" else QUESTIONS.get(band, "")
    return {
        "band": band,
        "decision": "calm" if band == "calm" else "distress",
        "confidence": round(strength if ask else 1 - strength, 4),
        "reason": reason,
        "ask": ask,
        "question": question,
        "risk": level,
        "voice_band": vb,
        "words_band": wb,
        "source": "policy",
    }


def adjudicate(voice_probability: float | None, language: dict, transcript: str, *, voice_threshold: float = 0.5) -> dict:
    """The live judge: crisis screen (keyword floor and risk model), then the bands."""
    risk = screen(transcript) if transcript.strip() else {"risk": "none", "source": "", "layers": []}
    verdict = decide(voice_probability, language, transcript, voice_threshold=voice_threshold, risk=risk)
    verdict["risk_source"] = risk.get("source", "")
    return verdict


__all__ = ["BANDS", "CRISIS_CARE", "RISK_ORDER", "adjudicate", "decide", "is_crisis", "voice_band", "words_band"]
