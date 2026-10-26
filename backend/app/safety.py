"""Layered crisis screen, run on every message a person sends.

Layer 1 is a written keyword floor. It catches explicit statements of
suicide or self-harm and cannot be overruled. Layer 2 is a language model
that labels each message none, concern or high, so indirect risk ("I don't
want to be here anymore") is caught too. When no model is available, a short
list of indirect phrases stands in for it and can only raise concern.

The final risk is the higher of the two layers. A model can raise the risk
but never lower what the keyword floor found.
"""

from __future__ import annotations

import re

from .llm import complete, parse_json

RISK_ORDER = {"none": 0, "concern": 1, "high": 2}

# Explicit statements. Any match is high risk.
HIGH_PATTERNS = (
    r"\bkill(ing)? myself\b",
    r"\bsuicid(e|al)\b",
    r"\bwant(ed)? to die\b",
    r"\bwanna die\b",
    r"\bend(ing)? (my|it all|my own) li(fe|ves)\b",
    r"\bend it all\b",
    r"\btake my (own )?life\b",
    r"\bhurt(ing)? myself\b",
    r"\bharm(ing)? myself\b",
    r"\bself[- ]harm",
    r"\bcut(ting)? myself\b",
    r"\boverdos(e|ed|ing)\b",
    r"\bnot (want|wanting) to (be alive|live)\b",
    r"\bdon'?t want to (be alive|live)( anymore)?\b",
    r"\bbetter off dead\b",
    r"\bno reason to (live|be alive)\b",
)

# Contexts in which a keyword is clearly not about the speaker.
NOT_ABOUT_SELF = (
    r"\bsuicide squad\b",
    r"\bsuicide doors?\b",
    r"\bsuicide (prevention|hotline|awareness)\b",
    r"\bsuicide (run|drill)s?\b",
    r"\bsuicide (bombing|bomber|mission)\b",
    r"\b(he|she|they|someone|my \w+) (wants?|wanted) to die\b",
    r"\b(want(ed)?|wanna) to die of (embarrassment|shame|laughter|laughing)\b",
    r"\bwanna die of (embarrassment|shame|laughter|laughing)\b",
)

# Indirect phrases, used only when no model is available. They raise concern.
CONCERN_PHRASES = (
    "don't want to be here",
    "dont want to be here",
    "do not want to be here",
    "can't go on",
    "cant go on",
    "cannot go on",
    "better off without me",
    "no point anymore",
    "no point in anything",
    "what's the point",
    "whats the point",
    "disappear forever",
    "go to sleep and not wake up",
    "not wake up",
    "nobody would miss me",
    "no one would miss me",
    "nobody would notice if i",
    "give up on everything",
    "giving up on everything",
    "can't do this anymore",
    "cant do this anymore",
    "tired of living",
    "tired of being alive",
    "burden to everyone",
    "say goodbye",
    "saying goodbye",
    "make it stop",
    "end the pain",
)


def keyword_risk(text: str) -> dict:
    lowered = " ".join(text.lower().replace("’", "'").split())
    scrubbed = lowered
    for pattern in NOT_ABOUT_SELF:
        scrubbed = re.sub(pattern, " ", scrubbed)
    for pattern in HIGH_PATTERNS:
        match = re.search(pattern, scrubbed)
        if match:
            return {"risk": "high", "source": "keyword floor", "evidence": match.group(0)}
    return {"risk": "none", "source": "keyword floor", "evidence": ""}


def phrase_risk(text: str) -> dict:
    lowered = " ".join(text.lower().replace("’", "'").split())
    hit = next((phrase for phrase in CONCERN_PHRASES if phrase in lowered), "")
    return {"risk": "concern" if hit else "none", "source": "indirect phrase list", "evidence": hit}


SYSTEM = (
    "You screen one message from a person talking to a support chatbot for risk of suicide or self-harm. "
    "Labels: high = the person states or clearly implies intent, a plan, or a wish to die or hurt themselves; "
    "concern = indirect or ambiguous signs, such as hopelessness, feeling like a burden, wanting to disappear, "
    "or saying goodbye, without a clear statement; none = no sign of risk to themselves. "
    "Ordinary venting, stress, sadness, anger, sarcasm, jokes, song or film titles, and talk about other people "
    "or the news are none unless they also show risk to the speaker. "
    'Return JSON: {"risk": "none" | "concern" | "high", "reason": one short sentence}.'
)


def model_risk(text: str, history: list[dict] | None = None) -> dict | None:
    context = ""
    recent = [str(turn.get("text", "")).strip() for turn in (history or [])[-4:] if turn.get("role") != "assistant"]
    if any(recent):
        context = "Earlier messages from the same person:\n" + "\n".join(f"- {line}" for line in recent if line) + "\n\n"
    raw = complete(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{context}Message to label:\n{text}"}],
        json_mode=True,
    )
    parsed = parse_json(raw or "")
    if not parsed:
        return None
    risk = str(parsed.get("risk", "")).strip().lower()
    if risk not in RISK_ORDER:
        return None
    return {"risk": risk, "source": "risk model", "evidence": str(parsed.get("reason", "")).strip()}


def screen(text: str, history: list[dict] | None = None, *, allow_model: bool = True) -> dict:
    """Highest risk from the keyword floor and the model (or the phrase list without a model)."""
    floor = keyword_risk(text)
    second = model_risk(text, history) if allow_model and text.strip() else None
    if second is None:
        second = phrase_risk(text)
    layers = [floor, second]
    top = max(layers, key=lambda layer: RISK_ORDER[layer["risk"]])
    return {"risk": top["risk"], "source": top["source"] if top["risk"] != "none" else "", "layers": layers}
