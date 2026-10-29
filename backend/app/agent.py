"""The support conversation that starts after a check-in or support reading.

The flow follows peer-support practice (open question, reflective listening,
validation): open, reflect, ask what would help, offer one grounding idea or a
resource, check in, then keep listening. A small state travels with each turn:
the turn number, the current risk level, the step, and short notes of what the
person has shared. The client sends it back with the next message.

Every message goes through the crisis screen first. High risk always gets the
988 referral. Concern gets one direct, caring safety question.

The person never sees model scores here. Those stay in the studio views.
"""

from __future__ import annotations

import re

from .judge import CRISIS_CARE
from .llm import complete
from .safety import RISK_ORDER, screen

STEPS = ("open", "reflect", "ask_help", "offer", "check_in", "listen")
DISCLAIMER = "I'm not a counselor, but I'm here to listen."
GROUNDING = (
    "If it helps, try one slow breath: in for four counts, out for six, a few times.",
    "Something small that can help: name five things you can see and four you can hear around you right now.",
    "If your body feels tense, try pressing your feet into the floor and noticing the weight of your body for a moment.",
)
RESOURCES = (
    "Talking to someone you trust, or a counselor, can help with this too. "
    "If things ever feel like too much, you can call or text 988 in the US, any time."
)
STEP_GUIDE = {
    "open": "Ask one open question about what is going on.",
    "reflect": "Reflect back, in your own words, what they said and how it might feel. Validate it. Do not give advice yet.",
    "ask_help": "Reflect briefly, then ask what would help right now, or what usually helps them when things feel like this.",
    "offer": "Offer exactly one small grounding idea, or one resource (a trusted person, a counselor, or 988 in the US), whichever fits what they said.",
    "check_in": "Check how they are feeling now, compared with when they started talking.",
    "listen": "Keep listening. Reflect and ask one gentle follow-up question.",
}
FINE = re.compile(r"\b(i'?m|i am|feeling|doing)( pretty| really)? (fine|ok|okay|good|alright|all right|great)\b|\ball good\b")
WHY = re.compile(r"\bwhy\b.*\b(distress|call|think|say|ask)\b|\bwhat made you\b")


def new_state(band: str = "support") -> dict:
    # The judge's question already opened the conversation, so the first reply reflects.
    return {"active": True, "band": band, "turn": 0, "step": "reflect", "risk": "none", "asked_safety": False, "disclaimed": False, "shared": []}


def _advance(step: str) -> str:
    return STEPS[min(STEPS.index(step) + 1, len(STEPS) - 1)] if step in STEPS else "listen"


def _note(message: str) -> str:
    text = " ".join(message.split())
    return text if len(text) <= 120 else text[:117] + "..."


def _fallback(step: str, turn: int) -> str:
    if step == "open":
        return "What's been going on for you?"
    if step == "reflect":
        return "That sounds really hard, and it makes sense that it's weighing on you. Can you tell me a bit more about it?"
    if step == "ask_help":
        return "Thank you for telling me. What do you think would help most right now, even a little?"
    if step == "offer":
        return f"{GROUNDING[turn % len(GROUNDING)]} {RESOURCES}"
    if step == "check_in":
        return "How are you feeling now, compared with when we started talking?"
    return "I'm still here. What's on your mind?"


def _model_reply(message: str, state: dict, history: list[dict], step: str) -> str | None:
    shared = "; ".join(state.get("shared", [])[-5:]) or "nothing yet"
    system = (
        "You are a supportive listener in a course project called DistressNet, talking with someone who may be having a hard time. "
        "You are not a clinician or counselor. Never diagnose, name a disorder, or suggest medication or treatment. "
        "Never mention models, scores, probabilities, bands, or how the system decided anything. "
        "Never give any information that could be used for self-harm. "
        "If they mention wanting to die or hurting themselves, tell them to call or text 988 in the US, or local emergency services. "
        "Write two to four short sentences in plain, warm language. Ask at most one question. "
        f"What they have shared so far: {shared}. "
        f"This turn: {STEP_GUIDE[step]}"
    )
    messages: list[dict] = [{"role": "system", "content": system}]
    for turn in history[-6:]:
        text = str(turn.get("text", "")).strip()
        if text:
            messages.append({"role": "assistant" if turn.get("role") == "assistant" else "user", "content": text})
    messages.append({"role": "user", "content": message})
    text = (complete(messages, temperature=0.3) or "").strip()
    if not text or re.search(r"\b(probabilit|score|model|network|band)\w*\b", text.lower()):
        return None
    return text


def respond(message: str, state: dict | None, history: list[dict] | None = None, *, allow_model: bool = True) -> tuple[str, dict]:
    """One turn of the support conversation. Returns the reply and the next state."""
    history = history or []
    state = dict(state or new_state())
    state["shared"] = list(state.get("shared", []))
    state["turn"] = int(state.get("turn", 0)) + 1
    lowered = message.lower().replace("’", "'")

    risk = screen(message, history, allow_model=allow_model)
    if RISK_ORDER[risk["risk"]] > RISK_ORDER.get(state.get("risk", "none"), 0):
        state["risk"] = risk["risk"]
    state["last_risk"] = risk["risk"]

    if risk["risk"] == "high":
        state["step"] = "crisis"
        return CRISIS_CARE, state

    if risk["risk"] == "concern" and not state.get("asked_safety"):
        state["asked_safety"] = True
        state["shared"] = (state["shared"] + [_note(message)])[-5:]
        return (
            "Thank you for telling me that. I want to check, because it matters: are you thinking about hurting yourself, "
            "or about not being alive? Whatever the answer, you can call or text 988 in the US to talk to someone right now.",
            state,
        )

    if state.get("step") == "crisis":
        # After a crisis referral, keep listening and keep the referral in view.
        text = _model_reply(message, state, history, "listen") if allow_model else None
        return f"{text or 'I am still here with you.'} Please reach out to 988 (call or text) or someone near you right now.", state

    if state.get("band") == "check-in" and state["turn"] == 1 and FINE.search(lowered) and risk["risk"] == "none":
        state["active"] = False
        return "I'm glad to hear that. If anything changes, I'm here, and you can come back any time.", state

    if WHY.search(lowered):
        return (
            "Something in how you sounded, or in what you said, suggested today might be hard. "
            "I could be wrong, and you know yourself better than any program does. How are you really doing?",
            state,
        )

    step = state.get("step", "reflect")
    if step not in STEPS:
        step = "listen"
    state["shared"] = (state["shared"] + [_note(message)])[-5:]
    text = _model_reply(message, state, history, step) if allow_model else None
    if text is None:
        text = _fallback(step, state["turn"])
    if not state.get("disclaimed"):
        if "counselor" not in text.lower():
            text = f"{DISCLAIMER} {text}"
        state["disclaimed"] = True
    state["step"] = _advance(step)
    return text, state
