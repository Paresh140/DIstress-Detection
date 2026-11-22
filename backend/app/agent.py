"""The support conversation that starts after a check-in or support reading.

The flow follows peer-support practice (open question, reflective listening,
validation): open, reflect, ask what would help, offer one idea or a resource,
check in, then keep listening. Along the way the person can steer it: ask for
ideas, say they only want to vent, ask who they can talk to, or try one of the
short exercises the chat runs with them (paced breathing, 5-4-3-2-1 grounding,
a 0-10 mood check that is asked again later so they can see the change).

A small state travels with each turn: the turn number, the current risk level,
the step, the mode (listen or ideas), the topics they have mentioned, their
mood ratings, and short notes of what they have shared. The client sends it
back with the next message. Each reply may also carry quick replies and one
card (an exercise, the mood scale, or the list of people to contact), which
the server takes out of the state before it answers.

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
    "offer": "Offer exactly one small idea from the list below, or one resource (a trusted person, a counselor, or 988 in the US), whichever fits what they said.",
    "check_in": "Check how they are feeling now, compared with when they started talking.",
    "listen": "Keep listening. Reflect what they just said and ask one gentle follow-up question.",
}

# What people often bring, with a few general, non-clinical ideas for each.
TOPICS = {
    "school": {
        "pattern": r"\b(exams?|tests?|stud(y|ying)|assignments?|homework|grades?|class(es)?|college|school|semester|finals|thesis|professor)\b",
        "about": "school pressure",
        "tips": (
            "Pick one small piece of work you can finish in about 25 minutes, then take a 5-minute break. Starting small is easier than facing the whole pile.",
            "Write down the two or three things that matter most this week. Everything else can wait, and seeing it on paper often makes it feel smaller.",
            "Sleep does more for memory than one more late hour of studying, so protecting your sleep is part of preparing.",
            "If you are behind, telling a professor or advisor early usually goes better than waiting. Many will work with you.",
        ),
    },
    "work": {
        "pattern": r"\b(work|job|boss|manager|shifts?|coworkers?|colleagues?|office|workload|fired|laid off)\b",
        "about": "work stress",
        "tips": (
            "Try to finish the day by writing down tomorrow's first task. It helps your mind let go of work in the evening.",
            "Short breaks away from the screen, even two minutes of walking, help your focus and your mood more than pushing straight through.",
            "If the workload is the problem, it can help to ask your manager which tasks matter most, so you are not carrying everything at once.",
        ),
    },
    "sleep": {
        "pattern": r"\b(sleep(ing)?|insomnia|can'?t sleep|awake|exhausted|tired|nightmares?|rest)\b",
        "about": "sleep",
        "tips": (
            "Keep the same wake-up time every day, even after a bad night. It is one of the steadiest ways to reset sleep.",
            "If you have been lying awake for about 20 minutes, get up and do something calm in dim light, then go back to bed when you feel sleepy.",
            "Putting screens away for the last half hour before bed, and keeping caffeine to the morning, both make falling asleep easier.",
        ),
    },
    "anxiety": {
        "pattern": r"\b(anxious|anxiety|panic(king)?|nervous|worr(y|ied|ying)|overthink(ing)?|on edge|racing thoughts|heart (is )?(pounding|racing)|scared|afraid)\b",
        "about": "anxiety",
        "tips": (
            "Breathing out more slowly than you breathe in, for example in for 4 and out for 6, tells your body it is safe. We can do it together if you like.",
            "A wave of panic feels awful, but it rises, peaks and passes, often within about ten minutes. Riding it out without fighting it can make it shorter.",
            "When a worry loops, try asking: what is one small thing in my control right now? Then do just that one thing.",
            "Grounding yourself in your senses (what you can see, hear and touch) pulls attention out of the worry and back into the room.",
        ),
    },
    "low": {
        "pattern": r"\b(sad|down|low|empty|hopeless|cry(ing)?|depressed|numb|unmotivated|worthless|miserable|unhappy|not feeling (well|good|great)|not (okay|ok))\b",
        "about": "feeling low",
        "tips": (
            "When mood is low, motivation usually follows action instead of coming first. One tiny step, like a shower or a five-minute walk, counts.",
            "A little daylight and movement, even briefly, can lift energy a bit.",
            "Try speaking to yourself the way you would to a friend who felt like this. Most of us are much harsher with ourselves.",
            "If this low feeling has lasted more than a couple of weeks, talking to a doctor or counselor is a good next step. It is common, and it can get better.",
        ),
    },
    "lonely": {
        "pattern": r"\b(lonely|loneliness|alone|isolated|no friends|nobody (cares|gets it)|no one (cares|gets it)|left out)\b",
        "about": "loneliness",
        "tips": (
            "One small reach-out counts: a short text to someone you have not talked to in a while.",
            "Being around people, even without talking much, like studying in a library or sitting in a café, can ease loneliness a little.",
            "Clubs, classes or volunteering give you the same people every week, which is how most friendships actually start.",
        ),
    },
    "relationship": {
        "pattern": r"\b(break ?up|broke up|girlfriend|boyfriend|partner|relationship|ex|divorce|cheat(ed|ing)?|dating)\b",
        "about": "a relationship",
        "tips": (
            "After a breakup or a fight, it is normal for feelings to come in waves. Letting yourself feel them, instead of pushing them away, usually helps them pass.",
            "Leaning on friends and keeping some routine (meals, sleep, getting outside) gives you something steady while things feel shaky.",
            "If you need to talk it through with the other person, waiting until you are calmer and using 'I feel' statements tends to go better.",
        ),
    },
    "family": {
        "pattern": r"\b(family|mom|mum|dad|parents?|mother|father|sister|brother|siblings?|home life)\b",
        "about": "family",
        "tips": (
            "Family stress can be heavy because it is hard to get distance from it. Small breaks, a walk or time in your own space, are allowed.",
            "If a conversation keeps turning into a fight, it is okay to say you need a pause and come back to it later.",
            "Talking to someone outside the family, a friend or a counselor, can help you sort out what you feel.",
        ),
    },
    "grief": {
        "pattern": r"\b(died|passed away|death|grief|grieving|funeral|lost my|miss (him|her|them))\b",
        "about": "grief",
        "tips": (
            "Grief often comes in waves, and there is no right way or timeline for it.",
            "Talking about the person, looking at photos or writing them a letter can help you carry the loss.",
            "Basic care (eating, sleeping, being with people) matters even more while you are grieving.",
        ),
    },
    "anger": {
        "pattern": r"\b(angry|anger|furious|mad|rage|frustrat(ed|ing)|annoyed|irritated|pissed)\b",
        "about": "anger",
        "tips": (
            "When anger spikes, stepping away for a few minutes before replying to anyone helps you say what you actually mean.",
            "Physical activity, even a brisk walk or a few minutes of stairs, burns off the adrenaline that comes with anger.",
            "Once you are calmer, it can help to name what is underneath it: often hurt, stress or feeling ignored.",
        ),
    },
    "money": {
        "pattern": r"\b(money|rent|bills?|debt|broke|afford|loan|financial|finances)\b",
        "about": "money worries",
        "tips": (
            "Writing down exactly what is owed and when it is due often makes money stress feel more manageable than carrying it in your head.",
            "Many schools, landlords and companies have hardship plans or emergency funds, and asking early is usually fine.",
        ),
    },
    "body": {
        "pattern": r"\b(my body|tense|tension|headaches?|stomach|chest|shaking|trembling|dizzy|feel sick|nausea|body feels)\b",
        "about": "how your body feels",
        "tips": (
            "Stress often sits in the shoulders, jaw and stomach. Let your shoulders drop, unclench your jaw and take one slow breath out.",
            "If something feels physically wrong, new or worrying, a doctor or nurse is the right person to check it.",
        ),
    },
}
GENERAL_TIPS = (
    "Take one slow breath, in for 4 and out for 6. It is a small reset you can do anywhere.",
    "Write down what is on your mind. Getting it out of your head and onto paper often makes it feel more manageable.",
    "Reach out to one person you trust, even with a short message. You do not have to explain everything.",
    "Do one small kind thing for your body: a glass of water, something to eat, a few minutes outside.",
)

BREATHING = {"type": "breathing", "inhale": 4, "exhale": 6, "rounds": 4}
GROUNDING_CARD = {
    "type": "grounding",
    "steps": [
        "Name 5 things you can see around you.",
        "Notice 4 things you can feel, like your feet on the floor or the chair under you.",
        "Listen for 3 things you can hear.",
        "Find 2 things you can smell, or two smells you like.",
        "Notice 1 thing you can taste, or take a sip of water.",
    ],
}
CONTACTS = {
    "type": "resources",
    "items": [
        {"name": "988 Suicide & Crisis Lifeline", "detail": "Call or text 988, any time, in the US.", "href": "tel:988", "action": "Call 988"},
        {"name": "988 by text", "detail": "Text 988 if talking feels hard.", "href": "sms:988", "action": "Text 988"},
        {"name": "Crisis Text Line", "detail": "Text HOME to 741741 to reach a trained volunteer.", "href": "sms:741741&body=HOME", "action": "Text HOME"},
        {"name": "Emergency", "detail": "If you are in danger right now, call 911 or your local emergency number.", "href": "tel:911", "action": "Call 911"},
        {"name": "Someone near you", "detail": "A friend, family member, campus counseling center or doctor you trust."},
    ],
}
MOOD_CARD = {"type": "mood", "question": "On a scale from 0 (very bad) to 10 (very good), how are you feeling right now?"}

SUPPORT_PROMPTS = ["Give me some ideas", "Try a breathing exercise", "I just want to vent", "Who can I talk to?"]
IDEA_PROMPTS = ["Tell me more ideas", "Try a breathing exercise", "Try grounding (5-4-3-2-1)", "I'm feeling a bit better"]
LISTEN_PROMPTS = ["There's more on my mind", "Give me some ideas", "Try a breathing exercise", "I'm feeling a bit better"]
CRISIS_PROMPTS = ["Show me who I can contact", "I'm safe right now", "Keep talking with me"]
EXERCISE_PROMPTS = ["I'd rather just talk", "Who can I talk to?"]

FINE = re.compile(r"\b(i'?m|i am|feeling|doing)( pretty| really)? (fine|ok|okay|good|alright|all right|great)\b|\ball good\b")
WHY = re.compile(r"\bwhy\b.*\b(distress|call|think|say|ask)\b|\bwhat made you\b")
BREATHE = re.compile(r"\bbreathing\b|\bbreathe\b|\bcalm (me )?down\b")
GROUND = re.compile(r"\bground(ing)?\b|5-4-3-2-1|54321")
FINISHED = re.compile(r"\bfinished the (breathing|grounding) exercise\b")
CONTACT = re.compile(
    r"\bwho (can|could|should) i (talk|reach out|turn) to\b|\b(hotlines?|helplines?|crisis lines?|therapists?|counsel(or|ors|ing)|professional help|talk to (someone|somebody|a person)|someone to talk to|contact)\b|\bresources?\b"
)
IDEAS = re.compile(
    r"\b(tips?|ideas?|advice|suggest\w*|strateg\w*|what (can|should|could) i do|how (can|do|should) i (cope|deal|handle|stop|calm|fix|feel better))\b"
)
VENT = re.compile(r"\b(want|need) to vent\b|\bjust (listen|vent)\b|\bdon'?t (want|need) advice\b|\bno advice\b")
BETTER = re.compile(r"\b(feel(ing)?|a bit|a little|much|lot) better\b|\b(that|it|this) helped\b|\bhelped (a lot|a bit|a little)\b")
NOT_BETTER = re.compile(r"\b(not|no|never|nothing)\b[^.]{0,20}\b(better|helped)\b|n't[^.]{0,20}\b(better|helped)\b")
MOOD = re.compile(r"\b(10|[0-9])\s*(?:/|out of)\s*10\b")
SAFE = re.compile(r"\bi'?m safe\b|\bi am safe\b")
JUST_TALK = re.compile(r"\b(rather|just|want to) (just )?talk\b|\bkeep talking\b|\bmore on my mind\b")


def new_state(band: str = "support", step: str = "reflect") -> dict:
    # The judge's question already opened the conversation, so the first reply reflects.
    return {
        "active": True,
        "band": band,
        "turn": 0,
        "step": step,
        "risk": "none",
        "asked_safety": False,
        "disclaimed": False,
        "shared": [],
        "mode": "",
        "topics": [],
        "tips_given": {},
        "mood_start": None,
        "mood_now": None,
        "asked_mood": False,
    }


def _advance(step: str) -> str:
    return STEPS[min(STEPS.index(step) + 1, len(STEPS) - 1)] if step in STEPS else "listen"


def _note(message: str) -> str:
    text = " ".join(message.split())
    return text if len(text) <= 120 else text[:117] + "..."


def _topics(message: str) -> list[str]:
    return [key for key, topic in TOPICS.items() if re.search(topic["pattern"], message)]


def _next_tips(state: dict, count: int) -> list[str]:
    """Ideas for the latest topic they mentioned, never repeating one already given."""
    key = state["topics"][-1] if state.get("topics") else "general"
    pool = TOPICS[key]["tips"] if key in TOPICS else GENERAL_TIPS
    given = dict(state.get("tips_given", {}))
    start = int(given.get(key, 0))
    chosen = [pool[(start + i) % len(pool)] for i in range(min(count, len(pool)))]
    given[key] = start + len(chosen)
    state["tips_given"] = given
    return chosen


def _about(state: dict) -> str:
    key = state["topics"][-1] if state.get("topics") else ""
    return TOPICS[key]["about"] if key in TOPICS else ""


def _fallback(step: str, turn: int, state: dict | None = None, ideas: list[str] | None = None) -> str:
    about = _about(state or {})
    if step == "open":
        return "I'm glad you reached out. What's been going on for you?"
    if step == "reflect":
        if about:
            return f"That sounds really hard. Dealing with {about} can wear you down, and it makes sense that it's weighing on you. Can you tell me a bit more about it?"
        return "That sounds really hard, and it makes sense that it's weighing on you. Can you tell me a bit more about it?"
    if step == "ask_help":
        return "Thank you for telling me. What do you think would help most right now, even a little?"
    if step == "offer":
        if about and ideas:
            return f"One thing that often helps with {about}: {ideas[0]} {RESOURCES}"
        return f"{GROUNDING[turn % len(GROUNDING)]} {RESOURCES}"
    if step == "check_in":
        return "How are you feeling now, compared with when we started talking?"
    return "I'm still here. What's on your mind?"


def _model_reply(message: str, state: dict, history: list[dict], step: str, ideas: list[str] | None = None) -> str | None:
    shared = "; ".join(state.get("shared", [])[-5:]) or "nothing yet"
    about = ", ".join(TOPICS[key]["about"] for key in state.get("topics", []) if key in TOPICS) or "not clear yet"
    mode = {
        "listen": "They asked you to just listen. Do not give advice or tips; reflect, validate, and stay curious.",
        "ideas": "They want practical ideas. Be concrete.",
    }.get(state.get("mode", ""), "")
    idea_text = " ".join(ideas or []) if step == "offer" or state.get("mode") == "ideas" else ""
    system = (
        "You are a warm, active, supportive listener in a course project called DistressNet, talking with someone who may be having a hard time. "
        "You are not a clinician or counselor. Never diagnose, name a disorder, or suggest medication or treatment. "
        "Never mention models, scores, probabilities, bands, or how the system decided anything. "
        "Never give any information that could be used for self-harm. "
        "If they mention wanting to die or hurting themselves, tell them to call or text 988 in the US, or local emergency services. "
        "Use what they actually said: name the specific situation and the feeling in your own words, so they feel heard. "
        "Do not repeat a sentence you already said earlier in this conversation, and do not start with 'I'm not a counselor'. "
        "Write three to five short sentences in plain, warm language. Ask at most one question, and end with it. "
        f"What they have shared so far: {shared}. What it seems to be about: {about}. "
        f"{mode} "
        f"This turn: {STEP_GUIDE[step]}"
        + (f" Ideas you may share, in your own words, keeping their meaning: {idea_text}" if idea_text else "")
    )
    messages: list[dict] = [{"role": "system", "content": system}]
    for turn in history[-6:]:
        text = str(turn.get("text", "")).strip()
        if text:
            messages.append({"role": "assistant" if turn.get("role") == "assistant" else "user", "content": text})
    messages.append({"role": "user", "content": message})
    text = (complete(messages, temperature=0.5) or "").strip()
    if not text or re.search(r"\b(probabilit|score|model|network|band)\w*\b", text.lower()):
        return None
    return text


def _mood_reply(score: int, state: dict) -> str:
    first = state.get("mood_start") is None
    if first:
        state["mood_start"] = score
        state["mood_now"] = score
        if score <= 3:
            return (
                f"Thank you for telling me, a {score} sounds like a really tough place to be right now. "
                "We can take this slowly. What's weighing on you the most?"
            )
        if score <= 6:
            return f"A {score}, so somewhere in the middle. Thanks for being honest. What's been pulling it down?"
        return f"A {score}, I'm glad it's not lower. Is there something on your mind you'd like to talk through?"
    before = int(state["mood_start"])
    state["mood_now"] = score
    if score > before:
        return (
            f"You went from {before} to {score}. That's a real change, and you did that. "
            "What do you think helped most? It's worth remembering for next time."
        )
    if score == before:
        return (
            f"Still a {score}. That's okay; some feelings take longer to shift, and talking about it still counts. "
            "Would it help to try something different, or would you rather keep talking?"
        )
    return (
        f"It went from {before} to {score}. I'm sorry it's feeling heavier. Thank you for telling me honestly. "
        "Would it help to talk to someone you trust, or to someone at 988 (call or text)? I'm still here too."
    )


def respond(message: str, state: dict | None, history: list[dict] | None = None, *, allow_model: bool = True) -> tuple[str, dict]:
    """One turn of the support conversation. Returns the reply and the next state.

    The state may come back with "prompts" (quick replies) and "card" (an exercise,
    the mood scale, or the contacts) for this turn only.
    """
    history = history or []
    state = {**new_state(), **dict(state or {})}
    state["shared"] = list(state.get("shared", []))
    state["topics"] = list(state.get("topics", []))
    state["turn"] = int(state.get("turn", 0)) + 1
    state.pop("card", None)
    state.pop("prompts", None)
    lowered = message.lower().replace("’", "'")

    risk = screen(message, history, allow_model=allow_model)
    if RISK_ORDER[risk["risk"]] > RISK_ORDER.get(state.get("risk", "none"), 0):
        state["risk"] = risk["risk"]
    state["last_risk"] = risk["risk"]

    if risk["risk"] == "high":
        state["step"] = "crisis"
        state["card"] = CONTACTS
        state["prompts"] = CRISIS_PROMPTS
        return CRISIS_CARE, state

    if risk["risk"] == "concern" and not state.get("asked_safety"):
        state["asked_safety"] = True
        state["shared"] = (state["shared"] + [_note(message)])[-5:]
        state["prompts"] = ["I'm safe right now", "I'm not sure", "Show me who I can contact"]
        return (
            "Thank you for telling me that. I want to check, because it matters: are you thinking about hurting yourself, "
            "or about not being alive? Whatever the answer, you can call or text 988 in the US to talk to someone right now.",
            state,
        )

    for key in _topics(lowered):
        if key not in state["topics"]:
            state["topics"].append(key)

    if CONTACT.search(lowered):
        state["card"] = CONTACTS
        state["prompts"] = ["Keep talking with me", "Try a breathing exercise", "Give me some ideas"]
        return (
            "Reaching out is a strong step. Here are people you can contact any time. "
            "988 is free and confidential, and you can call or text; you don't have to be in a crisis to use it. "
            "A friend, family member, campus counseling center or doctor can help too.",
            state,
        )

    if state.get("step") == "crisis":
        # After a crisis referral, keep listening and keep the referral in view.
        state["prompts"] = CRISIS_PROMPTS
        if SAFE.search(lowered):
            text = "I'm really glad you're safe right now. I'm still here. Would you like to keep talking about what's been going on?"
        else:
            text = _model_reply(message, state, history, "listen") if allow_model else None
        return f"{text or 'I am still here with you.'} Please reach out to 988 (call or text) or someone near you right now.", state

    if state.get("band") == "check-in" and state["turn"] == 1 and FINE.search(lowered) and risk["risk"] == "none":
        state["active"] = False
        return "I'm glad to hear that. If anything changes, I'm here, and you can come back any time.", state

    if WHY.search(lowered):
        state["prompts"] = SUPPORT_PROMPTS
        return (
            "Something in how you sounded, or in what you said, suggested today might be hard. "
            "I could be wrong, and you know yourself better than any program does. How are you really doing?",
            state,
        )

    mood = MOOD.search(lowered)
    if mood:
        text = _mood_reply(int(mood.group(1)), state)
        state["prompts"] = SUPPORT_PROMPTS if state["mood_now"] <= state["mood_start"] else ["There's more on my mind", "I'm feeling a bit better"]
        if state["mood_now"] <= 2:
            state["card"] = CONTACTS
        return text, state

    if FINISHED.search(lowered):
        state["prompts"] = ["I'm feeling a bit better", "Not much different", "Try grounding (5-4-3-2-1)"]
        if state.get("mood_start") is not None:
            state["card"] = MOOD_CARD
            return (
                f"Well done for doing that. You rated your mood {state['mood_start']} earlier. How is it now?",
                state,
            )
        return "Well done for doing that. How do you feel now, even a little different?", state

    if GROUND.search(lowered):
        state["card"] = GROUNDING_CARD
        state["prompts"] = EXERCISE_PROMPTS
        return (
            "Let's do it together. Go through each step at your own pace; there's no wrong answer. "
            "This pulls your attention out of the worry and back into the room.",
            state,
        )

    if BREATHE.search(lowered) and not JUST_TALK.search(lowered):
        state["card"] = BREATHING
        state["prompts"] = EXERCISE_PROMPTS
        return (
            "Let's slow things down together. Follow the circle: breathe in as it grows for 4, and out as it shrinks for 6. "
            "Four rounds take about 40 seconds. A longer out-breath helps your body settle.",
            state,
        )

    if VENT.search(lowered):
        state["mode"] = "listen"
    elif IDEAS.search(lowered):
        state["mode"] = "ideas"

    if JUST_TALK.search(lowered) and not VENT.search(lowered):
        state["prompts"] = LISTEN_PROMPTS
        state["shared"] = (state["shared"] + [_note(message)])[-5:]
        text = _model_reply(message, state, history, "listen") if allow_model else None
        return text or "Of course. I'm listening. What's on your mind?", state

    if IDEAS.search(lowered):
        state["prompts"] = IDEA_PROMPTS
        about = _about(state)
        tips = _next_tips(state, 2)
        intro = f"Here are a couple of things that often help with {about}:" if about else "Here are a couple of small things that often help:"
        return intro + "\n\n" + "\n".join(f"• {tip}" for tip in tips) + "\n\nWould you like to try a short breathing exercise together, or tell me which of these feels doable?", state

    if BETTER.search(lowered) and not NOT_BETTER.search(lowered):
        state["prompts"] = ["There's more on my mind", "Give me some ideas", "Who can I talk to?"]
        if state.get("mood_start") is not None:
            state["card"] = MOOD_CARD
            return "I'm really glad to hear that. Want to rate it again so you can see the change?", state
        return (
            "I'm really glad to hear that. Remember what helped, so you can come back to it. "
            "Is there anything else on your mind?",
            state,
        )

    step = state.get("step", "reflect")
    if step not in STEPS:
        step = "listen"
    if state.get("mode") == "listen" and step == "offer":
        # They asked not to get advice, so the offer step listens instead.
        step = "listen"
    state["shared"] = (state["shared"] + [_note(message)])[-5:]
    ideas = _next_tips(state, 1) if step == "offer" or state.get("mode") == "ideas" else None
    text = _model_reply(message, state, history, step, ideas) if allow_model else None
    if text is None:
        text = _fallback(step, state["turn"], state, ideas)
    if not state.get("disclaimed"):
        if "counselor" not in text.lower():
            text = f"{DISCLAIMER} {text}"
        state["disclaimed"] = True
    if not state.get("asked_mood"):
        # Ask for a mood rating once, early, so a later rating can show the change.
        state["asked_mood"] = True
        state["card"] = MOOD_CARD
    elif step == "check_in" and state.get("mood_start") is not None:
        state["card"] = MOOD_CARD
    state["prompts"] = LISTEN_PROMPTS if state.get("mode") == "listen" else IDEA_PROMPTS if step in ("offer", "check_in") else SUPPORT_PROMPTS
    state["step"] = _advance(step)
    return text, state
