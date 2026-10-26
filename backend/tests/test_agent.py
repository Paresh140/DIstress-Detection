from app.agent import new_state, respond
from app.judge import CRISIS_CARE


def test_first_reply_says_it_is_not_a_counselor_and_never_shows_scores():
    text, state = respond("work has been awful this week", new_state())
    assert "counselor" in text.lower()
    assert not any(word in text.lower() for word in ("probability", "score", "network"))
    assert state["turn"] == 1 and state["step"] == "ask_help"
    assert state["shared"] == ["work has been awful this week"]


def test_flow_moves_through_the_steps():
    state = new_state()
    steps = []
    for message in ("rough day", "my boss yelled at me", "maybe a walk", "ok", "a bit better"):
        _, state = respond(message, state)
        steps.append(state["step"])
    assert steps == ["ask_help", "offer", "check_in", "listen", "listen"]


def test_crisis_message_gets_988_and_stays_in_crisis():
    text, state = respond("I want to end my life", new_state())
    assert text == CRISIS_CARE
    text, state = respond("I don't know", state)
    assert "988" in text and state["risk"] == "high"


def test_concern_asks_about_safety_once():
    text, state = respond("I can't go on like this", new_state())
    assert "hurting yourself" in text and "988" in text
    text, _ = respond("I can't go on like this", state)
    assert "hurting yourself" not in text


def test_check_in_that_is_fine_ends_the_conversation():
    text, state = respond("I'm fine, thanks", new_state("check-in"))
    assert state["active"] is False
