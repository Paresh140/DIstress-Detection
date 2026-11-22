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


def test_first_reply_asks_for_a_mood_rating_and_offers_quick_replies():
    _, state = respond("exams are stressing me out", new_state())
    assert state["card"]["type"] == "mood"
    assert state["prompts"] and state["topics"] == ["school"]


def test_ideas_follow_the_topic_and_do_not_repeat():
    state = new_state()
    _, state = respond("I can't sleep at night", state)
    first, state = respond("give me some ideas", state)
    second, state = respond("tell me more ideas", state)
    assert "sleep" in first and "•" in first
    assert first.split("\n\n")[1] != second.split("\n\n")[1]


def test_breathing_and_grounding_come_as_cards():
    _, state = respond("can we try a breathing exercise", new_state())
    assert state["card"]["type"] == "breathing"
    _, state = respond("try grounding (5-4-3-2-1)", state)
    assert state["card"]["type"] == "grounding" and len(state["card"]["steps"]) == 5


def test_mood_ratings_show_the_change():
    _, state = respond("My mood is 3 out of 10", new_state())
    assert state["mood_start"] == 3
    text, state = respond("I finished the breathing exercise", state)
    assert state["card"]["type"] == "mood" and "3" in text
    text, state = respond("My mood is 6 out of 10", state)
    assert "3 to 6" in text


def test_very_low_mood_shows_who_to_contact():
    _, state = respond("My mood is 1 out of 10", new_state())
    assert state["card"]["type"] == "resources"
    assert any("988" in item["detail"] for item in state["card"]["items"])


def test_asking_who_to_talk_to_shows_contacts():
    text, state = respond("who can I talk to?", new_state())
    assert state["card"]["type"] == "resources" and "988" in text


def test_venting_mode_skips_advice_at_the_offer_step():
    state = new_state()
    for message in ("I just want to vent", "my roommate is awful", "she never cleans"):
        text, state = respond(message, state)
    assert state["mode"] == "listen" and "often helps" not in text


def test_not_feeling_better_is_not_read_as_better():
    _, state = respond("My mood is 4 out of 10", new_state())
    text, state = respond("honestly nothing helped", state)
    assert "glad" not in text.lower()


def test_crisis_comes_with_contacts():
    _, state = respond("I want to kill myself", new_state())
    assert state["card"]["type"] == "resources" and state["prompts"]
