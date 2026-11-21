from app.chat import respond


def test_saying_how_you_feel_starts_the_support_conversation():
    text, state = respond("I'm really stressed about my exams", None, None)
    assert state is not None and state["active"] and "counselor" in text.lower()


def test_asking_to_talk_opens_with_a_question():
    text, state = respond("I'd like to talk about how I'm feeling", None, None)
    assert state["active"] and text.endswith("?")


def test_studio_questions_stay_with_the_studio():
    _, state = respond("What are the limits?", None, None)
    assert state is None
