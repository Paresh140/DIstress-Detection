import pytest

from app.safety import keyword_risk, screen


@pytest.mark.parametrize(
    "text",
    ["I want to die", "I've been thinking about suicide", "I keep wanting to hurt myself", "i want to end my life"],
)
def test_explicit_statements_are_high(text):
    assert keyword_risk(text)["risk"] == "high"


@pytest.mark.parametrize(
    "text",
    ["Suicide Squad was a bad movie", "the car has suicide doors", "I volunteer at a suicide prevention hotline", "I wanted to die of embarrassment"],
)
def test_false_triggers_are_not_high(text):
    assert keyword_risk(text)["risk"] == "none"


def test_phrase_list_raises_concern_without_a_model():
    assert screen("Everyone would be better off without me", allow_model=False)["risk"] == "concern"


def test_model_cannot_lower_the_floor(monkeypatch):
    from app import safety

    monkeypatch.setattr(safety, "model_risk", lambda text, history=None: {"risk": "none", "source": "risk model", "evidence": ""})
    assert screen("I want to kill myself")["risk"] == "high"
