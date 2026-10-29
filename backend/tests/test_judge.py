import pytest

from app.judge import CRISIS_CARE, adjudicate, decide, voice_band, words_band

CALM_WORDS = {"label": "calm", "probability": 0.1}
DISTRESS_WORDS = {"label": "distress", "probability": 0.9}
UNSURE_WORDS = {"label": "calm", "probability": 0.45}


def test_distressed_words_with_calm_voice_is_never_calm():
    verdict = decide(0.05, DISTRESS_WORDS, "I am not okay")
    assert verdict["band"] == "support"
    assert verdict["decision"] == "distress"


def test_distressed_voice_with_calm_words_is_never_calm():
    verdict = decide(0.95, CALM_WORDS, "I feel fine")
    assert verdict["band"] == "support"


def test_both_calm_is_calm():
    verdict = decide(0.1, CALM_WORDS, "I feel fine")
    assert verdict["band"] == "calm"
    assert not verdict["ask"]


def test_uncertain_side_gives_a_check_in():
    assert decide(0.1, UNSURE_WORDS, "eh")["band"] == "check-in"
    assert decide(0.42, CALM_WORDS, "fine", voice_threshold=0.5)["band"] == "check-in"


@pytest.mark.parametrize("voice", [None, 0.0, 0.3, 0.49, 0.5, 0.7, 1.0])
@pytest.mark.parametrize("words", [None, CALM_WORDS, UNSURE_WORDS, DISTRESS_WORDS])
def test_band_is_never_below_either_side(voice, words):
    transcript = "" if words is None else "some words"
    verdict = decide(voice, words or {}, transcript)
    order = ("calm", "check-in", "support")
    for side in (voice_band(voice), words_band((words or {}).get("probability"))):
        if side:
            assert order.index(verdict["band"]) >= order.index(side)


def test_voice_bands_follow_the_validation_threshold():
    assert voice_band(0.70, threshold=0.74) == "check-in"
    assert voice_band(0.74, threshold=0.74) == "support"
    assert voice_band(0.50, threshold=0.74) == "calm"


def test_explicit_crisis_bypasses_the_judge():
    verdict = adjudicate(0.0, CALM_WORDS, "honestly I want to kill myself")
    assert verdict["band"] == "crisis"
    assert verdict["question"] == CRISIS_CARE


def test_indirect_risk_starts_support_with_a_safety_question():
    verdict = adjudicate(0.0, CALM_WORDS, "I just don't want to be here anymore")
    assert verdict["band"] == "support"
    assert verdict["risk"] == "concern"
    assert "hurting yourself" in verdict["question"]
