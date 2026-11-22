from collections import Counter

from app import pipeline_eval, safety, safety_split


def test_wilson_interval_brackets_the_rate():
    lo, hi = safety_split.wilson(8, 10)
    assert lo < 0.8 < hi
    assert safety_split.wilson(0, 10)[0] == 0.0
    assert safety_split.wilson(10, 10)[1] == 1.0


def test_split_is_stratified_and_disjoint():
    rows = safety_split._jsonl(safety_split.EVAL / "safety_set.jsonl")
    dev, held = safety_split.split(rows)
    assert len(dev) == 60 and len(held) == 40
    assert not {r["text"] for r in dev} & {r["text"] for r in held}
    total = Counter(r["category"] for r in rows)
    for cat, n in Counter(r["category"] for r in dev).items():
        assert n == round(total[cat] * 0.6)


def test_dropping_rules_is_temporary():
    before = safety.HIGH_PATTERNS
    with safety_split.without_rules([f"high {before[0]}"]):
        assert before[0] not in safety.HIGH_PATTERNS
    assert safety.HIGH_PATTERNS == before


def test_judge_pair_keeps_the_higher_band():
    reader = pipeline_eval.Reader("phrases")
    calm_risk = {"risk": "none"}
    loud_voice = pipeline_eval.judge_pair(0.95, 0.5, "Everything's fine, just a normal Tuesday.", reader, calm_risk)
    assert loud_voice["band"] == "support" and loud_voice["non_canceling_held"]
    words = pipeline_eval.judge_pair(0.01, 0.5, "I'm scared and I can't breathe", reader, calm_risk)
    assert words["band"] == "support" and words["voice_band"] == "calm"


def test_wer_normalizes_numbers_and_apostrophes():
    from app.asr_eval import normalize, wer

    assert normalize("It's 11 o'clock.") == ["its", "eleven", "oclock"]
    out = wer([("It's eleven o'clock", "Its 11 oclock"), ("Kids are talking by the door", "kids talking by the door")])
    assert out["errors"] == 1 and out["reference_words"] == 9 and out["exact_match"] == 0.5


def test_ece_equal_mass_is_zero_for_perfect_calibration():
    import numpy as np

    from app.ece_eval import ece_equal_mass

    prob = np.repeat([0.2, 0.8], 50)
    truth = np.concatenate([np.r_[np.ones(10), np.zeros(40)], np.r_[np.ones(40), np.zeros(10)]])
    assert ece_equal_mass(prob, truth, 2) < 1e-9
