import numpy as np

from app.calibration import distress_probability, ece, evaluate, fit_temperature, tune_threshold


def test_temperature_shrinks_overconfident_logits():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    logits = np.where(y == 1, 1.0, -1.0) * 8 + rng.normal(0, 8, 400)
    t = fit_temperature(logits, y)
    assert t > 1
    assert ece(distress_probability(logits, t), y) < ece(distress_probability(logits), y)


def test_four_class_probability_is_one_minus_non_distress():
    logits = np.array([[2.0, 0.0, 0.0, 0.0]])
    p = distress_probability(logits)[0]
    assert np.isclose(p, 3 / (np.e**2 + 3))


def test_threshold_and_evaluate_run():
    y = np.array([0, 0, 1, 1, 1, 1])
    prob = np.array([0.6, 0.65, 0.7, 0.8, 0.9, 0.95])
    assert 0.65 < tune_threshold(prob, y) <= 0.7
    out = evaluate(np.log(prob / (1 - prob)), y, np.log(prob / (1 - prob)), y)
    assert out["calibrated"]["balanced_accuracy"] == 1.0
