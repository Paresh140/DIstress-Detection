from pathlib import Path

import numpy as np
import pytest

from app import enrollment as enroll
from app.corpora import Example


def fake_corpus(corpus: str, speakers: int, per_emotion: int = 4, layers: int = 3, bins: int = 4, dim: int = 6, seed: int = 0):
    """Clips whose distress signal sits on top of a large per-speaker offset and scale, like a new microphone."""
    rng = np.random.default_rng(seed)
    clips, feats = [], {}
    direction = np.random.default_rng(99).normal(size=(layers, 1, dim))
    for s in range(speakers):
        offset = rng.normal(0, 3, size=(layers, 1, dim))
        scale = rng.uniform(0.5, 2.0, size=(layers, 1, dim))
        for emotion in ("neutral", "calm", "sad", "angry", "fearful"):
            for i in range(per_emotion):
                label = int(emotion in ("sad", "angry", "fearful"))
                path = Path(f"/{corpus}/{s:02d}_{emotion}_{i}.wav")
                clips.append(Example(path, corpus, f"{s:02d}", emotion, "x", label))
                x = rng.normal(size=(layers, bins, dim)) + 1.5 * label * direction
                feats[str(path)] = (x * scale + offset).astype(np.float16)
    return clips, feats


def test_calm_stats_and_apply_give_zero_mean_unit_std():
    rng = np.random.default_rng(1)
    calm = rng.normal(5, 3, size=(5, 2, 4, 3))
    stats = enroll.calm_stats(calm)
    assert stats["mean"].shape == (2, 1, 3)
    z = enroll.apply(calm, stats)
    assert np.allclose(z.mean(axis=(0, 2)), 0, atol=1e-5)
    assert np.allclose(z.std(axis=(0, 2)), 1, atol=1e-4)


def test_std_floor_prevents_division_blowup():
    flat = np.ones((3, 2, 4, 3))
    stats = enroll.calm_stats(flat)
    assert (stats["std"] >= enroll.STD_FLOOR).all()


def test_enrollment_uses_only_calm_clips_and_scorable_drops_them():
    clips, feats = fake_corpus("RAVDESS", 3)
    chosen = enroll.pick_enrollment(clips, k=5)
    assert len(chosen) == 3
    by_path = {str(c.path): c for c in clips}
    for paths in chosen.values():
        assert len(paths) == 5
        assert all(by_path[p].emotion in ("neutral", "calm") for p in paths)
    left = enroll.scorable(clips, chosen)
    used = {p for ps in chosen.values() for p in ps}
    assert len(left) == len(clips) - 15 and not used & {str(c.path) for c in left}
    shifted = enroll.enrolled_features(feats, clips, chosen)
    assert set(shifted) == set(feats)


def test_speaker_without_enough_calm_clips_is_skipped():
    clips, _ = fake_corpus("RAVDESS", 2, per_emotion=1)  # 2 calm clips per speaker
    assert enroll.pick_enrollment(clips, k=5) == {}


def test_store_persists_per_session(tmp_path):
    store = enroll.EnrollmentStore(tmp_path)
    session = "abcdef123456"
    assert store.stats(session) is None
    for _ in range(enroll.MIN_LIVE_CLIPS - 1):
        store.add(session, np.ones((2, 4, 3)))
    assert store.stats(session) is None  # not enough readings yet
    store.add(session, np.zeros((2, 4, 3)))
    again = enroll.EnrollmentStore(tmp_path)  # a new process sees the same readings
    assert again.count(session) == enroll.MIN_LIVE_CLIPS
    assert again.stats(session)["mean"].shape == (2, 1, 3)
    assert again.count("other-session-1") == 0
    with pytest.raises(ValueError):
        store.add("../../etc", np.zeros((2, 4, 3)))
    again.clear(session)
    assert again.count(session) == 0


def test_enroll_eval_runs_and_enrollment_beats_none_on_shifted_speakers():
    from app import enroll_eval

    train, ftr = fake_corpus("RAVDESS", 8, seed=2)
    val, fva = fake_corpus("VAL", 3, seed=3)
    test, fte = fake_corpus("CREMA-D", 4, seed=4)
    feats = {**ftr, **fva, **fte}
    report = enroll_eval.run(feats, [("synthetic", train, val, test)], k=5, draws=1, sweep=(3,))
    summary = report["settings"]["synthetic"]["summary"]
    assert set(summary) == {"none", "speaker", "enroll-5"}
    assert summary["enroll-5"]["pooled_balanced_accuracy"]["mean"] > summary["none"]["pooled_balanced_accuracy"]["mean"]
    assert "enroll-3" in report["settings"]["synthetic"]["sweep"]
    assert "calm enrollment (5 clips)" in enroll_eval.table(report)


def test_enrolled_head_trains_saves_and_scores_live(tmp_path, monkeypatch):
    import torch

    from app import voice
    from app.speech_encoder import LayerMixHead

    train, ftr = fake_corpus("RAVDESS", 6, seed=5)
    val, fva = fake_corpus("VAL", 2, seed=6)
    test, fte = fake_corpus("TEST", 2, seed=7)
    feats = {**ftr, **fva, **fte}
    out = voice._train_enrolled(feats, train, val, test, "binary-mean")
    assert out["summary"]["enroll_clips"] == enroll.ENROLL_CLIPS
    head = LayerMixHead(3, 6, classes=2, pool="mean")
    path = tmp_path / "voice_head.pt"
    torch.save(
        {
            "state": head.state_dict(),
            "n_layers": 3,
            "dim": 6,
            "mean": torch.zeros(3, 1, 6),
            "std": torch.ones(3, 1, 6),
            "summary": {"encoder": "x", "head": "binary-mean", "temperature": 1.0, "threshold": 0.5},
            "enrolled": out["saved"],
        },
        path,
    )
    monkeypatch.setattr(voice, "load_encoder", lambda name: (None, torch.nn.Identity()))
    model = voice.VoiceModel(path)
    assert model.enrolled is not None
    speaker = [c for c in test if c.speaker == "00"]
    calm = enroll.calm_stats(np.stack([feats[str(c.path)] for c in speaker if c.emotion == "neutral"]))
    plain = model.score_features(feats[str(speaker[0].path)])
    calibrated = model.score_features(feats[str(speaker[0].path)], calm)
    assert plain["enrolled"] is False and calibrated["enrolled"] is True
    assert calibrated["threshold"] == out["saved"]["threshold"]


def test_enroll_api_stores_readings_per_session(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import main

    class Fake:
        enrolled = {"head": None}

        def features(self, path):
            return np.random.default_rng(0).normal(size=(2, 4, 3))

    monkeypatch.setattr(main, "_voice_model", lambda: Fake())
    monkeypatch.setattr(main, "_enrollments", enroll.EnrollmentStore(tmp_path))
    client = TestClient(main.app)
    headers = {"X-Session": "session-test-0001"}
    status = client.get("/api/enroll", headers=headers).json()
    assert status["available"] and status["clips"] == 0 and not status["active"] and status["script"]
    for i in range(enroll.MIN_LIVE_CLIPS):
        status = client.post("/api/enroll", headers=headers, files={"file": ("calm.wav", b"RIFF....", "audio/wav")}).json()
    assert status["clips"] == enroll.MIN_LIVE_CLIPS and status["active"]
    assert client.get("/api/enroll", headers={"X-Session": "another-session-01"}).json()["clips"] == 0
    assert main._calm_stats("session-test-0001") is not None
    assert client.delete("/api/enroll", headers=headers).json()["clips"] == 0
    assert client.post("/api/enroll", headers={"X-Session": "bad"}, files={"file": ("c.wav", b"x", "audio/wav")}).status_code == 400
