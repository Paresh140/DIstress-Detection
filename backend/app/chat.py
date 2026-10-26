"""Studio assistant. Answers from the training report and the latest reading.

After a check-in or support reading, messages go to the support conversation
in agent.py instead, which never shows scores to the person.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import agent
from .judge import CRISIS_CARE, is_crisis


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _span(actors: list[int]) -> str:
    ordered = sorted(int(actor) for actor in actors)
    if not ordered:
        return "none"
    if ordered[-1] - ordered[0] + 1 == len(ordered):
        return f"{ordered[0]:02d}–{ordered[-1]:02d}"
    return ", ".join(f"{actor:02d}" for actor in ordered)


def _has(text: str, words: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


def _study_reply() -> str:
    path = Path(__file__).resolve().parents[1] / "artifacts" / "cross_corpus_report.json"
    if not path.exists():
        return (
            "The audio study is RAVDESS and CREMA-D, because IEMOCAP needs a signed USC license and is not on this machine. "
            "Each corpus is tested two ways: speakers held out inside the corpus, and the whole other corpus held out. "
            "Text is not trained into the network. RAVDESS repeats two sentences in every emotion, and CREMA-D repeats twelve neutral sentences, "
            "so those transcripts do not carry the label. The chatbot reads the words separately, after the voice score."
        )
    study = json.loads(path.read_text())
    lines = [
        f"{run['name']}: balanced accuracy {_fmt(run['test_balanced_accuracy'])}, "
        f"distress F1 {_fmt(run['test_f1'])}, majority-class F1 {_fmt(run['majority_f1'])}"
        for run in study.get("runs", [])
    ]
    text = study.get("text", {}).get("finding", "")
    return (
        "The checkpoint for each run was chosen on validation speakers, then the test speakers were scored once. "
        + " ".join(lines)
        + " "
        + text
        + " "
        + study.get("iemocap", "")
    )


STUDIO_WORDS = (
    "layer",
    "layers",
    "trained",
    "training",
    "epoch",
    "accuracy",
    "network",
    "actor",
    "actors",
    "f1",
    "limit",
    "limits",
    "crema",
    "iemocap",
    "corpus",
    "cross",
    "fusion",
    "sentence",
    "sentences",
)


def respond(
    message: str,
    report: dict | None,
    analysis: dict | None,
    history: list[dict] | None = None,
    state: dict | None = None,
) -> tuple[str, dict | None]:
    """Reply and the next support state (None while no support conversation is running)."""
    text = message.strip()
    lower = text.lower()
    judge = (analysis or {}).get("judge") or {}
    studio_question = _has(lower, STUDIO_WORDS)
    if state is None and judge.get("ask"):
        state = agent.new_state("check-in" if judge.get("band") == "check-in" else "support")
    if text and state is not None and state.get("active") and not studio_question:
        return agent.respond(text, state, history)
    if text and is_crisis(lower):
        return CRISIS_CARE, {**(state or agent.new_state()), "active": True, "risk": "high", "step": "crisis"}
    return reply(message, report, analysis, history), state


def reply(message: str, report: dict | None, analysis: dict | None, history: list[dict] | None = None) -> str:
    text = message.strip()
    lower = text.lower()
    if not text:
        return "Ask about this reading, the layers, or how the audio branch was trained."

    if is_crisis(lower):
        return CRISIS_CARE

    if _has(lower, ("who are you", "what are you", "your name", "who you are")) or lower in {"who", "name"}:
        return (
            "I am the DistressNet assistant. "
            "A voice model scores how a clip sounds, a language model reads the words, and a judge puts the reading in one of three bands: "
            "calm, check-in, or support. "
            "I explain those readings, the layers, and the training run. "
            "If the judge asks for a check-in or support, I ask what is going on and listen. I am not a counselor and I do not diagnose."
        )

    if report is None:
        return (
            "The training report is not loaded yet, so I cannot quote a result. "
            "Once the audio branch has been trained, I can walk through the epochs, the held-out actors, and a clip."
        )

    best = report["best"]
    split = report["split"]
    confusion = report["confusion"]

    if _has(lower, ("hello", "hi", "hey")):
        return (
            "I can explain a clip, the DistressNet audio branch, or the training run. "
            f"The best held-out F1 so far is {_fmt(best['f1'])}, from epoch {best['epoch']}."
        )

    if _has(lower, ("crema", "iemocap", "corpus", "cross", "rotation", "sentence", "sentences", "fusion", "transcript")):
        return _study_reply()

    if _has(lower, ("limit", "limits", "clinical", "diagnosis", "diagnose", "patient", "medical")):
        return (
            "This is a course model on acted RAVDESS speech, not a clinical tool. "
            "The actors were asked to portray emotion, so a high score means the voice resembles the distress class "
            "in that dataset: sad, angry, or fearful speech. It does not diagnose anyone, and it has not seen real patients. "
            f"The held-out actors are {_span(split['test_actors'])}. "
        )

    if _has(lower, ("wrong", "mistake", "mistakes", "confusion", "error", "misclassified")):
        wrong = [item for item in report["predictions"] if not item["correct"]]
        examples = ", ".join(
            f"actor {item['actor']:02d} {item['emotion']} read as {item['predicted']} ({_fmt(item['probability'])})"
            for item in wrong[:3]
        )
        return (
            f"On the best checkpoint the held-out confusion counts are "
            f"{confusion['tp']} true distress, {confusion['tn']} true calm, "
            f"{confusion['fp']} false distress, and {confusion['fn']} missed distress. "
            + (f"A few misses: {examples}." if examples else "Every held-out clip in this run was classified correctly.")
        )

    if _has(lower, ("actor", "actors", "split", "held", "holdout")):
        return (
            f"Training used actors {_span(split['train_actors'])} ({split['train_clips']} clips: "
            f"{split['train_distress']} distress and {split['train_calm']} calm). "
            f"Actors {_span(split['test_actors'])} were held out ({split['test_clips']} clips) so the same person is never both in the update and in the score. "
            + (
                f"Actors {_span(split['val_actors'])} chose the checkpoint. "
                if split.get("val_actors")
                else ""
            )
            + (
                f"Actors {_span(split['unused_actors'])} are not in this run. "
                if split.get("unused_actors")
                else "All 24 RAVDESS actors are in this run. "
            )
            + "Song recordings are not in the loss."
        )

    if _has(lower, ("loss", "optimizer", "adam", "epoch", "epochs", "augment", "trained", "training")):
        return (
            f"{report['narrative']} "
            f"Updates used Adam at learning rate {report['learning_rate']}, weight decay {report['weight_decay']}, "
            f"batch size {report['batch_size']}, and seed {report['seed']}. "
            f"Positive-class weight was {_fmt(report['pos_weight'])} because calm clips are fewer than distress clips. "
            f"{report['augmentation']}"
        )

    if _has(lower, ("layer", "layers", "conv", "architecture", "network", "parameter", "parameters", "fusion", "vision", "embedding")):
        params = report["parameters"]
        return (
            f"The trained path is a log-mel spectrogram into three conv blocks (16, 32, then 64 filters), "
            f"then a 64-d embedding and a single distress logit. That audio branch has {params['audio']:,} parameters. "
            "A face branch and a fusion head were sketched early on but never trained, so they are future work and not part of this model. "
            "Open Network and scrub the epochs to see how activation and gradient size moved in the audio layers."
        )

    if _has(lower, ("label", "labels", "class", "emotion", "emotions", "sad", "angry", "fearful")):
        return (
            "Distress here is a class we defined, not a diagnosis. "
            "Sad, angry, and fearful RAVDESS speech are the positive class. "
            "Neutral and calm speech are the negative class. "
            "Happy, surprised, and disgust clips are in the download but they are excluded from training and testing."
        )

    if analysis is None and _has(lower, ("f1", "accuracy", "precision", "recall", "metric", "metrics", "score", "result")):
        return (
            f"Best checkpoint: epoch {best['epoch']}, test F1 {_fmt(best['f1'])}, "
            f"accuracy {_fmt(best['accuracy'])}, precision {_fmt(best['precision'])}, "
            f"recall {_fmt(best['recall'])}. "
            "Precision is the share of distress predictions that were truly distress. "
            "Recall is the share of true distress clips the model caught. "
            "The curves on the Training page are the whole run, not only the best epoch."
        )

    if analysis is not None and _has(
        lower,
        ("this", "clip", "reading", "mean", "means", "why", "explain", "probability", "confidence", "result", "score"),
    ):
        voice = analysis.get("voice") or {}
        language = analysis.get("language") or {}
        verdict = analysis.get("judge") or {}
        if voice.get("probability") is not None and language.get("probability") is not None:
            return (
                f"The judge calls this {verdict.get('decision', analysis['label'])}. "
                f"The voice network said {voice.get('label')} at {_fmt(voice['probability'])}. "
                f"The words were read as {language.get('label')} at {_fmt(language['probability'])} "
                f"({language.get('note', '')}). {verdict.get('reason', '')}"
            )
        label = analysis["label"]
        prob = analysis["probability"]
        side = "above" if prob >= 0.5 else "below"
        return (
            f"This clip is {side} the 0.50 threshold, so the audio head calls it {label} "
            f"with probability {_fmt(prob)} (logit {_fmt(analysis['logit'])}). "
            "The network never sees the waveform directly. It sees the log-mel image drawn beside the score: "
            "64 mel bands by 128 time frames, standardized per clip. "
            "A higher probability means the embedding landed on the distress side of the linear head. "
            f"On held-out actors the best F1 was {_fmt(best['f1'])}, so treat a single clip as a demo of that model, not a verdict."
        )

    if analysis is not None:
        return (
            f"The latest clip was read as {analysis['label']} at probability {_fmt(analysis['probability'])}. "
            f"The trained model’s best held-out F1 is {_fmt(best['f1'])} at epoch {best['epoch']}. "
            "Ask what the score means, how training worked, which actors were held out, or what the model got wrong."
        )

    return (
        f"The audio branch’s best held-out F1 is {_fmt(best['f1'])} at epoch {best['epoch']}. "
        "Load a clip on Monitor if you want that reading explained. "
        "I can also walk through the layers, the actor split, the loss, or the mistakes."
    )


def prompts(analysis: dict | None) -> list[str]:
    judge = (analysis or {}).get("judge", {})
    if judge.get("band") == "check-in":
        return ["I'm okay, thanks", "Not great, honestly", "Why are you asking?"]
    if judge.get("ask"):
        return [
            "I have had a hard day",
            "My body feels wrong",
            "Why did you call this distress?",
        ]
    if analysis is not None:
        return [
            "Explain this reading",
            "How was the network trained?",
            "What did the model get wrong?",
            "What are the limits?",
        ]
    return [
        "How was the network trained?",
        "What does each layer do?",
        "Which actors were held out?",
        "What are the limits?",
    ]
