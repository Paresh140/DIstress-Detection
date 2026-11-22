"""DistressNet studio API."""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import llm
from . import transcribe as stt
from .audio import file_to_logmel
from .chat import prompts, respond
from .enrollment import CALM_SCRIPT, ENROLL_CLIPS, MIN_LIVE_CLIPS, EnrollmentStore
from .judge import adjudicate
from .language import read_language
from .model import DistressNet, architecture_spec
from .storage import LOCAL_ARTIFACTS, artifact_file, clip_root
from .train import EPOCHS
from .voice import HEAD_FILE, REPORT_FILE, VoiceModel

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = LOCAL_ARTIFACTS


def _data_root() -> Path:
    return clip_root()


def _weights() -> Path:
    return artifact_file("audio_branch.pt")


def _report_path() -> Path:
    return artifact_file("training_report.json")
PROGRESS_PATH = ARTIFACTS / "train_live.jsonl"
TRAIN_LOG = ARTIFACTS / "train_live.log"
_train_proc: subprocess.Popen | None = None
CLIP_NAME = re.compile(r"^Actor_(0[1-9]|1\d|2[0-4])/03-01-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.wav$")

app = FastAPI(title="DistressNet")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: DistressNet | None = None
_model_mtime = -1.0


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model() -> DistressNet:
    global _model, _model_mtime
    weights = _weights()
    mtime = weights.stat().st_mtime if weights.exists() else 0.0
    if _model is None or mtime != _model_mtime:
        net = DistressNet()
        if weights.exists():
            state = torch.load(weights, map_location="cpu", weights_only=True)
            net.load_state_dict(state, strict=False)
        net.eval()
        _model = net.to(_device())
        _model_mtime = mtime
    return _model


_voice: VoiceModel | None = None
_voice_error = ""


def _voice_model() -> VoiceModel | None:
    """The WavLM head when artifacts/voice_head.pt exists, else None and the CNN is used."""
    global _voice, _voice_error
    path = artifact_file(HEAD_FILE)
    if not path.exists():
        return None
    if _voice is None or _voice.mtime != path.stat().st_mtime:
        try:
            _voice = VoiceModel(path)
            _voice_error = ""
        except Exception as exc:  # noqa: BLE001
            _voice, _voice_error = None, f"The WavLM voice model could not load, so the CNN is used. {exc}"
    return _voice


# Calm enrollment per browser session, on disk so it survives requests and restarts.
_enrollments = EnrollmentStore(ARTIFACTS / "enrollments")


def _calm_stats(session: str) -> dict | None:
    try:
        return _enrollments.stats(session) if session else None
    except ValueError:
        return None


def _enroll_status(session: str) -> dict:
    wavlm = _voice_model()
    try:
        clips = _enrollments.count(session) if session else 0
    except ValueError:
        clips = 0
    available = bool(wavlm is not None and wavlm.enrolled is not None)
    if wavlm is None:
        why = "Calibration needs the WavLM voice model (artifacts/voice_head.pt)."
    elif wavlm.enrolled is None:
        why = "This voice_head.pt has no calm-enrolled head. Re-run python -m app.voice (Kaggle) to add one."
    else:
        why = ""
    return {
        "available": available,
        "clips": clips,
        "needed": MIN_LIVE_CLIPS,
        "recommended": ENROLL_CLIPS,
        "active": available and clips >= MIN_LIVE_CLIPS,
        "script": list(CALM_SCRIPT),
        "note": why,
    }


def _report() -> dict | None:
    path = _report_path()
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _resolve_clip(file_name: str) -> Path:
    if not CLIP_NAME.match(file_name):
        raise HTTPException(status_code=400, detail="Unknown clip.")
    root = _data_root()
    path = (root / file_name).resolve()
    if not str(path).startswith(str(root.resolve())) or not path.is_file():
        raise HTTPException(status_code=404, detail="Clip not found.")
    return path


def _predict(path: Path | None, transcript: str = "", session: str = "") -> dict:
    voice_probability = None
    voice_threshold = 0.5
    voice_extra: dict = {}
    logit_value = 0.0
    tap_means: dict[str, float] = {}
    spec: list = []
    embedding: list = []
    speech_to_text = None
    if path is not None:
        mel = file_to_logmel(str(path))
        spec = np.round(mel.numpy(), 3).tolist()
        wavlm = _voice_model()
        if wavlm is not None:
            scored = wavlm.score(str(path), _calm_stats(session))
            voice_probability = scored["probability"]
            voice_threshold = scored["threshold"]
            voice_extra = {key: scored[key] for key in ("model", "enrolled", "classes", "time_weights", "layer_weights") if key in scored}
        else:
            if not _weights().exists():
                raise HTTPException(status_code=409, detail="No voice model is trained yet.")
            model = _load_model()
            batch = mel.unsqueeze(0).unsqueeze(0).to(_device())
            with torch.no_grad():
                logit, taps = model.audio.forward_features(batch)
            logit_value = float(logit.squeeze().cpu())
            voice_probability = float(torch.sigmoid(logit).squeeze().cpu())
            voice_extra = {"model": "log-mel CNN"}
            tap_means = {
                key: round(float(value.detach().abs().mean().cpu()), 4)
                for key, value in taps.items()
                if key != "input"
            }
            embedding = [round(float(v), 3) for v in taps["embed"].squeeze().detach().cpu().tolist()]
        if not transcript.strip() and stt.available():
            speech_to_text = stt.transcribe(str(path))
            if speech_to_text:
                transcript = speech_to_text["text"]
    language = read_language(transcript)
    verdict = adjudicate(voice_probability, language, transcript, voice_threshold=voice_threshold)
    voice_label = None
    if voice_probability is not None:
        voice_label = "distress" if voice_probability >= voice_threshold else "calm"
    return {
        "file": path.name if path else "words",
        "label": verdict["decision"],
        "band": verdict["band"],
        "probability": verdict["confidence"],
        "logit": round(logit_value, 4),
        "threshold": round(voice_threshold, 4),
        "modality": (
            "voice+words"
            if transcript.strip() and voice_probability is not None
            else "audio"
            if voice_probability is not None
            else "words"
        ),
        "transcript": transcript.strip(),
        "speech_to_text": speech_to_text,
        "voice": {
            "label": voice_label,
            "probability": None if voice_probability is None else round(voice_probability, 4),
            "logit": round(logit_value, 4),
            "threshold": round(voice_threshold, 4),
            **voice_extra,
        },
        "language": language,
        "judge": verdict,
        "taps": tap_means,
        "embedding": embedding,
        "spectrogram": spec,
    }


class ChatTurn(BaseModel):
    role: str = ""
    text: str = ""


class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    analysis: dict | None = None
    history: list[ChatTurn] = Field(default_factory=list)
    state: dict | None = None


class ClipBody(BaseModel):
    file: str
    transcript: str = ""


class WordsBody(BaseModel):
    transcript: str = Field(min_length=1, max_length=2000)


@app.get("/api/health")
def health():
    wavlm = artifact_file(HEAD_FILE).exists()
    return {
        "ok": True,
        "weights": _weights().exists() or wavlm,
        "report": _report_path().exists(),
        "voice_model": "wavlm" if wavlm and not _voice_error else "cnn",
        "voice_error": _voice_error,
        "llm": llm.available() or "none",
        "speech_to_text": stt.available(),
    }


@app.get("/api/network")
def network():
    spec = architecture_spec()
    report = _report()
    spec["trained"] = _weights().exists()
    if report:
        spec["history"] = [
            {
                "epoch": row["epoch"],
                "test_loss": row["test_loss"],
                "test_accuracy": row["test_accuracy"],
                "test_precision": row["test_precision"],
                "test_recall": row["test_recall"],
                "test_f1": row["test_f1"],
                "activation": row["activation"],
                "grad_norm": row["grad_norm"],
            }
            for row in report["history"]
        ]
        spec["best_epoch"] = report["best"]["epoch"]
        spec["best"] = report["best"]
    else:
        spec["history"] = []
        spec["best_epoch"] = None
    return spec


@app.get("/api/study")
def study():
    path = artifact_file("cross_corpus_report.json")
    if not path.exists():
        return {
            "ready": False,
            "summary": (
                "The cross-corpus audio study has not been written yet. "
                "IEMOCAP is not on this machine. The run will train on RAVDESS and test on CREMA-D, then the reverse, "
                "and also hold speakers out inside each corpus."
            ),
        }
    report = json.loads(path.read_text())
    report["updated"] = path.stat().st_mtime
    return report


# The Kaggle notebook that runs kaggle/DistressNet_Kaggle.ipynb with the RAVDESS and CREMA-D datasets attached.
KAGGLE_URL = os.environ.get("DISTRESSNET_KAGGLE_URL", "https://www.kaggle.com/code/sarthakpokhrel/notebook5cd1df997e")
# What the Kaggle notebook packs into distressnet_results.zip; anything else in an upload is ignored.
KAGGLE_RESULTS = (
    HEAD_FILE,
    REPORT_FILE,
    "audio_branch.pt",
    "training_report.json",
    "cv_report.json",
    "cv_report_crema-d.json",
    "cross_corpus_report.json",
    "reader_report.json",
    "comparison.md",
)
HEAD_KEYS = {"state", "n_layers", "dim", "mean", "std", "summary"}


@app.get("/api/voice")
def voice_report():
    """The live WavLM model: its report when one was written, else the summary saved in voice_head.pt."""
    path = artifact_file(REPORT_FILE)
    if path.is_file():
        report = json.loads(path.read_text(encoding="utf-8"))
        return {"ready": True, **report}
    voice = _voice_model()
    if voice is None:
        return {"ready": False, "error": _voice_error}
    return {"ready": True, "summary": voice.summary, "epochs": 0, "history": [], "predictions": []}


@app.get("/api/voice/kaggle")
def voice_kaggle():
    return {"url": KAGGLE_URL}


def _check_result(name: str, data: bytes) -> None:
    """Refuse a file that would break the studio, before it replaces the working one."""
    if name == "audio_branch.pt":
        saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or not saved or not all(k.startswith("audio.") for k in saved):
            raise ValueError("it is not a CNN audio branch saved by python -m app.train")
        DistressNet().load_state_dict(saved, strict=False)
    elif name.endswith(".pt"):
        saved = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or not HEAD_KEYS <= saved.keys():
            raise ValueError("it is not a WavLM voice head saved by python -m app.voice")
    elif name.endswith(".json"):
        json.loads(data)


@app.post("/api/voice/results")
async def load_kaggle_results(files: list[UploadFile] = File(...)):
    """Install distressnet_results.zip from Kaggle (or the files inside it) into backend/artifacts."""
    found: dict[str, bytes] = {}
    for upload in files:
        data = await upload.read()
        name = Path(upload.filename or "").name
        if name.endswith(".zip"):
            try:
                archive = zipfile.ZipFile(io.BytesIO(data))
            except zipfile.BadZipFile:
                raise HTTPException(status_code=400, detail=f"{name} is not a zip file.")
            for member in archive.infolist():
                # Only the base name counts, so a path inside the zip cannot write outside artifacts/.
                inner = Path(member.filename).name
                if inner in KAGGLE_RESULTS and not member.is_dir():
                    found[inner] = archive.read(member)
        elif name in KAGGLE_RESULTS:
            found[name] = data
    if not found:
        raise HTTPException(
            status_code=400,
            detail="No Kaggle results found. Choose distressnet_results.zip from the notebook's Output tab.",
        )
    for name, data in found.items():
        try:
            _check_result(name, data)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"{name} could not be read: {exc}")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for name, data in found.items():
        temp = ARTIFACTS / f".{name}.part"
        temp.write_bytes(data)
        temp.replace(ARTIFACTS / name)
    voice = _voice_model()
    return {
        "ok": True,
        "installed": sorted(found),
        "voice_model": "wavlm" if voice is not None else "cnn",
        "voice_error": _voice_error,
    }


def _read_progress() -> tuple[list[dict], dict | None]:
    history: list[dict] = []
    best = None
    if not PROGRESS_PATH.exists():
        return history, best
    for line in PROGRESS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") == "epoch":
            history.append(event["row"])
            best = event["best"]
    return history, best


@app.post("/api/train/start")
def start_training():
    global _train_proc
    if _train_proc is not None and _train_proc.poll() is None:
        raise HTTPException(status_code=409, detail="Training is already running.")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()
    log = TRAIN_LOG.open("w", encoding="utf-8")
    _train_proc = subprocess.Popen(
        [sys.executable, "-m", "app.train"],
        cwd=str(Path(__file__).resolve().parent.parent),
        env={**os.environ, "TRAIN_PROGRESS": str(PROGRESS_PATH)},
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    log.close()
    return {"ok": True, "epochs": EPOCHS}


@app.get("/api/train/live")
def train_live():
    history, best = _read_progress()
    running = _train_proc is not None and _train_proc.poll() is None
    code = None if _train_proc is None else _train_proc.poll()
    if running:
        phase = "training" if history else "loading"
    elif code not in (None, 0):
        phase = "error"
    elif history:
        phase = "done"
    else:
        phase = "idle"
    error = ""
    if phase == "error" and TRAIN_LOG.exists():
        tail = TRAIN_LOG.read_text(encoding="utf-8").strip().splitlines()
        error = tail[-1] if tail else "Training stopped before it finished."
    return {
        "running": running,
        "phase": phase,
        "epoch": history[-1]["epoch"] if history else 0,
        "epochs": EPOCHS,
        "history": history,
        "best": best,
        "error": error,
    }


@app.get("/api/training")
def training():
    report = _report()
    if report is None:
        raise HTTPException(status_code=404, detail="Training report not found.")
    return report


@app.get("/api/samples")
def samples():
    report = _report()
    if report is None:
        return {"samples": []}
    preds = report["predictions"]
    calm = next((item for item in preds if item["truth"] == "calm" and item["correct"]), None)
    distress = next((item for item in preds if item["truth"] == "distress" and item["correct"]), None)
    chosen = [item for item in (calm, distress) if item]
    return {
        "samples": [
            {
                "file": item["file"],
                "actor": item["actor"],
                "emotion": item["emotion"],
                "intensity": item["intensity"],
                "truth": item["truth"],
            }
            for item in chosen
        ]
    }


@app.get("/api/clips/{actor}/{name}")
def clip_audio(actor: str, name: str):
    path = _resolve_clip(f"{actor}/{name}")
    return FileResponse(path, media_type="audio/wav", filename=name)


@app.post("/api/analyze/clip")
def analyze_clip(body: ClipBody, x_session: str = Header("")):
    path = _resolve_clip(body.file)
    result = _predict(path, body.transcript, x_session)
    result["file"] = body.file
    return result


@app.post("/api/analyze/words")
def analyze_words(body: WordsBody):
    return _predict(None, body.transcript)


@app.get("/api/enroll")
def enroll_status(x_session: str = Header("")):
    return _enroll_status(x_session)


@app.post("/api/enroll")
async def enroll(file: UploadFile = File(...), x_session: str = Header("")):
    """Add one calm reading for this session. Its WavLM features are stored; the stats are used from the third reading on."""
    wavlm = _voice_model()
    status = _enroll_status(x_session)
    if not status["available"]:
        raise HTTPException(status_code=409, detail=status["note"])
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The recording was empty.")
    if len(data) > 12_000_000:
        raise HTTPException(status_code=400, detail="Clip is too large.")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "calm.wav"
        path.write_bytes(data)
        try:
            feats = wavlm.features(str(path))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Could not read that clip. {exc}") from exc
    try:
        _enrollments.add(x_session, feats)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Missing or bad session id.") from exc
    return _enroll_status(x_session)


@app.delete("/api/enroll")
def enroll_reset(x_session: str = Header("")):
    try:
        _enrollments.clear(x_session)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Missing or bad session id.") from exc
    return _enroll_status(x_session)


@app.post("/api/analyze/upload")
async def analyze_upload_file(file: UploadFile = File(...), transcript: str = Form(""), x_session: str = Header("")):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Upload a .wav clip.")
    data = await file.read()
    if len(data) > 12_000_000:
        raise HTTPException(status_code=400, detail="Clip is too large.")
    tmp = ARTIFACTS / "_upload.wav"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(data)
    try:
        result = _predict(tmp, transcript, x_session)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not read that clip. {exc}") from exc
    result["file"] = file.filename
    return result


@app.post("/api/chat")
def chat(body: ChatBody):
    history = [{"role": turn.role, "text": turn.text} for turn in body.history[-8:]]
    text, state = respond(body.message, _report(), body.analysis, history, body.state)
    # Quick replies and a card (exercise, mood scale, contacts) belong to this turn only.
    turn = dict(state) if state else {}
    card = turn.pop("card", None)
    quick = turn.pop("prompts", None)
    return {"reply": text, "prompts": quick or prompts(body.analysis), "state": turn or None, "card": card}


def _slim(analysis: dict) -> dict:
    """The parts of a reading the chat needs (no spectrogram, taps or embedding)."""
    keys = ("file", "label", "band", "probability", "logit", "modality", "transcript", "voice", "language", "judge")
    return {key: analysis[key] for key in keys if key in analysis}


@app.post("/api/chat/voice")
async def chat_voice(file: UploadFile = File(...), payload: str = Form("{}"), x_session: str = Header("")):
    """One spoken turn: transcribe it, read the voice again, then answer like /api/chat.

    Every turn is scored, so the voice keeps counting after the first clip, and the
    crisis screen sees the words exactly as it does for typed messages.
    """
    import tempfile

    try:
        body = json.loads(payload or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Bad chat payload.") from exc
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="The recording was empty.")
    if len(data) > 12_000_000:
        raise HTTPException(status_code=400, detail="That recording is too long.")
    if not stt.available():
        raise HTTPException(status_code=503, detail="Speech to text is not installed (pip install faster-whisper).")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "turn.wav"
        path.write_bytes(data)
        try:
            reading = _predict(path, "", x_session)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Could not read that recording. {exc}") from exc

    heard = reading.get("transcript", "").strip()
    turn_reading = _slim(reading)
    if not heard:
        return {
            "heard": "",
            "reply": "Sorry, I didn't catch that. Could you say it again?",
            "prompts": [],
            "state": body.get("state"),
            "card": None,
            "reading": turn_reading,
        }

    history = [
        {"role": str(turn.get("role", "")), "text": str(turn.get("text", ""))}
        for turn in (body.get("history") or [])[-8:]
        if isinstance(turn, dict)
    ]
    state = body.get("state")
    # Before a support conversation starts, this turn's reading can open one, just
    # like the first clip. Once it is running, the agent leads on the words.
    analysis = turn_reading if not (state or {}).get("active") else (body.get("analysis") or turn_reading)
    text, state = respond(heard[:2000], _report(), analysis, history, state)
    turn = dict(state) if state else {}
    card = turn.pop("card", None)
    quick = turn.pop("prompts", None)
    return {
        "heard": heard,
        "reply": text,
        "prompts": quick or prompts(analysis),
        "state": turn or None,
        "card": card,
        "reading": turn_reading,
    }
