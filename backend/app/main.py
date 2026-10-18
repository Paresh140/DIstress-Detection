"""DistressNet studio API."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .audio import file_to_logmel
from .chat import prompts, reply
from .judge import adjudicate
from .language import read_language
from .model import DistressNet, architecture_spec
from .storage import LOCAL_ARTIFACTS, artifact_file, clip_root, drive_folder
from .train import EPOCHS

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


def _predict(path: Path | None, transcript: str = "") -> dict:
    voice_probability = None
    logit_value = 0.0
    tap_means: dict[str, float] = {}
    spec: list = []
    embedding: list = []
    if path is not None:
        if not _weights().exists():
            raise HTTPException(status_code=409, detail="Audio branch weights are not trained yet.")
        mel = file_to_logmel(str(path))
        model = _load_model()
        batch = mel.unsqueeze(0).unsqueeze(0).to(_device())
        with torch.no_grad():
            logit, taps = model.audio.forward_features(batch)
        logit_value = float(logit.squeeze().cpu())
        voice_probability = float(torch.sigmoid(logit).squeeze().cpu())
        tap_means = {
            key: round(float(value.detach().abs().mean().cpu()), 4)
            for key, value in taps.items()
            if key != "input"
        }
        spec = np.round(mel.numpy(), 3).tolist()
        embedding = [round(float(v), 3) for v in taps["embed"].squeeze().detach().cpu().tolist()]
    language = read_language(transcript)
    verdict = adjudicate(voice_probability, language, transcript)
    voice_label = None
    if voice_probability is not None:
        voice_label = "distress" if voice_probability >= 0.5 else "calm"
    return {
        "file": path.name if path else "words",
        "label": verdict["decision"],
        "probability": verdict["confidence"],
        "logit": round(logit_value, 4),
        "threshold": 0.5,
        "modality": (
            "voice+words"
            if transcript.strip() and voice_probability is not None
            else "audio"
            if voice_probability is not None
            else "words"
        ),
        "transcript": transcript.strip(),
        "voice": {
            "label": voice_label,
            "probability": None if voice_probability is None else round(voice_probability, 4),
            "logit": round(logit_value, 4),
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


class ClipBody(BaseModel):
    file: str
    transcript: str = ""


class WordsBody(BaseModel):
    transcript: str = Field(min_length=1, max_length=2000)


@app.get("/api/health")
def health():
    return {"ok": True, "weights": _weights().exists(), "report": _report_path().exists()}


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
    path = ARTIFACTS / "cross_corpus_report.json"
    if not path.exists():
        return {
            "ready": False,
            "summary": (
                "The cross-corpus audio study has not been written yet. "
                "IEMOCAP is not on this machine. The run will train on RAVDESS and test on CREMA-D, then the reverse, "
                "and also hold speakers out inside each corpus."
            ),
        }
    return json.loads(path.read_text())


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


COLAB_URL = "https://colab.research.google.com/drive/191x-_g3EAUwqb1ay1ZnT-Ovqzc1syroU"
GPU_EPOCHS = 24


CONSOLE_LOG = LOCAL_ARTIFACTS / "colab_console.json"
GPU_SESSION = LOCAL_ARTIFACTS / "gpu_session.json"


def _gpu_paths() -> tuple[Path, Path] | None:
    folder = drive_folder()
    if folder is None:
        return None
    artifacts = folder / "artifacts"
    return artifacts / "train_status.json", artifacts / "train_live.jsonl"


def _colab_console() -> dict | None:
    if not CONSOLE_LOG.is_file():
        return None
    payload = json.loads(CONSOLE_LOG.read_text(encoding="utf-8"))
    epochs = payload.get("epochs") or []
    if len(epochs) < GPU_EPOCHS:
        return None
    return payload


@app.post("/api/train/gpu")
def prepare_gpu():
    paths = _gpu_paths()
    if paths is None:
        raise HTTPException(status_code=404, detail="The DistressNet folder is not on this Mac's Google Drive.")
    status_path, progress_path = paths
    status_path.parent.mkdir(parents=True, exist_ok=True)
    for path in (status_path, progress_path):
        if path.exists():
            path.unlink()
    GPU_SESSION.write_text(json.dumps({"started": time.time()}))
    return {"ok": True, "url": COLAB_URL, "epochs": GPU_EPOCHS}


def _console_replay() -> dict | None:
    payload = _colab_console()
    if payload is None:
        return None
    history = [
        {
            "epoch": int(row["epoch"]),
            "train_loss": float(row["train_loss"]),
            "val_balanced_accuracy": float(row["val_balanced_accuracy"]),
            "test_f1": float(row["test_f1"]),
            "grad_norm": {},
            "activation": {},
        }
        for row in payload["epochs"]
    ]
    chosen = payload["best"]
    best = {
        "epoch": int(chosen["epoch"]),
        "f1": float(chosen["f1"]),
        "accuracy": float(chosen["accuracy"]),
        "precision": float(chosen["precision"]),
        "recall": float(chosen["recall"]),
        "loss": None,
    }
    display = {
        "dataset": "RAVDESS speech audio, actors 01–24",
        "citation": "Livingstone & Russo, 2018, PLoS ONE, doi:10.1371/journal.pone.0196391",
        "seed": 7,
        "device": "cuda",
        "epochs_ran": GPU_EPOCHS,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "weight_decay": 0.0001,
        "loss": "BCEWithLogitsLoss",
        "pos_weight": 0.5,
        "threshold": 0.5,
        "augmentation": "Random frequency mask (up to 8 bands) and time mask (up to 16 frames) on the log-mel input.",
        "narrative": payload["narrative"],
        "logged_only": True,
        "label": {
            "distress": ["sad", "angry", "fearful"],
            "calm": ["neutral", "calm"],
            "excluded": ["happy", "surprised", "disgust", "song"],
        },
        "split": {
            "train_actors": list(range(1, 19)),
            "val_actors": [19, 20],
            "test_actors": [21, 22, 23, 24],
            "unused_actors": [],
            "train_clips": 648,
            "test_clips": 144,
            "train_distress": 432,
            "train_calm": 216,
        },
        "parameters": {"audio": 163041, "vision_untrained": 89408, "fusion_untrained": 4161},
        "history": history,
        "best": best,
        "predictions": [],
    }
    return {
        "history": history,
        "best": best,
        "narrative": payload["narrative"],
        "logged_only": True,
        "display": display,
    }


@app.get("/api/train/gpu")
def gpu_live():
    paths = _gpu_paths()
    history: list[dict] = []
    best = None
    phase = "waiting"
    error = ""
    logged_only = False
    narrative = ""
    display = None
    if paths is None:
        phase = "error"
        error = "The DistressNet folder is not on this Mac's Google Drive."
    else:
        status_path, progress_path = paths
        if progress_path.exists():
            history, best = _read_progress_file(progress_path)
        status_phase = ""
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status_phase = str(status.get("phase") or "waiting")
            phase = status_phase
            error = str(status.get("detail") or "")
        elif history:
            phase = "training"
        session_started = 0.0
        if GPU_SESSION.is_file():
            session_started = float(json.loads(GPU_SESSION.read_text(encoding="utf-8")).get("started") or 0)
        replay_allowed = session_started < CONSOLE_LOG.stat().st_mtime if CONSOLE_LOG.is_file() else False
        drive_report = status_path.with_name("training_report.json")
        if not history and status_phase in {"downloading", "training"} and replay_allowed and not drive_report.is_file():
            replay = _console_replay()
            if replay is not None:
                history = replay["history"]
                best = replay["best"]
                narrative = replay["narrative"]
                display = replay["display"]
                logged_only = True
                phase = "done"
                error = ""
    running = phase in {"waiting", "downloading", "training"}
    return {
        "running": running,
        "phase": phase,
        "epoch": history[-1]["epoch"] if history else 0,
        "epochs": GPU_EPOCHS,
        "history": history,
        "best": best,
        "error": error,
        "url": COLAB_URL,
        "logged_only": logged_only,
        "narrative": narrative,
        "display": display,
    }


def _read_progress_file(path: Path) -> tuple[list[dict], dict | None]:
    history: list[dict] = []
    best = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") == "epoch":
            history.append(event["row"])
            best = event["best"]
    return history, best


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
def analyze_clip(body: ClipBody):
    path = _resolve_clip(body.file)
    result = _predict(path, body.transcript)
    result["file"] = body.file
    return result


@app.post("/api/analyze/words")
def analyze_words(body: WordsBody):
    return _predict(None, body.transcript)


@app.post("/api/analyze/upload")
async def analyze_upload_file(file: UploadFile = File(...), transcript: str = Form("")):
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Upload a .wav clip.")
    data = await file.read()
    if len(data) > 12_000_000:
        raise HTTPException(status_code=400, detail="Clip is too large.")
    tmp = ARTIFACTS / "_upload.wav"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(data)
    try:
        result = _predict(tmp, transcript)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not read that clip. {exc}") from exc
    result["file"] = file.filename
    return result


@app.post("/api/chat")
def chat(body: ChatBody):
    history = [{"role": turn.role, "text": turn.text} for turn in body.history[-8:]]
    text = reply(body.message, _report(), body.analysis, history)
    return {"reply": text, "prompts": prompts(body.analysis)}
