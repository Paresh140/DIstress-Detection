"""Where the app reads clips and saved results: data/ravdess and backend/artifacts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_RAVDESS = ROOT / "data" / "ravdess"
LOCAL_ARTIFACTS = ROOT / "backend" / "artifacts"


def clip_root() -> Path:
    return LOCAL_RAVDESS


def artifact_file(name: str) -> Path:
    return LOCAL_ARTIFACTS / name
