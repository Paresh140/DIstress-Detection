"""Local files, or the Google Drive copy once Colab has written one."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_RAVDESS = ROOT / "data" / "ravdess"
LOCAL_ARTIFACTS = ROOT / "backend" / "artifacts"


def drive_folder() -> Path | None:
    cloud = Path.home() / "Library/CloudStorage"
    if not cloud.is_dir():
        return None
    matches = sorted(cloud.glob("GoogleDrive-*/My Drive/DistressNet"))
    return matches[0] if matches else None


def clip_root() -> Path:
    folder = drive_folder()
    if folder is not None and any((folder / "ravdess").glob("Actor_*/*.wav")):
        return folder / "ravdess"
    return LOCAL_RAVDESS


def artifact_file(name: str) -> Path:
    folder = drive_folder()
    if folder is not None:
        remote = folder / "artifacts" / name
        if remote.is_file() and _readable(remote):
            return remote
    return LOCAL_ARTIFACTS / name


def _readable(path: Path) -> bool:
    # macOS can list a Drive file yet refuse to open it when the app lacks Drive access.
    try:
        with path.open("rb"):
            return True
    except OSError:
        return False
