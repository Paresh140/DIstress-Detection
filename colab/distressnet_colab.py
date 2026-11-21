"""Full RAVDESS speech training for a Google Colab GPU.

Run this with a GPU runtime. It writes clips, weights, and the training report
into My Drive/DistressNet for the account that mounts Drive.
"""

from __future__ import annotations

import json
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 7
EPOCHS = 24
BATCH = 32
LR = 1e-3
TRAIN_ACTORS = set(range(1, 19))
VAL_ACTORS = {19, 20}
TEST_ACTORS = {21, 22, 23, 24}
DISTRESS = {4, 5, 6}
CALM = {1, 2}
NAMES = {1: "neutral", 2: "calm", 4: "sad", 5: "angry", 6: "fearful"}


def mount_and_prepare() -> tuple[Path, Path, Path]:
    from google.colab import drive

    drive.mount("/content/drive")
    root = Path("/content/drive/MyDrive/DistressNet")
    ravdess = root / "ravdess"
    artifacts = root / "artifacts"
    ravdess.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    return root, ravdess, artifacts


def download_speech(ravdess: Path) -> None:
    from huggingface_hub import snapshot_download

    print("Downloading all 24 actors of RAVDESS speech into Drive…")
    snapshot_download(
        repo_id="birgermoell/ravdess",
        repo_type="dataset",
        allow_patterns=["**/03-01-*.wav"],
        local_dir=str(ravdess / "_hf"),
    )
    for path in (ravdess / "_hf").rglob("03-01-*.wav"):
        actor = path.name.split("-")[-1][:2]
        dest = ravdess / f"Actor_{actor}"
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / path.name
        if not target.exists():
            shutil.copy2(path, target)
    shutil.rmtree(ravdess / "_hf", ignore_errors=True)
    count = len(list(ravdess.glob("Actor_*/*.wav")))
    if count >= 1400:
        print(f"Speech clips on Drive: {count}")
        return
    print(f"Hugging Face returned {count} files. Trying the Zenodo speech archive…")
    import urllib.request

    zip_path = ravdess / "Audio_Speech_Actors_01-24.zip"
    urllib.request.urlretrieve(
        "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1",
        zip_path,
    )
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(ravdess)
    zip_path.unlink(missing_ok=True)
    for path in list(ravdess.rglob("03-01-*.wav")):
        actor = path.name.split("-")[-1][:2]
        dest = ravdess / f"Actor_{actor}"
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / path.name
        if path.resolve() != target.resolve():
            shutil.move(str(path), target)
    print("Speech clips on Drive:", len(list(ravdess.glob("Actor_*/*.wav"))))


def logmel(path: Path) -> torch.Tensor:
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32)
    target = 16_000
    if sr != target:
        g = int(np.gcd(target, sr))
        data = resample_poly(data, target // g, sr // g).astype(np.float32)
    n = target * 3
    if len(data) >= n:
        start = (len(data) - n) // 2
        data = data[start : start + n]
    else:
        data = np.pad(data, (0, n - len(data)))
    data = data / (np.max(np.abs(data)) + 1e-8)
    spec = torch.stft(
        torch.from_numpy(data.astype(np.float32)),
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=torch.hann_window(512),
        return_complex=True,
        center=True,
    ).abs()
    n_mels, n_fft = 64, 512
    n_freqs = n_fft // 2 + 1

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(0), hz_to_mel(target / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / target).astype(int)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        center = max(center, left + 1)
        right = max(right, center + 1)
        for j in range(left, center):
            if 0 <= j < n_freqs:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if 0 <= j < n_freqs:
                fb[i, j] = (right - j) / (right - center)
    logmel = np.log(fb @ spec.numpy() + 1e-6).astype(np.float32)
    dst = np.linspace(0, logmel.shape[1] - 1, 128)
    src = np.linspace(0, logmel.shape[1] - 1, logmel.shape[1])
    resized = np.vstack([np.interp(dst, src, row) for row in logmel]).astype(np.float32)
    tensor = torch.from_numpy(resized)
    return (tensor - tensor.mean()) / (tensor.std() + 1e-6)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.relu(self.bn(self.conv(x))))


class AudioBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(1, 16)
        self.conv2 = ConvBlock(16, 32)
        self.conv3 = ConvBlock(32, 64)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.head = nn.Linear(64, 1)

    def forward_features(self, x: torch.Tensor):
        taps = {}
        h = self.conv1(x)
        taps["conv1"] = h
        h = self.conv2(h)
        taps["conv2"] = h
        h = self.conv3(h)
        taps["conv3"] = h
        h = torch.relu(self.fc1(torch.flatten(self.pool(h), 1)))
        h = self.drop(h)
        emb = torch.relu(self.fc2(h))
        taps["embed"] = emb
        logit = self.head(emb)
        taps["head"] = logit
        return logit, taps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit, _ = self.forward_features(x)
        return logit


class MelSet(Dataset):
    def __init__(self, rows: list[dict], train: bool):
        self.rows = rows
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        mel = self.rows[index]["mel"].clone()
        if self.train:
            freq = random.randint(0, 8)
            if freq:
                start = random.randint(0, mel.shape[0] - freq)
                mel[start : start + freq, :] = 0
            time = random.randint(0, 16)
            if time:
                start = random.randint(0, mel.shape[1] - time)
                mel[:, start : start + time] = 0
        return mel.unsqueeze(0), torch.tensor(self.rows[index]["label"], dtype=torch.float32)


def discover(ravdess: Path) -> list[dict]:
    clips = []
    for path in sorted(ravdess.glob("Actor_*/*.wav")):
        parts = path.stem.split("-")
        if len(parts) != 7:
            continue
        modality, channel, emotion, intensity, _statement, _rep, actor = (int(p) for p in parts)
        if modality != 3 or channel != 1 or emotion not in DISTRESS | CALM:
            continue
        clips.append(
            {
                "path": path,
                "actor": actor,
                "emotion": emotion,
                "intensity": intensity,
                "label": 1 if emotion in DISTRESS else 0,
            }
        )
    return clips


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / max(len(truth), 1),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate(model, loader, dev):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    losses, truths, probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            logit = model(x).squeeze(1)
            losses.append(float(loss_fn(logit, y).cpu()))
            probs.append(torch.sigmoid(logit).cpu().numpy())
            truths.append(y.cpu().numpy())
    truth = np.concatenate(truths)
    prob = np.concatenate(probs)
    pred = (prob >= 0.5).astype(np.int64)
    model.train()
    return float(np.mean(losses)), metrics(truth.astype(np.int64), pred), truth, prob


def group_grad(model) -> dict[str, float]:
    buckets = {"conv1": 0.0, "conv2": 0.0, "conv3": 0.0, "embed": 0.0, "head": 0.0}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        value = float(param.grad.detach().norm().cpu())
        if "conv1" in name:
            buckets["conv1"] += value
        elif "conv2" in name:
            buckets["conv2"] += value
        elif "conv3" in name:
            buckets["conv3"] += value
        elif "fc1" in name or "fc2" in name:
            buckets["embed"] += value
        elif "head" in name:
            buckets["head"] += value
    return buckets


def write_status(artifacts: Path, phase: str, detail: str = "") -> None:
    (artifacts / "train_status.json").write_text(
        json.dumps({"phase": phase, "epochs": EPOCHS, "detail": detail})
    )


def emit_epoch(artifacts: Path, row: dict, best: dict) -> None:
    with (artifacts / "train_live.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "epoch", "row": row, "best": best}) + "\n")


def train(ravdess: Path, artifacts: Path) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    clips = discover(ravdess)
    if len({clip["actor"] for clip in clips}) < 24:
        raise SystemExit(f"Expected actors 01–24, found {sorted({clip['actor'] for clip in clips})}")
    rows = []
    for clip in clips:
        rows.append({**clip, "mel": logmel(clip["path"])})
    splits = {
        "train": [row for row in rows if row["actor"] in TRAIN_ACTORS],
        "val": [row for row in rows if row["actor"] in VAL_ACTORS],
        "test": [row for row in rows if row["actor"] in TEST_ACTORS],
    }
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev)
    model = AudioBranch().to(dev)
    n_pos = sum(row["label"] for row in splits["train"])
    n_neg = len(splits["train"]) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader = DataLoader(MelSet(splits["train"], True), batch_size=BATCH, shuffle=True)
    loaders = {
        name: DataLoader(MelSet(splits[name], False), batch_size=BATCH)
        for name in ("train", "val", "test")
    }
    probe = torch.stack([row["mel"] for row in splits["train"][:BATCH]]).unsqueeze(1)
    history = []
    best = {"score": -1.0, "state": None, "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses, grad_sum, steps = [], {key: 0.0 for key in ("conv1", "conv2", "conv3", "embed", "head")}, 0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(1), y)
            loss.backward()
            grads = group_grad(model)
            for key in grad_sum:
                grad_sum[key] += grads[key]
            opt.step()
            losses.append(float(loss.detach().cpu()))
            steps += 1
        train_loss, train_metrics, _, _ = evaluate(model, loaders["train"], dev)
        _val_loss, val_metrics, _, _ = evaluate(model, loaders["val"], dev)
        test_loss, test_metrics, _, _ = evaluate(model, loaders["test"], dev)
        model.eval()
        with torch.no_grad():
            _, taps = model.forward_features(probe.to(dev))
        activation = {key: float(taps[key].detach().abs().mean().cpu()) for key in grad_sum}
        row = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(len(losses), 1),
            "eval_train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "grad_norm": {key: value / max(steps, 1) for key, value in grad_sum.items()},
            "activation": activation,
        }
        history.append(row)
        if val_metrics["balanced_accuracy"] > best["score"]:
            best = {
                "score": val_metrics["balanced_accuracy"],
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "test": test_metrics,
                "test_loss": test_loss,
            }
        emit_epoch(
            artifacts,
            row,
            {
                "epoch": best["epoch"],
                "f1": best["test"]["f1"],
                "accuracy": best["test"]["accuracy"],
                "precision": best["test"]["precision"],
                "recall": best["test"]["recall"],
                "loss": best["test_loss"],
            },
        )
        print(
            f"epoch {epoch:02d}  train {row['train_loss']:.3f}  "
            f"val bal {val_metrics['balanced_accuracy']:.3f}  test f1 {test_metrics['f1']:.3f}"
        )
    model.load_state_dict(best["state"])
    model.to(dev)
    _, final_metrics, truth, prob = evaluate(model, loaders["test"], dev)
    pred = (prob >= 0.5).astype(np.int64)
    predictions = []
    for row, truth_i, prob_i, pred_i in zip(splits["test"], truth, prob, pred):
        predictions.append(
            {
                "file": f"Actor_{row['actor']:02d}/{row['path'].name}",
                "actor": row["actor"],
                "emotion": NAMES[row["emotion"]],
                "intensity": "strong" if row["intensity"] == 2 else "normal",
                "truth": "distress" if int(truth_i) == 1 else "calm",
                "predicted": "distress" if int(pred_i) == 1 else "calm",
                "probability": round(float(prob_i), 4),
                "correct": bool(int(pred_i) == int(truth_i)),
            }
        )
    chosen = next(item for item in history if item["epoch"] == best["epoch"])
    report = {
        "dataset": "RAVDESS speech audio, actors 01–24",
        "citation": "Livingstone & Russo, 2018, PLoS ONE, doi:10.1371/journal.pone.0196391",
        "machine": "Google Colab GPU",
        "seed": SEED,
        "device": str(dev),
        "epochs_ran": EPOCHS,
        "batch_size": BATCH,
        "learning_rate": LR,
        "optimizer": "Adam",
        "weight_decay": 1e-4,
        "loss": "BCEWithLogitsLoss",
        "pos_weight": float(pos_weight.cpu()),
        "threshold": 0.5,
        "augmentation": "Random frequency mask (up to 8 bands) and time mask (up to 16 frames) on the log-mel input.",
        "label": {
            "distress": ["sad", "angry", "fearful"],
            "calm": ["neutral", "calm"],
            "excluded": ["happy", "surprised", "disgust", "song"],
        },
        "split": {
            "train_actors": sorted(TRAIN_ACTORS),
            "val_actors": sorted(VAL_ACTORS),
            "test_actors": sorted(TEST_ACTORS),
            "unused_actors": [],
            "train_clips": len(splits["train"]),
            "test_clips": len(splits["test"]),
            "train_distress": int(n_pos),
            "train_calm": int(n_neg),
        },
        "parameters": {"audio": sum(p.numel() for p in model.parameters()), "vision_untrained": 89408, "fusion_untrained": 4161},
        "history": history,
        "best": {
            "epoch": best["epoch"],
            "f1": chosen["test_f1"],
            "accuracy": chosen["test_accuracy"],
            "precision": chosen["test_precision"],
            "recall": chosen["test_recall"],
            "loss": chosen["test_loss"],
        },
        "confusion": {key: final_metrics[key] for key in ("tn", "fp", "fn", "tp")},
        "predictions": predictions,
    }
    report["narrative"] = (
        f"This run trained on a Colab GPU for {EPOCHS} epochs. "
        f"Actors 01–18 supplied {len(splits['train'])} speech clips. "
        f"Actors 19–20 chose the checkpoint by balanced accuracy. "
        f"Actors 21–24 scored it once: F1 {chosen['test_f1']:.2f}, "
        f"accuracy {chosen['test_accuracy']:.2f}, precision {chosen['test_precision']:.2f}, "
        f"recall {chosen['test_recall']:.2f}, at epoch {best['epoch']}. "
        "Distress is sad, angry, and fearful speech. Calm is neutral and calm speech. "
        "The face branch and the fusion head were not trained."
    )
    weights = {f"audio.{key}": value for key, value in best["state"].items()}
    torch.save(weights, artifacts / "audio_branch.pt")
    (artifacts / "training_report.json").write_text(json.dumps(report, indent=2))
    write_status(artifacts, "done")
    print(report["narrative"])
    print("Saved", artifacts / "audio_branch.pt")


if __name__ == "__main__":
    _root, ravdess, artifacts = mount_and_prepare()
    progress = artifacts / "train_live.jsonl"
    if progress.exists():
        progress.unlink()
    try:
        write_status(artifacts, "downloading", "Downloading all 24 actors of RAVDESS speech into Drive.")
        download_speech(ravdess)
        write_status(artifacts, "training", "The GPU is training on actors 01–18.")
        train(ravdess, artifacts)
    except Exception as exc:
        write_status(artifacts, "error", str(exc))
        raise
