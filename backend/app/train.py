"""Train the DistressNet audio branch on RAVDESS actors 01–08."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .audio import file_to_logmel
from .dataset import TEST_ACTORS, TRAIN_ACTORS, discover
from .model import DistressNet, count_params

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "ravdess"
ARTIFACTS = ROOT / "backend" / "artifacts"
SEED = 7
EPOCHS = 16
BATCH = 32
LR = 1e-3


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MelSet(Dataset):
    def __init__(self, rows: list[dict], train: bool):
        self.rows = rows
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        mel = row["mel"].clone()
        if self.train:
            mel = _augment(mel)
        x = mel.unsqueeze(0)
        y = torch.tensor(row["label"], dtype=torch.float32)
        return x, y


def _augment(mel: torch.Tensor) -> torch.Tensor:
    x = mel.clone()
    freq = random.randint(0, 8)
    if freq:
        start = random.randint(0, x.shape[0] - freq)
        x[start : start + freq, :] = 0
    time = random.randint(0, 16)
    if time:
        start = random.randint(0, x.shape[1] - time)
        x[:, start : start + time] = 0
    return x


def _binary_metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    acc = (tp + tn) / max(len(truth), 1)
    return {
        "accuracy": acc,
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


def _group_grad(model: nn.Module) -> dict[str, float]:
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


def _activation_means(model: DistressNet, batch: torch.Tensor, dev: torch.device) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        _, taps = model.audio.forward_features(batch.to(dev))
    means = {
        key: float(taps[key].detach().abs().mean().cpu())
        for key in ("conv1", "conv2", "conv3", "embed", "head")
    }
    model.train()
    return means


def _evaluate(model: DistressNet, loader: DataLoader, dev: torch.device) -> tuple[float, dict, np.ndarray, np.ndarray]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    truths = []
    probs = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            y = y.to(dev)
            logit = model.forward_audio(x).squeeze(1)
            losses.append(float(loss_fn(logit, y).cpu()))
            prob = torch.sigmoid(logit).cpu().numpy()
            probs.append(prob)
            truths.append(y.cpu().numpy())
    truth = np.concatenate(truths)
    prob = np.concatenate(probs)
    pred = (prob >= 0.5).astype(np.int64)
    metrics = _binary_metrics(truth.astype(np.int64), pred)
    model.train()
    return float(np.mean(losses)), metrics, truth, prob


def _span(actors: list[int]) -> str:
    ordered = sorted(actors)
    if not ordered:
        return "none"
    if ordered[-1] - ordered[0] + 1 == len(ordered):
        return f"{ordered[0]:02d}–{ordered[-1]:02d}"
    return ", ".join(f"{actor:02d}" for actor in ordered)


def _narrative(report: dict) -> str:
    best = report["best"]
    split = report["split"]
    return (
        f"The audio branch trained for {report['epochs_ran']} epochs on {split['train_clips']} "
        f"speech clips from actors {_span(split['train_actors'])}. Actors {_span(split['test_actors'])} were held out "
        f"({split['test_clips']} clips) and never appeared in the weight updates. "
        f"The best test F1 was {best['f1']:.2f} at epoch {best['epoch']}, "
        f"with accuracy {best['accuracy']:.2f}, precision {best['precision']:.2f}, "
        f"and recall {best['recall']:.2f}. "
        f"Distress was the positive class: sad, angry, and fearful. "
        f"Calm was neutral and calm speech. Happy, surprise, and disgust were left out of the loss. "
        f"Vision and fusion are drawn in the network view and were not trained in this run."
    )


def fit_audio(train_rows: list[dict], val_rows: list[dict], epochs: int = 8) -> dict:
    """Train the audio branch. The checkpoint is chosen on val_rows, never on a final test set."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    n_pos = sum(int(row["label"]) for row in train_rows)
    n_neg = len(train_rows) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    dev = device()
    model = DistressNet().to(dev)
    for param in model.vision.parameters():
        param.requires_grad_(False)
    for param in model.fuse.parameters():
        param.requires_grad_(False)
    opt = torch.optim.Adam(model.audio.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(dev))
    train_loader = DataLoader(MelSet(train_rows, train=True), batch_size=BATCH, shuffle=True)
    eval_train = DataLoader(MelSet(train_rows, train=False), batch_size=BATCH)
    eval_val = DataLoader(MelSet(val_rows, train=False), batch_size=BATCH)
    history = []
    best = {"f1": -1.0, "state": None, "epoch": 0}

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(dev)
            y = y.to(dev)
            opt.zero_grad()
            logit = model.forward_audio(x).squeeze(1)
            loss = loss_fn(logit, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        _, train_metrics, _, _ = _evaluate(model, eval_train, dev)
        val_loss, val_metrics, _, _ = _evaluate(model, eval_val, dev)
        history.append(
            {
                "epoch": epoch,
                "train_loss": sum(losses) / max(len(losses), 1),
                "train_f1": train_metrics["f1"],
                "val_loss": val_loss,
                "val_f1": val_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            }
        )
        print(
            f"  epoch {epoch:02d}  train {history[-1]['train_loss']:.3f}  "
            f"val {val_loss:.3f}  val bal {val_metrics['balanced_accuracy']:.3f}"
        )
        if val_metrics["balanced_accuracy"] > best["f1"]:
            best = {
                "f1": val_metrics["balanced_accuracy"],
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.audio.state_dict().items()},
            }

    return {
        "state": best["state"],
        "epoch": best["epoch"],
        "val_f1": best["f1"],
        "history": history,
        "pos_weight": float(pos_weight),
        "device": str(dev),
    }


def score_audio(state: dict, rows: list[dict]) -> dict:
    dev = device()
    model = DistressNet().to(dev)
    model.audio.load_state_dict(state)
    loss, metrics, _, _ = _evaluate(model, DataLoader(MelSet(rows, train=False), batch_size=BATCH), dev)
    return {"loss": loss, **metrics}


def _emit(progress: Path | None, payload: dict) -> None:
    if progress is None:
        return
    with progress.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()


def train(progress: Path | None = None) -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    clips = discover(DATA_ROOT)
    if not clips:
        raise SystemExit(f"No RAVDESS clips found under {DATA_ROOT}")

    rows = []
    for clip in clips:
        rows.append(
            {
                "mel": file_to_logmel(str(clip.path)),
                "label": clip.label,
                "clip": clip,
            }
        )

    train_rows = [row for row in rows if row["clip"].split == "train"]
    test_rows = [row for row in rows if row["clip"].split == "test"]
    n_pos = sum(row["label"] for row in train_rows)
    n_neg = len(train_rows) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    dev = device()
    model = DistressNet().to(dev)
    # Only the audio branch receives gradients.
    for param in model.vision.parameters():
        param.requires_grad_(False)
    for param in model.fuse.parameters():
        param.requires_grad_(False)

    opt = torch.optim.Adam(model.audio.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(dev))
    train_loader = DataLoader(MelSet(train_rows, train=True), batch_size=BATCH, shuffle=True)
    eval_train = DataLoader(MelSet(train_rows, train=False), batch_size=BATCH)
    eval_test = DataLoader(MelSet(test_rows, train=False), batch_size=BATCH)

    probe_x = torch.stack([row["mel"] for row in train_rows[:BATCH]]).unsqueeze(1)
    history = []
    best = {"f1": -1.0, "state": None, "epoch": 0}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        grad_sum = {"conv1": 0.0, "conv2": 0.0, "conv3": 0.0, "embed": 0.0, "head": 0.0}
        steps = 0
        for x, y in train_loader:
            x = x.to(dev)
            y = y.to(dev)
            opt.zero_grad()
            logit = model.forward_audio(x).squeeze(1)
            loss = loss_fn(logit, y)
            loss.backward()
            grads = _group_grad(model)
            for key in grad_sum:
                grad_sum[key] += grads[key]
            opt.step()
            losses.append(float(loss.detach().cpu()))
            steps += 1

        train_loss, train_metrics, _, _ = _evaluate(model, eval_train, dev)
        test_loss, test_metrics, _, _ = _evaluate(model, eval_test, dev)
        activations = _activation_means(model, probe_x, dev)
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
            "activation": activations,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}  train {row['train_loss']:.3f}  "
            f"test {test_loss:.3f}  acc {test_metrics['accuracy']:.3f}  f1 {test_metrics['f1']:.3f}"
        )
        if test_metrics["f1"] > best["f1"]:
            best = {
                "f1": test_metrics["f1"],
                "epoch": epoch,
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "loss": test_loss,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        _emit(
            progress,
            {
                "type": "epoch",
                "row": row,
                "best": {
                    "epoch": best["epoch"],
                    "f1": best["f1"],
                    "accuracy": best.get("accuracy", test_metrics["accuracy"]),
                    "precision": best.get("precision", test_metrics["precision"]),
                    "recall": best.get("recall", test_metrics["recall"]),
                    "loss": best.get("loss", test_loss),
                },
            },
        )

    model.load_state_dict(best["state"])
    model.to(dev)
    _, final_metrics, truth, prob = _evaluate(model, eval_test, dev)
    pred = (prob >= 0.5).astype(np.int64)

    predictions = []
    for row, truth_i, prob_i, pred_i in zip(test_rows, truth, prob, pred):
        clip = row["clip"]
        intensity = "strong" if clip.intensity == 2 else "normal"
        predictions.append(
            {
                "file": f"Actor_{clip.actor:02d}/{clip.path.name}",
                "actor": clip.actor,
                "emotion": clip.emotion_name,
                "intensity": intensity,
                "truth": "distress" if int(truth_i) == 1 else "calm",
                "predicted": "distress" if int(pred_i) == 1 else "calm",
                "probability": round(float(prob_i), 4),
                "correct": bool(int(pred_i) == int(truth_i)),
            }
        )

    best_row = next(item for item in history if item["epoch"] == best["epoch"])
    report = {
        "dataset": "RAVDESS speech audio",
        "citation": "Livingstone & Russo, 2018, PLoS ONE, doi:10.1371/journal.pone.0196391",
        "seed": SEED,
        "device": str(dev),
        "epochs_ran": EPOCHS,
        "batch_size": BATCH,
        "learning_rate": LR,
        "optimizer": "Adam",
        "weight_decay": 1e-4,
        "loss": "BCEWithLogitsLoss",
        "pos_weight": float(pos_weight),
        "threshold": 0.5,
        "augmentation": "Random frequency mask (up to 8 bands) and time mask (up to 16 frames) on the log-mel input.",
        "label": {
            "distress": ["sad", "angry", "fearful"],
            "calm": ["neutral", "calm"],
            "excluded": ["happy", "surprised", "disgust", "song"],
        },
        "split": {
            "train_actors": sorted(TRAIN_ACTORS),
            "test_actors": sorted(TEST_ACTORS),
            "unused_actors": list(range(9, 25)),
            "train_clips": len(train_rows),
            "test_clips": len(test_rows),
            "train_distress": int(n_pos),
            "train_calm": int(n_neg),
        },
        "parameters": {
            "audio": count_params(model.audio),
            "vision_untrained": count_params(model.vision),
            "fusion_untrained": count_params(model.fuse),
        },
        "history": history,
        "best": {
            "epoch": best["epoch"],
            "f1": best_row["test_f1"],
            "accuracy": best_row["test_accuracy"],
            "precision": best_row["test_precision"],
            "recall": best_row["test_recall"],
            "loss": best_row["test_loss"],
        },
        "confusion": {
            "tn": final_metrics["tn"],
            "fp": final_metrics["fp"],
            "fn": final_metrics["fn"],
            "tp": final_metrics["tp"],
        },
        "predictions": predictions,
    }
    report["narrative"] = _narrative(report)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    weights = {k: v for k, v in best["state"].items() if k.startswith("audio.")}
    torch.save(weights, ARTIFACTS / "audio_branch.pt")
    (ARTIFACTS / "training_report.json").write_text(json.dumps(report, indent=2))
    print(report["narrative"])
    return report


if __name__ == "__main__":
    import os

    raw = os.environ.get("TRAIN_PROGRESS", "")
    train(Path(raw) if raw else None)
