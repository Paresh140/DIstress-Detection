"""A frozen pretrained speech encoder (WavLM or wav2vec 2.0) with a small trained head.

The encoder never trains. Each clip goes through it once, every hidden layer is
mean-pooled over time, and the pooled vectors are cached on disk. The head
learns a softmax weight per layer (as in the SUPERB benchmark), mixes the
layers, and maps the result to one distress logit through a small MLP. Only the
head trains, so a fold takes seconds once the cache exists.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.signal import resample_poly

from .audio import TARGET_SR, load_wav
from .train import SEED, _binary_metrics, device

DEFAULT_ENCODER = "microsoft/wavlm-base-plus"
MAX_SECONDS = 6.0  # RAVDESS speech clips run about 3 to 5 seconds
HEAD_EPOCHS = 100
HEAD_BATCH = 32
HEAD_LR = 3e-3
HEAD_WEIGHT_DECAY = 1e-3
HEAD_HIDDEN = 128
HEAD_DROPOUT = 0.3


def load_16k(path: str) -> np.ndarray:
    y, sr = load_wav(path)
    if sr != TARGET_SR:
        g = int(np.gcd(TARGET_SR, sr))
        y = resample_poly(y, TARGET_SR // g, sr // g).astype(np.float32)
    n = int(TARGET_SR * MAX_SECONDS)
    if len(y) > n:
        start = (len(y) - n) // 2
        y = y[start : start + n]
    return y.astype(np.float32)


def load_encoder(name: str = DEFAULT_ENCODER):
    from transformers import AutoFeatureExtractor, AutoModel

    extractor = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    return extractor, model


@torch.no_grad()
def pooled_layers(waveform: np.ndarray, extractor, model, dev: torch.device) -> np.ndarray:
    """Mean over time of every hidden state, including the CNN output: [layers, dim]."""
    inputs = extractor(waveform, sampling_rate=TARGET_SR, return_tensors="pt")
    out = model(inputs["input_values"].to(dev), output_hidden_states=True)
    hidden = torch.stack(out.hidden_states, dim=1)[0]  # [layers, frames, dim]
    return hidden.mean(dim=1).float().cpu().numpy()


def embed_clips(paths: list[str], encoder: str, cache_file: Path, extractor=None, model=None) -> np.ndarray:
    """Pooled layer features for each path, [clips, layers, dim]. Reuses and extends cache_file."""
    cached: dict[str, np.ndarray] = {}
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as data:
            cached = dict(zip(data["paths"].tolist(), data["feats"]))
    missing = [path for path in paths if path not in cached]
    if missing:
        if model is None:
            extractor, model = load_encoder(encoder)
        dev = device()
        model = model.to(dev).eval()
        print(f"{encoder}: embedding {len(missing)} clips on {dev}")
        for i, path in enumerate(missing, start=1):
            cached[path] = pooled_layers(load_16k(path), extractor, model, dev).astype(np.float16)
            if i % 100 == 0:
                print(f"  {i}/{len(missing)}")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(cached)
        np.savez(cache_file, paths=np.array(keys), feats=np.stack([cached[k] for k in keys]))
    return np.stack([cached[path] for path in paths]).astype(np.float32)


class LayerMixHead(nn.Module):
    def __init__(self, n_layers: int, dim: int, hidden: int = HEAD_HIDDEN, dropout: float = HEAD_DROPOUT):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = (self.layer_weights()[None, :, None] * x).sum(dim=1)
        return self.mlp(mixed).squeeze(1)


def _standardizer(train_x: np.ndarray):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    return lambda x: torch.from_numpy(((x - mean) / std).astype(np.float32))


@torch.no_grad()
def _predict(head: LayerMixHead, x: torch.Tensor, dev: torch.device) -> np.ndarray:
    head.eval()
    return torch.sigmoid(head(x.to(dev))).cpu().numpy()


def fit_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    epochs: int = HEAD_EPOCHS,
) -> dict:
    """Train the head on pooled features. The epoch is chosen on the validation speakers."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    dev = device()
    scale = _standardizer(train_x)
    xt, xv = scale(train_x), scale(val_x)
    yt = torch.from_numpy(train_y.astype(np.float32))
    yv = torch.from_numpy(val_y.astype(np.float32)).to(dev)
    n_pos = int(train_y.sum())
    pos_weight = torch.tensor([(len(train_y) - n_pos) / max(n_pos, 1)], dtype=torch.float32, device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    head = LayerMixHead(train_x.shape[1], train_x.shape[2]).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY)
    best = {"score": -1.0, "val_loss": float("inf"), "state": None, "epoch": 0}
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(yt))
        for start in range(0, len(order), HEAD_BATCH):
            idx = order[start : start + HEAD_BATCH]
            opt.zero_grad()
            loss = loss_fn(head(xt[idx].to(dev)), yt[idx].to(dev))
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            val_logit = head(xv.to(dev))
            val_loss = float(loss_fn(val_logit, yv))
        pred = (torch.sigmoid(val_logit) >= 0.5).long().cpu().numpy()
        score = _binary_metrics(val_y.astype(np.int64), pred)["balanced_accuracy"]
        # Ties on balanced accuracy go to the lower validation loss, as in fit_audio.
        if (score, -val_loss) > (best["score"], -best["val_loss"]):
            best = {
                "score": score,
                "val_loss": val_loss,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }

    head.load_state_dict(best["state"])
    return {
        "head": head,
        "scale": scale,
        "epoch": best["epoch"],
        "val_balanced_accuracy": best["score"],
        "layer_weights": [round(float(w), 4) for w in head.layer_weights().detach().cpu()],
    }


def predict(fitted: dict, x: np.ndarray) -> np.ndarray:
    return _predict(fitted["head"], fitted["scale"](x), device())
