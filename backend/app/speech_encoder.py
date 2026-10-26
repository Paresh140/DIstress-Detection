"""A frozen pretrained speech encoder (WavLM or wav2vec 2.0) with a small trained head.

The encoder never trains. Each clip goes through it once, and every hidden
layer is average-pooled into BINS equal stretches of time, so a clip becomes
[layers, BINS, dim]. Those arrays are cached on disk. The head learns a softmax
weight per layer (as in the SUPERB benchmark), mixes the layers, pools the time
bins with attentive statistics pooling (or a plain mean), and an MLP maps the
result to four emotion logits: non-distress (neutral or calm), sad, angry,
fearful. P(distress) is the sum of the three negative classes. Only the head
trains, so a fold takes seconds once the cache exists.
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
MAX_SECONDS = 8.0  # RAVDESS speech clips run about 3 to 5 seconds, CREMA-D 2 to 4
BINS = 16  # time stretches kept per layer, about 0.2 to 0.5 s each
HEAD_EPOCHS = 100
HEAD_BATCH = 32
HEAD_LR = 1e-3
HEAD_WEIGHT_DECAY = 1e-2
HEAD_HIDDEN = 256
HEAD_DROPOUT = 0.3

# Four-class targets. Index 0 is everything that is not distress.
CLASSES = ("non-distress", "sad", "angry", "fearful")
EMOTION_CLASS = {"neutral": 0, "calm": 0, "sad": 1, "angry": 2, "fearful": 3}
HEADS = {
    "four-attentive": {"classes": 4, "pool": "attentive"},
    "four-mean": {"classes": 4, "pool": "mean"},
    "binary-attentive": {"classes": 2, "pool": "attentive"},
    "binary-mean": {"classes": 2, "pool": "mean"},  # the head in the first WavLM runs
}
DEFAULT_HEAD = "four-attentive"


def targets(clips: list, head: str = DEFAULT_HEAD) -> np.ndarray:
    """Class index per clip for a four-class head, or the 0/1 distress label for a binary one."""
    if HEADS[head]["classes"] == 4:
        return np.array([EMOTION_CLASS[clip.emotion] for clip in clips], dtype=np.int64)
    return np.array([clip.label for clip in clips], dtype=np.int64)


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
    """Every hidden state (including the CNN output) averaged into BINS stretches: [layers, BINS, dim]."""
    inputs = extractor(waveform, sampling_rate=TARGET_SR, return_tensors="pt")
    out = model(inputs["input_values"].to(dev), output_hidden_states=True)
    hidden = torch.stack(out.hidden_states, dim=1)[0]  # [layers, frames, dim]
    binned = nn.functional.adaptive_avg_pool1d(hidden.transpose(1, 2), BINS)  # [layers, dim, BINS]
    return binned.transpose(1, 2).float().cpu().numpy()


def cache_path(artifacts: Path, encoder: str) -> Path:
    return artifacts / "ssl_cache" / f"{encoder.strip('/').replace('/', '__')}__bins{BINS}.npz"


def embed_clips(paths: list[str], encoder: str, cache_file: Path, extractor=None, model=None) -> np.ndarray:
    """Binned layer features for each path, [clips, layers, BINS, dim] in float16. Reuses and extends cache_file."""
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
            if i % 250 == 0:
                print(f"  {i}/{len(missing)}")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(cached)
        np.savez(cache_file, paths=np.array(keys), feats=np.stack([cached[k] for k in keys]))
    return np.stack([cached[path] for path in paths])


class LayerMixHead(nn.Module):
    """Softmax layer weights, then attentive statistics pooling (or a mean) over time, then an MLP."""

    def __init__(
        self,
        n_layers: int,
        dim: int,
        classes: int = 4,
        pool: str = "attentive",
        hidden: int = HEAD_HIDDEN,
        dropout: float = HEAD_DROPOUT,
    ):
        super().__init__()
        self.pool = pool
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        if pool == "attentive":
            self.attention = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))
        pooled = 2 * dim if pool == "attentive" else dim
        out = 1 if classes == 2 else classes
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pooled, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.layer_logits, dim=0)

    def time_weights(self, mixed: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.attention(mixed).squeeze(-1), dim=1)  # [batch, BINS]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, layers, BINS, dim]
        mixed = (self.layer_weights()[None, :, None, None] * x).sum(dim=1)  # [batch, BINS, dim]
        if self.pool == "attentive":
            alpha = self.time_weights(mixed).unsqueeze(-1)
            mean = (alpha * mixed).sum(dim=1)
            std = ((alpha * mixed.pow(2)).sum(dim=1) - mean.pow(2)).clamp_min(1e-6).sqrt()
            pooled = torch.cat([mean, std], dim=1)
        else:
            pooled = mixed.mean(dim=1)
        out = self.mlp(pooled)
        return out.squeeze(1) if out.shape[1] == 1 else out


class Standardizer:
    """Per layer and channel mean and std from the training clips, applied batch by batch."""

    def __init__(self, train_x: np.ndarray, chunk: int = 256):
        axes = (0, 2) if train_x.ndim == 4 else (0,)
        total = sq = 0.0
        count = 0
        for start in range(0, len(train_x), chunk):
            block = train_x[start : start + chunk].astype(np.float64)
            total = total + block.sum(axis=axes, keepdims=True)[0]
            sq = sq + (block**2).sum(axis=axes, keepdims=True)[0]
            count += len(block) * (block.shape[2] if block.ndim == 4 else 1)
        mean = total / count
        std = np.sqrt(np.maximum(sq / count - mean**2, 0.0)) + 1e-6
        self.mean = torch.from_numpy(mean.astype(np.float32))
        self.std = torch.from_numpy(std.astype(np.float32))

    def __call__(self, x: np.ndarray, dev: torch.device) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(x)).to(dev).float()
        return (t - self.mean.to(dev)) / self.std.to(dev)


@torch.no_grad()
def _logits(head: LayerMixHead, scale: Standardizer, x: np.ndarray, dev: torch.device) -> torch.Tensor:
    head.eval()
    return torch.cat([head(scale(x[i : i + 256], dev)) for i in range(0, len(x), 256)])


def _distress_prob(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return torch.sigmoid(logits)
    return 1.0 - torch.softmax(logits, dim=1)[:, 0]


def fit_head(
    train_x: np.ndarray,
    train_t: np.ndarray,
    val_x: np.ndarray,
    val_t: np.ndarray,
    epochs: int = HEAD_EPOCHS,
    head: str = DEFAULT_HEAD,
) -> dict:
    """Train the head on binned features. The epoch is chosen on the validation speakers.

    train_t and val_t come from targets(): class indices for a four-class head,
    0/1 labels for a binary one. Selection uses the binary distress decision
    (P(distress) >= 0.5) so every head is chosen the same way.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    spec = HEADS[head]
    dev = device()
    scale = Standardizer(train_x)
    yt = torch.from_numpy(train_t.astype(np.int64))
    yv = torch.from_numpy(val_t.astype(np.int64)).to(dev)
    val_label = (val_t > 0).astype(np.int64)
    if spec["classes"] == 2:
        n_pos = int(train_t.sum())
        pos_weight = torch.tensor([(len(train_t) - n_pos) / max(n_pos, 1)], dtype=torch.float32, device=dev)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_fn = lambda logit, y: bce(logit, y.float())  # noqa: E731
    else:
        counts = np.bincount(train_t, minlength=spec["classes"]).astype(np.float64)
        weight = torch.tensor(len(train_t) / (spec["classes"] * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
        loss_fn = nn.CrossEntropyLoss(weight=weight)

    model = LayerMixHead(train_x.shape[1], train_x.shape[-1], classes=spec["classes"], pool=spec["pool"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY)
    best = {"score": -1.0, "val_loss": float("inf"), "state": None, "epoch": 0}
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(yt))
        for start in range(0, len(order), HEAD_BATCH):
            idx = order[start : start + HEAD_BATCH].numpy()
            opt.zero_grad()
            loss = loss_fn(model(scale(train_x[np.sort(idx)], dev)), yt[np.sort(idx)].to(dev))
            loss.backward()
            opt.step()
        val_logit = _logits(model, scale, val_x, dev)
        val_loss = float(loss_fn(val_logit, yv))
        pred = (_distress_prob(val_logit) >= 0.5).long().cpu().numpy()
        score = _binary_metrics(val_label, pred)["balanced_accuracy"]
        # Ties on balanced accuracy go to the lower validation loss, as in fit_audio.
        if (score, -val_loss) > (best["score"], -best["val_loss"]):
            best = {
                "score": score,
                "val_loss": val_loss,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

    model.load_state_dict(best["state"])
    return {
        "head": model,
        "scale": scale,
        "spec": head,
        "epoch": best["epoch"],
        "val_balanced_accuracy": best["score"],
        "layer_weights": [round(float(w), 4) for w in model.layer_weights().detach().cpu()],
    }


def predict_logits(fitted: dict, x: np.ndarray) -> np.ndarray:
    return _logits(fitted["head"], fitted["scale"], x, device()).cpu().numpy()


def predict(fitted: dict, x: np.ndarray) -> np.ndarray:
    """Uncalibrated P(distress) per clip."""
    return _distress_prob(torch.from_numpy(predict_logits(fitted, x))).numpy()
