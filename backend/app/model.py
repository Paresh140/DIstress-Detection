"""DistressNet: a small convolutional network on log-mel spectrograms.

The vision branch is part of the architecture shown in the studio. This
training run fits the audio branch only, on RAVDESS speech.
"""

from __future__ import annotations

import torch
from torch import nn

from .audio import N_FRAMES, N_MELS


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
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

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        taps: dict[str, torch.Tensor] = {"input": x}
        h = self.conv1(x)
        taps["conv1"] = h
        h = self.conv2(h)
        taps["conv2"] = h
        h = self.conv3(h)
        taps["conv3"] = h
        h = self.pool(h)
        flat = torch.flatten(h, 1)
        h = torch.relu(self.fc1(flat))
        taps["fc1"] = h
        h = self.drop(h)
        emb = torch.relu(self.fc2(h))
        taps["embed"] = emb
        logit = self.head(emb)
        taps["head"] = logit
        return logit, taps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit, _ = self.forward_features(x)
        return logit


class VisionBranch(nn.Module):
    """Face branch. Defined so the studio can show the full network.

    Weights are randomly initialized. They are not trained in this run.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(3, 16)
        self.conv2 = ConvBlock(16, 32)
        self.conv3 = ConvBlock(32, 64)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 4 * 4, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv3(self.conv2(self.conv1(x)))
        return torch.relu(self.fc(torch.flatten(self.pool(h), 1)))


class DistressNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio = AudioBranch()
        self.vision = VisionBranch()
        self.fuse = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward_audio(self, x: torch.Tensor) -> torch.Tensor:
        return self.audio(x)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def architecture_spec() -> dict:
    net = DistressNet()
    audio_layers = [
        {
            "id": "input",
            "name": "Log-mel input",
            "kind": "input",
            "shape": f"1 × {N_MELS} × {N_FRAMES}",
            "detail": "3 seconds of speech at 16 kHz, mapped onto 64 mel bands and 128 time frames, then standardized.",
            "params": 0,
        },
        {
            "id": "conv1",
            "name": "Conv block 16",
            "kind": "conv",
            "shape": "16 × 32 × 64",
            "detail": "3×3 convolution, 16 filters, batch norm, ReLU, 2×2 max pool.",
            "params": count_params(net.audio.conv1),
        },
        {
            "id": "conv2",
            "name": "Conv block 32",
            "kind": "conv",
            "shape": "32 × 16 × 32",
            "detail": "3×3 convolution, 32 filters, batch norm, ReLU, 2×2 max pool.",
            "params": count_params(net.audio.conv2),
        },
        {
            "id": "conv3",
            "name": "Conv block 64",
            "kind": "conv",
            "shape": "64 × 8 × 16",
            "detail": "3×3 convolution, 64 filters, batch norm, ReLU, 2×2 max pool.",
            "params": count_params(net.audio.conv3),
        },
        {
            "id": "embed",
            "name": "Embedding",
            "kind": "dense",
            "shape": "64",
            "detail": "Adaptive pool to 4×4, then a 1024 → 128 → 64 fully connected stack with dropout 0.3.",
            "params": count_params(net.audio.fc1) + count_params(net.audio.fc2),
        },
        {
            "id": "head",
            "name": "Distress head",
            "kind": "head",
            "shape": "1",
            "detail": "Linear logit. Probability is the sigmoid of this value. Threshold 0.50.",
            "params": count_params(net.audio.head),
        },
    ]
    return {
        "name": "DistressNet",
        "task": "Binary distress vs calm, from speech",
        "loss": "BCEWithLogitsLoss with pos_weight = n_calm / n_distress",
        "optimizer": "Adam",
        "threshold": 0.5,
        "audio_layers": audio_layers,
        # The vision branch and fusion head were never trained. They are future work and are not shown as part of the model.
        "vision_layers": [],
        "future_work": "A face branch and a fusion head were sketched early and never trained; they are future work, not part of DistressNet.",
        "audio_params": count_params(net.audio),
        "vision_params": count_params(net.vision),
        "fusion_params": count_params(net.fuse),
        "trained": "audio",
    }
