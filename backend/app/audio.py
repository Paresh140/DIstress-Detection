"""Log-mel spectrograms for RAVDESS speech clips."""

from __future__ import annotations

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

TARGET_SR = 16_000
CLIP_SECONDS = 3.0
N_FFT = 512
HOP = 256
N_MELS = 64
N_FRAMES = 128


def load_wav(path: str) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        peak = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / float(peak)
    else:
        data = data.astype(np.float32)
    return data, int(sr)


def _fix_length(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        start = (len(y) - n) // 2
        return y[start : start + n]
    return np.pad(y, (0, n - len(y)))


def mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    mels = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for j in range(left, center):
            if 0 <= j < n_freqs:
                fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if 0 <= j < n_freqs:
                fb[i, j] = (right - j) / (right - center)
    return fb


_FB = mel_filterbank(TARGET_SR, N_FFT, N_MELS)
_WINDOW = torch.hann_window(N_FFT)


def waveform_to_logmel(y: np.ndarray, sr: int) -> torch.Tensor:
    if sr != TARGET_SR:
        g = int(np.gcd(TARGET_SR, sr))
        y = resample_poly(y, TARGET_SR // g, sr // g).astype(np.float32)
    n = int(TARGET_SR * CLIP_SECONDS)
    y = _fix_length(y.astype(np.float32), n)
    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak
    audio = torch.from_numpy(y)
    spec = torch.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window=_WINDOW,
        return_complex=True,
        center=True,
    )
    mag = spec.abs().numpy()
    mel = _FB @ mag
    logmel = np.log(mel + 1e-6).astype(np.float32)
    src = np.linspace(0, logmel.shape[1] - 1, logmel.shape[1])
    dst = np.linspace(0, logmel.shape[1] - 1, N_FRAMES)
    resized = np.vstack([np.interp(dst, src, row) for row in logmel]).astype(np.float32)
    tensor = torch.from_numpy(resized)
    tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-6)
    return tensor


def file_to_logmel(path: str) -> torch.Tensor:
    y, sr = load_wav(path)
    return waveform_to_logmel(y, sr)
