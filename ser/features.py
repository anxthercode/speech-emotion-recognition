"""HTK Mel, power dB, orthonormal DCT-II MFCC. No learned normalization here."""
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import gcd

import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 64
    fmin: float = 50.0
    fmax: float = 7600.0
    window_seconds: float = 4
    eval_hop_seconds: float = 2
    log_floor: float = 1e-10
    n_mfcc: int = 13
    delta_radius: int = 2

    def __post_init__(self):
        if not (0 <= self.fmin < self.fmax <= self.sample_rate / 2):
            raise ValueError("Invalid Mel frequency range")
        if not (0 < self.win_length <= self.n_fft <= self.window_samples):
            raise ValueError("Invalid FFT/window lengths")
        if not (self.hop_length > 0 and 0 < self.n_mfcc <= self.n_mels):
            raise ValueError("Invalid hop or MFCC count")
        if not (0 < self.eval_hop_samples <= self.window_samples and self.delta_radius > 0):
            raise ValueError("Invalid evaluation hop or delta radius")
        if not (np.isfinite(self.log_floor) and self.log_floor > 0):
            raise ValueError("Invalid log floor")

    @property
    def window_samples(self):
        return int(self.sample_rate * self.window_seconds)

    @property
    def eval_hop_samples(self):
        return int(self.sample_rate * self.eval_hop_seconds)

    def parameters(self):
        return {**asdict(self), "window": "periodic_hann", "center": False,
                "power": 2, "mel_scale": "htk", "filter_norm": "slaney",
                "log_reference_power": 1, "mfcc_dct": "II_ortho",
                "mfcc_include_c0": True, "delta_order": 2,
                "preemphasis": False, "peak_normalization": False}


DEFAULT_AUDIO = AudioConfig()


def mono_resample(samples: np.ndarray, source_rate: int, config=DEFAULT_AUDIO):
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Expected nonempty finite mono or multichannel audio")
    if source_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if source_rate != config.sample_rate:
        divisor = gcd(source_rate, config.sample_rate)
        samples = resample_poly(samples, config.sample_rate // divisor, source_rate // divisor)
    return np.asarray(samples, dtype=np.float32)


@lru_cache(maxsize=8)
def mel_filterbank(config=DEFAULT_AUDIO):
    mel_min = 2595 * np.log10(1 + config.fmin / 700)
    mel_max = 2595 * np.log10(1 + config.fmax / 700)
    edges = 700 * (10 ** (np.linspace(mel_min, mel_max, config.n_mels + 2) / 2595) - 1)
    frequencies = np.fft.rfftfreq(config.n_fft, 1 / config.sample_rate)
    rising = (frequencies[None, :] - edges[:-2, None]) / np.diff(edges)[:-1, None]
    falling = (edges[2:, None] - frequencies[None, :]) / np.diff(edges)[1:, None]
    weights = np.maximum(0, np.minimum(rising, falling))
    weights *= (2 / (edges[2:] - edges[:-2]))[:, None]
    weights.setflags(write=False)
    return weights


def log_mel_frames(mono, config=DEFAULT_AUDIO):
    """Return [mel, time] on actual audio, padding only if shorter than one FFT."""
    signal = np.asarray(mono, dtype=np.float64)
    if signal.ndim != 1 or not len(signal) or not np.isfinite(signal).all():
        raise ValueError("Expected nonempty finite mono audio")
    if len(signal) < config.n_fft:
        signal = np.pad(signal, (0, config.n_fft - len(signal)))
    frames = np.lib.stride_tricks.sliding_window_view(signal, config.n_fft)[::config.hop_length]
    window = np.zeros(config.n_fft)
    start = (config.n_fft - config.win_length) // 2
    window[start:start + config.win_length] = np.hanning(config.win_length + 1)[:-1]
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    mel = mel_filterbank(config) @ power.T
    return (10 * np.log10(np.maximum(mel, config.log_floor))).astype(np.float32)


def log_mel_window(mono, start=0, config=DEFAULT_AUDIO):
    if start < 0 or start >= len(mono):
        raise ValueError("Window start is outside audio")
    segment = np.asarray(mono[start:start + config.window_samples], dtype=np.float64)
    segment = np.pad(segment, (0, config.window_samples - len(segment)))
    return log_mel_frames(segment, config)


def evaluation_starts(length, config=DEFAULT_AUDIO):
    if length <= 0:
        raise ValueError("Empty audio")
    last = max(0, length - config.window_samples)
    starts = list(range(0, last + 1, config.eval_hop_samples))
    if starts[-1] != last:
        starts.append(last)
    return np.array(starts, dtype=np.int64)


def delta(frames, radius=2):
    """Regression derivative with edge replication. Input [time, features]."""
    padded = np.pad(frames.astype(np.float64), ((radius, radius), (0, 0)), mode="edge")
    result = np.zeros_like(frames, dtype=np.float64)
    for offset in range(1, radius + 1):
        result += offset * (padded[radius + offset:radius + offset + len(frames)]
                            - padded[radius - offset:radius - offset + len(frames)])
    return (result / (2 * sum(n * n for n in range(1, radius + 1)))).astype(np.float32)


def mfcc_features(mono, config=DEFAULT_AUDIO):
    # Full recording, without the CNN's four-second zero padding or cropping.
    static = dct(log_mel_frames(mono, config), type=2, axis=0, norm="ortho")[:config.n_mfcc].T
    first = delta(static, config.delta_radius)
    second = delta(first, config.delta_radius)
    return np.concatenate((static, first, second), axis=1).astype(np.float32)


def extract_features(mono, config=DEFAULT_AUDIO):
    starts = evaluation_starts(len(mono), config)
    mfcc = mfcc_features(mono, config)
    return {
        "waveform": np.asarray(mono, dtype=np.float32),
        "window_starts": starts,
        "log_mel": np.stack([log_mel_window(mono, int(start), config) for start in starts]),
        "mfcc": mfcc,
        "svm": np.concatenate((mfcc.mean(axis=0), mfcc.std(axis=0))).astype(np.float32),
    }
