"""Repeatable, full-clip SNR mixing, before padding and without clipping."""
import hashlib

import numpy as np


def stable_seed(file_id, noise_id, replica_seed):
    key = f"{file_id}|{noise_id}|{replica_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def noise_segment(length, seed, background=None):
    rng = np.random.Generator(np.random.PCG64(seed))
    if background is None:
        return rng.standard_normal(length), 0
    if len(background) < length:
        raise ValueError("Background must be at least as long as the speech recording")
    offset = int(rng.integers(0, len(background) - length + 1))
    return np.asarray(background[offset:offset + length], dtype=np.float64), offset


def measured_snr(clean, mixed):
    clean = np.asarray(clean, dtype=np.float64)
    residual = np.asarray(mixed, dtype=np.float64) - clean
    signal_power, noise_power = np.mean(clean ** 2), np.mean(residual ** 2)
    if min(signal_power, noise_power) <= 0:
        raise ValueError("SNR requires nonzero signal and noise")
    return float(10 * np.log10(signal_power / noise_power))


def mix_at_snr(clean, noise, snr_db):
    clean, noise = np.asarray(clean, dtype=np.float64), np.asarray(noise, dtype=np.float64)
    if clean.ndim != 1 or clean.shape != noise.shape or not len(clean):
        raise ValueError("Signal and noise must have matching nonempty mono shapes")
    if not np.isfinite(clean).all() or not np.isfinite(noise).all() or not np.isfinite(snr_db):
        raise ValueError("Nonfinite signal, noise or SNR")
    signal_power, noise_power = np.mean(clean ** 2), np.mean(noise ** 2)
    if min(signal_power, noise_power) <= 0:
        raise ValueError("Cannot mix silent signal or noise")
    alpha = float(np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10))))
    mixed = (clean + alpha * noise).astype(np.float32)
    actual = measured_snr(clean, mixed)
    if not np.isfinite(mixed).all() or abs(actual - snr_db) > 0.01:
        raise ValueError("Mixed waveform failed finite/SNR checks")
    return mixed, alpha, actual
