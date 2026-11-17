"""Decode the recording and display its full-length, unstandardized Log-Mel."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from ser.features import DEFAULT_AUDIO, log_mel_frames, mel_filterbank, mono_resample

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif")
AUDIO_FORMATS = "WAV, MP3, FLAC, OGG и AIFF"
AUDIO_FILTER = "Аудиофайлы (" + " ".join(f"*{ext}" for ext in AUDIO_EXTENSIONS) + ")"

SAMPLE_RATE = DEFAULT_AUDIO.sample_rate
N_FFT = DEFAULT_AUDIO.n_fft
WIN_LENGTH = DEFAULT_AUDIO.win_length
HOP_LENGTH = DEFAULT_AUDIO.hop_length
N_MELS = DEFAULT_AUDIO.n_mels
FMIN, FMAX = DEFAULT_AUDIO.fmin, DEFAULT_AUDIO.fmax
PARAMETERS = {
    "sample_rate": SAMPLE_RATE, "n_fft": N_FFT, "win_length": WIN_LENGTH,
    "hop_length": HOP_LENGTH, "n_mels": N_MELS, "fmin": FMIN, "fmax": FMAX,
    "window": "periodic_hann", "center": False, "power": 2,
    "mel_scale": "htk", "filter_norm": "slaney", "log_floor": 1e-10,
    "preview": "full_recording", "preview_window_start_s": 0,
    "standardization": "not_applied",
}


@dataclass
class AudioPreview:
    path: str
    sample_rate: int
    channels: int
    duration_s: float
    waveform: np.ndarray
    log_mel: np.ndarray

    def metadata(self) -> dict:
        return {
            "file": self.path, "source_sample_rate": self.sample_rate,
            "channels": self.channels, "duration_s": self.duration_s,
            "log_mel_shape": list(self.log_mel.shape), "preprocessing": PARAMETERS,
            "model_connected": False, "prediction": None,
        }


def log_mel_preview(mono: np.ndarray) -> np.ndarray:
    return log_mel_frames(mono)


def load_preview(path: str) -> AudioPreview:
    source = Path(path).resolve()
    if source.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError(f"Поддерживаемые форматы: {AUDIO_FORMATS}.")
    with source.open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("Это указатель Git LFS. Для воспроизведения нужен загруженный аудиофайл.")
    if source.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Максимальный размер аудиофайла — 64 МБ.")
    try:
        info = sf.info(str(source))
    except sf.LibsndfileError as error:
        raise ValueError(f"Не удалось прочитать аудио: {source.name}.") from error
    if info.frames <= 0 or not 0 < info.duration <= 60:
        raise ValueError("Для просмотра выберите запись длительностью от 0 до 60 секунд.")
    if info.channels > 8:
        raise ValueError("Поддерживается до 8 аудиоканалов.")
    if not 8000 <= info.samplerate <= 192000:
        raise ValueError("Поддерживаемая частота аудио — от 8 до 192 кГц.")
    samples, rate = sf.read(str(source), dtype="float32", always_2d=True)
    if not np.isfinite(samples).all():
        raise ValueError("В аудиоданных обнаружены некорректные значения NaN или Inf.")
    mono = samples.mean(axis=1)
    features = log_mel_preview(mono_resample(mono, rate))
    return AudioPreview(str(source), rate, info.channels, info.duration, mono, features)
