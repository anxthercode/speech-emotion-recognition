"""Read prepared, unnormalized features and reproducible test noise without Torch."""
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .features import AudioConfig, extract_features, mono_resample
from .noise import measured_snr, noise_segment


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class PreparedDataset:
    def __init__(self, root="data/prepared", split="speaker", subset="train", *, _allow_incomplete=False):
        self.root = Path(root)
        pointer = self.root / "current.json"
        if pointer.exists():
            self.root = self.root / json.loads(pointer.read_text(encoding="utf-8"))["directory"]
        self.report = json.loads((self.root / "report.json").read_text(encoding="utf-8"))
        if (not self.report["complete"] and not _allow_incomplete) or split not in ("speaker", "random"):
            raise ValueError("Unfinished dataset or unknown protocol")
        if subset not in ("train", "validation", "test"):
            raise ValueError("Unknown subset")
        self.config = AudioConfig(**self.report["config"]["audio"])
        self.split, self.subset = split, subset
        self.records = [row for row in read_csv(self.root / "splits" / f"{split}.csv") if row["subset"] == subset]
        self.ids = {row["file_id"] for row in self.records}
        self.normalization = json.loads((self.root / "normalization" / f"{split}.json").read_text(encoding="utf-8"))
        self._noise_rows = None
        self._background = None

    def __len__(self):
        return len(self.records)

    def load(self, file_id):
        if file_id not in self.ids:
            raise KeyError(f"Recording does not belong to {self.split}/{self.subset}: {file_id}")
        with np.load(self.root / "features" / f"{file_id}.npz", allow_pickle=False) as source:
            return {key: source[key] for key in source.files}

    def noisy_audio(self, file_id, condition, snr_db):
        if self.subset != "test":
            raise ValueError("Prepared noise belongs exclusively to test")
        clean = self.load(file_id)["waveform"]
        if self._noise_rows is None:
            self._noise_rows = {(row["file_id"], row["condition"], float(row["snr_db"])): row
                                for row in read_csv(self.root / "noise_manifest.csv") if row["split"] == self.split}
        row = self._noise_rows[(file_id, condition, float(snr_db))]
        background = None
        if condition == "background":
            if self._background is None:
                raw = (self.root / "background.wav").read_bytes()
                if hashlib.sha256(raw).hexdigest() != self.report["background"]["sha256"]:
                    raise ValueError("Background source checksum mismatch")
                samples, rate = sf.read(BytesIO(raw), dtype="float32", always_2d=True)
                self._background = mono_resample(samples, rate, self.config)
            background = self._background
        noise, offset = noise_segment(len(clean), int(row["seed"]), background)
        if offset != int(row["offset_samples"]):
            raise ValueError("Noise offset mismatch")
        mixed = (clean.astype(np.float64) + float(row["alpha"]) * noise).astype(np.float32)
        if abs(measured_snr(clean, mixed) - float(snr_db)) > 0.01:
            raise ValueError("Reconstructed SNR mismatch")
        if hashlib.sha256(mixed.tobytes()).hexdigest() != row["mixture_sha256"]:
            raise ValueError("Reconstructed mixture checksum mismatch; check package versions")
        return mixed

    def features(self, file_id, condition="clean", snr_db=None, standardize=False):
        if condition == "clean":
            if snr_db is not None:
                raise ValueError("Clean condition has no SNR")
            features = self.load(file_id)
        else:
            features = extract_features(self.noisy_audio(file_id, condition, snr_db), self.config)
        if standardize:
            for name in ("log_mel", "mfcc", "svm"):
                mean = np.array(self.normalization[name]["mean"], dtype=np.float32)
                std = np.array(self.normalization[name]["std"], dtype=np.float32)
                if name == "log_mel":
                    mean, std = mean[None, :, None], std[None, :, None]
                features[name] = (features[name] - mean) / std
        return features
