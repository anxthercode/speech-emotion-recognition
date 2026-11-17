"""Load the final stage-2 models and score prepared or freshly extracted features."""
from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from .features import AudioConfig, extract_features, mono_resample
from .noise import mix_at_snr, noise_segment, stable_seed
from .splits import LABELS

PROJECT = Path(__file__).resolve().parents[1]
MODELS = ("svm", "gmm", "cnn")


def final_runs(project=PROJECT):
    project = Path(project)
    root = project / json.loads((project / "runs/stage2_current.json").read_text(encoding="utf-8"))["directory"]
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if not summary["stage2_complete"]:
        raise ValueError("Полное обучение ещё не завершено.")
    runs = [run for name in MODELS for run in summary["models"][name]["runs"]]
    keys = {(r["model"], r["split"], r["seed"]) for r in runs}
    config = summary["manifest"]["config"]
    expected = {(name, split, seed) for name in MODELS for split in ("speaker", "random")
                for seed in ([config["selection_seed"]] if name == "svm" else config["final_seeds"])}
    if keys != expected or len(runs) != len(expected):
        raise ValueError("В отчёте отсутствуют финальные запуски или есть дубликаты.")
    return runs


def demo_run(runs, model, split):
    return max((r for r in runs if r["model"] == model and r["split"] == split),
               key=lambda r: (r["validation"]["macro_f1"], -r["seed"]))


class Predictor:
    def __init__(self, run, device="cpu", project=PROJECT):
        from .training_common import digest

        self.run = run
        self.device = device
        directory = Path(project) / run["run_directory"]
        path = directory / run["checkpoint"]
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if (digest(path) != run["checkpoint_sha256"] or self.metadata["classes"] != LABELS
                or any(self.metadata[k] != run[k] for k in ("model", "split", "seed", "parameters"))):
            raise ValueError(f"Метаданные или контрольная сумма модели не совпадают: {path}")
        self.name = run["model"]
        preprocessing = self.metadata["preprocessing"]
        self.config = AudioConfig(**{field.name: preprocessing[field.name] for field in fields(AudioConfig)})
        key = {"cnn": "log_mel", "gmm": "mfcc", "svm": "svm"}[self.name]
        self.key = key
        norm = self.metadata["normalization"][key]
        self.mean = np.asarray(norm["mean"], dtype=np.float32)
        self.std = np.asarray(norm["std"], dtype=np.float32)
        if self.name == "cnn":
            import torch
            from .cnn import EmotionCNN

            self.torch = torch
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint["metadata"] != self.metadata:
                raise ValueError("Метаданные CNN не совпадают с чекпойнтом.")
            self.model = EmotionCNN(run["parameters"]["dropout"])
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.to(device=device, memory_format=torch.channels_last).eval()
            self.score_type = "softmax_probability"
        elif self.name == "gmm":
            from .gmm import GMMClassifier
            self.model = GMMClassifier.load(path)
            self.score_type = "mean_log_posterior_score"
        else:
            import joblib
            self.model = joblib.load(path)
            if list(self.model.classes_) != list(range(6)):
                raise ValueError("Неожиданный порядок классов SVM.")
            self.score_type = "svm_ovr_decision_score"

    def predict_batch(self, features):
        values = [item[self.key] for item in features]
        if self.name == "svm":
            x = (np.stack(values) - self.mean) / self.std
            return self.model.predict(x), self.model.decision_function(x)
        if self.name == "gmm":
            scores = self.model.decision_function([(x - self.mean) / self.std for x in values])
        else:
            torch = self.torch
            windows = (np.concatenate(values) - self.mean[None, :, None]) / self.std[None, :, None]
            windows = torch.from_numpy(windows[:, None]).to(device=self.device, memory_format=torch.channels_last)
            amp = self.device == "cuda" and self.metadata["amp_enabled"]
            with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.float16, enabled=amp):
                probabilities = torch.cat([self.model(batch).float().softmax(1) for batch in windows.split(32)]).cpu().numpy()
            lengths = [len(x) for x in values]
            scores = np.add.reduceat(probabilities.astype(np.float64), np.cumsum([0, *lengths[:-1]])) / np.array(lengths)[:, None]
        return scores.argmax(axis=1), scores

    def predict_waveform(self, waveform, rate):
        return self._predict_mono(mono_resample(waveform, rate, self.config))

    def _predict_mono(self, mono):
        features = extract_features(mono, self.config)
        predicted, scores = self.predict_batch([features])
        return {"model": self.name, "split": self.run["split"], "seed": self.run["seed"],
                "emotion": list(LABELS)[int(predicted[0])], "score_type": self.score_type,
                "scores": dict(zip(LABELS, scores[0].tolist()))}

    def compare_waveform(self, waveform, rate, condition="clean", snr=None):
        mono = mono_resample(waveform, rate, self.config)
        result = {"clean": self._predict_mono(mono)}
        if condition == "clean":
            return result
        if condition not in {"white", "background"} or snr not in {0, 10, 20}:
            raise ValueError("Выберите белый или фоновый шум и SNR 0, 10 или 20 дБ.")
        background = None
        if condition == "background":
            import soundfile as sf

            root = PROJECT / "data/prepared"
            directory = json.loads((root / "current.json").read_text(encoding="utf-8"))["directory"]
            samples, rate = sf.read(root / directory / "background.wav", dtype="float32", always_2d=True)
            background = mono_resample(samples, rate, self.config)
            if len(background) < len(mono):
                background = np.tile(background, (len(mono) + len(background) - 1) // len(background))
        seed = stable_seed("interactive_demo", condition, 17)
        noise, _ = noise_segment(len(mono), seed, background)
        mixed, _, actual = mix_at_snr(mono, noise, snr)
        return {**result, "noisy": self._predict_mono(mixed), "condition": condition, "snr_db": snr,
                "measured_snr_db": actual}
