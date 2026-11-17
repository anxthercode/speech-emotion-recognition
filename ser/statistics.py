"""Streaming population statistics; callers provide training records only."""
import numpy as np


class Moments:
    def __init__(self, features):
        self.count = 0
        self.mean = np.zeros(features, dtype=np.float64)
        self.m2 = np.zeros(features, dtype=np.float64)

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.mean) or not np.isfinite(values).all():
            raise ValueError("Invalid statistics batch")
        if not len(values):
            return
        count = len(values)
        mean = values.mean(axis=0)
        m2 = ((values - mean) ** 2).sum(axis=0)
        difference = mean - self.mean
        total = self.count + count
        self.m2 += m2 + difference ** 2 * self.count * count / total
        self.mean += difference * count / total
        self.count = total

    def result(self):
        if not self.count:
            raise ValueError("Cannot fit normalization without training observations")
        return {"count": self.count, "mean": self.mean.tolist(),
                "std": np.maximum(np.sqrt(self.m2 / self.count), 1e-6).tolist()}


def fit_training_statistics(rows, load_features, n_mels=64, n_mfcc=13):
    groups = {"log_mel": Moments(n_mels), "mfcc": Moments(n_mfcc * 3), "svm": Moments(n_mfcc * 6)}
    train_ids = []
    for row in rows:
        if row["subset"] != "train":
            continue
        train_ids.append(row["file_id"])
        features = load_features(row["file_id"])
        # One deterministic first window per clip; long clips do not get extra weight.
        groups["log_mel"].update(features["log_mel"][0].T)
        groups["mfcc"].update(features["mfcc"])
        groups["svm"].update(features["svm"][None, :])
    return {"fit_subset": "train", "train_file_ids": train_ids, "ddof": 0,
            "cnn_window_policy": "first_4_seconds_right_zero_pad",
            **{name: moments.result() for name, moments in groups.items()}}
