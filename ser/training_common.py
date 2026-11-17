"""Training artifacts and metrics. Test is not loaded by stage 2."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from .dataset import PreparedDataset
from .splits import LABELS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".partial")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def metrics(labels, predictions):
    classes = list(range(6))
    return {"macro_f1": float(f1_score(labels, predictions, labels=classes, average="macro", zero_division=0)),
            "uar": float(recall_score(labels, predictions, labels=classes, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(labels, predictions)),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist()}


def save_predictions(path, records, scores, predictions, score_type):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        fields = ["file_id", "speaker_id", "y_true", "y_pred", "score_type", *[f"score_{c}" for c in LABELS]]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row, score, prediction in zip(records, scores, predictions):
            writer.writerow({"file_id": row["file_id"], "speaker_id": row["speaker_id"], "y_true": row["label_id"],
                             "y_pred": int(prediction), "score_type": score_type,
                             **{f"score_{code}": float(value) for code, value in zip(LABELS, score)}})


def metadata(dataset, model, parameters, seed):
    return {"model": model, "split": dataset.split, "seed": seed, "parameters": parameters,
            "dataset_fingerprint": dataset.report["fingerprint"], "classes": LABELS,
            "preprocessing": dataset.config.parameters(), "normalization": dataset.normalization,
            "split_sha256": digest(dataset.root / "splits" / f"{dataset.split}.csv"),
            "selection_subset": "validation", "test_evaluated": False}


class ClassicalData:
    def __init__(self, root, split, subset, frames_per_file=80, limit_per_class=None):
        if subset not in ("train", "validation"):
            raise ValueError("Stage 2 may only read train and validation")
        self.dataset = PreparedDataset(root, split, subset)
        self.records = limited_records(self.dataset.records, limit_per_class)
        self.labels = np.array([int(row["label_id"]) for row in self.records])
        norm = self.dataset.normalization
        mfcc_mean, mfcc_std = (np.array(norm["mfcc"][name], dtype=np.float32) for name in ("mean", "std"))
        svm_mean, svm_std = (np.array(norm["svm"][name], dtype=np.float32) for name in ("mean", "std"))
        self.sequences, vectors = [], []
        for row in self.records:
            with np.load(self.dataset.root / "features" / f"{row['file_id']}.npz", allow_pickle=False) as data:
                mfcc = data["mfcc"]
                if subset == "train" and len(mfcc) > frames_per_file:
                    # Same deterministic coverage and cap for every training recording.
                    mfcc = mfcc[np.linspace(0, len(mfcc) - 1, frames_per_file, dtype=int)]
                self.sequences.append((mfcc - mfcc_mean) / mfcc_std)
                vectors.append((data["svm"] - svm_mean) / svm_std)
        self.vectors = np.array(vectors)


def limited_records(records, per_class):
    if per_class is None:
        return records
    selected = {row["file_id"] for label in range(6)
                for row in [r for r in records if int(r["label_id"]) == label][:per_class]}
    return [row for row in records if row["file_id"] in selected]
