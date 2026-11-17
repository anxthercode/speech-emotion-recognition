"""Evaluate the final stage-2 checkpoints on clean and noisy held-out audio."""
import argparse
import csv
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from threadpoolctl import threadpool_limits
import torch

from ser.dataset import PreparedDataset
from ser.inference import MODELS, Predictor, final_runs
from ser.training_common import digest, metrics, write_json
from ser.splits import LABELS

CONDITIONS = [("clean", None)] + [(kind, snr) for kind in ("white", "background") for snr in (20, 10, 0)]
METRICS = ("macro_f1", "uar", "accuracy")


def csv_output(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(output, device):
    runs = final_runs()
    output.mkdir(parents=True, exist_ok=True)
    per_seed, timings, per_class = [], [], []
    fingerprint = None
    prediction_fields = ["model", "split", "seed", "condition", "snr_db", "file_id", "speaker_id",
                         "y_true", "y_pred", "score_type", *[f"score_{code}" for code in LABELS]]
    pending = output / "test_predictions.csv.partial"
    with pending.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=prediction_fields)
        writer.writeheader()
        for split in ("speaker", "random"):
            data = PreparedDataset(split=split, subset="test")
            fingerprint = data.report["fingerprint"]
            split_hash = digest(data.root / "splits" / f"{split}.csv")
            predictors = [Predictor(r, device if r["model"] == "cnn" else "cpu") for r in runs if r["split"] == split]
            for p in predictors:
                if (p.metadata["dataset_fingerprint"] != fingerprint or p.metadata["split_sha256"] != split_hash
                        or p.metadata["normalization"] != data.normalization or p.config != data.config):
                    raise ValueError("Prepared test data do not match the checkpoint provenance.")
            labels = [int(row["label_id"]) for row in data.records]
            for kind, snr in CONDITIONS:
                predictions = [[] for _ in predictors]
                inference_seconds = [0.0] * len(predictors)
                preprocessing_seconds = 0.0
                for start in range(0, len(data), 32):
                    records = data.records[start:start + 32]
                    started = time.perf_counter()
                    features = [data.features(row["file_id"], kind, snr) for row in records]
                    preprocessing_seconds += time.perf_counter() - started
                    for i, p in enumerate(predictors):
                        started = time.perf_counter()
                        predicted, scores = p.predict_batch(features)
                        inference_seconds[i] += time.perf_counter() - started
                        predictions[i].extend(predicted.tolist())
                        for row, target, values in zip(records, predicted, scores):
                            writer.writerow({"model": p.name, "split": split, "seed": p.run["seed"], "condition": kind,
                                             "snr_db": snr, "file_id": row["file_id"], "speaker_id": row["speaker_id"],
                                             "y_true": row["label_id"], "y_pred": int(target), "score_type": p.score_type,
                                             **{f"score_{code}": value for code, value in zip(LABELS, values)}})
                for i, p in enumerate(predictors):
                    key = {"model": p.name, "split": split, "seed": p.run["seed"], "condition": kind, "snr_db": snr}
                    result = metrics(labels, predictions[i])
                    confusion = np.asarray(result["confusion_matrix"])
                    normalized = confusion / confusion.sum(axis=1, keepdims=True)
                    per_seed.append({**key, "n_test": len(data), **result, "confusion_normalized": normalized.tolist()})
                    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions[i], labels=list(range(6)), zero_division=0)
                    for j, code in enumerate(LABELS):
                        per_class.append({**key, "class": code, "precision": precision[j], "recall": recall[j],
                                          "f1": f1[j], "support": support[j]})
                    timings.append({**key, "device": p.device, "batch_size": 32,
                                    "preprocessing_seconds_shared": preprocessing_seconds,
                                    "inference_seconds": inference_seconds[i]})
                print(f"{split}/{kind}/{snr}: {len(data)} recordings, {len(predictors)} checkpoints", flush=True)
    pending.replace(output / "test_predictions.csv")
    summary = []
    for name in MODELS:
        for split in ("speaker", "random"):
            clean = None
            for kind, snr in CONDITIONS:
                rows = [r for r in per_seed if (r["model"], r["split"], r["condition"], r["snr_db"]) == (name, split, kind, snr)]
                values = {metric: float(np.mean([r[metric] for r in rows])) for metric in METRICS}
                deviations = {f"{metric}_std": float(np.std([r[metric] for r in rows], ddof=1)) if len(rows) > 1 else None
                              for metric in METRICS}
                if kind == "clean":
                    clean = values["macro_f1"]
                summary.append({"model": name, "split": split, "condition": kind, "snr_db": snr,
                                "n_test": rows[0]["n_test"], "n_seeds": len(rows), **values, **deviations,
                                "f1_drop_pp": (clean - values["macro_f1"]) * 100})
    csv_output(output / "test_summary.csv", summary)
    csv_output(output / "test_per_seed.csv", [{k: v for k, v in r.items() if not k.startswith("confusion")} for r in per_seed])
    csv_output(output / "test_per_class.csv", per_class)
    csv_output(output / "timings.csv", timings)
    write_json(output / "evaluation.json", {"dataset_fingerprint": fingerprint, "runs": runs,
                                            "results": per_seed, "summary": summary})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=Path("reports/stage3"))
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Use --device cpu.")
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        evaluate(args.output, args.device)


if __name__ == "__main__":
    main()
