"""Fit the NumPy GMM or sklearn SVM on train; score validation only."""
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC

from .gmm import GMMClassifier
from .training_common import metadata, metrics, save_predictions, write_json, digest


def train_classical(name, train, validation, parameters, seed, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    info = metadata(train.dataset, name, parameters, seed)
    info.update(train_files=len(train.records), validation_files=len(validation.records))
    write_json(output / "metadata.json", info)
    started = time.perf_counter()
    if name == "svm":
        model = SVC(kernel="rbf", C=parameters["C"], gamma=parameters["gamma"],
                    cache_size=parameters["cache_size_mb"], decision_function_shape="ovr")
        model.fit(train.vectors, train.labels)
        fit_seconds = time.perf_counter() - started
        scores = model.decision_function(validation.vectors)
        predicted = model.predict(validation.vectors)
        # SVC.predict uses pairwise voting. It need not equal argmax of OVR display scores.
        checkpoint = output / "model.joblib"
        joblib.dump(model, checkpoint)
        restored = joblib.load(checkpoint)
        np.testing.assert_array_equal(predicted, restored.predict(validation.vectors))
        np.testing.assert_allclose(scores, restored.decision_function(validation.vectors), rtol=0, atol=0)
        score_type = "svm_ovr_decision_score"
    elif name == "gmm":
        model = GMMClassifier(**parameters, seed=seed)
        model.fit(train.sequences, train.labels,
                  progress=lambda label, mixture: print(f"GMM {train.dataset.split} seed={seed} class={label} "
                      f"K={parameters['n_components']} logL={mixture.lower_bound_:.3f}", flush=True))
        fit_seconds = time.perf_counter() - started
        scores = model.decision_function(validation.sequences)
        predicted = scores.argmax(axis=1)
        checkpoint = output / "model.npz"
        model.save(checkpoint)
        histories = [{"class": index, "selected_initialization": mixture.best_initialization_,
                      "initializations": mixture.initializations_} for index, mixture in enumerate(model.models_)]
        write_json(output / "em_history.json", histories)
        restored_scores = GMMClassifier.load(checkpoint).decision_function(validation.sequences)
        np.testing.assert_allclose(scores, restored_scores, rtol=0, atol=0)
        score_type = "mean_log_posterior_score"
    else:
        raise ValueError(f"Unknown model: {name}")
    result = {"complete": True, "model": name, "split": train.dataset.split, "seed": seed,
              "parameters": parameters, "validation": metrics(validation.labels, predicted),
              "fit_seconds": fit_seconds, "total_seconds": time.perf_counter() - started,
              "checkpoint": checkpoint.name, "checkpoint_sha256": digest(checkpoint)}
    save_predictions(output / "validation_predictions.csv", validation.records, scores, predicted, score_type)
    write_json(output / "result.json", result)
    print(f"{name.upper()} {train.dataset.split}: validation F1={result['validation']['macro_f1']:.4f}, "
          f"fit={fit_seconds:.1f}s", flush=True)
    return result
