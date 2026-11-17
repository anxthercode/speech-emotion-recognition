"""Stage 2 CLI: validation-only model selection, final training, smoke and timing runs."""
import argparse
import gc
import hashlib
import itertools
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import platform
import time

from filelock import FileLock
import numpy as np
import scipy
import sklearn
import torch
from threadpoolctl import threadpool_limits

from ser.dataset import PreparedDataset
from ser.train_classical import train_classical
from ser.train_cnn import CNNData, train_cnn
from ser.training_common import ClassicalData, digest, write_json


def parameter_grid(name, config, mode):
    settings = config[name]
    if name == "cnn":
        grid = [{"learning_rate": lr, "dropout": dropout,
                 **{key: settings[key] for key in ("weight_decay", "batch_size", "max_epochs", "patience", "amp")}}
                for lr, dropout in itertools.product(settings["learning_rates"], settings["dropouts"])]
        if mode != "full":
            grid = [{**grid[0], "max_epochs": 2, "patience": 2}]
    elif name == "gmm":
        grid = [{"n_components": k, "variance_floor": floor,
                 **{key: settings[key] for key in ("max_iter", "tol", "n_init", "batch_size")}}
                for k, floor in itertools.product(settings["components"], settings["variance_floors"])]
        if mode != "full":
            grid = [{**grid[0], "n_components": 8 if mode == "benchmark" else 2,
                     "max_iter": 5, "n_init": 1, "tol": 0}]
    else:
        grid = [{"C": c, "gamma": gamma, "cache_size_mb": settings["cache_size_mb"]}
                for c, gamma in itertools.product(settings["C"], settings["gamma"])]
        if mode != "full":
            grid = [{"C": 10, "gamma": "scale", "cache_size_mb": settings["cache_size_mb"]}]
    return grid


def load_data(name, root, split, config, device, mode):
    started = time.perf_counter()
    if name == "cnn":
        training = CNNData(root, split, "train", device, 8 if mode == "smoke" else None)
        validation = CNNData(root, split, "validation", device, 4 if mode == "smoke" else None)
    else:
        training = ClassicalData(root, split, "train", config["gmm"]["frames_per_file"], 8 if mode == "smoke" else None)
        validation = ClassicalData(root, split, "validation", config["gmm"]["frames_per_file"], 4 if mode == "smoke" else None)
    print(f"Loaded {name}/{split}: train={len(training.records)}, validation={len(validation.records)}, "
          f"{time.perf_counter() - started:.1f}s", flush=True)
    return training, validation


def run_job(name, training, validation, parameters, seed, directory):
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (result["parameters"] != parameters or result["seed"] != seed
                or digest(directory / result["checkpoint"]) != result["checkpoint_sha256"]):
            raise ValueError(f"Saved run does not match its checkpoint/settings: {directory}")
        print(f"Reusing completed run: {directory.name}", flush=True)
    elif name == "cnn":
        result = train_cnn(training, validation, parameters, seed, directory)
    else:
        result = train_classical(name, training, validation, parameters, seed, directory)
    return {**result, "run_directory": directory.as_posix()}


def run_model(name, args, config, root, device):
    output = root / name
    output.mkdir(exist_ok=True)
    with FileLock(str(output / "training.lock"), timeout=0):
        grid = parameter_grid(name, config, args.mode)
        training, validation = load_data(name, args.data, "speaker", config, device, args.mode)
        trials = []
        for index, parameters in enumerate(grid):
            print(f"{name.upper()} configuration {index + 1}/{len(grid)}: {parameters}", flush=True)
            directory = output / "speaker" / f"trial_{index:02d}_seed{config['selection_seed']}"
            trials.append(run_job(name, training, validation, parameters, config["selection_seed"], directory))
            write_json(output / "selection.json", {"subset": "validation", "split": "speaker", "trials": trials})
        best = max(enumerate(trials), key=lambda item: (
            item[1]["validation"]["macro_f1"],
            -item[1]["validation"].get("loss", 0), item[1]["validation"]["accuracy"], -item[0]))[1]
        write_json(output / "selected.json", {"parameters": best["parameters"],
                   "validation": best["validation"], "source_run": best["run_directory"], "test_evaluated": False})
        final = [best]
        if args.mode == "full":
            seeds = [config["selection_seed"]] if name == "svm" else config["final_seeds"]
            for seed in seeds:
                if seed != config["selection_seed"]:
                    final.append(run_job(name, training, validation, best["parameters"], seed,
                                         output / "speaker" / f"final_seed{seed}"))
            del training, validation
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            training, validation = load_data(name, args.data, "random", config, device, args.mode)
            for seed in seeds:
                final.append(run_job(name, training, validation, best["parameters"], seed,
                                     output / "random" / f"final_seed{seed}"))
        write_json(output / "completed.json", {"mode": args.mode, "model": name, "runs": final})
        del training, validation
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def summarize(root, args, manifest):
    completed = {}
    for name in ("svm", "gmm", "cnn"):
        path = root / name / "completed.json"
        if path.exists():
            completed[name] = json.loads(path.read_text(encoding="utf-8"))
    complete = len(completed) == 3
    report = {"all_models_complete": complete, "stage2_complete": complete and args.mode == "full",
              "mode": args.mode, "test_evaluated": False, "run_root": root.as_posix(),
              "manifest": manifest, "models": completed}
    write_json(root / "summary.json", report)
    write_json(Path("reports") / f"stage2_{args.mode}.json", report)
    if report["stage2_complete"]:
        write_json(Path("runs") / "stage2_current.json", {"directory": root.as_posix()})
    if args.mode == "benchmark":
        estimates = {}
        for name, completion in completed.items():
            result = completion["runs"][0]
            if name == "cnn":
                # Second epoch excludes CUDA initialization; early stopping can reduce the cap.
                seconds = result["epoch_seconds"][-1]
                runs = len(parameter_grid(name, manifest["config"], "full")) + 5
                estimates[name] = {"measured_seconds_per_epoch": seconds,
                                   "one_run_epoch_cap_minutes": seconds * manifest["config"]["cnn"]["max_epochs"] / 60,
                                   "whole_stage_epoch_cap_minutes": seconds * manifest["config"]["cnn"]["max_epochs"] * runs / 60}
            elif name == "gmm":
                settings = manifest["config"]["gmm"]
                seconds = result["fit_seconds"] * settings["max_iter"] / result["parameters"]["max_iter"] * settings["n_init"]
                estimates[name] = {"rough_one_run_minutes": seconds / 60,
                                   "rough_whole_stage_minutes": seconds * (len(parameter_grid(name, manifest["config"], "full")) + 5) / 60,
                                   "note": "Extrapolated from K=8; K, convergence and initialization change runtime."}
            else:
                estimates[name] = {"one_fit_seconds": result["fit_seconds"],
                                   "rough_whole_stage_minutes": result["total_seconds"] * (len(parameter_grid(name, manifest["config"], "full")) + 1) / 60,
                                   "note": "C and gamma affect runtime; this is one measured configuration."}
        write_json(Path("reports") / "training_time_estimate.json", estimates)
        print(json.dumps(estimates, indent=2), flush=True)
    print(f"Saved reports/stage2_{args.mode}.json; stage2_complete={report['stage2_complete']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("svm", "gmm", "cnn"), default=["svm", "gmm", "cnn"])
    parser.add_argument("--mode", choices=("full", "smoke", "benchmark"), default="full")
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if config["selection_split"] != "speaker" or config["selection_seed"] not in config["final_seeds"]:
            raise ValueError("Expected selection on speaker and inclusion of its seed in final_seeds")
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this Python environment; run tools.check_training")
        torch.set_num_threads(config["cpu_threads"])
        torch.set_num_interop_threads(config["cpu_threads"])
        data = PreparedDataset(args.data, "speaker", "train")
        sources = ["ser/gmm.py", "ser/cnn.py", "ser/train_cnn.py", "ser/train_classical.py", "ser/training_common.py",
                   "ser/features.py", "ser/dataset.py", "tools/train_models.py"]
        source_root = Path(__file__).resolve().parent.parent
        manifest = {"mode": args.mode, "config": config, "dataset_fingerprint": data.report["fingerprint"],
                    "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                                 "torch": str(torch.__version__), "sklearn": sklearn.__version__},
                    "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                    "implementation": {source: digest(source_root / source) for source in sources}}
        identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
        root = Path("runs") / f"stage2_{args.mode}_{identity}"
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "manifest.json", manifest)
        print(f"Mode={args.mode}; device={device}; output={root}", flush=True)
        with threadpool_limits(limits=config["cpu_threads"]):
            for name in dict.fromkeys(args.models):
                run_model(name, args, config, root, device)
        with FileLock(str(root / "summary.lock"), timeout=10):
            summarize(root, args, manifest)
    except KeyboardInterrupt:
        parser.exit(130, "Stopped. Run the same command to reuse completed jobs and resume CNN from the last saved epoch.\n")
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
