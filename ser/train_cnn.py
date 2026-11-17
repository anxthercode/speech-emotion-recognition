"""CNN training with file-level validation and resumable epoch checkpoints."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .cnn import EmotionCNN
from .dataset import PreparedDataset
from .features import log_mel_window
from .training_common import limited_records, metadata, metrics, save_predictions, write_json, digest


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def save_torch(path, value):
    pending = Path(path).with_suffix(".pt.partial")
    torch.save(value, pending)
    pending.replace(path)


class CNNData:
    def __init__(self, root, split, subset, device, limit_per_class=None):
        if subset not in ("train", "validation"):
            raise ValueError("Stage 2 cannot read the test subset")
        self.dataset = PreparedDataset(root, split, subset)
        self.records = limited_records(self.dataset.records, limit_per_class)
        self.labels = np.array([int(row["label_id"]) for row in self.records], dtype=np.int64)
        self.long_waveforms = {}
        windows, owners = [], []
        for index, row in enumerate(self.records):
            with np.load(self.dataset.root / "features" / f"{row['file_id']}.npz", allow_pickle=False) as source:
                mel = source["log_mel"]
                if subset == "train":
                    windows.append(mel[:1])
                    owners.append(index)
                    if len(mel) > 1:
                        self.long_waveforms[index] = source["waveform"]
                else:
                    windows.append(mel)
                    owners.extend([index] * len(mel))
        array = np.concatenate(windows)[:, None]
        del windows
        mean = np.array(self.dataset.normalization["log_mel"]["mean"], dtype=np.float32)[None, None, :, None]
        std = np.array(self.dataset.normalization["log_mel"]["std"], dtype=np.float32)[None, None, :, None]
        array -= mean
        array /= std
        self.x = torch.from_numpy(array).to(device=device, memory_format=torch.channels_last)
        self.targets = torch.as_tensor(self.labels, device=device)
        self.owners = np.array(owners)
        self.mean, self.std = mean[0], std[0]

    def random_crops(self, seed, epoch):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        config = self.dataset.config
        for index, waveform in self.long_waveforms.items():
            start = int(rng.integers(len(waveform) - config.window_samples + 1))
            mel = log_mel_window(waveform, start, config)[None]
            self.x[index].copy_(torch.from_numpy((mel - self.mean) / self.std).to(self.x.device))


@torch.inference_mode()
def evaluate(model, data, batch_size, amp):
    model.eval()
    probabilities = []
    for start in range(0, len(data.x), batch_size):
        with torch.autocast(device_type=data.x.device.type, dtype=torch.float16, enabled=amp):
            logits = model(data.x[start:start + batch_size])
        probabilities.append(logits.float().softmax(1).cpu().numpy())
    window_probabilities = np.concatenate(probabilities)
    file_probabilities = np.zeros((len(data.records), 6), dtype=np.float64)
    np.add.at(file_probabilities, data.owners, window_probabilities)
    file_probabilities /= np.bincount(data.owners)[:, None]
    loss = float(-np.log(np.maximum(file_probabilities[np.arange(len(data.labels)), data.labels], 1e-12)).mean())
    return {**metrics(data.labels, file_probabilities.argmax(axis=1)), "loss": loss}, file_probabilities


def train_cnn(train, validation, parameters, seed, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = train.x.device
    seed_everything(seed)
    amp = bool(parameters["amp"] and device.type == "cuda")
    model = EmotionCNN(parameters["dropout"]).to(device=device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = nn.CrossEntropyLoss()
    base = metadata(train.dataset, "cnn", parameters, seed)
    base.update(device=str(device), amp_enabled=amp, train_files=len(train.records), validation_files=len(validation.records),
                parameter_count=sum(p.numel() for p in model.parameters()))
    write_json(output / "metadata.json", base)
    history, best_metrics, best_epoch, stale = [], None, 0, 0
    last = output / "last.pt"
    if last.exists():
        checkpoint = torch.load(last, map_location=device, weights_only=True)
        if checkpoint["metadata"] != base:
            raise ValueError("Resume settings differ from the saved run")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        history, best_metrics = checkpoint["history"], checkpoint["best_metrics"]
        best_epoch, stale = checkpoint["best_epoch"], checkpoint["stale"]
        torch.set_rng_state(checkpoint["rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        print(f"Resuming CNN from epoch {len(history) + 1}", flush=True)
    started = time.perf_counter()
    for epoch in range(len(history) + 1, parameters["max_epochs"] + 1):
        if stale >= parameters["patience"]:
            break
        epoch_started = time.perf_counter()
        train.random_crops(seed, epoch)
        generator = torch.Generator().manual_seed(seed * 10000 + epoch)
        order = torch.randperm(len(train.records), generator=generator).to(device)
        model.train()
        total_loss = torch.zeros((), device=device)
        for start in range(0, len(order), parameters["batch_size"]):
            batch = order[start:start + parameters["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = criterion(model(train.x[batch]), train.targets[batch])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite CNN loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach() * len(batch)
        validation_metrics, _ = evaluate(model, validation, parameters["batch_size"], amp)
        epoch_record = {"epoch": epoch, "train_loss": float(total_loss.item() / len(train.records)),
                        "validation_loss": validation_metrics["loss"],
                        **{f"validation_{key}": validation_metrics[key] for key in ("macro_f1", "uar", "accuracy")},
                        "seconds": time.perf_counter() - epoch_started}
        history.append(epoch_record)
        improved = best_metrics is None or (-validation_metrics["macro_f1"], validation_metrics["loss"]) < (-best_metrics["macro_f1"], best_metrics["loss"])
        if improved:
            best_metrics, best_epoch, stale = validation_metrics, epoch, 0
            save_torch(output / "best.pt", {"metadata": base, "state_dict": model.state_dict(),
                                            "epoch": epoch, "validation": validation_metrics})
        else:
            stale += 1
        save_torch(last, {"metadata": base, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scaler": scaler.state_dict(), "history": history, "best_metrics": best_metrics,
                          "best_epoch": best_epoch, "stale": stale, "rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
        write_json(output / "history.json", history)
        print(f"CNN {train.dataset.split} seed={seed} epoch={epoch:02d} "
              f"loss={epoch_record['train_loss']:.4f} val_F1={validation_metrics['macro_f1']:.4f} "
              f"{epoch_record['seconds']:.1f}s best={best_epoch}", flush=True)
    best = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["state_dict"])
    measured, probabilities = evaluate(model, validation, parameters["batch_size"], amp)
    if not np.isclose(measured["macro_f1"], best_metrics["macro_f1"], atol=1e-10):
        raise ValueError("Restored CNN validation predictions changed")
    save_predictions(output / "validation_predictions.csv", validation.records, probabilities,
                     probabilities.argmax(axis=1), "softmax_probability")
    result = {"complete": True, "model": "cnn", "split": train.dataset.split, "seed": seed,
              "parameters": parameters, "validation": measured, "best_epoch": best_epoch,
              "epochs": len(history), "epoch_seconds": [row["seconds"] for row in history],
              "fit_seconds": sum(row["seconds"] for row in history),
              "this_session_seconds": time.perf_counter() - started,
              "checkpoint": "best.pt", "checkpoint_sha256": digest(output / "best.pt")}
    write_json(output / "result.json", result)
    return result
