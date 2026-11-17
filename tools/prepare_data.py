"""Stage 1: audited CREMA-D -> splits, features, train statistics and noise recipes."""
import argparse
from collections import Counter
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import scipy
import soundfile as sf

from ser.dataset import PreparedDataset
from ser.features import AudioConfig, extract_features, mono_resample
from ser.noise import mix_at_snr, noise_segment, stable_seed
from ser.splits import LABELS, make_splits, read_metadata, summarize
from ser.statistics import fit_training_statistics


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_cache(path, source_sha, config):
    with np.load(path, allow_pickle=False) as archive:
        features = {key: archive[key] for key in archive.files}
    if features.pop("source_sha256").item() != source_sha:
        raise ValueError(f"Stale feature cache: {path}")
    frames = 1 + (config.window_samples - config.n_fft) // config.hop_length
    if features["log_mel"].shape != (len(features["window_starts"]), config.n_mels, frames):
        raise ValueError(f"Invalid Log-Mel cache: {path}")
    if features["mfcc"].ndim != 2 or features["mfcc"].shape[1] != config.n_mfcc * 3:
        raise ValueError(f"Invalid MFCC cache: {path}")
    if features["svm"].shape != (config.n_mfcc * 6,):
        raise ValueError(f"Invalid SVM cache: {path}")
    if not all(np.isfinite(value).all() for value in features.values()):
        raise ValueError(f"Nonfinite cache: {path}")
    return features


def prepare(args):
    started = time.perf_counter()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    audio = AudioConfig(**config["audio"])
    if config["snr_db"] != [20, 10, 0]:
        raise ValueError("The coursework protocol requires SNR [20, 10, 0]")
    records = read_metadata(args.metadata)
    labels_path = args.metadata.parent / "labels.json"
    if json.loads(labels_path.read_text(encoding="utf-8")) != LABELS:
        raise ValueError("Audited class ordering differs from the pipeline")
    background_source = json.loads(Path(config["background_source"]).read_text(encoding="utf-8"))
    background_path = Path(background_source["path"])
    if not background_path.exists():
        raise ValueError("Background missing; run python -m tools.fetch_background first")
    background_raw = background_path.read_bytes()
    if sha256(background_raw) != background_source["sha256"]:
        raise ValueError("Background checksum mismatch")
    background_samples, rate = sf.read(BytesIO(background_raw), dtype="float32", always_2d=True)
    background = mono_resample(background_samples, rate, audio)
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "soundfile": sf.__version__}
    source_root = Path(__file__).resolve().parent.parent
    implementation = {name: sha256((source_root / name).read_bytes()) for name in (
        "ser/features.py", "ser/noise.py", "ser/splits.py", "ser/statistics.py", "tools/prepare_data.py")}
    provenance = {"config": config, "metadata_sha256": sha256(args.metadata.read_bytes()),
                  "background": background_source, "versions": versions, "implementation": implementation}
    fingerprint = sha256(json_bytes(provenance))
    root = args.out_dir / fingerprint[:16]
    root.mkdir(parents=True, exist_ok=True)
    print(f"Output: {root}", flush=True)
    splits = make_splits(records, config["split_seed"], config["validation_speakers"], config["test_speakers"])
    for protocol, rows in splits.items():
        write_csv(root / "splits" / f"{protocol}.csv", rows,
                  ("file_id", "relative_path", "speaker_id", "emotion", "label_id", "sha256", "subset", "split_seed"))
    write_json(root / "config.json", config)
    write_json(root / "labels.json", LABELS)
    write_json(root / "preprocessing.json", audio.parameters())
    shutil.copyfile(background_path, root / "background.wav")
    feature_dir = root / "features"
    feature_dir.mkdir(exist_ok=True)
    features_index = []
    by_id = {row["file_id"]: row for row in records}
    audio_root = args.audio_dir.resolve()
    for index, row in enumerate(records, 1):
        path = (audio_root / row["relative_path"]).resolve()
        if not path.is_relative_to(audio_root) or Path(row["file_id"]).name != row["file_id"]:
            raise ValueError("Audio path escapes the dataset")
        raw = path.read_bytes()
        if sha256(raw) != row["sha256"]:
            raise ValueError(f"Audio changed after audit: {path}")
        cache = feature_dir / f"{row['file_id']}.npz"
        if cache.exists():
            features = load_cache(cache, row["sha256"], audio)
        else:
            samples, rate = sf.read(BytesIO(raw), dtype="float32", always_2d=True)
            features = extract_features(mono_resample(samples, rate, audio), audio)
            temporary = cache.with_suffix(".npz.partial")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **features, source_sha256=np.array(row["sha256"]))
            temporary.replace(cache)
            features = load_cache(cache, row["sha256"], audio)
        features_index.append({"file_id": row["file_id"], "label_id": row["label_id"],
                               "source_sha256": row["sha256"], "cache_path": cache.relative_to(root).as_posix(),
                               "n_samples": len(features["waveform"]), "cnn_windows": len(features["log_mel"]),
                               "mfcc_frames": len(features["mfcc"]), "cache_bytes": cache.stat().st_size})
        if index % 500 == 0 or index == len(records):
            print(f"Features verified: {index}/{len(records)}", flush=True)
    write_csv(root / "feature_index.csv", features_index, tuple(features_index[0]))

    def load(file_id):
        return load_cache(feature_dir / f"{file_id}.npz", by_id[file_id]["sha256"], audio)

    for protocol, rows in splits.items():
        print(f"Fitting train-only normalization: {protocol}", flush=True)
        statistics = fit_training_statistics(rows, load, audio.n_mels, audio.n_mfcc)
        statistics["split_sha256"] = sha256((root / "splits" / f"{protocol}.csv").read_bytes())
        write_json(root / "normalization" / f"{protocol}.json", statistics)

    test_protocols = {}
    for protocol, rows in splits.items():
        for row in rows:
            if row["subset"] == "test":
                test_protocols.setdefault(row["file_id"], []).append(protocol)
    speaker_test = [row for row in splits["speaker"] if row["subset"] == "test"]
    examples = {next(row["file_id"] for row in speaker_test if row["emotion"] == emotion) for emotion in LABELS}
    noise_rows, example_rows = [], []
    max_error, peak, replay_verified = 0.0, 0.0, 0
    for index, (file_id, protocols) in enumerate(sorted(test_protocols.items()), 1):
        clean = load(file_id)["waveform"]
        variants = {("clean", ""): clean} if file_id in examples else {}
        for condition in ("white", "background"):
            noise_id = "gaussian_pcg64" if condition == "white" else background_source["noise_id"]
            seed = stable_seed(file_id, noise_id, config["noise_seed"])
            noise, offset = noise_segment(len(clean), seed, None if condition == "white" else background)
            for snr_db in config["snr_db"]:
                mixed, alpha, measured = mix_at_snr(clean, noise, snr_db)
                max_error = max(max_error, abs(measured - snr_db))
                peak = max(peak, float(np.abs(mixed).max()))
                for protocol in protocols:
                    noise_rows.append({
                        "split": protocol, "file_id": file_id, "condition": condition,
                        "snr_db": snr_db, "replica_seed": config["noise_seed"], "noise_id": noise_id,
                        "noise_sha256": "" if condition == "white" else background_source["sha256"],
                        "seed": seed, "offset_samples": offset, "alpha": alpha,
                        "measured_snr_db": measured, "n_samples": len(clean),
                        "mixture_sha256": sha256(mixed.tobytes()),
                    })
                if file_id in examples:
                    variants[(condition, snr_db)] = mixed
        if variants:
            # All seven demo versions of a recording share one playback gain.
            gain = min(1.0, 0.98 / max(float(np.abs(value).max()) for value in variants.values()))
            sample_dir = root / "samples"
            sample_dir.mkdir(exist_ok=True)
            for (condition, snr_db), waveform in variants.items():
                output = sample_dir / f"{file_id}_{condition}{snr_db}.wav"
                sf.write(output, waveform * gain, audio.sample_rate, subtype="FLOAT")
                example_rows.append({"file_id": file_id, "emotion": by_id[file_id]["emotion"],
                                     "condition": condition, "snr_db": snr_db,
                                     "playback_gain": gain, "path": output.relative_to(root).as_posix()})
        if index % 500 == 0 or index == len(test_protocols):
            print(f"Noise SNR verified: {index}/{len(test_protocols)} recordings", flush=True)
    write_csv(root / "noise_manifest.csv", noise_rows, tuple(noise_rows[0]))
    write_csv(root / "samples.csv", example_rows, tuple(example_rows[0]))
    report = {
        "complete": False, "fingerprint": fingerprint, **provenance,
        "accepted_files": len(records), "accepted_speakers": len({row["speaker_id"] for row in records}),
        "classes": dict(sorted(Counter(row["emotion"] for row in records).items())),
        "splits": {name: summarize(rows) for name, rows in splits.items()},
        "features": {"files": len(features_index), "log_mel_shape_per_window": [audio.n_mels, 1 + (audio.window_samples - audio.n_fft) // audio.hop_length],
                     "cnn_windows": sum(row["cnn_windows"] for row in features_index),
                     "mfcc_features_per_frame": audio.n_mfcc * 3, "svm_features_per_file": audio.n_mfcc * 6,
                     "cache_bytes": sum(row["cache_bytes"] for row in features_index)},
        "noise": {"conditions": ["white", "background"], "snr_db": config["snr_db"],
                  "replicas": 1, "unique_test_recordings": len(test_protocols), "manifest_rows": len(noise_rows),
                  "max_snr_error_db": max_error, "max_unclipped_abs_sample": peak,
                  "demo_wav_files": len(example_rows), "sample_policy": "shared playback gain per file; experiments use original float mixtures"},
    }
    # This report is provisional until serialized recipes have been independently replayed.
    write_json(root / "report.json", report)
    for protocol in splits:
        dataset = PreparedDataset(root, protocol, "test", _allow_incomplete=True)
        for row in dataset.records:
            for condition in ("white", "background"):
                for snr in config["snr_db"]:
                    dataset.noisy_audio(row["file_id"], condition, snr)
                    replay_verified += 1
        print(f"Serialized noise recipes replayed: {protocol}", flush=True)
    report["noise"]["replayed_mixtures"] = replay_verified
    report["complete"] = True
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    write_json(root / "report.json", report)
    write_json(args.out_dir / "current.json", {"directory": root.name, "fingerprint": fingerprint})
    write_json(args.report_dir / "stage1.json", {"output_directory": root.as_posix(), **report})
    write_report(args.report_dir / "stage1.md", root, report)
    print(f"Stage 1 complete: {len(records)} files; {len(noise_rows)} verified noise recipes", flush=True)
    print(f"Report: {args.report_dir / 'stage1.md'}", flush=True)


def write_report(path, root, report):
    lines = ["# Этап 1 — результат подготовки данных", "",
             f"Пригодны {report['accepted_files']} записей, {report['accepted_speakers']} диктор. "
             "Исключения определены аудитом до разбиения.", "",
             "| Протокол | Часть | Файлы | Дикторы |", "|---|---|---:|---:|"]
    for protocol, summary in report["splits"].items():
        for subset, counts in summary["subsets"].items():
            lines.append(f"| {protocol} | {subset} | {counts['files']} | {counts['speakers']} |")
    lines.extend(["", "Пересечений файлов и SHA-256 между частями нет. В speaker-протоколе "
                  "также нет общих дикторов. В random-протоколе общие дикторы ожидаемы.", "",
                  f"Артефакты: `{root.as_posix()}`. Указатель на завершённый набор: `data/prepared/current.json`.", "",
                  "- Log-Mel: 64 × 397 на окно; длинные записи покрываются окнами с обязательным последним.",
                  "- MFCC: 13 коэффициентов, включая C0, + delta + delta-delta = 39 на реальный кадр.",
                  "- SVM: среднее и стандартное отклонение 39 признаков = 78 чисел на файл.",
                  "- Нормализация вычислена отдельно по train каждого протокола. Кеш содержит ненормированные признаки.",
                  f"- Кеш признаков: {report['features']['cache_bytes'] / 1024 ** 2:.1f} MiB.",
                  f"- Шумовые рецепты: {report['noise']['manifest_rows']}; все повторно восстановлены и проверены по SHA-256.",
                  f"- Максимальная ошибка SNR: {report['noise']['max_snr_error_db']:.8f} dB.",
                  f"- WAV для прослушивания: {report['noise']['demo_wav_files']} (6 эмоций × 7 условий).", "",
                  "Белый шум — PCG64 Gaussian. Реальный фон — AirConditioner_1.wav из Microsoft MS-SNSD. "
                  "Источник, commit, SHA-256 и сведения о лицензировании сохранены в `configs/background_source.json`.", "",
                  "Ограничения: одна реализация каждого шума на запись и один фоновый исходник. "
                  "SNR измеряется по всей исходной записи с естественными паузами, до дополнения нулями. "
                  "Экспериментальные смеси хранятся как рецепты без клиппинга; демонстрационные WAV имеют "
                  "общий коэффициент громкости для всех условий одного файла.", "",
                  "Модели ещё не обучены. Метрики распознавания появятся на этапах 2–3.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.json"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/raw/CREMA-D/AudioWAV"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    try:
        prepare(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")


if __name__ == "__main__":
    main()
