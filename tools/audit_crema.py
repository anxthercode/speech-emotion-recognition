"""Decode CREMA-D WAV files and build metadata before dataset splitting."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile as sf

from .check_crema import classify


LABEL_IDS = {code: index for index, code in enumerate(("ANG", "DIS", "FEA", "HAP", "NEU", "SAD"))}
FILENAME = re.compile(r"(?P<speaker_id>[0-9]{4})_(?P<sentence_id>[A-Z]{3})_(?P<emotion>ANG|DIS|FEA|HAP|NEU|SAD)_(?P<intensity>LO|MD|HI|XX)")
MIN_DURATION_S = 0.25
MIN_RMS = 1e-4
LONG_REVIEW_S = 8.0
DOCUMENTED_AUDIO_ISSUE = "1076_MTI_SAD_XX"
FIELDS = (
    "file_id", "relative_path", "speaker_id", "sentence_id", "emotion", "label_id",
    "intensity", "sample_rate", "channels", "n_samples", "duration_s", "rms",
    "sha256", "valid", "exclusion_reason", "audit_flags", "duplicate_of",
)


def inspect_record(path: Path) -> dict:
    record = dict.fromkeys(FIELDS, "")
    record.update(file_id=path.stem, relative_path=path.name)
    reasons = []
    flags = []
    match = FILENAME.fullmatch(path.stem)
    if match:
        record.update(match.groupdict())
        record["label_id"] = LABEL_IDS[record["emotion"]]
    else:
        reasons.append("invalid_filename")

    raw = path.read_bytes()
    record["sha256"] = hashlib.sha256(raw).hexdigest()
    kind = classify(raw[:512])
    if kind == "wav_header":
        try:
            with sf.SoundFile(BytesIO(raw)) as source:
                record["sample_rate"] = source.samplerate
                record["channels"] = source.channels
                samples = source.read(dtype="float64", always_2d=True)
        except sf.SoundFileError:
            reasons.append("decode_error")
        else:
            frame_count = len(samples)
            duration = frame_count / record["sample_rate"]
            record.update(n_samples=frame_count, duration_s=duration)
            if frame_count == 0:
                reasons.append("empty_audio")
            else:
                if duration < MIN_DURATION_S:
                    reasons.append("too_short")
                if duration > LONG_REVIEW_S:
                    flags.append("long_clip_review")
                if np.isfinite(samples).all():
                    mono = samples.mean(axis=1)
                    rms = float(np.sqrt(np.mean(mono ** 2)))
                    record["rms"] = rms
                    if rms < MIN_RMS:
                        reasons.append("near_silent")
                else:
                    reasons.append("non_finite_audio")
    else:
        reasons.append("lfs_pointer" if kind == "lfs_pointer" else "invalid_wav_header")

    if path.stem == DOCUMENTED_AUDIO_ISSUE:
        flags.append("upstream_audio_issue")
    record.update(valid=not reasons, exclusion_reason="|".join(reasons), audit_flags="|".join(flags))
    return record


def handle_duplicates(records: list[dict]) -> list[dict]:
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record)

    duplicates = []
    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        labels = {(row["speaker_id"], row["emotion"]) for row in group if row["speaker_id"]}
        conflict = len(labels) > 1
        eligible = [row for row in group if row["valid"]]
        canonical = eligible[0] if eligible and not conflict else None
        for row in eligible:
            if row is canonical:
                continue
            row["valid"] = False
            row["exclusion_reason"] = "duplicate_label_conflict" if conflict else "exact_duplicate"
            if canonical is not None:
                row["duplicate_of"] = canonical["file_id"]
        duplicates.append({
            "sha256": digest,
            "files": [row["relative_path"] for row in group],
            "label_conflict": conflict,
            "retained_file": canonical["relative_path"] if canonical is not None else None,
        })
    return duplicates


def csv_text(records: list[dict]) -> str:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue()


def save_outputs(output_dir: Path, outputs: dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    try:
        for name, content in outputs.items():
            target = output_dir / name
            with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=output_dir, delete=False) as stream:
                pending.append((Path(stream.name), target))
                stream.write(content)
        for temp, target in pending:
            temp.replace(target)
    finally:
        for temp, _ in pending:
            temp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_dir", type=Path, help="Directory containing the source WAV files")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--revision", help="Optional source CREMA-D commit for provenance")
    args = parser.parse_args()

    paths = sorted(args.audio_dir.glob("*.wav"))
    if not paths:
        parser.error(f"No WAV files found in {args.audio_dir}")

    records = []
    for index, path in enumerate(paths, start=1):
        try:
            records.append(inspect_record(path))
        except OSError as error:
            parser.exit(1, f"Could not read {path}: {error}\n")
        if index % 1000 == 0 or index == len(paths):
            print(f"Audited {index}/{len(paths)} files", flush=True)

    duplicates = handle_duplicates(records)
    accepted = [row for row in records if row["valid"]]
    excluded = [row for row in records if not row["valid"]]
    reasons = Counter(reason for row in excluded for reason in row["exclusion_reason"].split("|"))
    report = {
        "audio_dir": str(args.audio_dir.resolve()),
        "dataset_revision": args.revision,
        "relative_path_base": "audio_dir",
        "audited_files": len(records),
        "accepted_files": len(accepted),
        "excluded_files": len(excluded),
        "accepted_speakers": len({row["speaker_id"] for row in accepted}),
        "accepted_classes": dict(sorted(Counter(row["emotion"] for row in accepted).items())),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "thresholds": {"min_duration_s": MIN_DURATION_S, "min_mono_rms": MIN_RMS, "long_clip_review_s": LONG_REVIEW_S},
        "review_flags": [
            {"file": row["relative_path"], "flags": row["audit_flags"], "valid": row["valid"]}
            for row in records if row["audit_flags"]
        ],
        "duplicate_groups": duplicates,
    }
    if accepted:
        durations = [row["duration_s"] for row in accepted]
        report["accepted_duration_s"] = {
            "min": min(durations), "median": float(np.median(durations)), "max": max(durations),
        }
    summary = json.dumps(report, ensure_ascii=True, indent=2) + "\n"
    save_outputs(args.out_dir, {
        "metadata.csv": csv_text(records),
        "exclusions.csv": csv_text(excluded),
        "labels.json": json.dumps(LABEL_IDS, indent=2) + "\n",
        "crema_audio_audit.json": summary,
    })
    print(summary)
    return 0 if accepted and not reasons["lfs_pointer"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
