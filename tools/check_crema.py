"""Read-only preflight for a CREMA-D archive or its extracted AudioWAV folder.

No archive extraction or network access. RIFF headers are not a full audio audit.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import zipfile

LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def classify(header: bytes) -> str:
    if header.startswith(LFS_HEADER):
        return "lfs_pointer"
    if header[:4] in (b"RIFF", b"RF64") and header[8:12] == b"WAVE":
        return "wav_header"
    return "unknown"


def inspect(source: Path) -> dict:
    counts = Counter()
    speakers = set()
    emotions = Counter()
    examples = []
    stored_bytes = 0
    expected_audio_bytes = 0

    def visit(name, size, header):
        nonlocal stored_bytes, expected_audio_bytes
        kind = classify(header)
        counts[kind] += 1
        stored_bytes += size
        parts = Path(name).stem.split("_")
        if len(parts) == 4:
            speakers.add(parts[0])
            emotions[parts[2]] += 1
        if kind == "lfs_pointer":
            match = re.search(rb"(?m)^size (\d+)\s*$", header)
            if match:
                expected_audio_bytes += int(match[1])
        if len(examples) < 3:
            examples.append({"file": name, "stored_bytes": size, "type": kind})

    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for entry in archive.infolist():
                path = entry.filename.replace("\\", "/")
                if not entry.is_dir() and "/AudioWAV/" in "/"+path and path.lower().endswith(".wav"):
                    with archive.open(entry) as stream:
                        visit(path, entry.file_size, stream.read(512))
    elif source.is_dir():
        for file in sorted(source.glob("*.wav")):
            with file.open("rb") as stream:
                visit(file.name, file.stat().st_size, stream.read(512))
    else:
        raise ValueError("Укажите ZIP-архив или существующую папку AudioWAV.")
    complete_headers = counts["wav_header"] == 7442 and counts["unknown"] == 0 and counts["lfs_pointer"] == 0
    return {
        "source": str(source.resolve()), "wav_entries": sum(counts.values()),
        "speakers_in_names": len(speakers), "classes_in_names": dict(sorted(emotions.items())),
        "types": dict(counts), "stored_wav_bytes": stored_bytes,
        "expected_audio_bytes_from_pointers": expected_audio_bytes,
        "complete_wav_headers": complete_headers,
        "full_audio_audit_performed": False, "examples": examples,
        "next_step": "decode_and_audit_audio" if complete_headers else "fetch_real_audio_with_git_lfs" if counts["lfs_pointer"] else "check_missing_or_invalid_files",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = inspect(args.source)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized+"\n", encoding="utf-8")
    return 0 if report["complete_wav_headers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
