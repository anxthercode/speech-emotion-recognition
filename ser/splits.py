"""Fixed speaker holdout and exact-size, stratified clip-level control split."""
from collections import Counter
import csv
from itertools import combinations

import numpy as np

LABELS = {code: i for i, code in enumerate(("ANG", "DIS", "FEA", "HAP", "NEU", "SAD"))}
SUBSETS = ("train", "validation", "test")


def read_metadata(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = sorted((row for row in rows if row["valid"].lower() == "true"), key=lambda row: row["file_id"])
    if not records:
        raise ValueError("No audited valid recordings")
    for key in ("file_id", "sha256"):
        if len({row[key] for row in records}) != len(records):
            raise ValueError(f"Duplicate {key} in valid metadata; rerun the audit")
    for row in records:
        if LABELS.get(row["emotion"]) != int(row["label_id"]):
            raise ValueError(f"Inconsistent label: {row['file_id']}")
    return records


def stratified_take(records, size, rng):
    """Largest-remainder allocation preserves the exact requested file count."""
    labels = sorted({int(row["label_id"]) for row in records})
    groups = [[row for row in records if int(row["label_id"]) == label] for label in labels]
    expected = np.array([len(group) for group in groups]) * size / len(records)
    quotas = np.floor(expected).astype(int)
    # Stable tie break follows the fixed class order.
    for index in np.argsort(-(expected - quotas), kind="stable")[:size - int(quotas.sum())]:
        quotas[index] += 1
    taken, remaining = [], []
    for group, count in zip(groups, quotas):
        indices = rng.permutation(len(group))
        taken.extend(group[i] for i in indices[:count])
        remaining.extend(group[i] for i in indices[count:])
    return taken, remaining


def make_splits(records, seed=42, validation_speakers=14, test_speakers=14):
    records = sorted(records, key=lambda row: row["file_id"])
    actors = sorted({row["speaker_id"] for row in records})
    if min(validation_speakers, test_speakers) < 1 or validation_speakers + test_speakers >= len(actors):
        raise ValueError("Need nonempty train, validation and test speaker groups")
    rng = np.random.Generator(np.random.PCG64(seed))
    shuffled = rng.permutation(actors)
    test = set(shuffled[:test_speakers])
    validation = set(shuffled[test_speakers:test_speakers + validation_speakers])
    assignment = {actor: "test" if actor in test else "validation" if actor in validation else "train"
                  for actor in actors}
    speaker = [{**row, "subset": assignment[row["speaker_id"]], "split_seed": seed} for row in records]
    sizes = Counter(row["subset"] for row in speaker)
    rng = np.random.Generator(np.random.PCG64(seed))
    test_rows, remaining = stratified_take(records, sizes["test"], rng)
    validation_rows, train_rows = stratified_take(remaining, sizes["validation"], rng)
    random = sorted([{**row, "subset": subset, "split_seed": seed}
                     for subset, group in zip(SUBSETS, (train_rows, validation_rows, test_rows))
                     for row in group], key=lambda row: row["file_id"])
    result = {"speaker": speaker, "random": random}
    for protocol, rows in result.items():
        validate_split(records, rows, grouped=protocol == "speaker")
    return result


def validate_split(records, rows, grouped):
    original = {row["file_id"]: row for row in records}
    if len(rows) != len(original) or {row["file_id"] for row in rows} != set(original):
        raise ValueError("Split must include every accepted recording exactly once")
    if any(row["subset"] not in SUBSETS for row in rows):
        raise ValueError("Unknown subset")
    for row in rows:
        if any(str(row[key]) != str(original[row["file_id"]][key])
               for key in ("speaker_id", "label_id", "sha256")):
            raise ValueError("Split changed original metadata")
    groups = {subset: [row for row in rows if row["subset"] == subset] for subset in SUBSETS}
    for group in groups.values():
        if {int(row["label_id"]) for row in group} != set(LABELS.values()):
            raise ValueError("Every subset must contain all six classes")
    for left, right in combinations(SUBSETS, 2):
        for key in ("file_id", "sha256", *(('speaker_id',) if grouped else ())):
            if {row[key] for row in groups[left]} & {row[key] for row in groups[right]}:
                raise ValueError(f"{key} leakage between {left} and {right}")


def summarize(rows):
    groups = {subset: [row for row in rows if row["subset"] == subset] for subset in SUBSETS}
    speakers = {subset: sorted({row["speaker_id"] for row in group}) for subset, group in groups.items()}
    return {
        "subsets": {subset: {"files": len(group), "speakers": len(speakers[subset]),
                             "speaker_ids": speakers[subset],
                             "classes": dict(sorted(Counter(row["emotion"] for row in group).items()))}
                    for subset, group in groups.items()},
        "speaker_overlap": {f"{a}/{b}": len(set(speakers[a]) & set(speakers[b]))
                            for a, b in combinations(SUBSETS, 2)},
    }
