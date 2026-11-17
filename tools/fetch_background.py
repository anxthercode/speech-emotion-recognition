"""Download the small, pinned noise source and verify its SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("configs/background_source.json"))
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    target = Path(source["path"])
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != source["sha256"]:
            parser.error(f"Existing background has a different hash: {target}")
        print(f"Verified {target}")
        return
    with urllib.request.urlopen(source["url"], timeout=60) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000 or hashlib.sha256(raw).hexdigest() != source["sha256"]:
        parser.error("Background download failed size/hash verification")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    print(f"Downloaded and verified {target}")


if __name__ == "__main__":
    main()
