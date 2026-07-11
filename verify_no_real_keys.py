from __future__ import annotations

import getpass
from pathlib import Path


ROOTS = [
    Path(r"E:\ai\image2-gen"),
    Path(r"E:\ai\HermesHome\claude-authenticity-detector"),
]
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".mp3", ".zip", ".pyc"}


def main() -> None:
    keys = [
        getpass.getpass("Key A (hidden): ").encode(),
        getpass.getpass("Key B (hidden): ").encode(),
        getpass.getpass("Key C (hidden): ").encode(),
    ]
    counts = [0, 0, 0]
    files = [set(), set(), set()]
    for root in ROOTS:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            for index, key in enumerate(keys):
                found = raw.count(key)
                if found:
                    counts[index] += found
                    files[index].add(str(path))
    for label, count, matched_files in zip("ABC", counts, files, strict=True):
        print(label, "exact_matches", count, "files", len(matched_files))
    raise SystemExit(1 if any(counts) else 0)


if __name__ == "__main__":
    main()
