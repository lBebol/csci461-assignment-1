#!/usr/bin/env python3
"""Rebuild the included dataset from the official UCI archive, without pandas."""

import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip"
FILES = {
    "Youtube01-Psy.csv": "Psy",
    "Youtube02-KatyPerry.csv": "KatyPerry",
    "Youtube03-LMFAO.csv": "LMFAO",
    "Youtube04-Eminem.csv": "Eminem",
    "Youtube05-Shakira.csv": "Shakira",
}
FIELDS = ["COMMENT_ID", "AUTHOR", "DATE", "CONTENT", "CLASS", "video"]


def main():
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive_bytes = response.read()
    rows = []
    source_files = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for filename, video in FILES.items():
            with archive.open(filename) as source:
                reader = csv.DictReader(io.TextIOWrapper(source, encoding="utf-8-sig"))
                if reader.fieldnames != FIELDS[:-1]:
                    raise ValueError(f"Unexpected columns in {filename}: {reader.fieldnames}")
                records = [{**row, "video": video} for row in reader]
            rows.extend(records)
            source_files.append({"file": filename, "video": video, "rows": len(records)})
    if len(rows) != 1956 or {row["CLASS"] for row in rows} != {"0", "1"}:
        raise ValueError("The upstream dataset has changed; review it before proceeding.")

    destination = ROOT / "data"
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "dataset.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "source_url": URL,
        "source_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "dataset_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "rows": len(rows),
        "files": source_files,
        "license": "CC-BY-4.0",
        "citation": "Alberto, T. & Lochter, J. (2015). YouTube Spam Collection. UCI. DOI:10.24432/C58885",
    }
    (destination / "SOURCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved {len(rows):,} comments to {csv_path}")
    print(f"SHA-256: {provenance['dataset_sha256']}")


if __name__ == "__main__":
    main()
