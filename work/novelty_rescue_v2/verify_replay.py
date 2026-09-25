#!/usr/bin/env python3
"""Verify that the publishable extension tables reproduce byte for byte."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "work/novelty_rescue_v2/replay_reference"
CURRENT = ROOT / "outputs/novelty_rescue_v2"
OUTPUT = CURRENT / "replay_verification.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    reference_files = sorted(
        [*REFERENCE.glob("*.csv"), *REFERENCE.glob("*.json")], key=lambda path: path.name
    )
    records = []
    for reference in reference_files:
        current = CURRENT / reference.name
        records.append(
            {
                "file": reference.name,
                "reference_sha256": digest(reference),
                "replay_sha256": digest(current) if current.exists() else None,
                "byte_exact": current.exists() and reference.read_bytes() == current.read_bytes(),
            }
        )
    result = {
        "reference_files": len(records),
        "byte_exact_files": sum(record["byte_exact"] for record in records),
        "all_publishable_tables_and_json_byte_exact": all(
            record["byte_exact"] for record in records
        ),
        "records": records,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    if not result["all_publishable_tables_and_json_byte_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
