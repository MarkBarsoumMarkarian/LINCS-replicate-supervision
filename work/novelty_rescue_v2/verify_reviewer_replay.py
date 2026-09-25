#!/usr/bin/env python3
"""Verify the separate post-freeze reviewer-response replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "work/novelty_rescue_v2/reviewer_replay_reference"
CURRENT = ROOT / "outputs/novelty_rescue_v2"
OUTPUT = CURRENT / "reviewer_replay_verification.json"
FILES = ["reviewer_schedule_audit.csv", "reviewer_response_results.json"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    records = []
    for name in FILES:
        reference = REFERENCE / name
        current = CURRENT / name
        records.append(
            {
                "file": name,
                "reference_sha256": digest(reference),
                "replay_sha256": digest(current) if current.exists() else None,
                "byte_exact": current.exists() and reference.read_bytes() == current.read_bytes(),
            }
        )
    result = {
        "status": "post-freeze sensitivity replay",
        "reference_files": len(records),
        "byte_exact_files": sum(record["byte_exact"] for record in records),
        "all_reviewer_response_artifacts_byte_exact": all(
            record["byte_exact"] for record in records
        ),
        "records": records,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    if not result["all_reviewer_response_artifacts_byte_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
