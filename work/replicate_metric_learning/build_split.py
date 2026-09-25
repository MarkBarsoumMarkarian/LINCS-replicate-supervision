#!/usr/bin/env python3
"""Create the outcome-blind compound split for metric learning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ELIGIBILITY = ROOT / "outputs/lincs_local_pipeline/post_extraction_eligibility.csv"
OUT = ROOT / "outputs/replicate_metric_learning"
SALT = "20260924"

EXPOSED = {
    "BRD-A19037878", "BRD-K68202742", "BRD-K81418486", "BRD-A75409952",
    "BRD-A19500257", "BRD-K96799727", "BRD-K64890080", "BRD-K90382497",
    "BRD-K12539581", "BRD-K77987382", "BRD-A45889380", "BRD-A84481105",
    "BRD-A79768653", "BRD-K35960502", "BRD-K50140147", "BRD-K97764662",
}


def key(pert_id: str) -> str:
    return hashlib.sha256(f"{SALT}|{pert_id}".encode()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ELIGIBILITY)
    if frame["pert_id"].nunique() != 168:
        raise RuntimeError("expected exactly 168 eligible compounds")
    frame = frame.drop_duplicates("pert_id").copy()
    frame["previously_exposed"] = frame["pert_id"].isin(EXPOSED)
    frame["split_hash"] = frame["pert_id"].map(key)

    unexposed = frame[~frame["previously_exposed"]].sort_values("split_hash")
    validation_ids = set(unexposed.iloc[:34]["pert_id"])
    test_ids = set(unexposed.iloc[34:68]["pert_id"])
    frame["split"] = "train"
    frame.loc[frame["pert_id"].isin(validation_ids), "split"] = "validation"
    frame.loc[frame["pert_id"].isin(test_ids), "split"] = "test"
    frame = frame.sort_values(["split", "pert_id"])

    counts = frame.groupby("split")["pert_id"].nunique().to_dict()
    expected = {"train": 100, "validation": 34, "test": 34}
    if counts != expected:
        raise RuntimeError(f"unexpected split counts: {counts}")
    if frame.loc[frame["previously_exposed"], "split"].ne("train").any():
        raise RuntimeError("an exposed compound escaped the training split")

    split_path = OUT / "compound_split.csv"
    frame.to_csv(split_path, index=False, lineterminator="\n")
    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    (OUT / "compound_split.sha256").write_text(
        f"{digest}  compound_split.csv\n", encoding="utf-8"
    )
    record = {
        "expression_opened": False,
        "selection_uses_activity_or_mechanism": False,
        "split_counts": counts,
        "previously_exposed_forced_to_train": len(EXPOSED),
        "split_sha256": digest,
    }
    (OUT / "split_record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
