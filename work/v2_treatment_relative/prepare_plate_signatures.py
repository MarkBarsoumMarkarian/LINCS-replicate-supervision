#!/usr/bin/env python3
"""Build plate-level, vehicle-relative signatures without pairwise comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_H5 = ROOT / "outputs/lincs_local_pipeline/lincs_mcf7_10um_6h_level2_subset.h5"
INPUT_META = ROOT / "outputs/lincs_local_pipeline/extracted_instance_metadata.csv"
OUT = ROOT / "work/v2_treatment_relative/derived"
SIGNATURES = OUT / "plate_relative_signatures.csv.gz"
DIAGNOSTICS = OUT / "preparation_diagnostics.json"

DEVELOPMENT = {
    "BRD-K81418486": "vorinostat",
    "BRD-A19037878": "trichostatin-a",
    "BRD-A75409952": "wortmannin",
    "BRD-A19500257": "geldanamycin",
}


def decode(values: np.ndarray) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(INPUT_META, low_memory=False)
    with h5py.File(INPUT_H5, "r") as handle:
        inst_ids = decode(handle["inst_id"][:])
        gene_ids = decode(handle["gene_id"][:])
        values = handle["values"][:].astype(np.float64)

    if metadata["inst_id"].tolist() != inst_ids:
        raise RuntimeError("H5 columns and metadata rows are not exactly aligned")
    if not np.isfinite(values).all() or np.min(values) <= 0:
        raise RuntimeError("Level-2 values must be finite and positive for log2 transform")

    log_values = np.log2(values)
    gene_columns = [f"g_{g}" for g in gene_ids]
    rows: list[dict[str, object]] = []
    plates_without_vehicle: list[str] = []
    vehicle_counts: dict[str, int] = {}

    for plate, plate_frame in metadata.groupby("rna_plate", sort=True):
        plate_indices = plate_frame.index.to_numpy()
        vehicle_local = (plate_frame["pert_type"].to_numpy() == "ctl_vehicle")
        vehicle_counts[str(plate)] = int(vehicle_local.sum())
        if not vehicle_local.any():
            plates_without_vehicle.append(str(plate))
            continue
        vehicle_center = np.median(log_values[plate_indices[vehicle_local], :], axis=0)
        treatments = plate_frame[plate_frame["pert_type"] == "trt_cp"]
        for pert_id, compound_frame in treatments.groupby("pert_id", sort=True):
            compound_values = log_values[compound_frame.index.to_numpy(), :]
            signature = compound_values.mean(axis=0) - vehicle_center
            row: dict[str, object] = {
                "pert_id": str(pert_id),
                "pert_iname": str(compound_frame["pert_iname"].iloc[0]),
                "rna_plate": str(plate),
                "n_treatment_wells": int(len(compound_frame)),
                "n_vehicle_wells": int(vehicle_local.sum()),
                "development_compound": bool(pert_id in DEVELOPMENT),
            }
            row.update(dict(zip(gene_columns, signature.tolist())))
            rows.append(row)

    result = pd.DataFrame(rows)
    if result.duplicated(["pert_id", "rna_plate"]).any():
        raise RuntimeError("plate-level compound rows are not unique")
    result.to_csv(SIGNATURES, index=False, compression="gzip")

    counts = result.groupby(["pert_id", "pert_iname"], as_index=False).agg(
        n_plates=("rna_plate", "nunique"),
        n_treatment_wells=("n_treatment_wells", "sum"),
    )
    development_counts = counts[counts["pert_id"].isin(DEVELOPMENT)].to_dict("records")
    diagnostics = {
        "input_profiles": int(values.shape[0]),
        "genes": int(values.shape[1]),
        "output_plate_signatures": int(len(result)),
        "output_compounds": int(result["pert_id"].nunique()),
        "output_plates": int(result["rna_plate"].nunique()),
        "plates_without_vehicle": plates_without_vehicle,
        "vehicle_wells_per_plate": {
            "minimum": int(min(vehicle_counts.values())),
            "median": float(np.median(list(vehicle_counts.values()))),
            "maximum": int(max(vehicle_counts.values())),
        },
        "development_counts": development_counts,
        "pairwise_outcomes_calculated": False,
        "representation": "plate mean log2 treatment minus same-plate vehicle median",
    }
    DIAGNOSTICS.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()

