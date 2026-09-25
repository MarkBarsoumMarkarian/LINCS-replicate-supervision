#!/usr/bin/env python3
"""Score frozen plate-relative signatures with gseapy ssGSEA."""

from __future__ import annotations

import argparse
from pathlib import Path

import gseapy as gp
import pandas as pd


DEVELOPMENT = {
    "BRD-K81418486",
    "BRD-A19037878",
    "BRD-A75409952",
    "BRD-A19500257",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signatures", required=True)
    parser.add_argument("--gmt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scope", choices=["development", "all"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signatures = pd.read_csv(args.signatures, low_memory=False)
    if args.scope == "development":
        signatures = signatures[signatures["pert_id"].isin(DEVELOPMENT)].copy()
    gene_columns = [column for column in signatures if column.startswith("g_")]
    sample_ids = [
        f"{row.pert_id}|{row.rna_plate}"
        for row in signatures[["pert_id", "rna_plate"]].itertuples(index=False)
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("ssGSEA sample identifiers are not unique")

    expression = signatures[gene_columns].T
    expression.index = [column[2:] for column in gene_columns]
    expression.columns = sample_ids
    expression.insert(0, "gene_id", expression.index.astype(str))
    expression = expression.reset_index(drop=True)

    result = gp.ssgsea(
        data=expression,
        gene_sets=args.gmt,
        outdir=None,
        sample_norm_method="rank",
        correl_norm_type="rank",
        min_size=5,
        max_size=500,
        permutation_num=0,
        weight=0.25,
        no_plot=True,
        threads=4,
        seed=20_260_924,
        verbose=False,
    )
    long = result.res2d.rename(columns={"Name": "sample_id", "Term": "pathway"})
    long["ES"] = pd.to_numeric(long["ES"])
    scores = long.pivot(index="sample_id", columns="pathway", values="ES")
    scores = scores.reindex(sample_ids)
    metadata = signatures[
        ["pert_id", "pert_iname", "rna_plate", "n_treatment_wells", "n_vehicle_wells"]
    ].copy()
    metadata.insert(0, "sample_id", sample_ids)
    output = metadata.merge(scores.reset_index(), on="sample_id", validate="one_to_one")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print(
        {
            "scope": args.scope,
            "samples": len(output),
            "pathways": scores.shape[1],
            "gseapy_version": gp.__version__,
            "output": str(destination),
        }
    )


if __name__ == "__main__":
    main()
