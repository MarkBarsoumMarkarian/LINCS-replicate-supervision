#!/usr/bin/env python3
"""Apply the frozen Phase I representations across five new GSE70138 cell lines."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (
    GSE70138_GCTX,
    GSE70138_META,
    SEED,
    V2_OUT,
    build_context_signatures,
    cluster_bootstrap_difference,
    fit_frozen_transforms,
    load_filzen,
    load_phase1,
    make_representations,
    normalize_rows,
    phase1_pathway_scaling,
    read_gse70138_metadata,
    retrieval_metrics,
    score_hallmark_ssgsea,
    set_seed,
    write_json,
)


CELL_LINES = ["A375", "A549", "HA1E", "HT29", "PC3"]
EXPECTED_COMPOUNDS = {"A375": 70, "A549": 102, "HA1E": 78, "HT29": 77, "PC3": 63}
BOOTSTRAPS = 2_000


def main() -> None:
    set_seed(SEED)
    V2_OUT.mkdir(parents=True, exist_ok=True)
    signature_dir = V2_OUT / "multicontext_signatures"
    signature_dir.mkdir(parents=True, exist_ok=True)
    phase1, genes, indices = load_phase1()
    transforms = fit_frozen_transforms(phase1, genes, indices)
    filzen = load_filzen(genes)
    pathways, pathway_mean, pathway_sd = phase1_pathway_scaling(phase1, indices)
    phase1_ids = set(phase1["pert_id"].astype(str))
    raw_phase2_available = GSE70138_GCTX.exists() and GSE70138_META.exists()
    if raw_phase2_available:
        metadata, gene_ids, gene_order = read_gse70138_metadata()
    else:
        metadata = None
        gene_ids = None
        gene_order = None

    metric_rows = []
    query_tables: dict[tuple[str, str], pd.DataFrame] = {}
    eligibility_rows = []
    gzip_options = {"method": "gzip", "compresslevel": 6, "mtime": 0}

    for position, cell_line in enumerate(CELL_LINES, start=1):
        cached_path = (
            signature_dir
            / f"gse70138_{cell_line.lower()}_10um_24h_unseen_ge6.csv.gz"
        )
        if raw_phase2_available:
            print(
                f"[{position}/5] {cell_line}: extracting leakage-safe plate signatures",
                flush=True,
            )
            signatures = build_context_signatures(
                metadata,
                gene_ids,
                gene_order,
                cell_line=cell_line,
                dose=10.0,
                time_hours=24.0,
                minimum_plates=6,
            )
        else:
            if not cached_path.exists():
                raise FileNotFoundError(
                    f"Missing raw GSE70138 inputs and cached signatures: {cached_path}"
                )
            print(
                f"[{position}/5] {cell_line}: using archived leakage-safe signatures",
                flush=True,
            )
            signatures = pd.read_csv(cached_path, low_memory=False)
        counts = signatures.groupby("pert_id")["rna_plate"].nunique()
        eligible_ids = set(
            counts[(counts >= 6) & ~counts.index.astype(str).isin(phase1_ids)].index.astype(str)
        )
        signatures = signatures[
            signatures["pert_id"].astype(str).isin(eligible_ids)
        ].reset_index(drop=True)
        if len(eligible_ids) != EXPECTED_COMPOUNDS[cell_line]:
            raise RuntimeError(
                f"{cell_line} metadata eligibility changed: expected "
                f"{EXPECTED_COMPOUNDS[cell_line]}, found {len(eligible_ids)}"
            )
        if raw_phase2_available:
            signatures.to_csv(
                cached_path,
                index=False,
                compression=gzip_options,
            )
        eligibility_rows.append(
            {
                "cell_line": cell_line,
                "eligible_unseen_compounds": len(eligible_ids),
                "profiles": len(signatures),
                "plates": signatures["rna_plate"].nunique(),
                "minimum_plates_per_compound": int(
                    signatures.groupby("pert_id")["rna_plate"].nunique().min()
                ),
            }
        )
        raw = signatures[genes].to_numpy(dtype=np.float32)
        representations = make_representations(raw, transforms, filzen)
        representations["ssgsea_hallmark"] = score_hallmark_ssgsea(
            signatures, genes, pathways, pathway_mean, pathway_sd
        )
        representations["contrastive_lda_ensemble"] = np.hstack(
            [
                normalize_rows(representations["replicate_contrastive"]) / np.sqrt(2),
                normalize_rows(representations["shrinkage_lda"]) / np.sqrt(2),
            ]
        )
        compounds = signatures["pert_id"].astype(str).to_numpy()
        plates = signatures["rna_plate"].astype(str).to_numpy()
        for method, values in representations.items():
            metrics, queries = retrieval_metrics(values, compounds, plates)
            metric_rows.append({"cell_line": cell_line, "method": method, **metrics})
            query_tables[(cell_line, method)] = queries
        print(
            f"[{position}/5] {cell_line}: {len(eligible_ids)} compounds, "
            f"{len(signatures)} profiles evaluated",
            flush=True,
        )

    eligibility = pd.DataFrame(eligibility_rows)
    eligibility.to_csv(V2_OUT / "multicontext_eligibility.csv", index=False)
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(V2_OUT / "multicontext_metrics.csv", index=False)
    pd.concat(
        [
            frame.assign(cell_line=cell_line, method=method)
            for (cell_line, method), frame in query_tables.items()
        ],
        ignore_index=True,
    ).to_csv(V2_OUT / "multicontext_query_results.csv", index=False)

    nonlearned = ["raw_cosine", "pca_64", "limma_weighted", "ssgsea_hallmark"]
    comparison_rows = []
    per_context_differences: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for offset, cell_line in enumerate(CELL_LINES):
        context = metrics_frame[metrics_frame["cell_line"] == cell_line]
        best = str(
            context[context["method"].isin(nonlearned)]
            .sort_values(["mean_average_precision", "method"], ascending=[False, True])
            .iloc[0]["method"]
        )
        lda_query = query_tables[(cell_line, "shrinkage_lda")]
        baseline_query = query_tables[(cell_line, best)]
        difference, low, high = cluster_bootstrap_difference(
            lda_query,
            baseline_query,
            seed=SEED + 100 + offset,
            draws=BOOTSTRAPS,
        )
        comparison_rows.append(
            {
                "cell_line": cell_line,
                "selected_method": "shrinkage_lda",
                "strongest_nonlearned_baseline": best,
                "map_difference": difference,
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
            }
        )
        query_difference = (
            lda_query["average_precision"].to_numpy()
            - baseline_query["average_precision"].to_numpy()
        )
        per_context_differences[cell_line] = (
            lda_query["pert_id"].to_numpy(), query_difference
        )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(V2_OUT / "multicontext_comparisons.csv", index=False)

    rng = np.random.default_rng(SEED + 200)
    bootstrap = np.empty(BOOTSTRAPS, dtype=float)
    for draw in range(BOOTSTRAPS):
        context_means = []
        for cell_line in CELL_LINES:
            compounds, differences = per_context_differences[cell_line]
            unique = np.unique(compounds)
            grouped = {name: differences[compounds == name] for name in unique}
            sampled = rng.choice(unique, size=len(unique), replace=True)
            context_means.append(
                np.mean(np.concatenate([grouped[name] for name in sampled]))
            )
        bootstrap[draw] = np.mean(context_means)
    observed = float(comparisons["map_difference"].mean())
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    positive_contexts = int((comparisons["map_difference"] > 0).sum())
    passed = low > 0 and positive_contexts >= 4
    result = {
        "training_context": "GSE92742 MCF7 10 micromolar 6 hours",
        "external_contexts": CELL_LINES,
        "external_compounds_by_context": EXPECTED_COMPOUNDS,
        "model_refitting_on_phase2": False,
        "macro_map_improvement_over_contextwise_strongest_nonlearned": observed,
        "stratified_compound_bootstrap_95_interval": [float(low), float(high)],
        "positive_contexts": positive_contexts,
        "required_positive_contexts": 4,
        "bootstrap_draws": BOOTSTRAPS,
        "multicontext_endpoint": "PASS" if passed else "FAIL",
    }
    write_json(V2_OUT / "multicontext_result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
