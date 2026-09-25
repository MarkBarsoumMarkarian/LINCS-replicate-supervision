#!/usr/bin/env python3
"""Run MODZ and held-out MoA/target retrieval on the fixed Phase II cohort."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import (
    SEED,
    V2_OUT,
    cluster_bootstrap_difference,
    fit_frozen_transforms,
    load_filzen,
    load_phase1,
    load_repurposing_annotations,
    make_representations,
    modz_query_to_compound,
    normalize_rows,
    phase1_pathway_scaling,
    retrieval_metrics,
    score_hallmark_ssgsea,
    set_seed,
    write_json,
)


EXTERNAL_SIGNATURES = (
    V2_OUT.parent
    / "replicate_metric_learning/external_gse70138/"
    "phase2_mcf7_10um_24h_plate_signatures.csv.gz"
)
BOOTSTRAPS = 2_000
PERMUTATIONS = 2_000


def mean_query_to_compound(
    values: np.ndarray, compounds: np.ndarray, plates: np.ndarray
) -> tuple[dict[str, float], pd.DataFrame]:
    values = values.astype(np.float64, copy=False)
    names = np.array(sorted(np.unique(compounds)))
    reciprocal = np.empty(len(values), dtype=float)
    top1 = np.empty(len(values), dtype=float)
    hit5 = np.empty(len(values), dtype=float)
    for query in range(len(values)):
        candidates = []
        for compound in names:
            mask = (compounds == compound) & (plates != plates[query])
            mask &= np.arange(len(values)) != query
            if not mask.any():
                raise RuntimeError("mean-consensus candidate lacks a cross-plate profile")
            candidates.append(values[mask].mean(axis=0))
        scores = normalize_rows(np.asarray(candidates)) @ normalize_rows(values[query : query + 1])[0]
        order = np.argsort(scores, kind="stable")[::-1]
        rank = int(np.flatnonzero(names[order] == compounds[query])[0]) + 1
        reciprocal[query] = 1.0 / rank
        top1[query] = float(rank == 1)
        hit5[query] = float(rank <= 5)
    metrics = {
        "mean_reciprocal_rank": float(reciprocal.mean()),
        "top1_accuracy": float(top1.mean()),
        "top5_hit_rate": float(hit5.mean()),
        "queries": int(len(values)),
        "compounds": int(len(names)),
    }
    query_frame = pd.DataFrame(
        {
            "pert_id": compounds,
            "rna_plate": plates,
            "reciprocal_rank": reciprocal,
            "top1_correct": top1,
            "top5_hit": hit5,
        }
    )
    return metrics, query_frame


def collapse_compounds(
    values: np.ndarray, compounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    names = np.array(sorted(np.unique(compounds)))
    collapsed = np.vstack(
        [normalize_rows(values[compounds == name]).mean(axis=0) for name in names]
    )
    return names, normalize_rows(collapsed)


def token_sets(values: pd.Series) -> list[frozenset[str]]:
    return [
        frozenset(token for token in str(value).split("|") if token)
        if pd.notna(value)
        else frozenset()
        for value in values
    ]


def biological_query_results(
    similarity: np.ndarray,
    names: np.ndarray,
    annotations: list[frozenset[str]],
) -> tuple[dict[str, float], pd.DataFrame]:
    frequency: dict[str, int] = {}
    for annotation in annotations:
        for token in annotation:
            frequency[token] = frequency.get(token, 0) + 1
    eligible_tokens = {token for token, count in frequency.items() if count >= 2}
    filtered = [annotation & eligible_tokens for annotation in annotations]
    annotated = np.array([bool(annotation) for annotation in annotations])
    rows = []
    for query in range(len(names)):
        if not annotated[query]:
            continue
        candidate_mask = annotated.copy()
        candidate_mask[query] = False
        candidate_indices = np.flatnonzero(candidate_mask)
        relevant = np.array(
            [bool(filtered[query] & filtered[index]) for index in candidate_indices]
        )
        if not relevant.any():
            continue
        order = np.argsort(similarity[query, candidate_indices], kind="stable")[::-1]
        ranked = relevant[order]
        ranks = np.flatnonzero(ranked) + 1
        average_precision = float(np.mean(np.arange(1, len(ranks) + 1) / ranks))
        rows.append(
            {
                "pert_id": names[query],
                "average_precision": average_precision,
                "top1_correct": float(ranked[0]),
                "top5_hit": float(ranked[:5].any()),
                "relevant_compounds": int(relevant.sum()),
                "eligible_annotations": "|".join(sorted(filtered[query])),
            }
        )
    frame = pd.DataFrame(rows)
    metrics = {
        "mean_average_precision": float(frame["average_precision"].mean()),
        "nearest_neighbor_accuracy": float(frame["top1_correct"].mean()),
        "top5_hit_rate": float(frame["top5_hit"].mean()),
        "eligible_queries": int(len(frame)),
        "annotated_compounds": int(annotated.sum()),
        "eligible_annotation_terms": int(len(eligible_tokens)),
    }
    return metrics, frame


def paired_query_bootstrap(
    selected: pd.DataFrame, baseline: pd.DataFrame, rng: np.random.Generator
) -> tuple[float, float, float]:
    joined = selected[["pert_id", "average_precision"]].merge(
        baseline[["pert_id", "average_precision"]],
        on="pert_id",
        suffixes=("_selected", "_baseline"),
        validate="one_to_one",
    )
    difference = (
        joined["average_precision_selected"] - joined["average_precision_baseline"]
    ).to_numpy()
    draws = np.empty(BOOTSTRAPS, dtype=float)
    for draw in range(BOOTSTRAPS):
        draws[draw] = rng.choice(difference, size=len(difference), replace=True).mean()
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(difference.mean()), float(low), float(high)


def main() -> None:
    set_seed(SEED)
    V2_OUT.mkdir(parents=True, exist_ok=True)
    phase1, genes, indices = load_phase1()
    transforms = fit_frozen_transforms(phase1, genes, indices)
    filzen = load_filzen(genes)

    print("[1/3] Evaluating task-matched consensus retrieval", flush=True)
    modz_rows = []
    modz_queries = []
    for cohort_name, frame, cohort_indices in [
        ("phase1_test", phase1, indices["test"]),
    ]:
        raw = frame.iloc[cohort_indices][genes].to_numpy(dtype=np.float32)
        standardized = (raw - transforms["mean"]) / transforms["sd"]
        compounds = frame.iloc[cohort_indices]["pert_id"].astype(str).to_numpy()
        plates = frame.iloc[cohort_indices]["rna_plate"].astype(str).to_numpy()
        for method, function in [
            ("raw_mean_consensus", mean_query_to_compound),
            ("raw_modz_consensus", modz_query_to_compound),
        ]:
            metrics, queries = function(standardized, compounds, plates)
            modz_rows.append({"cohort": cohort_name, "method": method, **metrics})
            modz_queries.append(queries.assign(cohort=cohort_name, method=method))

    external = pd.read_csv(EXTERNAL_SIGNATURES, low_memory=False)
    phase1_ids = set(phase1["pert_id"].astype(str))
    counts = external.groupby("pert_id")["rna_plate"].nunique()
    primary_ids = set(
        counts[(counts >= 6) & ~counts.index.astype(str).isin(phase1_ids)].index.astype(str)
    )
    external = external[external["pert_id"].astype(str).isin(primary_ids)].reset_index(drop=True)
    if len(primary_ids) != 81 or len(external) != 569:
        raise RuntimeError("fixed MCF7 external cohort changed")
    external_raw = external[genes].to_numpy(dtype=np.float32)
    external_standardized = (external_raw - transforms["mean"]) / transforms["sd"]
    external_compounds = external["pert_id"].astype(str).to_numpy()
    external_plates = external["rna_plate"].astype(str).to_numpy()
    for method, function in [
        ("raw_mean_consensus", mean_query_to_compound),
        ("raw_modz_consensus", modz_query_to_compound),
    ]:
        metrics, queries = function(external_standardized, external_compounds, external_plates)
        modz_rows.append({"cohort": "phase2_mcf7", "method": method, **metrics})
        modz_queries.append(queries.assign(cohort="phase2_mcf7", method=method))
    pd.DataFrame(modz_rows).to_csv(V2_OUT / "modz_consensus_metrics.csv", index=False)
    pd.concat(modz_queries, ignore_index=True).to_csv(
        V2_OUT / "modz_consensus_query_results.csv", index=False
    )

    print("[2/3] Building frozen representations and annotation table", flush=True)
    representations = make_representations(external_raw, transforms, filzen)
    pathways, pathway_mean, pathway_sd = phase1_pathway_scaling(phase1, indices)
    representations["ssgsea_hallmark"] = score_hallmark_ssgsea(
        external, genes, pathways, pathway_mean, pathway_sd
    )
    names = None
    compound_representations = {}
    for method, values in representations.items():
        local_names, collapsed = collapse_compounds(values, external_compounds)
        if names is None:
            names = local_names
        elif not np.array_equal(names, local_names):
            raise RuntimeError("compound collapse order changed")
        compound_representations[method] = collapsed
    assert names is not None
    annotation = load_repurposing_annotations()
    annotation = pd.DataFrame({"pert_id": names}).merge(
        annotation[["pert_id", "pert_iname", "moa", "target"]],
        on="pert_id",
        how="left",
        validate="one_to_one",
    )
    annotation.to_csv(V2_OUT / "primary_external_annotations.csv", index=False)

    print("[3/3] Opening locked MoA and target retrieval endpoints", flush=True)
    metric_rows = []
    query_tables: dict[tuple[str, str], pd.DataFrame] = {}
    for label in ["moa", "target"]:
        annotations = token_sets(annotation[label])
        for method, values in compound_representations.items():
            similarity = normalize_rows(values) @ normalize_rows(values).T
            metrics, queries = biological_query_results(similarity, names, annotations)
            metric_rows.append({"label": label, "method": method, **metrics})
            query_tables[(label, method)] = queries
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(V2_OUT / "biological_retrieval_metrics.csv", index=False)
    pd.concat(
        [frame.assign(label=label, method=method) for (label, method), frame in query_tables.items()],
        ignore_index=True,
    ).to_csv(V2_OUT / "biological_retrieval_query_results.csv", index=False)

    bootstrap_rng = np.random.default_rng(SEED + 1)
    moa_lda = query_tables[("moa", "shrinkage_lda")]
    comparisons = []
    for baseline in [
        "raw_cosine", "ssgsea_hallmark", "replicate_contrastive",
        "filzen_continuous", "filzen_binary",
    ]:
        difference, low, high = paired_query_bootstrap(
            moa_lda, query_tables[("moa", baseline)], bootstrap_rng
        )
        comparisons.append(
            {
                "label": "moa",
                "selected_method": "shrinkage_lda",
                "baseline": baseline,
                "map_difference": difference,
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "bootstrap_draws": BOOTSTRAPS,
            }
        )
    pd.DataFrame(comparisons).to_csv(
        V2_OUT / "biological_retrieval_comparisons.csv", index=False
    )

    real_annotations = token_sets(annotation["moa"])
    lda_similarity = compound_representations["shrinkage_lda"] @ compound_representations["shrinkage_lda"].T
    observed_moa = float(
        metrics_frame[
            (metrics_frame["label"] == "moa")
            & (metrics_frame["method"] == "shrinkage_lda")
        ]["mean_average_precision"].iloc[0]
    )
    permutation_rng = np.random.default_rng(SEED + 2)
    null = np.empty(PERMUTATIONS, dtype=float)
    for iteration in range(PERMUTATIONS):
        order = permutation_rng.permutation(len(real_annotations))
        permuted = [real_annotations[index] for index in order]
        null[iteration] = biological_query_results(
            lda_similarity, names, permuted
        )[0]["mean_average_precision"]
    pd.DataFrame(
        {"permutation": np.arange(1, PERMUTATIONS + 1), "mean_average_precision": null}
    ).to_csv(V2_OUT / "moa_permutation_null.csv", index=False)

    raw_comparison = next(row for row in comparisons if row["baseline"] == "raw_cosine")
    pvalue = float((1 + np.sum(null >= observed_moa)) / (PERMUTATIONS + 1))
    passed = raw_comparison["bootstrap_95_low"] > 0 and pvalue < 0.05
    result = {
        "cohort": "GSE70138 MCF7 10 micromolar 24 hours; 81 Phase-I-unseen compounds with >=6 plates",
        "moa_annotated_compounds": int(annotation["moa"].notna().sum()),
        "target_annotated_compounds": int(annotation["target"].notna().sum()),
        "shrinkage_lda_moa_map": observed_moa,
        "shrinkage_lda_minus_raw_map": raw_comparison["map_difference"],
        "paired_bootstrap_95_interval": [
            raw_comparison["bootstrap_95_low"], raw_comparison["bootstrap_95_high"]
        ],
        "permutation_null_mean": float(null.mean()),
        "permutation_null_95_interval": [
            float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))
        ],
        "permutation_pvalue_one_sided": pvalue,
        "primary_biological_endpoint": "PASS" if passed else "FAIL",
    }
    write_json(V2_OUT / "primary_biological_result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
