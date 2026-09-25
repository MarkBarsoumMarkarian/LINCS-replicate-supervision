#!/usr/bin/env python3
"""Frozen external validation of the Phase-I representation on GSE70138."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import gseapy as gp
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import hypergeom
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "work/replicate_metric_learning/external_gse70138"
RAW = RUN / "raw"
OUT = ROOT / "outputs/replicate_metric_learning/external_gse70138"
FIG = OUT / "figures"

GCTX = RAW / "GSE70138_Broad_LINCS_Level2_GEX_n345976x978_2017-03-06.gctx"
META = RAW / "GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz"
PHASE1_SIG = ROOT / "work/v2_treatment_relative/derived/plate_relative_signatures.csv.gz"
SPLIT = ROOT / "outputs/replicate_metric_learning/compound_split.csv"
CHECKPOINT = ROOT / "outputs/replicate_metric_learning/contrastive_encoder.pt"
MODERATION = ROOT / "work/v2_treatment_relative/derived/limma/limma_gene_moderation.csv"
PHASE1_SSGSEA = ROOT / "work/v2_treatment_relative/derived/ssgsea/all_scores.csv"
GMT = ROOT / "work/v2_treatment_relative/gene_sets/h.all.v2026.1.Hs.entrez.gmt"

SEED = 20_260_925
HIDDEN = 128
EMBEDDING = 32
DROPOUT = 0.10
BOOTSTRAP_DRAWS = 2_000
PERMUTATIONS = 200


def decode(values: np.ndarray) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def cosine_similarity(values: np.ndarray) -> np.ndarray:
    values = normalize_rows(values.astype(np.float64, copy=False))
    return values @ values.T


class Encoder(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, EMBEDDING),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.network(values), dim=1)


def neural_embedding(model: Encoder, values: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(values.astype(np.float32, copy=False))).numpy()


def query_metrics(
    similarity: np.ndarray,
    compound_ids: np.ndarray,
    plates: np.ndarray,
    include_pair_auc: bool = True,
) -> tuple[dict[str, float], pd.DataFrame]:
    n = len(compound_ids)
    ap = np.empty(n, dtype=float)
    top1 = np.empty(n, dtype=float)
    hit5 = np.empty(n, dtype=float)
    for index in range(n):
        eligible = (np.arange(n) != index) & (plates != plates[index])
        order = np.flatnonzero(eligible)[np.argsort(similarity[index, eligible])[::-1]]
        relevant = compound_ids[order] == compound_ids[index]
        ranks = np.flatnonzero(relevant) + 1
        if len(ranks) == 0:
            raise RuntimeError(f"query {index} has no cross-plate positive")
        ap[index] = np.mean(np.arange(1, len(ranks) + 1) / ranks)
        top1[index] = float(relevant[0])
        hit5[index] = float(relevant[:5].any())
    metrics: dict[str, float] = {
        "mean_average_precision": float(ap.mean()),
        "nearest_neighbor_accuracy": float(top1.mean()),
        "top5_hit_rate": float(hit5.mean()),
        "queries": int(n),
    }
    if include_pair_auc:
        upper = np.triu_indices(n, k=1)
        cross_plate = plates[upper[0]] != plates[upper[1]]
        labels = (compound_ids[upper[0]] == compound_ids[upper[1]])[cross_plate]
        metrics["pair_auroc"] = float(
            roc_auc_score(labels, similarity[upper][cross_plate])
        )
        metrics["positive_cross_plate_pairs"] = int(labels.sum())
        metrics["negative_cross_plate_pairs"] = int((~labels).sum())
    query = pd.DataFrame(
        {
            "pert_id": compound_ids,
            "rna_plate": plates,
            "average_precision": ap,
            "top1_correct": top1,
            "top5_hit": hit5,
        }
    )
    return metrics, query


def load_gmt(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        result[fields[0]] = set(fields[2:])
    return result


def eligible_compounds(
    metadata: pd.DataFrame,
    time_hours: float,
    minimum_plates: int,
    excluded: set[str],
) -> set[str]:
    treatment = metadata[
        (metadata["cell_id"] == "MCF7")
        & (metadata["pert_type"] == "trt_cp")
        & (metadata["pert_time"] == time_hours)
        & np.isclose(metadata["pert_dose"], 10.0)
        & (metadata["pert_dose_unit"].str.lower() == "um")
    ]
    vehicles = metadata[
        (metadata["cell_id"] == "MCF7")
        & (metadata["pert_type"] == "ctl_vehicle")
        & (metadata["pert_time"] == time_hours)
    ]
    treatment = treatment[treatment["det_plate"].isin(set(vehicles["det_plate"]))]
    counts = treatment.groupby("pert_id")["det_plate"].nunique()
    return set(counts[(counts >= minimum_plates) & ~counts.index.isin(excluded)].index)


def build_signatures(
    metadata: pd.DataFrame,
    time_hours: float,
    minimum_plates: int,
    gene_ids: list[str],
    gene_order: np.ndarray,
) -> pd.DataFrame:
    treatment = metadata[
        (metadata["cell_id"] == "MCF7")
        & (metadata["pert_type"] == "trt_cp")
        & (metadata["pert_time"] == time_hours)
        & np.isclose(metadata["pert_dose"], 10.0)
        & (metadata["pert_dose_unit"].str.lower() == "um")
    ].copy()
    vehicles = metadata[
        (metadata["cell_id"] == "MCF7")
        & (metadata["pert_type"] == "ctl_vehicle")
        & (metadata["pert_time"] == time_hours)
    ].copy()
    treatment = treatment[treatment["det_plate"].isin(set(vehicles["det_plate"]))]
    counts = treatment.groupby("pert_id")["det_plate"].nunique()
    retained = set(counts[counts >= minimum_plates].index)
    treatment = treatment[treatment["pert_id"].isin(retained)]
    used_plates = set(treatment["det_plate"])
    vehicles = vehicles[vehicles["det_plate"].isin(used_plates)]
    selected = pd.concat([treatment, vehicles]).sort_values("matrix_index")
    indices = selected["matrix_index"].to_numpy(dtype=int)
    with h5py.File(GCTX, "r") as handle:
        values = handle["0/DATA/0/matrix"][indices, :].astype(np.float64)
    values = values[:, gene_order]
    if not np.isfinite(values).all() or np.min(values) <= 0:
        raise RuntimeError("external Level-2 values must be finite and positive")
    location = {matrix_index: local for local, matrix_index in enumerate(indices)}
    log_values = np.log2(values)
    gene_columns = [f"g_{gene}" for gene in gene_ids]
    rows: list[dict[str, object]] = []
    for plate, plate_treatment in treatment.groupby("det_plate", sort=True):
        plate_vehicle = vehicles[vehicles["det_plate"] == plate]
        vehicle_local = [location[index] for index in plate_vehicle["matrix_index"]]
        center = np.median(log_values[vehicle_local], axis=0)
        for pert_id, compound in plate_treatment.groupby("pert_id", sort=True):
            compound_local = [location[index] for index in compound["matrix_index"]]
            signature = log_values[compound_local].mean(axis=0) - center
            row: dict[str, object] = {
                "pert_id": str(pert_id),
                "pert_iname": str(compound["pert_iname"].iloc[0]),
                "rna_plate": str(plate),
                "pert_time": float(time_hours),
                "n_treatment_wells": int(len(compound)),
                "n_vehicle_wells": int(len(plate_vehicle)),
            }
            row.update(dict(zip(gene_columns, signature.tolist())))
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.duplicated(["pert_id", "rna_plate"]).any():
        raise RuntimeError("external compound-by-plate signatures are not unique")
    return result


def ssgsea_scores(signatures: pd.DataFrame, gene_columns: list[str]) -> pd.DataFrame:
    sample_ids = (
        signatures["pert_id"].astype(str) + "|" + signatures["rna_plate"].astype(str)
    ).tolist()
    expression = signatures[gene_columns].T
    expression.index = [column[2:] for column in gene_columns]
    expression.columns = sample_ids
    expression.insert(0, "gene_id", expression.index.astype(str))
    result = gp.ssgsea(
        data=expression.reset_index(drop=True),
        gene_sets=str(GMT),
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
    return long.pivot(index="sample_id", columns="pathway", values="ES").reindex(sample_ids)


def bootstrap_difference(
    selected: pd.DataFrame,
    baseline: pd.DataFrame,
    draws: int,
) -> tuple[float, float, float]:
    if not selected[["pert_id", "rna_plate"]].equals(
        baseline[["pert_id", "rna_plate"]]
    ):
        raise RuntimeError("paired query tables do not align")
    difference = selected["average_precision"].to_numpy() - baseline[
        "average_precision"
    ].to_numpy()
    compounds = selected["pert_id"].to_numpy()
    unique = np.unique(compounds)
    grouped = {name: difference[compounds == name] for name in unique}
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty(draws, dtype=float)
    for draw in range(draws):
        sample = rng.choice(unique, size=len(unique), replace=True)
        bootstrap[draw] = np.mean(np.concatenate([grouped[name] for name in sample]))
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return float(difference.mean()), float(lower), float(upper)


def evaluate_cohort(
    frame: pd.DataFrame,
    representations: dict[str, np.ndarray],
    indices: np.ndarray,
    include_ssgsea: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    compound_ids = frame.iloc[indices]["pert_id"].to_numpy()
    plates = frame.iloc[indices]["rna_plate"].to_numpy()
    similarities = {
        name: cosine_similarity(values[indices]) for name, values in representations.items()
    }
    similarities["contrastive_lda_ensemble"] = (
        similarities["replicate_contrastive"] + similarities["shrinkage_lda"]
    ) / 2
    if include_ssgsea is not None:
        similarities["ssgsea_hallmark"] = cosine_similarity(include_ssgsea[indices])
    rows = []
    queries: dict[str, pd.DataFrame] = {}
    for name, similarity in similarities.items():
        metrics, query = query_metrics(similarity, compound_ids, plates)
        rows.append({"method": name, **metrics})
        queries[name] = query
    return (
        pd.DataFrame(rows).sort_values("mean_average_precision", ascending=False),
        queries,
        similarities,
    )


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("[1/4] Verifying source alignment and building Phase II signatures", flush=True)
    metadata = pd.read_csv(META, sep="\t", low_memory=False)
    with h5py.File(GCTX, "r") as handle:
        gctx_ids = decode(handle["0/META/COL/id"][:])
        gene_ids = decode(handle["0/META/ROW/id"][:])
    if metadata["inst_id"].duplicated().any() or len(set(gctx_ids)) != len(gctx_ids):
        raise RuntimeError("instance identifiers are not unique")
    if set(metadata["inst_id"].astype(str)) != set(gctx_ids):
        raise RuntimeError("GCTX and metadata instance identifiers do not match")
    gctx_position = {inst_id: position for position, inst_id in enumerate(gctx_ids)}
    metadata["matrix_index"] = metadata["inst_id"].astype(str).map(gctx_position)
    if metadata["matrix_index"].isna().any():
        raise RuntimeError("an instance could not be mapped to the GCTX matrix")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    gene_columns = list(checkpoint["gene_columns"])
    expected_gene_ids = [column[2:] for column in gene_columns]
    if len(set(gene_ids)) != len(gene_ids) or set(gene_ids) != set(expected_gene_ids):
        raise RuntimeError("Phase II genes do not exactly match frozen Phase I genes")
    phase2_gene_position = {gene_id: position for position, gene_id in enumerate(gene_ids)}
    gene_order = np.array(
        [phase2_gene_position[gene_id] for gene_id in expected_gene_ids], dtype=int
    )

    phase1 = pd.read_csv(PHASE1_SIG, low_memory=False)
    phase1_compounds = set(phase1["pert_id"])
    split = pd.read_csv(SPLIT)[["pert_id", "split"]]
    phase1 = phase1.merge(split, on="pert_id", how="left", validate="many_to_one")
    train_index = np.flatnonzero(phase1["split"].to_numpy() == "train")
    train_raw = phase1.iloc[train_index][gene_columns].to_numpy(dtype=np.float32)
    train_mean = np.asarray(checkpoint["train_mean"], dtype=np.float32)
    train_sd = np.asarray(checkpoint["train_sd"], dtype=np.float32)
    calculated_mean = train_raw.mean(axis=0)
    calculated_sd = train_raw.std(axis=0, ddof=1)
    calculated_sd = np.where(calculated_sd > 1e-8, calculated_sd, 1.0)
    if not np.allclose(train_mean, calculated_mean, rtol=1e-5, atol=1e-6):
        raise RuntimeError("checkpoint mean does not match frozen training split")
    if not np.allclose(train_sd, calculated_sd, rtol=1e-5, atol=1e-6):
        raise RuntimeError("checkpoint SD does not match frozen training split")

    signatures24 = build_signatures(metadata, 24.0, 3, expected_gene_ids, gene_order)
    signatures3 = build_signatures(metadata, 3.0, 3, expected_gene_ids, gene_order)
    gzip_options = {"method": "gzip", "compresslevel": 6, "mtime": 0}
    signatures24.to_csv(
        OUT / "phase2_mcf7_10um_24h_plate_signatures.csv.gz",
        index=False,
        compression=gzip_options,
    )
    signatures3.to_csv(
        OUT / "phase2_mcf7_10um_3h_plate_signatures.csv.gz",
        index=False,
        compression=gzip_options,
    )

    all24_counts = signatures24.groupby("pert_id")["rna_plate"].nunique()
    primary_ids = set(
        all24_counts[(all24_counts >= 6) & ~all24_counts.index.isin(phase1_compounds)].index
    )
    if len(primary_ids) != 81:
        raise RuntimeError(f"primary eligibility changed: expected 81, found {len(primary_ids)}")

    train_standardized = (train_raw - train_mean) / train_sd
    train_names = sorted(phase1.iloc[train_index]["pert_id"].unique())
    label_map = {name: position for position, name in enumerate(train_names)}
    train_labels = np.array(
        [label_map[name] for name in phase1.iloc[train_index]["pert_id"]], dtype=int
    )
    pca = PCA(n_components=64, random_state=20_260_924).fit(train_standardized)
    lda = LinearDiscriminantAnalysis(
        solver="eigen", shrinkage="auto", n_components=EMBEDDING
    ).fit(train_standardized, train_labels)
    model = Encoder(len(gene_columns))
    model.load_state_dict(checkpoint["state_dict"])

    moderation = pd.read_csv(MODERATION)
    if moderation["gene_id"].astype(str).tolist() != expected_gene_ids:
        raise RuntimeError("limma weights do not align with frozen genes")
    weights = moderation["precision_weight"].to_numpy(dtype=float)
    weights = weights / weights.mean()

    def representations(frame: pd.DataFrame) -> dict[str, np.ndarray]:
        raw = frame[gene_columns].to_numpy(dtype=np.float32)
        standardized = (raw - train_mean) / train_sd
        return {
            "raw_cosine": standardized,
            "pca_64": pca.transform(standardized),
            "limma_weighted": (raw - train_mean) * np.sqrt(weights),
            "replicate_contrastive": neural_embedding(model, standardized),
            "shrinkage_lda": lda.transform(standardized),
        }

    rep24 = representations(signatures24)
    rep3 = representations(signatures3)

    external_pathway = ssgsea_scores(signatures24, gene_columns)
    external_pathway3 = ssgsea_scores(signatures3, gene_columns)
    phase1_pathway = pd.read_csv(PHASE1_SSGSEA, low_memory=False)
    pathway_columns = [
        column
        for column in phase1_pathway
        if column
        not in {
            "sample_id", "pert_id", "pert_iname", "rna_plate",
            "n_treatment_wells", "n_vehicle_wells",
        }
    ]
    phase1_sample = phase1["pert_id"].astype(str) + "|" + phase1["rna_plate"].astype(str)
    phase1_pathway = pd.DataFrame({"sample_id": phase1_sample}).merge(
        phase1_pathway[["sample_id", *pathway_columns]],
        on="sample_id",
        validate="one_to_one",
    )
    pathway_train = phase1_pathway.iloc[train_index][pathway_columns].to_numpy(float)
    pathway_mean = pathway_train.mean(axis=0)
    pathway_sd = pathway_train.std(axis=0, ddof=1)
    pathway_sd = np.where(pathway_sd > 1e-8, pathway_sd, 1.0)
    external_pathway = external_pathway[pathway_columns].to_numpy(float)
    external_pathway = (external_pathway - pathway_mean) / pathway_sd
    external_pathway3 = external_pathway3[pathway_columns].to_numpy(float)
    external_pathway3 = (external_pathway3 - pathway_mean) / pathway_sd

    print("[2/4] Opening the frozen primary external endpoint", flush=True)
    primary_index = np.flatnonzero(signatures24["pert_id"].isin(primary_ids).to_numpy())
    primary_metrics, primary_queries, primary_sims = evaluate_cohort(
        signatures24, rep24, primary_index, external_pathway
    )
    primary_metrics.to_csv(OUT / "primary_external_metrics.csv", index=False)

    baseline_names = ["raw_cosine", "pca_64", "limma_weighted", "ssgsea_hallmark"]
    best_baseline = str(
        primary_metrics[primary_metrics["method"].isin(baseline_names)]
        .sort_values(["mean_average_precision", "method"], ascending=[False, True])
        .iloc[0]["method"]
    )
    selected_name = "contrastive_lda_ensemble"
    selected_query = primary_queries[selected_name]
    baseline_query = primary_queries[best_baseline]
    difference, lower, upper = bootstrap_difference(
        selected_query, baseline_query, BOOTSTRAP_DRAWS
    )
    lda_difference, lda_lower, lda_upper = bootstrap_difference(
        primary_queries["shrinkage_lda"], baseline_query, BOOTSTRAP_DRAWS
    )
    selected_query.assign(
        pert_iname=signatures24.iloc[primary_index]["pert_iname"].to_numpy(),
        baseline_average_precision=baseline_query["average_precision"],
        paired_map_improvement=(
            selected_query["average_precision"] - baseline_query["average_precision"]
        ),
    ).to_csv(OUT / "primary_external_query_results.csv", index=False)

    print("[3/4] Running controls, ablations, and sensitivity analyses", flush=True)
    threshold_rows = []
    for threshold in [3, 4, 5, 6, 7]:
        ids = set(
            all24_counts[
                (all24_counts >= threshold) & ~all24_counts.index.isin(phase1_compounds)
            ].index
        )
        cohort_index = np.flatnonzero(signatures24["pert_id"].isin(ids).to_numpy())
        metrics, _, _ = evaluate_cohort(signatures24, rep24, cohort_index)
        for row in metrics.to_dict("records"):
            threshold_rows.append(
                {
                    "minimum_plates": threshold,
                    "compounds": len(ids),
                    "profiles": len(cohort_index),
                    **row,
                }
            )
    threshold_frame = pd.DataFrame(threshold_rows)
    threshold_frame.to_csv(OUT / "replicate_threshold_sensitivity.csv", index=False)

    ids3 = eligible_compounds(metadata, 3.0, 3, phase1_compounds)
    index3 = np.flatnonzero(signatures3["pert_id"].isin(ids3).to_numpy())
    metrics3, _, _ = evaluate_cohort(
        signatures3, rep3, index3, external_pathway3
    )
    metrics3.insert(1, "compounds", len(ids3))
    metrics3.insert(2, "profiles", len(index3))
    metrics3.to_csv(OUT / "secondary_3h_metrics.csv", index=False)

    seen_ids = set(
        all24_counts[(all24_counts >= 6) & all24_counts.index.isin(phase1_compounds)].index
    )
    seen_index = np.flatnonzero(signatures24["pert_id"].isin(seen_ids).to_numpy())
    seen_metrics, _, _ = evaluate_cohort(signatures24, rep24, seen_index)
    seen_metrics.insert(1, "compounds", len(seen_ids))
    seen_metrics.insert(2, "profiles", len(seen_index))
    seen_metrics.to_csv(OUT / "phase1_overlap_control_metrics.csv", index=False)

    primary_compounds = signatures24.iloc[primary_index]["pert_id"].to_numpy()
    primary_plates = signatures24.iloc[primary_index]["rna_plate"].to_numpy()
    ensemble_similarity = primary_sims[selected_name]
    permutation_rng = np.random.default_rng(SEED + 1)
    permutation_map = np.empty(PERMUTATIONS, dtype=float)
    for iteration in range(PERMUTATIONS):
        order = permutation_rng.permutation(len(primary_index))
        permuted = ensemble_similarity[np.ix_(order, order)]
        perm_metrics, _ = query_metrics(
            permuted, primary_compounds, primary_plates, include_pair_auc=False
        )
        permutation_map[iteration] = perm_metrics["mean_average_precision"]
    pd.DataFrame(
        {"permutation": np.arange(1, PERMUTATIONS + 1), "mean_average_precision": permutation_map}
    ).to_csv(OUT / "permutation_null.csv", index=False)

    print("[4/4] Producing interpretation and consolidated evidence", flush=True)
    neural_importance = np.linalg.norm(
        model.network[0].weight.detach().numpy(), axis=0
    )
    lda_importance = np.linalg.norm(lda.scalings_[:, :EMBEDDING], axis=1)
    neural_rank = pd.Series(neural_importance).rank(ascending=False, method="average")
    lda_rank = pd.Series(lda_importance).rank(ascending=False, method="average")
    importance = pd.DataFrame(
        {
            "gene_id": expected_gene_ids,
            "neural_weight_norm": neural_importance,
            "lda_loading_norm": lda_importance,
            "neural_rank": neural_rank,
            "lda_rank": lda_rank,
            "combined_rank": (neural_rank + lda_rank) / 2,
        }
    ).sort_values("combined_rank")
    importance.to_csv(OUT / "gene_importance.csv", index=False)

    gene_sets = load_gmt(GMT)
    top_genes = set(importance.head(100)["gene_id"])
    universe = set(expected_gene_ids)
    enrichment_rows = []
    for pathway, members in gene_sets.items():
        members = members & universe
        overlap = top_genes & members
        if not members:
            continue
        pvalue = hypergeom.sf(len(overlap) - 1, len(universe), len(members), len(top_genes))
        enrichment_rows.append(
            {
                "pathway": pathway,
                "pathway_genes_in_landmarks": len(members),
                "top100_overlap": len(overlap),
                "overlap_gene_ids": ";".join(sorted(overlap)),
                "hypergeometric_pvalue": float(pvalue),
            }
        )
    enrichment = pd.DataFrame(enrichment_rows).sort_values("hypergeometric_pvalue")
    enrichment["bh_fdr"] = np.minimum.accumulate(
        (enrichment["hypergeometric_pvalue"] * len(enrichment) / np.arange(1, len(enrichment) + 1))
        .to_numpy()[::-1]
    )[::-1]
    enrichment.to_csv(OUT / "top_gene_hallmark_enrichment.csv", index=False)

    compound_results = selected_query.assign(
        baseline_average_precision=baseline_query["average_precision"],
        improvement=selected_query["average_precision"] - baseline_query["average_precision"],
    ).groupby("pert_id", as_index=False).agg(
        ensemble_map=("average_precision", "mean"),
        baseline_map=("baseline_average_precision", "mean"),
        improvement=("improvement", "mean"),
        profiles=("rna_plate", "size"),
    )
    names = signatures24[["pert_id", "pert_iname"]].drop_duplicates("pert_id")
    compound_results = compound_results.merge(names, on="pert_id", validate="one_to_one")
    compound_results.sort_values("improvement", ascending=False).to_csv(
        OUT / "compound_level_improvement.csv", index=False
    )

    method_labels = {
        "raw_cosine": "Raw cosine",
        "pca_64": "PCA (64 components)",
        "limma_weighted": "Limma-weighted cosine",
        "ssgsea_hallmark": "Hallmark ssGSEA",
        "replicate_contrastive": "Contrastive neural",
        "shrinkage_lda": "Shrinkage LDA",
        "contrastive_lda_ensemble": "Frozen neural + LDA ensemble",
    }
    method_order = primary_metrics.sort_values("mean_average_precision").copy()
    method_order["display_method"] = method_order["method"].map(method_labels)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = ["#8296A8" if x != selected_name else "#E08A3E" for x in method_order["method"]]
    ax.barh(method_order["display_method"], method_order["mean_average_precision"], color=colors)
    ax.set_xlabel("Cross-plate mean average precision")
    ax.set_title("Frozen external validation: GSE70138 MCF7, 10 µM, 24 h")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "external_method_performance.png", dpi=220)
    fig.savefig(FIG / "external_method_performance.pdf")
    plt.close(fig)

    plot = threshold_frame[
        threshold_frame["method"].isin([selected_name, "raw_cosine", "shrinkage_lda", "replicate_contrastive"])
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for method, group in plot.groupby("method"):
        ax.plot(
            group["minimum_plates"],
            group["mean_average_precision"],
            marker="o",
            label=method_labels[method],
        )
    ax.set_xlabel("Minimum independent Phase II plates per compound")
    ax.set_ylabel("Mean average precision")
    ax.set_title("External robustness to replicate threshold")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "replicate_threshold_sensitivity.png", dpi=220)
    fig.savefig(FIG / "replicate_threshold_sensitivity.pdf")
    plt.close(fig)

    top = compound_results.nlargest(15, "improvement").sort_values("improvement")
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.barh(top["pert_iname"], top["improvement"], color="#2F7F78")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(
        f"Per-compound mAP improvement over {method_labels[best_baseline]}"
    )
    ax.set_title("Largest external gains among unseen compounds")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "top_compound_improvements.png", dpi=220)
    fig.savefig(FIG / "top_compound_improvements.pdf")
    plt.close(fig)

    selected_row = primary_metrics[primary_metrics["method"] == selected_name].iloc[0]
    baseline_row = primary_metrics[primary_metrics["method"] == best_baseline].iloc[0]
    result = {
        "contract": "EXTERNAL_VALIDATION_CONTRACT.md",
        "primary_cohort": {
            "dataset": "GSE70138",
            "cell_line": "MCF7",
            "dose_micromolar": 10,
            "time_hours": 24,
            "minimum_plates": 6,
            "phase1_unseen_compounds": len(primary_ids),
            "profiles": len(primary_index),
        },
        "selected_method": selected_name,
        "selected_map": float(selected_row["mean_average_precision"]),
        "selected_nearest_neighbor_accuracy": float(selected_row["nearest_neighbor_accuracy"]),
        "selected_top5_hit_rate": float(selected_row["top5_hit_rate"]),
        "selected_pair_auroc": float(selected_row["pair_auroc"]),
        "best_nonlearned_baseline": best_baseline,
        "best_baseline_map": float(baseline_row["mean_average_precision"]),
        "map_improvement": difference,
        "map_improvement_bootstrap_95_interval": [lower, upper],
        "strongest_external_method": "shrinkage_lda",
        "shrinkage_lda_map": float(
            primary_metrics.loc[
                primary_metrics["method"] == "shrinkage_lda", "mean_average_precision"
            ].iloc[0]
        ),
        "shrinkage_lda_improvement_over_best_baseline": lda_difference,
        "shrinkage_lda_improvement_bootstrap_95_interval": [lda_lower, lda_upper],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "permutation_null_map_mean": float(permutation_map.mean()),
        "permutation_null_map_95_interval": [
            float(np.quantile(permutation_map, 0.025)),
            float(np.quantile(permutation_map, 0.975)),
        ],
        "permutation_pvalue_one_sided": float(
            (1 + np.sum(permutation_map >= selected_row["mean_average_precision"]))
            / (PERMUTATIONS + 1)
        ),
        "external_endpoint": "PASS" if lower > 0 else "FAIL",
        "interpretation_boundary": (
            "external perturbation-identity retrieval across LINCS phases and a time-point shift; "
            "not mechanism or clinical equivalence"
        ),
    }
    (OUT / "external_validation_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    hashes = {}
    for path in sorted(OUT.glob("*.csv")) + [OUT / "external_validation_result.json"]:
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
