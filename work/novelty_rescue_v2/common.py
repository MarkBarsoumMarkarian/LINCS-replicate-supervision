#!/usr/bin/env python3
"""Shared, deterministic utilities for the Version 2 benchmark."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
V1_OUT = ROOT / "outputs/replicate_metric_learning"
V2_OUT = ROOT / "outputs/novelty_rescue_v2"
PHASE1_SIGNATURES = (
    ROOT / "work/v2_treatment_relative/derived/plate_relative_signatures.csv.gz"
)
PHASE1_SPLIT = V1_OUT / "compound_split.csv"
PHASE1_CHECKPOINT = V1_OUT / "contrastive_encoder.pt"
PHASE1_MODERATION = (
    ROOT / "work/v2_treatment_relative/derived/limma/limma_gene_moderation.csv"
)
PHASE1_SSGSEA = ROOT / "work/v2_treatment_relative/derived/ssgsea/all_scores.csv"
HALLMARK_GMT = (
    ROOT / "work/v2_treatment_relative/gene_sets/h.all.v2026.1.Hs.entrez.gmt"
)
GSE70138_GCTX = (
    ROOT
    / "work/replicate_metric_learning/external_gse70138/raw/"
    "GSE70138_Broad_LINCS_Level2_GEX_n345976x978_2017-03-06.gctx"
)
GSE70138_META = (
    ROOT
    / "work/replicate_metric_learning/external_gse70138/raw/"
    "GSE70138_Broad_LINCS_inst_info_2017-03-06.txt.gz"
)
REPURPOSING_DRUGS = ROOT / "work/dataset_audit/raw/repurposing_drugs_20200324.txt"
REPURPOSING_SAMPLES = ROOT / "work/dataset_audit/raw/repurposing_samples_20200324.txt"

SEED = 20_260_926
HIDDEN = 128
EMBEDDING = 32
DROPOUT = 0.10


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def cosine_similarity(values: np.ndarray) -> np.ndarray:
    normalized = normalize_rows(values.astype(np.float64, copy=False))
    return normalized @ normalized.T


class OriginalEncoder(nn.Module):
    """Architecture used by the frozen Version 1 checkpoint."""

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


class FilzenBarcode(nn.Module):
    """Modern PyTorch reconstruction of the Filzen et al. barcode network."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.first = nn.Linear(n_features, 400)
        self.second = nn.Linear(400, 100)

    def forward(self, values: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        first = self.first(values)
        if noisy:
            first = first + torch.randn_like(first) * 0.5
        first = torch.sigmoid(first)
        if noisy:
            first = nn.functional.dropout(first, p=0.5, training=True)
        second = self.second(first)
        if noisy:
            second = second + torch.randn_like(second) * 0.5
        second = torch.sigmoid(second)
        if noisy:
            second = nn.functional.dropout(second, p=0.5, training=True)
        return second


def neural_embedding(model: nn.Module, values: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(values.astype(np.float32, copy=False))
        return model(tensor).cpu().numpy()


def load_phase1() -> tuple[pd.DataFrame, list[str], dict[str, np.ndarray]]:
    frame = pd.read_csv(PHASE1_SIGNATURES, low_memory=False)
    split = pd.read_csv(PHASE1_SPLIT)[["pert_id", "split"]]
    frame = frame.merge(split, on="pert_id", how="left", validate="many_to_one")
    if frame["split"].isna().any():
        raise RuntimeError("Phase I profiles are missing split assignments")
    genes = [column for column in frame if column.startswith("g_")]
    indices = {
        name: np.flatnonzero(frame["split"].to_numpy() == name)
        for name in ["train", "validation", "test"]
    }
    return frame, genes, indices


def fit_frozen_transforms(
    phase1: pd.DataFrame, genes: list[str], indices: dict[str, np.ndarray]
) -> dict[str, object]:
    checkpoint = torch.load(PHASE1_CHECKPOINT, map_location="cpu", weights_only=False)
    if list(checkpoint["gene_columns"]) != genes:
        raise RuntimeError("Version 1 checkpoint genes do not match Phase I signatures")
    raw = phase1[genes].to_numpy(dtype=np.float32)
    train = indices["train"]
    mean = np.asarray(checkpoint["train_mean"], dtype=np.float32)
    sd = np.asarray(checkpoint["train_sd"], dtype=np.float32)
    calculated_mean = raw[train].mean(axis=0)
    calculated_sd = raw[train].std(axis=0, ddof=1)
    calculated_sd = np.where(calculated_sd > 1e-8, calculated_sd, 1.0)
    if not np.allclose(mean, calculated_mean, rtol=1e-5, atol=1e-6):
        raise RuntimeError("frozen training mean changed")
    if not np.allclose(sd, calculated_sd, rtol=1e-5, atol=1e-6):
        raise RuntimeError("frozen training SD changed")
    standardized = (raw - mean) / sd

    names = sorted(phase1.iloc[train]["pert_id"].astype(str).unique())
    label_map = {name: position for position, name in enumerate(names)}
    labels = np.array(
        [label_map[name] for name in phase1.iloc[train]["pert_id"].astype(str)],
        dtype=int,
    )
    pca = PCA(n_components=64, random_state=20_260_924).fit(standardized[train])
    lda = LinearDiscriminantAnalysis(
        solver="eigen", shrinkage="auto", n_components=EMBEDDING
    ).fit(standardized[train], labels)
    original = OriginalEncoder(len(genes))
    original.load_state_dict(checkpoint["state_dict"])
    original.eval()

    moderation = pd.read_csv(PHASE1_MODERATION)
    if moderation["gene_id"].astype(str).tolist() != [gene[2:] for gene in genes]:
        raise RuntimeError("limma weights do not align with Phase I genes")
    precision = moderation["precision_weight"].to_numpy(dtype=float)
    precision = precision / precision.mean()
    return {
        "raw": raw,
        "standardized": standardized,
        "mean": mean,
        "sd": sd,
        "pca": pca,
        "lda": lda,
        "original": original,
        "precision": precision,
        "train_labels": labels,
    }


def load_filzen(genes: list[str]) -> FilzenBarcode:
    checkpoint = torch.load(
        V2_OUT / "filzen_barcode_checkpoint.pt", map_location="cpu", weights_only=False
    )
    if list(checkpoint["gene_columns"]) != genes:
        raise RuntimeError("Filzen checkpoint genes do not match")
    model = FilzenBarcode(len(genes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def make_representations(
    raw: np.ndarray,
    transforms: dict[str, object],
    filzen: FilzenBarcode | None = None,
) -> dict[str, np.ndarray]:
    standardized = (raw - transforms["mean"]) / transforms["sd"]
    original = transforms["original"]
    representations = {
        "raw_cosine": standardized,
        "pca_64": transforms["pca"].transform(standardized),
        "limma_weighted": (raw - transforms["mean"])
        * np.sqrt(transforms["precision"]),
        "replicate_contrastive": neural_embedding(original, standardized),
        "shrinkage_lda": transforms["lda"].transform(standardized),
    }
    if filzen is not None:
        continuous = neural_embedding(filzen, standardized)
        representations["filzen_continuous"] = continuous
        representations["filzen_binary"] = (continuous >= 0.5).astype(np.float32)
    return representations


def phase1_pathway_scaling(
    phase1: pd.DataFrame, indices: dict[str, np.ndarray]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    scores = pd.read_csv(PHASE1_SSGSEA, low_memory=False)
    metadata = {
        "sample_id", "pert_id", "pert_iname", "rna_plate",
        "n_treatment_wells", "n_vehicle_wells",
    }
    pathways = [column for column in scores if column not in metadata]
    sample = phase1["pert_id"].astype(str) + "|" + phase1["rna_plate"].astype(str)
    aligned = pd.DataFrame({"sample_id": sample}).merge(
        scores[["sample_id", *pathways]], on="sample_id", validate="one_to_one"
    )
    training = aligned.iloc[indices["train"]][pathways].to_numpy(dtype=float)
    mean = training.mean(axis=0)
    sd = training.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return pathways, mean, sd


def score_hallmark_ssgsea(
    signatures: pd.DataFrame,
    genes: list[str],
    pathways: list[str],
    mean: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    import gseapy as gp

    sample_ids = (
        signatures["pert_id"].astype(str) + "|" + signatures["rna_plate"].astype(str)
    ).tolist()
    expression = signatures[genes].T
    expression.index = [gene[2:] for gene in genes]
    expression.columns = sample_ids
    expression.insert(0, "gene_id", expression.index.astype(str))
    result = gp.ssgsea(
        data=expression.reset_index(drop=True),
        gene_sets=str(HALLMARK_GMT),
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
    matrix = long.pivot(index="sample_id", columns="pathway", values="ES")
    matrix = matrix.reindex(index=sample_ids, columns=pathways)
    if matrix.isna().any().any():
        raise RuntimeError("Hallmark ssGSEA output is incomplete")
    return (matrix.to_numpy(dtype=float) - mean) / sd


def retrieval_metrics(
    representation: np.ndarray,
    compounds: np.ndarray,
    plates: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    similarity = cosine_similarity(representation)
    n = len(compounds)
    ap = np.empty(n, dtype=float)
    top1 = np.empty(n, dtype=float)
    hit5 = np.empty(n, dtype=float)
    for query in range(n):
        eligible = (np.arange(n) != query) & (plates != plates[query])
        order = np.flatnonzero(eligible)[
            np.argsort(similarity[query, eligible], kind="stable")[::-1]
        ]
        relevant = compounds[order] == compounds[query]
        ranks = np.flatnonzero(relevant) + 1
        if len(ranks) == 0:
            raise RuntimeError(f"query {query} has no cross-plate positive")
        ap[query] = np.mean(np.arange(1, len(ranks) + 1) / ranks)
        top1[query] = float(relevant[0])
        hit5[query] = float(relevant[:5].any())
    metrics = {
        "mean_average_precision": float(ap.mean()),
        "nearest_neighbor_accuracy": float(top1.mean()),
        "top5_hit_rate": float(hit5.mean()),
        "queries": int(n),
        "compounds": int(len(np.unique(compounds))),
    }
    queries = pd.DataFrame(
        {
            "pert_id": compounds,
            "rna_plate": plates,
            "average_precision": ap,
            "top1_correct": top1,
            "top5_hit": hit5,
        }
    )
    return metrics, queries


def cluster_bootstrap_difference(
    selected: pd.DataFrame,
    baseline: pd.DataFrame,
    seed: int = SEED,
    draws: int = 2_000,
) -> tuple[float, float, float]:
    key = ["pert_id", "rna_plate"]
    if not selected[key].equals(baseline[key]):
        raise RuntimeError("paired query tables are not aligned")
    difference = (
        selected["average_precision"].to_numpy()
        - baseline["average_precision"].to_numpy()
    )
    compounds = selected["pert_id"].to_numpy()
    unique = np.unique(compounds)
    grouped = {compound: difference[compounds == compound] for compound in unique}
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        samples[draw] = np.mean(np.concatenate([grouped[name] for name in chosen]))
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(difference.mean()), float(low), float(high)


def read_gse70138_metadata() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    metadata = pd.read_csv(GSE70138_META, sep="\t", low_memory=False)
    metadata["pert_id"] = metadata["pert_id"].astype(str)
    with h5py.File(GSE70138_GCTX, "r") as handle:
        matrix_ids = decode(handle["0/META/COL/id"][:])
        gene_ids = decode(handle["0/META/ROW/id"][:])
    if metadata["inst_id"].duplicated().any() or len(set(matrix_ids)) != len(matrix_ids):
        raise RuntimeError("GSE70138 instance IDs are not unique")
    if set(metadata["inst_id"].astype(str)) != set(matrix_ids):
        raise RuntimeError("GSE70138 metadata and GCTX instance IDs differ")
    position = {name: index for index, name in enumerate(matrix_ids)}
    metadata["matrix_index"] = metadata["inst_id"].astype(str).map(position)
    phase1, genes, _ = load_phase1()
    expected = [gene[2:] for gene in genes]
    if set(gene_ids) != set(expected):
        raise RuntimeError("GSE70138 genes differ from Phase I")
    gene_position = {name: index for index, name in enumerate(gene_ids)}
    gene_order = np.array([gene_position[name] for name in expected], dtype=int)
    return metadata, expected, gene_order


def build_context_signatures(
    metadata: pd.DataFrame,
    gene_ids: list[str],
    gene_order: np.ndarray,
    cell_line: str,
    dose: float = 10.0,
    time_hours: float = 24.0,
    minimum_plates: int = 3,
) -> pd.DataFrame:
    treatment = metadata[
        (metadata["cell_id"] == cell_line)
        & (metadata["pert_type"] == "trt_cp")
        & np.isclose(metadata["pert_time"], time_hours)
        & np.isclose(metadata["pert_dose"], dose)
        & (metadata["pert_dose_unit"].astype(str).str.lower() == "um")
    ].copy()
    vehicles = metadata[
        (metadata["cell_id"] == cell_line)
        & (metadata["pert_type"] == "ctl_vehicle")
        & np.isclose(metadata["pert_time"], time_hours)
    ].copy()
    treatment = treatment[treatment["det_plate"].isin(set(vehicles["det_plate"]))]
    counts = treatment.groupby("pert_id")["det_plate"].nunique()
    treatment = treatment[treatment["pert_id"].isin(counts[counts >= minimum_plates].index)]
    vehicles = vehicles[vehicles["det_plate"].isin(set(treatment["det_plate"]))]
    selected = pd.concat([treatment, vehicles]).sort_values("matrix_index")
    indices = selected["matrix_index"].to_numpy(dtype=int)
    with h5py.File(GSE70138_GCTX, "r") as handle:
        values = handle["0/DATA/0/matrix"][indices, :].astype(np.float64)
    values = values[:, gene_order]
    if not np.isfinite(values).all() or np.min(values) <= 0:
        raise RuntimeError(f"invalid Level 2 values in {cell_line}")
    local = {matrix_index: index for index, matrix_index in enumerate(indices)}
    log_values = np.log2(values)
    columns = [f"g_{gene}" for gene in gene_ids]
    rows: list[dict[str, object]] = []
    for plate, plate_treatment in treatment.groupby("det_plate", sort=True):
        plate_vehicle = vehicles[vehicles["det_plate"] == plate]
        vehicle_rows = [local[index] for index in plate_vehicle["matrix_index"]]
        center = np.median(log_values[vehicle_rows], axis=0)
        for pert_id, compound in plate_treatment.groupby("pert_id", sort=True):
            compound_rows = [local[index] for index in compound["matrix_index"]]
            signature = log_values[compound_rows].mean(axis=0) - center
            row: dict[str, object] = {
                "pert_id": str(pert_id),
                "pert_iname": str(compound["pert_iname"].iloc[0]),
                "cell_id": cell_line,
                "rna_plate": str(plate),
                "pert_dose": float(dose),
                "pert_time": float(time_hours),
                "n_treatment_wells": int(len(compound)),
                "n_vehicle_wells": int(len(plate_vehicle)),
            }
            row.update(dict(zip(columns, signature.tolist())))
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.duplicated(["pert_id", "rna_plate"]).any():
        raise RuntimeError(f"duplicate compound-by-plate signatures in {cell_line}")
    return result


def modz_consensus(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return values[0]
    ranked = np.apply_along_axis(rankdata, 1, values)
    correlations = np.corrcoef(ranked)
    correlations = np.nan_to_num(correlations, nan=0.0)
    correlations = np.maximum(correlations, 0.01)
    np.fill_diagonal(correlations, 0.0)
    weights = correlations.sum(axis=1)
    weights = weights / weights.sum()
    return weights @ values


def modz_query_to_compound(
    values: np.ndarray, compounds: np.ndarray, plates: np.ndarray
) -> tuple[dict[str, float], pd.DataFrame]:
    values = values.astype(np.float64, copy=False)
    unique = np.array(sorted(np.unique(compounds)))
    reciprocal = np.empty(len(values), dtype=float)
    top1 = np.empty(len(values), dtype=float)
    hit5 = np.empty(len(values), dtype=float)
    for query in range(len(values)):
        candidate_vectors = []
        candidate_names = []
        for compound in unique:
            mask = (compounds == compound) & (plates != plates[query])
            mask &= np.arange(len(values)) != query
            if not mask.any():
                raise RuntimeError("MODZ candidate lacks a cross-plate profile")
            candidate_vectors.append(modz_consensus(values[mask]))
            candidate_names.append(compound)
        candidates = normalize_rows(np.asarray(candidate_vectors))
        query_vector = normalize_rows(values[query : query + 1])[0]
        scores = candidates @ query_vector
        order = np.argsort(scores, kind="stable")[::-1]
        relevant = np.asarray(candidate_names)[order] == compounds[query]
        rank = int(np.flatnonzero(relevant)[0]) + 1
        reciprocal[query] = 1.0 / rank
        top1[query] = float(rank == 1)
        hit5[query] = float(rank <= 5)
    metrics = {
        "mean_reciprocal_rank": float(reciprocal.mean()),
        "top1_accuracy": float(top1.mean()),
        "top5_hit_rate": float(hit5.mean()),
        "queries": int(len(values)),
        "compounds": int(len(unique)),
    }
    queries = pd.DataFrame(
        {
            "pert_id": compounds,
            "rna_plate": plates,
            "reciprocal_rank": reciprocal,
            "top1_correct": top1,
            "top5_hit": hit5,
        }
    )
    return metrics, queries


def load_repurposing_annotations() -> pd.DataFrame:
    drugs = pd.read_csv(REPURPOSING_DRUGS, sep="\t", skiprows=9, dtype=str)
    samples = pd.read_csv(REPURPOSING_SAMPLES, sep="\t", skiprows=9, dtype=str)
    samples["pert_id"] = samples["broad_id"].str[:13]
    sample_names = samples[["pert_id", "pert_iname"]].drop_duplicates()
    merged = sample_names.merge(
        drugs[["pert_iname", "moa", "target"]], on="pert_iname", how="left"
    )

    def collapse(series: Iterable[str]) -> str:
        tokens: set[str] = set()
        for value in series:
            if pd.isna(value):
                continue
            tokens.update(
                token.strip().lower()
                for token in str(value).split("|")
                if token.strip()
            )
        return "|".join(sorted(tokens))

    return (
        merged.groupby("pert_id", as_index=False)
        .agg(pert_iname=("pert_iname", "first"), moa=("moa", collapse), target=("target", collapse))
        .replace({"": np.nan})
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
