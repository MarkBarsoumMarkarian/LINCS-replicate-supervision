#!/usr/bin/env python3
"""Train replicate-supervised metrics and open the locked test split once."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
SIGNATURES = ROOT / "work/v2_treatment_relative/derived/plate_relative_signatures.csv.gz"
SPLIT = ROOT / "outputs/replicate_metric_learning/compound_split.csv"
MODERATION = ROOT / "work/v2_treatment_relative/derived/limma/limma_gene_moderation.csv"
SSGSEA = ROOT / "work/v2_treatment_relative/derived/ssgsea/all_scores.csv"
OUT = ROOT / "outputs/replicate_metric_learning"

SEED = 20_260_924
HIDDEN = 128
EMBEDDING = 32
TEMPERATURE = 0.10
DROPOUT = 0.10
MAX_EPOCHS = 120
PATIENCE = 18
COMPOUNDS_PER_BATCH = 16
PROFILES_PER_COMPOUND = 4
BOOTSTRAP_DRAWS = 2_000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


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
        embedded = self.network(values)
        return nn.functional.normalize(embedded, dim=1)


def supervised_contrastive_loss(
    embedding: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    similarity = embedding @ embedding.T / TEMPERATURE
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    logits = similarity.masked_fill(identity, float("-inf"))
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1)
    return -(
        (log_probability.masked_fill(~positive, 0.0)).sum(dim=1) / positive_count
    ).mean()


def balanced_batch(
    class_indices: dict[int, np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    classes = rng.choice(
        np.array(sorted(class_indices)), size=COMPOUNDS_PER_BATCH, replace=False
    )
    selected = [
        rng.choice(class_indices[int(label)], size=PROFILES_PER_COMPOUND, replace=False)
        for label in classes
    ]
    return np.concatenate(selected)


def retrieval_details(
    similarity: np.ndarray, compound_ids: np.ndarray, plates: np.ndarray
) -> tuple[dict[str, float], pd.DataFrame]:
    n = len(compound_ids)
    ap = np.empty(n, dtype=float)
    top1 = np.empty(n, dtype=float)
    hit5 = np.empty(n, dtype=float)
    for index in range(n):
        eligible = (np.arange(n) != index) & (plates != plates[index])
        order = np.flatnonzero(eligible)[
            np.argsort(similarity[index, eligible])[::-1]
        ]
        relevant = compound_ids[order] == compound_ids[index]
        ranks = np.flatnonzero(relevant) + 1
        if len(ranks) == 0:
            raise RuntimeError(f"query {index} has no cross-plate positive")
        ap[index] = np.mean(np.arange(1, len(ranks) + 1) / ranks)
        top1[index] = float(relevant[0])
        hit5[index] = float(relevant[:5].any())

    upper = np.triu_indices(n, k=1)
    cross_plate = plates[upper[0]] != plates[upper[1]]
    pair_scores = similarity[upper][cross_plate]
    pair_labels = (compound_ids[upper[0]] == compound_ids[upper[1]])[cross_plate]
    metrics = {
        "mean_average_precision": float(ap.mean()),
        "nearest_neighbor_accuracy": float(top1.mean()),
        "top5_hit_rate": float(hit5.mean()),
        "pair_auroc": float(roc_auc_score(pair_labels, pair_scores)),
        "queries": int(n),
        "positive_cross_plate_pairs": int(pair_labels.sum()),
        "negative_cross_plate_pairs": int((~pair_labels).sum()),
    }
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


def cosine_similarity(embedding: np.ndarray) -> np.ndarray:
    normalized = normalize_rows(embedding.astype(np.float64, copy=False))
    return normalized @ normalized.T


def neural_embedding(model: Encoder, values: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(values.astype(np.float32, copy=False))
        return model(tensor).cpu().numpy()


def evaluate_embedding(
    embedding: np.ndarray, compound_ids: np.ndarray, plates: np.ndarray
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    similarity = cosine_similarity(embedding)
    metrics, query = retrieval_details(similarity, compound_ids, plates)
    return metrics, query, similarity


def evaluate_similarity(
    similarity: np.ndarray, compound_ids: np.ndarray, plates: np.ndarray
) -> tuple[dict[str, float], pd.DataFrame]:
    return retrieval_details(similarity, compound_ids, plates)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    frame = pd.read_csv(SIGNATURES, low_memory=False)
    split = pd.read_csv(SPLIT)[["pert_id", "split"]]
    frame = frame.merge(split, on="pert_id", how="left", validate="many_to_one")
    if frame["split"].isna().any():
        raise RuntimeError("profiles are missing split assignments")
    frame["sample_id"] = frame["pert_id"] + "|" + frame["rna_plate"]
    gene_columns = [column for column in frame if column.startswith("g_")]
    raw = frame[gene_columns].to_numpy(dtype=np.float32)

    indices = {
        name: np.flatnonzero(frame["split"].to_numpy() == name)
        for name in ["train", "validation", "test"]
    }
    train_mean = raw[indices["train"]].mean(axis=0)
    train_sd = raw[indices["train"]].std(axis=0, ddof=1)
    train_sd = np.where(train_sd > 1e-8, train_sd, 1.0)
    standardized = (raw - train_mean) / train_sd

    label_names = sorted(frame.loc[indices["train"], "pert_id"].unique())
    label_map = {name: position for position, name in enumerate(label_names)}
    train_labels = np.array(
        [label_map[value] for value in frame.loc[indices["train"], "pert_id"]],
        dtype=np.int64,
    )
    class_indices = {
        label: np.flatnonzero(train_labels == label) for label in np.unique(train_labels)
    }
    if min(map(len, class_indices.values())) < PROFILES_PER_COMPOUND:
        raise RuntimeError("a training compound has too few profiles for balanced batches")

    model = Encoder(len(gene_columns))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps_per_epoch = max(12, math.ceil(len(indices["train"]) / 64))
    rng = np.random.default_rng(SEED)
    validation_compounds = frame.loc[indices["validation"], "pert_id"].to_numpy()
    validation_plates = frame.loc[indices["validation"], "rna_plate"].to_numpy()
    best_map = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            local = balanced_batch(class_indices, rng)
            global_indices = indices["train"][local]
            batch_values = torch.from_numpy(
                standardized[global_indices].astype(np.float32, copy=False)
            )
            batch_labels = torch.from_numpy(train_labels[local])
            optimizer.zero_grad(set_to_none=True)
            embedding = model(batch_values)
            loss = supervised_contrastive_loss(embedding, batch_labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        validation_embedding = neural_embedding(
            model, standardized[indices["validation"]]
        )
        validation_metrics, _, _ = evaluate_embedding(
            validation_embedding, validation_compounds, validation_plates
        )
        epoch_map = validation_metrics["mean_average_precision"]
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "validation_mean_average_precision": epoch_map,
                "validation_nearest_neighbor_accuracy": validation_metrics[
                    "nearest_neighbor_accuracy"
                ],
            }
        )
        if epoch_map > best_map + 1e-7:
            best_map = epoch_map
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            print(
                f"epoch={epoch} loss={np.mean(losses):.5f} "
                f"validation_mAP={epoch_map:.5f} best=yes",
                flush=True,
            )
        else:
            stale += 1
            if epoch % 5 == 0:
                print(
                    f"epoch={epoch} loss={np.mean(losses):.5f} "
                    f"validation_mAP={epoch_map:.5f} stale={stale}",
                    flush=True,
                )
        if stale >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("neural model failed to produce a validation result")
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)

    # Fit all baseline and learned transforms without using the test labels.
    pca = PCA(n_components=64, random_state=SEED)
    pca.fit(standardized[indices["train"]])
    lda = LinearDiscriminantAnalysis(
        solver="eigen", shrinkage="auto", n_components=EMBEDDING
    )
    lda.fit(standardized[indices["train"]], train_labels)

    moderation = pd.read_csv(MODERATION)
    expected_genes = [column[2:] for column in gene_columns]
    if moderation["gene_id"].astype(str).tolist() != expected_genes:
        raise RuntimeError("limma weights do not align with gene features")
    limma_weight = moderation["precision_weight"].to_numpy(dtype=float)
    limma_weight = limma_weight / limma_weight.mean()
    limma_features = (raw - train_mean) * np.sqrt(limma_weight)

    ssgsea = pd.read_csv(SSGSEA, low_memory=False)
    pathway_columns = [
        column
        for column in ssgsea
        if column
        not in {
            "sample_id", "pert_id", "pert_iname", "rna_plate",
            "n_treatment_wells", "n_vehicle_wells",
        }
    ]
    ssgsea = frame[["sample_id"]].merge(
        ssgsea[["sample_id", *pathway_columns]],
        on="sample_id",
        validate="one_to_one",
    )
    pathway_raw = ssgsea[pathway_columns].to_numpy(dtype=float)
    pathway_mean = pathway_raw[indices["train"]].mean(axis=0)
    pathway_sd = pathway_raw[indices["train"]].std(axis=0, ddof=1)
    pathway_sd = np.where(pathway_sd > 1e-8, pathway_sd, 1.0)
    pathway_standardized = (pathway_raw - pathway_mean) / pathway_sd

    def representations(split_name: str) -> dict[str, np.ndarray]:
        split_indices = indices[split_name]
        return {
            "raw_cosine": standardized[split_indices],
            "pca_64": pca.transform(standardized[split_indices]),
            "limma_weighted": limma_features[split_indices],
            "ssgsea_hallmark": pathway_standardized[split_indices],
            "replicate_contrastive": neural_embedding(
                model, standardized[split_indices]
            ),
            "shrinkage_lda": lda.transform(standardized[split_indices]),
        }

    validation_representations = representations("validation")
    validation_similarities = {
        name: cosine_similarity(values)
        for name, values in validation_representations.items()
    }
    validation_similarities["contrastive_lda_ensemble"] = (
        validation_similarities["replicate_contrastive"]
        + validation_similarities["shrinkage_lda"]
    ) / 2

    validation_rows = []
    for name, similarity in validation_similarities.items():
        metrics, _ = evaluate_similarity(
            similarity, validation_compounds, validation_plates
        )
        validation_rows.append({"method": name, **metrics})
    validation_frame = pd.DataFrame(validation_rows).sort_values(
        "mean_average_precision", ascending=False
    )
    validation_frame.to_csv(OUT / "validation_metrics.csv", index=False)

    learned_names = [
        "replicate_contrastive", "shrinkage_lda", "contrastive_lda_ensemble"
    ]
    learned_validation = validation_frame[
        validation_frame["method"].isin(learned_names)
    ].sort_values(["mean_average_precision", "method"], ascending=[False, True])
    selected_method = str(learned_validation.iloc[0]["method"])
    selection_record = {
        "selected_method": selected_method,
        "selection_metric": "validation mean average precision",
        "selected_validation_map": float(
            learned_validation.iloc[0]["mean_average_precision"]
        ),
        "neural_best_epoch": best_epoch,
        "neural_best_validation_map": float(best_map),
        "test_opened_during_selection": False,
    }
    (OUT / "model_selection.json").write_text(
        json.dumps(selection_record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection_record, indent=2), flush=True)

    # Single opening of the locked test set after selection is recorded.
    test_compounds = frame.loc[indices["test"], "pert_id"].to_numpy()
    test_plates = frame.loc[indices["test"], "rna_plate"].to_numpy()
    test_representations = representations("test")
    test_similarities = {
        name: cosine_similarity(values) for name, values in test_representations.items()
    }
    test_similarities["contrastive_lda_ensemble"] = (
        test_similarities["replicate_contrastive"]
        + test_similarities["shrinkage_lda"]
    ) / 2

    test_rows = []
    query_tables: dict[str, pd.DataFrame] = {}
    for name, similarity in test_similarities.items():
        metrics, query = evaluate_similarity(similarity, test_compounds, test_plates)
        query_tables[name] = query
        test_rows.append({"method": name, **metrics})
    test_frame = pd.DataFrame(test_rows).sort_values(
        "mean_average_precision", ascending=False
    )
    test_frame.to_csv(OUT / "test_metrics.csv", index=False)

    baseline_names = ["raw_cosine", "pca_64", "limma_weighted", "ssgsea_hallmark"]
    baseline_frame = test_frame[test_frame["method"].isin(baseline_names)].sort_values(
        ["mean_average_precision", "method"], ascending=[False, True]
    )
    best_baseline = str(baseline_frame.iloc[0]["method"])
    selected_query = query_tables[selected_method]
    baseline_query = query_tables[best_baseline]
    if not selected_query[["pert_id", "rna_plate"]].equals(
        baseline_query[["pert_id", "rna_plate"]]
    ):
        raise RuntimeError("paired query tables do not align")
    query_difference = (
        selected_query["average_precision"].to_numpy()
        - baseline_query["average_precision"].to_numpy()
    )
    query_compounds = selected_query["pert_id"].to_numpy()
    unique_compounds = np.unique(query_compounds)
    bootstrap_rng = np.random.default_rng(SEED + 1)
    bootstrap = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    compound_differences = {
        compound: query_difference[query_compounds == compound]
        for compound in unique_compounds
    }
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = bootstrap_rng.choice(
            unique_compounds, size=len(unique_compounds), replace=True
        )
        bootstrap[draw] = np.mean(
            np.concatenate([compound_differences[compound] for compound in sampled])
        )
    observed_difference = float(query_difference.mean())
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    benchmark_success = bool(lower > 0)
    final_result = {
        **selection_record,
        "test_opened_once_after_selection": True,
        "best_nonlearned_baseline": best_baseline,
        "selected_test_map": float(
            test_frame.loc[
                test_frame["method"] == selected_method,
                "mean_average_precision",
            ].iloc[0]
        ),
        "best_baseline_test_map": float(
            baseline_frame.iloc[0]["mean_average_precision"]
        ),
        "map_improvement": observed_difference,
        "map_improvement_bootstrap_95_interval": [float(lower), float(upper)],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "benchmark_endpoint": "PASS" if benchmark_success else "FAIL",
    }
    (OUT / "benchmark_result.json").write_text(
        json.dumps(final_result, indent=2) + "\n", encoding="utf-8"
    )
    selected_query.assign(
        baseline_average_precision=baseline_query["average_precision"],
        paired_map_improvement=query_difference,
    ).to_csv(OUT / "test_query_results.csv", index=False)

    checkpoint = {
        "state_dict": model.state_dict(),
        "gene_columns": gene_columns,
        "train_mean": train_mean,
        "train_sd": train_sd,
        "seed": SEED,
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, OUT / "contrastive_encoder.pt")
    hashes = {}
    for path in [
        OUT / "compound_split.csv",
        OUT / "model_selection.json",
        OUT / "test_metrics.csv",
        OUT / "benchmark_result.json",
        OUT / "contrastive_encoder.pt",
    ]:
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "artifact_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )

    print("\nValidation metrics:\n", validation_frame.to_string(index=False), flush=True)
    print("\nTest metrics:\n", test_frame.to_string(index=False), flush=True)
    print("\nFinal result:\n", json.dumps(final_result, indent=2), flush=True)


if __name__ == "__main__":
    main()
