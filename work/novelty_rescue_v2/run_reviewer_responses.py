#!/usr/bin/env python3
"""Post-freeze sensitivities for neural schedules, comparators, and score ties."""

from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import pandas as pd
import torch

from common import (
    FilzenBarcode,
    OriginalEncoder,
    SEED,
    V2_OUT,
    load_phase1,
    retrieval_metrics,
    set_seed,
    write_json,
)


AUDIT_POINTS = [(20, 3), (20, 6), (20, 9), (20, 12), (100, 12)]
CONTRASTIVE_MAX_EPOCHS = 120
CONTRASTIVE_PATIENCE = 18
FILZEN_MAX_EPOCHS = 80
FILZEN_PATIENCE = 14
FILZEN_STEPS_PER_EPOCH = 16
BOOTSTRAPS = 2_000


def hash_order(value: str) -> str:
    return hashlib.sha256(f"20260926|{value}".encode()).hexdigest()


def evaluate(values: np.ndarray, compounds: np.ndarray, plates: np.ndarray) -> float:
    return float(retrieval_metrics(values, compounds, plates)[0]["mean_average_precision"])


def supervised_contrastive_loss(
    embedding: torch.Tensor, labels: torch.Tensor, temperature: float = 0.10
) -> torch.Tensor:
    similarity = embedding @ embedding.T / temperature
    identity = torch.eye(len(labels), dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    logits = similarity.masked_fill(identity, float("-inf"))
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(
        log_probability.masked_fill(~positive, 0.0).sum(dim=1)
        / positive.sum(dim=1)
    ).mean()


def train_contrastive_main_schedule(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    validation_values: np.ndarray,
    validation_compounds: np.ndarray,
    validation_plates: np.ndarray,
    seed: int,
) -> tuple[float, int, int]:
    set_seed(seed)
    model = OriginalEncoder(train_values.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    class_indices = {
        int(label): np.flatnonzero(train_labels == label)
        for label in np.unique(train_labels)
    }
    profiles_per_class = min(4, min(map(len, class_indices.values())))
    compounds_per_batch = min(16, len(class_indices))
    steps_per_epoch = max(12, math.ceil(len(train_values) / 64))
    rng = np.random.default_rng(seed)
    best_map = -np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    classes = np.asarray(sorted(class_indices))
    for epoch in range(1, CONTRASTIVE_MAX_EPOCHS + 1):
        model.train()
        for _ in range(steps_per_epoch):
            chosen = rng.choice(classes, size=compounds_per_batch, replace=False)
            local = []
            labels = []
            for label in chosen:
                selected = rng.choice(
                    class_indices[int(label)], size=profiles_per_class, replace=False
                )
                local.extend(selected.tolist())
                labels.extend([int(label)] * profiles_per_class)
            x = torch.from_numpy(train_values[np.asarray(local)].astype(np.float32))
            y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_contrastive_loss(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            embedded = model(
                torch.from_numpy(validation_values.astype(np.float32))
            ).numpy()
        score = evaluate(embedded, validation_compounds, validation_plates)
        if score > best_map + 1e-7:
            best_map = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= CONTRASTIVE_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("main-schedule contrastive fit failed")
    return float(best_map), best_epoch, steps_per_epoch


def sample_filzen_pairs(
    class_indices: dict[int, np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.asarray(sorted(class_indices))
    left = []
    right = []
    target = []
    for _ in range(24):
        label = int(rng.choice(classes))
        pair = rng.choice(class_indices[label], size=2, replace=False)
        left.append(pair[0]); right.append(pair[1]); target.append(1.0)
    for _ in range(48):
        label_left, label_right = rng.choice(classes, size=2, replace=False)
        left.append(rng.choice(class_indices[int(label_left)]))
        right.append(rng.choice(class_indices[int(label_right)]))
        target.append(0.0)
    order = rng.permutation(len(left))
    return (
        np.asarray(left)[order],
        np.asarray(right)[order],
        np.asarray(target, dtype=np.float32)[order],
    )


def train_filzen_main_schedule(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    validation_values: np.ndarray,
    validation_compounds: np.ndarray,
    validation_plates: np.ndarray,
    seed: int,
) -> tuple[float, int]:
    set_seed(seed)
    model = FilzenBarcode(train_values.shape[1])
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    class_indices = {
        int(label): np.flatnonzero(train_labels == label)
        for label in np.unique(train_labels)
    }
    rng = np.random.default_rng(seed)
    best_map = -np.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, FILZEN_MAX_EPOCHS + 1):
        model.train()
        for _ in range(FILZEN_STEPS_PER_EPOCH):
            left, right, target = sample_filzen_pairs(class_indices, rng)
            x_left = torch.from_numpy(train_values[left].astype(np.float32))
            x_right = torch.from_numpy(train_values[right].astype(np.float32))
            y = torch.from_numpy(target)
            optimizer.zero_grad(set_to_none=True)
            z_left = model(x_left, noisy=True)
            z_right = model(x_right, noisy=True)
            squared = ((z_left - z_right) ** 2).sum(dim=1)
            pair_loss = y * squared + (1.0 - y) * torch.relu(5.0 - squared)
            l1 = sum(parameter.abs().sum() for parameter in model.parameters()) * 1e-6
            loss = pair_loss.mean() + l1
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            embedded = model(
                torch.from_numpy(validation_values.astype(np.float32)), noisy=False
            ).numpy()
        score = evaluate(embedded, validation_compounds, validation_plates)
        if score > best_map + 1e-7:
            best_map = score
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= FILZEN_PATIENCE:
            break
    return float(best_map), best_epoch


def paired_annotation_comparison(
    label: str, baseline_method: str, baseline_name: str, seed: int
) -> dict[str, object]:
    query = pd.read_csv(V2_OUT / "biological_retrieval_query_results.csv")
    lda = query[(query.label == label) & (query.method == "shrinkage_lda")]
    baseline = query[(query.label == label) & (query.method == baseline_method)]
    joined = lda[["pert_id", "average_precision"]].merge(
        baseline[["pert_id", "average_precision"]],
        on="pert_id",
        suffixes=("_lda", "_baseline"),
        validate="one_to_one",
    )
    difference = (
        joined.average_precision_lda - joined.average_precision_baseline
    ).to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        draws[draw] = rng.choice(difference, size=len(difference), replace=True).mean()
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "comparison": f"shrinkage LDA minus {baseline_name}",
        "status": "post-freeze credibility sensitivity",
        "eligible_queries": int(len(joined)),
        "lda_map": float(lda.average_precision.mean()),
        "baseline_map": float(baseline.average_precision.mean()),
        "map_difference": float(difference.mean()),
        "bootstrap_95_interval": [float(low), float(high)],
        "bootstrap_draws": BOOTSTRAPS,
    }


def target_ssgsea_comparison() -> dict[str, object]:
    return paired_annotation_comparison(
        "target", "ssgsea_hallmark", "Hallmark ssGSEA", SEED + 401
    )


def mechanism_limma_comparison() -> dict[str, object]:
    return paired_annotation_comparison(
        "moa", "limma_weighted", "limma-weighted cosine", SEED + 402
    )


def tie_aware_retrieval_metrics(
    representation: np.ndarray,
    compounds: np.ndarray,
    plates: np.ndarray,
) -> dict[str, float]:
    """Expected retrieval metrics under uniform ordering within exact score ties."""
    normalized = representation / np.maximum(
        np.linalg.norm(representation, axis=1, keepdims=True), 1e-12
    )
    similarity = normalized @ normalized.T
    average_precision = []
    top1_probability = []
    top5_probability = []
    for query in range(len(compounds)):
        eligible = (np.arange(len(compounds)) != query) & (plates != plates[query])
        scores = similarity[query, eligible]
        relevant = (compounds[eligible] == compounds[query]).astype(int)
        order = np.argsort(scores, kind="stable")[::-1]
        scores = scores[order]
        relevant = relevant[order]
        total_relevant = int(relevant.sum())
        preceding_items = 0
        preceding_relevant = 0
        expected_ap_numerator = 0.0
        remaining_top5 = 5
        probability_no_top5_hit = 1.0
        first_group = True
        for score in np.unique(scores)[::-1]:
            group = relevant[scores == score]
            group_size = len(group)
            group_relevant = int(group.sum())
            if first_group:
                top1_probability.append(group_relevant / group_size)
                first_group = False
            if remaining_top5 > 0:
                selected = min(remaining_top5, group_size)
                if group_relevant:
                    if group_size - group_relevant < selected:
                        probability_none = 0.0
                    else:
                        probability_none = math.comb(
                            group_size - group_relevant, selected
                        ) / math.comb(group_size, selected)
                    probability_no_top5_hit *= probability_none
                remaining_top5 -= selected
            for position in range(1, group_size + 1):
                paired_relevant = 0.0
                if group_size > 1:
                    paired_relevant = (
                        (position - 1)
                        * group_relevant
                        * (group_relevant - 1)
                        / (group_size * (group_size - 1))
                    )
                expected_numerator = (
                    preceding_relevant * group_relevant / group_size
                    + group_relevant / group_size
                    + paired_relevant
                )
                expected_ap_numerator += expected_numerator / (
                    preceding_items + position
                )
            preceding_items += group_size
            preceding_relevant += group_relevant
        average_precision.append(expected_ap_numerator / total_relevant)
        top5_probability.append(1.0 - probability_no_top5_hit)
    return {
        "mean_average_precision": float(np.mean(average_precision)),
        "nearest_neighbor_accuracy": float(np.mean(top1_probability)),
        "top5_hit_rate": float(np.mean(top5_probability)),
    }


def filzen_binary_tie_audit(
    phase1: pd.DataFrame,
    genes: list[str],
    indices: dict[str, np.ndarray],
) -> dict[str, object]:
    checkpoint = torch.load(
        V2_OUT / "filzen_barcode_checkpoint.pt", map_location="cpu", weights_only=False
    )
    model = FilzenBarcode(len(genes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    raw = phase1[genes].to_numpy(dtype=np.float32)
    standardized = (raw - checkpoint["train_mean"]) / checkpoint["train_sd"]
    result: dict[str, object] = {
        "status": "post-freeze tie-aware sensitivity",
        "tie_policy": "expected metrics under uniform random ordering within exact cosine-score ties",
    }
    for split in ["validation", "test"]:
        selected = indices[split]
        with torch.no_grad():
            continuous = model(
                torch.from_numpy(standardized[selected].astype(np.float32)),
                noisy=False,
            ).numpy()
        binary = (continuous >= 0.5).astype(np.float32)
        compounds = phase1.iloc[selected].pert_id.astype(str).to_numpy()
        plates = phase1.iloc[selected].rna_plate.astype(str).to_numpy()
        result[split] = {
            "profiles": int(len(selected)),
            "unique_binary_vectors": int(len(np.unique(binary, axis=0))),
            **tie_aware_retrieval_metrics(binary, compounds, plates),
        }
    return result


def main() -> None:
    set_seed(SEED)
    V2_OUT.mkdir(parents=True, exist_ok=True)
    phase1, genes, indices = load_phase1()
    raw = phase1[genes].to_numpy(dtype=np.float32)
    training_frame = phase1.iloc[indices["train"]].copy()
    training_frame["global_index"] = indices["train"]
    validation = indices["validation"]
    validation_compounds = phase1.iloc[validation].pert_id.astype(str).to_numpy()
    validation_plates = phase1.iloc[validation].rna_plate.astype(str).to_numpy()
    ordered_compounds = sorted(
        training_frame.pert_id.astype(str).unique(), key=hash_order
    )
    short_grid = pd.read_csv(V2_OUT / "scaling_analysis.csv")
    rows = []

    for run, (compound_count, plate_count) in enumerate(AUDIT_POINTS, start=1):
        selected_compounds = set(ordered_compounds[:compound_count])
        compound_frame = training_frame[
            training_frame.pert_id.astype(str).isin(selected_compounds)
        ]
        selected_rows = []
        for compound, group in compound_frame.groupby("pert_id", sort=True):
            ordered = group.assign(
                _hash=group.rna_plate.astype(str).map(
                    lambda plate: hash_order(f"{compound}|{plate}")
                )
            ).sort_values("_hash")
            selected_rows.extend(ordered.head(plate_count).global_index.astype(int))
        selected = np.asarray(selected_rows, dtype=int)
        train_raw = raw[selected]
        mean = train_raw.mean(axis=0)
        sd = train_raw.std(axis=0, ddof=1)
        sd = np.where(sd > 1e-8, sd, 1.0)
        train_values = (train_raw - mean) / sd
        validation_values = (raw[validation] - mean) / sd
        selected_names = phase1.iloc[selected].pert_id.astype(str).to_numpy()
        label_names = sorted(np.unique(selected_names))
        label_map = {name: position for position, name in enumerate(label_names)}
        labels = np.asarray([label_map[name] for name in selected_names], dtype=int)
        point_seed = SEED + compound_count * 100 + plate_count

        contrastive_map, contrastive_epoch, contrastive_steps = (
            train_contrastive_main_schedule(
                train_values,
                labels,
                validation_values,
                validation_compounds,
                validation_plates,
                point_seed,
            )
        )
        filzen_map, filzen_epoch = train_filzen_main_schedule(
            train_values,
            labels,
            validation_values,
            validation_compounds,
            validation_plates,
            point_seed + 1,
        )
        point = short_grid[
            (short_grid.training_compounds == compound_count)
            & (short_grid.plates_per_compound == plate_count)
        ].set_index("method")
        lda_map = float(point.loc["shrinkage_lda", "validation_map"])
        for method, short_map, audited_map, epoch, steps, maximum, patience in [
            (
                "replicate_contrastive",
                float(point.loc["replicate_contrastive", "validation_map"]),
                contrastive_map,
                contrastive_epoch,
                contrastive_steps,
                CONTRASTIVE_MAX_EPOCHS,
                CONTRASTIVE_PATIENCE,
            ),
            (
                "filzen_continuous",
                float(point.loc["filzen_continuous", "validation_map"]),
                filzen_map,
                filzen_epoch,
                FILZEN_STEPS_PER_EPOCH,
                FILZEN_MAX_EPOCHS,
                FILZEN_PATIENCE,
            ),
        ]:
            rows.append(
                {
                    "training_compounds": compound_count,
                    "plates_per_compound": plate_count,
                    "method": method,
                    "short_schedule_validation_map": short_map,
                    "main_schedule_validation_map": audited_map,
                    "main_schedule_selected_epoch": epoch,
                    "main_schedule_steps_per_epoch": steps,
                    "main_schedule_max_epochs": maximum,
                    "main_schedule_patience": patience,
                    "shrinkage_lda_validation_map": lda_map,
                    "lda_leads_main_schedule_fit": bool(lda_map > audited_map),
                }
            )
        pd.DataFrame(rows).to_csv(
            V2_OUT / "reviewer_schedule_audit.csv", index=False
        )
        print(
            f"[{run}/{len(AUDIT_POINTS)}] compounds={compound_count} plates={plate_count} "
            f"LDA={lda_map:.4f} contrastive={contrastive_map:.4f} Filzen={filzen_map:.4f}",
            flush=True,
        )

    audit = pd.DataFrame(rows)
    validation_metrics = pd.read_csv(
        V2_OUT.parent / "replicate_metric_learning/validation_metrics.csv"
    ).set_index("method")
    filzen_metrics = pd.read_csv(V2_OUT / "filzen_phase1_metrics.csv")
    filzen_validation = filzen_metrics[
        (filzen_metrics.split == "validation")
        & (filzen_metrics.method == "filzen_continuous")
    ].iloc[0]
    full = audit[
        (audit.training_compounds == 100) & (audit.plates_per_compound == 12)
    ]
    result = {
        "status": "post-freeze credibility sensitivity",
        "audit_points": [[int(a), int(b)] for a, b in AUDIT_POINTS],
        "main_schedule_definition": {
            "contrastive_max_epochs": CONTRASTIVE_MAX_EPOCHS,
            "contrastive_patience": CONTRASTIVE_PATIENCE,
            "contrastive_steps_per_epoch": "max(12, ceiling(training profiles / 64))",
            "contrastive_compounds_per_batch": 16,
            "contrastive_profiles_per_compound": "up to 4; 3 at the 3-plate point",
            "filzen_max_epochs": FILZEN_MAX_EPOCHS,
            "filzen_patience": FILZEN_PATIENCE,
            "filzen_steps_per_epoch": FILZEN_STEPS_PER_EPOCH,
            "filzen_pairs_per_step": "24 positive and 48 negative",
        },
        "lda_leads_all_audited_main_schedule_fits": bool(
            audit.lda_leads_main_schedule_fit.all()
        ),
        "full_100_compound_12_plate_point": {
            row.method: {
                "short_schedule_map": float(row.short_schedule_validation_map),
                "main_schedule_map": float(row.main_schedule_validation_map),
                "shrinkage_lda_map": float(row.shrinkage_lda_validation_map),
            }
            for row in full.itertuples()
        },
        "all_available_plate_reference": {
            "shrinkage_lda_validation_map": float(
                validation_metrics.loc["shrinkage_lda", "mean_average_precision"]
            ),
            "replicate_contrastive_validation_map": float(
                validation_metrics.loc[
                    "replicate_contrastive", "mean_average_precision"
                ]
            ),
            "filzen_continuous_validation_map": float(
                filzen_validation.mean_average_precision
            ),
            "note": "Uses all available training plates and each model's main analysis schedule.",
        },
        "target_comparator": target_ssgsea_comparison(),
        "mechanism_comparator": mechanism_limma_comparison(),
        "filzen_binary_tie_sensitivity": filzen_binary_tie_audit(
            phase1, genes, indices
        ),
    }
    write_json(V2_OUT / "reviewer_response_results.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
