#!/usr/bin/env python3
"""Training-only scaling analysis for linear and neural replicate supervision."""

from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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


COMPOUND_COUNTS = [20, 40, 60, 80, 100]
PLATE_COUNTS = [3, 6, 9, 12]
MAX_EPOCHS = 25
PATIENCE = 5
STEPS_PER_EPOCH = 6


def hash_order(value: str) -> str:
    return hashlib.sha256(f"20260926|{value}".encode()).hexdigest()


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


def balanced_indices(
    class_indices: dict[int, np.ndarray], profiles_per_class: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    chosen = rng.choice(np.array(sorted(class_indices)), size=min(12, len(class_indices)), replace=False)
    local = []
    labels = []
    for label in chosen:
        selected = rng.choice(class_indices[int(label)], size=profiles_per_class, replace=False)
        local.extend(selected.tolist())
        labels.extend([int(label)] * profiles_per_class)
    return np.asarray(local, dtype=int), np.asarray(labels, dtype=np.int64)


def evaluate(
    values: np.ndarray, compounds: np.ndarray, plates: np.ndarray
) -> float:
    return retrieval_metrics(values, compounds, plates)[0]["mean_average_precision"]


def train_original(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    validation_values: np.ndarray,
    validation_compounds: np.ndarray,
    validation_plates: np.ndarray,
    seed: int,
) -> tuple[float, int]:
    set_seed(seed)
    model = OriginalEncoder(train_values.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    class_indices = {
        int(label): np.flatnonzero(train_labels == label) for label in np.unique(train_labels)
    }
    profiles_per_class = min(4, min(map(len, class_indices.values())))
    rng = np.random.default_rng(seed)
    best_map = -np.inf
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for _ in range(STEPS_PER_EPOCH):
            local, labels = balanced_indices(class_indices, profiles_per_class, rng)
            x = torch.from_numpy(train_values[local].astype(np.float32))
            y = torch.from_numpy(labels)
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_contrastive_loss(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            embedded = model(torch.from_numpy(validation_values.astype(np.float32))).numpy()
        score = evaluate(embedded, validation_compounds, validation_plates)
        if score > best_map + 1e-7:
            best_map = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("contrastive scaling fit failed")
    return float(best_map), best_epoch


def sample_filzen_pairs(
    class_indices: dict[int, np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.array(sorted(class_indices))
    left = []
    right = []
    target = []
    for _ in range(12):
        label = int(rng.choice(classes))
        pair = rng.choice(class_indices[label], size=2, replace=False)
        left.append(pair[0]); right.append(pair[1]); target.append(1.0)
    for _ in range(24):
        label_left, label_right = rng.choice(classes, size=2, replace=False)
        left.append(rng.choice(class_indices[int(label_left)]))
        right.append(rng.choice(class_indices[int(label_right)]))
        target.append(0.0)
    order = rng.permutation(len(left))
    return np.asarray(left)[order], np.asarray(right)[order], np.asarray(target, dtype=np.float32)[order]


def train_filzen(
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
        int(label): np.flatnonzero(train_labels == label) for label in np.unique(train_labels)
    }
    rng = np.random.default_rng(seed)
    best_map = -np.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for _ in range(STEPS_PER_EPOCH):
            left, right, target = sample_filzen_pairs(class_indices, rng)
            x_left = torch.from_numpy(train_values[left].astype(np.float32))
            x_right = torch.from_numpy(train_values[right].astype(np.float32))
            y = torch.from_numpy(target)
            optimizer.zero_grad(set_to_none=True)
            z_left = model(x_left, noisy=True)
            z_right = model(x_right, noisy=True)
            squared = ((z_left - z_right) ** 2).sum(dim=1)
            pair_loss = y * squared + (1 - y) * torch.relu(5.0 - squared)
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
        if stale >= PATIENCE:
            break
    return float(best_map), best_epoch


def main() -> None:
    set_seed(SEED)
    V2_OUT.mkdir(parents=True, exist_ok=True)
    phase1, genes, indices = load_phase1()
    raw = phase1[genes].to_numpy(dtype=np.float32)
    training_frame = phase1.iloc[indices["train"]].copy()
    training_frame["global_index"] = indices["train"]
    validation = indices["validation"]
    validation_compounds = phase1.iloc[validation]["pert_id"].astype(str).to_numpy()
    validation_plates = phase1.iloc[validation]["rna_plate"].astype(str).to_numpy()
    ordered_compounds = sorted(
        training_frame["pert_id"].astype(str).unique(), key=hash_order
    )
    rows = []
    total = len(COMPOUND_COUNTS) * len(PLATE_COUNTS)
    run = 0

    for compound_count in COMPOUND_COUNTS:
        selected_compounds = set(ordered_compounds[:compound_count])
        compound_frame = training_frame[
            training_frame["pert_id"].astype(str).isin(selected_compounds)
        ]
        for plate_count in PLATE_COUNTS:
            run += 1
            selected_rows = []
            for compound, group in compound_frame.groupby("pert_id", sort=True):
                ordered = group.assign(
                    _hash=group["rna_plate"].astype(str).map(
                        lambda plate: hash_order(f"{compound}|{plate}")
                    )
                ).sort_values("_hash")
                selected_rows.extend(ordered.head(plate_count)["global_index"].astype(int))
            selected = np.asarray(selected_rows, dtype=int)
            train_raw = raw[selected]
            mean = train_raw.mean(axis=0)
            sd = train_raw.std(axis=0, ddof=1)
            sd = np.where(sd > 1e-8, sd, 1.0)
            train_values = (train_raw - mean) / sd
            validation_values = (raw[validation] - mean) / sd
            selected_names = phase1.iloc[selected]["pert_id"].astype(str).to_numpy()
            label_names = sorted(np.unique(selected_names))
            label_map = {name: position for position, name in enumerate(label_names)}
            labels = np.array([label_map[name] for name in selected_names], dtype=int)

            raw_map = evaluate(
                validation_values, validation_compounds, validation_plates
            )
            lda = LinearDiscriminantAnalysis(
                solver="eigen",
                shrinkage="auto",
                n_components=min(32, compound_count - 1),
            ).fit(train_values, labels)
            lda_map = evaluate(
                lda.transform(validation_values), validation_compounds, validation_plates
            )
            point_seed = SEED + compound_count * 100 + plate_count
            contrastive_map, contrastive_epoch = train_original(
                train_values,
                labels,
                validation_values,
                validation_compounds,
                validation_plates,
                point_seed,
            )
            filzen_map, filzen_epoch = train_filzen(
                train_values,
                labels,
                validation_values,
                validation_compounds,
                validation_plates,
                point_seed + 1,
            )
            rows.extend(
                [
                    {"training_compounds": compound_count, "plates_per_compound": plate_count, "method": "raw_cosine", "validation_map": raw_map, "selected_epoch": np.nan},
                    {"training_compounds": compound_count, "plates_per_compound": plate_count, "method": "shrinkage_lda", "validation_map": lda_map, "selected_epoch": np.nan},
                    {"training_compounds": compound_count, "plates_per_compound": plate_count, "method": "replicate_contrastive", "validation_map": contrastive_map, "selected_epoch": contrastive_epoch},
                    {"training_compounds": compound_count, "plates_per_compound": plate_count, "method": "filzen_continuous", "validation_map": filzen_map, "selected_epoch": filzen_epoch},
                ]
            )
            pd.DataFrame(rows).to_csv(V2_OUT / "scaling_analysis.csv", index=False)
            print(
                f"[{run:02d}/{total}] compounds={compound_count:3d} plates={plate_count:2d} "
                f"mAP raw={raw_map:.3f} LDA={lda_map:.3f} "
                f"contrastive={contrastive_map:.3f} Filzen={filzen_map:.3f}",
                flush=True,
            )

    result_frame = pd.DataFrame(rows)
    summary = {
        "evaluation_split": "unchanged Phase I validation compounds",
        "grid_points": total,
        "compound_counts": COMPOUND_COUNTS,
        "plates_per_compound": PLATE_COUNTS,
        "neural_max_epochs": MAX_EPOCHS,
        "neural_patience": PATIENCE,
        "neural_steps_per_epoch": STEPS_PER_EPOCH,
        "best_method_counts": result_frame.loc[
            result_frame.groupby(["training_compounds", "plates_per_compound"])[
                "validation_map"
            ].idxmax(),
            "method",
        ].value_counts().to_dict(),
        "full_grid_point": result_frame[
            (result_frame["training_compounds"] == 100)
            & (result_frame["plates_per_compound"] == 12)
        ][["method", "validation_map"]].set_index("method")["validation_map"].to_dict(),
    }
    write_json(V2_OUT / "scaling_analysis_result.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
