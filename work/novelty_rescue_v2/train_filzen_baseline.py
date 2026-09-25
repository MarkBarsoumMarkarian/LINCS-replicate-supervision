#!/usr/bin/env python3
"""Train and freeze the direct Filzen-style perturbation-barcode baseline."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    FilzenBarcode,
    SEED,
    V2_OUT,
    load_phase1,
    retrieval_metrics,
    set_seed,
    write_json,
)


MAX_EPOCHS = 80
PATIENCE = 14
STEPS_PER_EPOCH = 16
POSITIVE_PAIRS = 24
NEGATIVE_PAIRS = 48
MARGIN_SQUARED = 5.0


def sample_pairs(
    class_indices: dict[str, np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes = np.array(sorted(class_indices))
    positive_left = []
    positive_right = []
    for _ in range(POSITIVE_PAIRS):
        label = str(rng.choice(classes))
        left, right = rng.choice(class_indices[label], size=2, replace=False)
        positive_left.append(left)
        positive_right.append(right)
    negative_left = []
    negative_right = []
    for _ in range(NEGATIVE_PAIRS):
        label_left, label_right = rng.choice(classes, size=2, replace=False)
        negative_left.append(rng.choice(class_indices[str(label_left)]))
        negative_right.append(rng.choice(class_indices[str(label_right)]))
    left = np.array(positive_left + negative_left, dtype=int)
    right = np.array(positive_right + negative_right, dtype=int)
    target = np.array([1] * POSITIVE_PAIRS + [0] * NEGATIVE_PAIRS, dtype=np.float32)
    order = rng.permutation(len(left))
    return left[order], right[order], target[order]


def embed(model: FilzenBarcode, values: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(values.astype(np.float32)), noisy=False).numpy()


def main() -> None:
    V2_OUT.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    phase1, genes, indices = load_phase1()
    raw = phase1[genes].to_numpy(dtype=np.float32)
    train_mean = raw[indices["train"]].mean(axis=0)
    train_sd = raw[indices["train"]].std(axis=0, ddof=1)
    train_sd = np.where(train_sd > 1e-8, train_sd, 1.0)
    standardized = (raw - train_mean) / train_sd
    train_compounds = phase1.iloc[indices["train"]]["pert_id"].astype(str).to_numpy()
    class_indices = {
        compound: np.flatnonzero(train_compounds == compound)
        for compound in np.unique(train_compounds)
    }
    if min(map(len, class_indices.values())) < 2:
        raise RuntimeError("a training compound lacks an independent positive pair")

    model = FilzenBarcode(len(genes))
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(SEED)
    validation = indices["validation"]
    validation_compounds = phase1.iloc[validation]["pert_id"].astype(str).to_numpy()
    validation_plates = phase1.iloc[validation]["rna_plate"].astype(str).to_numpy()
    history = []
    best_map = -np.inf
    best_epoch = 0
    best_state = None
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for _ in range(STEPS_PER_EPOCH):
            left, right, target = sample_pairs(class_indices, rng)
            global_left = indices["train"][left]
            global_right = indices["train"][right]
            x_left = torch.from_numpy(standardized[global_left].astype(np.float32))
            x_right = torch.from_numpy(standardized[global_right].astype(np.float32))
            y = torch.from_numpy(target)
            optimizer.zero_grad(set_to_none=True)
            z_left = model(x_left, noisy=True)
            z_right = model(x_right, noisy=True)
            squared = ((z_left - z_right) ** 2).sum(dim=1)
            pair_loss = y * squared + (1.0 - y) * torch.relu(MARGIN_SQUARED - squared)
            l1 = sum(parameter.abs().sum() for parameter in model.parameters()) * 1e-6
            loss = pair_loss.mean() + l1
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        validation_embedding = embed(model, standardized[validation])
        metrics, _ = retrieval_metrics(
            validation_embedding, validation_compounds, validation_plates
        )
        epoch_map = metrics["mean_average_precision"]
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "validation_map_continuous": epoch_map,
            }
        )
        improved = epoch_map > best_map + 1e-7
        if improved:
            best_map = epoch_map
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch:03d} loss={np.mean(losses):.5f} "
            f"validation_mAP={epoch_map:.5f} best={'yes' if improved else 'no'}",
            flush=True,
        )
        if stale >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("Filzen-style model produced no validation result")
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(V2_OUT / "filzen_training_history.csv", index=False)
    checkpoint = {
        "state_dict": model.state_dict(),
        "gene_columns": genes,
        "train_mean": train_mean,
        "train_sd": train_sd,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_map": best_map,
        "architecture": "978-400-100 noisy sigmoid; dropout 0.5; margin squared 5",
    }
    torch.save(checkpoint, V2_OUT / "filzen_barcode_checkpoint.pt")

    rows = []
    query_tables = []
    for split_name in ["validation", "test"]:
        split_indices = indices[split_name]
        continuous = embed(model, standardized[split_indices])
        representations = {
            "filzen_continuous": continuous,
            "filzen_binary": (continuous >= 0.5).astype(np.float32),
        }
        compounds = phase1.iloc[split_indices]["pert_id"].astype(str).to_numpy()
        plates = phase1.iloc[split_indices]["rna_plate"].astype(str).to_numpy()
        for method, representation in representations.items():
            metrics, queries = retrieval_metrics(representation, compounds, plates)
            rows.append({"split": split_name, "method": method, **metrics})
            query_tables.append(queries.assign(split=split_name, method=method))
    pd.DataFrame(rows).to_csv(V2_OUT / "filzen_phase1_metrics.csv", index=False)
    pd.concat(query_tables, ignore_index=True).to_csv(
        V2_OUT / "filzen_phase1_query_results.csv", index=False
    )
    write_json(
        V2_OUT / "filzen_training_record.json",
        {
            "direct_prior": "Filzen et al. 2017; doi:10.1371/journal.pcbi.1005335",
            "status": "faithful modern reimplementation, not exact reproduction",
            "selection_split": "Phase I validation compounds",
            "selection_metric": "cross-plate perturbation-identity mAP",
            "best_epoch": best_epoch,
            "best_validation_map": best_map,
            "test_was_not_used_for_selection": True,
            "pair_ratio": "1 positive : 2 negative",
            "steps_per_epoch": STEPS_PER_EPOCH,
            "maximum_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
        },
    )
    print(json.dumps({"best_epoch": best_epoch, "best_validation_map": best_map}, indent=2))


if __name__ == "__main__":
    main()
