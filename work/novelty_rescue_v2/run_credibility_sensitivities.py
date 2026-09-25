#!/usr/bin/env python3
"""Bounded reviewer-requested sensitivity analyses used to finalize the paper."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf

from common import (
    SEED,
    V2_OUT,
    fit_frozen_transforms,
    load_phase1,
    normalize_rows,
    set_seed,
    write_json,
)


EXTERNAL_SIGNATURES = (
    V2_OUT.parent
    / "replicate_metric_learning/external_gse70138/"
    "phase2_mcf7_10um_24h_plate_signatures.csv.gz"
)
BOOTSTRAPS = 2_000


def query_to_compound_mean(
    values: np.ndarray, compounds: np.ndarray, plates: np.ndarray
) -> dict[str, float]:
    names = np.array(sorted(np.unique(compounds)))
    reciprocal = []
    top1 = []
    top5 = []
    for query in range(len(values)):
        candidates = []
        for compound in names:
            mask = (compounds == compound) & (plates != plates[query])
            if not mask.any():
                raise RuntimeError("candidate lacks a cross-plate profile")
            candidates.append(values[mask].mean(axis=0))
        scores = normalize_rows(np.asarray(candidates)) @ normalize_rows(values[query : query + 1])[0]
        order = np.argsort(scores, kind="stable")[::-1]
        rank = int(np.flatnonzero(names[order] == compounds[query])[0]) + 1
        reciprocal.append(1.0 / rank)
        top1.append(float(rank == 1))
        top5.append(float(rank <= 5))
    return {
        "mean_reciprocal_rank": float(np.mean(reciprocal)),
        "top1_accuracy": float(np.mean(top1)),
        "top5_hit_rate": float(np.mean(top5)),
        "queries": int(len(values)),
        "compounds": int(len(names)),
    }


def paired_bootstrap(selected: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, object]:
    joined = selected[["pert_id", "average_precision"]].merge(
        baseline[["pert_id", "average_precision"]],
        on="pert_id",
        suffixes=("_selected", "_baseline"),
        validate="one_to_one",
    )
    difference = (
        joined.average_precision_selected - joined.average_precision_baseline
    ).to_numpy()
    rng = np.random.default_rng(SEED + 301)
    draws = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        draws[draw] = rng.choice(difference, size=len(difference), replace=True).mean()
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "eligible_queries": int(len(joined)),
        "map_difference": float(difference.mean()),
        "bootstrap_95_interval": [float(low), float(high)],
        "bootstrap_draws": BOOTSTRAPS,
    }


def cross_context_global_compound_bootstrap() -> dict[str, object]:
    queries = pd.read_csv(V2_OUT / "multicontext_query_results.csv")
    contexts = ["A375", "A549", "HA1E", "HT29", "PC3"]
    grouped: dict[str, dict[str, np.ndarray]] = {}
    all_compounds: set[str] = set()
    observed = []
    for context in contexts:
        lda = queries[(queries.cell_line == context) & (queries.method == "shrinkage_lda")]
        baseline = queries[(queries.cell_line == context) & (queries.method == "ssgsea_hallmark")]
        joined = lda[["pert_id", "rna_plate", "average_precision"]].merge(
            baseline[["pert_id", "rna_plate", "average_precision"]],
            on=["pert_id", "rna_plate"],
            suffixes=("_lda", "_baseline"),
            validate="one_to_one",
        )
        joined["difference"] = joined.average_precision_lda - joined.average_precision_baseline
        grouped[context] = {
            str(compound): group.difference.to_numpy()
            for compound, group in joined.groupby("pert_id")
        }
        all_compounds.update(grouped[context])
        observed.append(float(joined.difference.mean()))
    names = np.array(sorted(all_compounds))
    rng = np.random.default_rng(SEED + 302)
    draws = np.empty(BOOTSTRAPS)
    for draw in range(BOOTSTRAPS):
        sampled = rng.choice(names, size=len(names), replace=True)
        context_means = []
        for context in contexts:
            arrays = [grouped[context][name] for name in sampled if name in grouped[context]]
            context_means.append(float(np.concatenate(arrays).mean()))
        draws[draw] = float(np.mean(context_means))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "unique_compounds_across_contexts": int(len(names)),
        "macro_map_difference": float(np.mean(observed)),
        "global_compound_cluster_bootstrap_95_interval": [float(low), float(high)],
        "bootstrap_draws": BOOTSTRAPS,
        "interpretation": "sensitivity preserving a compound as one cluster across cell lines",
    }


def spectrum(values: np.ndarray, compounds: np.ndarray) -> tuple[np.ndarray, float, float]:
    names = np.array(sorted(np.unique(compounds)))
    means = np.vstack([values[compounds == name].mean(axis=0) for name in names])
    residuals = np.vstack(
        [values[compounds == name] - values[compounds == name].mean(axis=0) for name in names]
    )
    estimator = LedoitWolf(assume_centered=True).fit(residuals)
    within = (estimator.covariance_ + estimator.covariance_.T) / 2
    between = np.cov(means, rowvar=False, ddof=1)
    between = (between + between.T) / 2
    eigenvalues = eigh(between, within, eigvals_only=True, check_finite=True)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    positive = eigenvalues[eigenvalues > 1e-12]
    participation = float(positive.sum() ** 2 / np.sum(positive**2))
    return eigenvalues, participation, float(estimator.shrinkage_)


def main() -> None:
    set_seed(SEED)
    phase1, genes, indices = load_phase1()
    transforms = fit_frozen_transforms(phase1, genes, indices)

    # Direct LDA comparison on the same query-to-compound consensus endpoint as MODZ.
    rows = []
    for cohort, frame, selected in [
        ("phase1_test", phase1, indices["test"]),
    ]:
        raw = frame.iloc[selected][genes].to_numpy(dtype=np.float32)
        standardized = (raw - transforms["mean"]) / transforms["sd"]
        lda = transforms["lda"].transform(standardized)
        compounds = frame.iloc[selected].pert_id.astype(str).to_numpy()
        plates = frame.iloc[selected].rna_plate.astype(str).to_numpy()
        rows.append({"cohort": cohort, "method": "shrinkage_lda_mean_consensus", **query_to_compound_mean(lda, compounds, plates)})

    external = pd.read_csv(EXTERNAL_SIGNATURES, low_memory=False)
    phase1_ids = set(phase1.pert_id.astype(str))
    counts = external.groupby("pert_id").rna_plate.nunique()
    eligible = set(counts[(counts >= 6) & ~counts.index.astype(str).isin(phase1_ids)].index.astype(str))
    external = external[external.pert_id.astype(str).isin(eligible)].reset_index(drop=True)
    raw = external[genes].to_numpy(dtype=np.float32)
    standardized = (raw - transforms["mean"]) / transforms["sd"]
    lda = transforms["lda"].transform(standardized)
    rows.append(
        {
            "cohort": "phase2_mcf7",
            "method": "shrinkage_lda_mean_consensus",
            **query_to_compound_mean(
                lda,
                external.pert_id.astype(str).to_numpy(),
                external.rna_plate.astype(str).to_numpy(),
            ),
        }
    )
    pd.DataFrame(rows).to_csv(V2_OUT / "lda_consensus_sensitivity.csv", index=False)

    biological = pd.read_csv(V2_OUT / "biological_retrieval_query_results.csv")
    target_lda = biological[(biological.label == "target") & (biological.method == "shrinkage_lda")]
    target_raw = biological[(biological.label == "target") & (biological.method == "raw_cosine")]
    target_result = paired_bootstrap(target_lda, target_raw)

    cross_context = cross_context_global_compound_bootstrap()

    standardized_all = transforms["standardized"].astype(np.float64)
    train_values = standardized_all[indices["train"]]
    train_compounds = phase1.iloc[indices["train"]].pert_id.astype(str).to_numpy()
    held_indices = np.concatenate([indices["validation"], indices["test"]])
    held_values = standardized_all[held_indices]
    held_compounds = phase1.iloc[held_indices].pert_id.astype(str).to_numpy()
    train_eigen, train_dimension, train_shrinkage = spectrum(train_values, train_compounds)
    held_eigen, held_dimension, held_shrinkage = spectrum(held_values, held_compounds)
    spectrum_comparison = pd.DataFrame(
        {
            "direction": np.arange(1, len(held_eigen) + 1),
            "training_generalized_eigenvalue": train_eigen[: len(held_eigen)],
            "heldout_generalized_eigenvalue": held_eigen,
        }
    )
    spectrum_comparison.to_csv(V2_OUT / "heldout_reproducibility_spectrum.csv", index=False)
    spectrum_result = {
        "training_compounds": int(len(np.unique(train_compounds))),
        "heldout_compounds": int(len(np.unique(held_compounds))),
        "training_effective_dimension": train_dimension,
        "heldout_effective_dimension": held_dimension,
        "training_covariance_shrinkage": train_shrinkage,
        "heldout_covariance_shrinkage": held_shrinkage,
        "training_top32_mass": float(train_eigen[:32].sum() / train_eigen.sum()),
        "heldout_top32_mass": float(held_eigen[:32].sum() / held_eigen.sum()),
        "rank_note": "positive-direction counts are bounded by compounds minus one and are not interpreted as a biological finding",
    }

    result = {
        "target_lda_minus_raw": target_result,
        "cross_context_global_compound_cluster": cross_context,
        "heldout_spectrum": spectrum_result,
    }
    write_json(V2_OUT / "credibility_sensitivities.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
