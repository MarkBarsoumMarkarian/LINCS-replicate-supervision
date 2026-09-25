#!/usr/bin/env python3
"""Estimate the training-only reproducibility spectrum of L1000 responses."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.covariance import LedoitWolf

from common import SEED, V2_OUT, fit_frozen_transforms, load_phase1, set_seed, write_json


def main() -> None:
    set_seed(SEED)
    V2_OUT.mkdir(parents=True, exist_ok=True)
    phase1, genes, indices = load_phase1()
    transforms = fit_frozen_transforms(phase1, genes, indices)
    train = indices["train"]
    values = transforms["standardized"][train].astype(np.float64)
    compounds = phase1.iloc[train]["pert_id"].astype(str).to_numpy()
    names = np.array(sorted(np.unique(compounds)))
    means = np.vstack([values[compounds == name].mean(axis=0) for name in names])
    residuals = np.vstack(
        [values[compounds == name] - values[compounds == name].mean(axis=0) for name in names]
    )

    print("Estimating shrinkage within-compound covariance", flush=True)
    within_estimator = LedoitWolf(assume_centered=True).fit(residuals)
    within = within_estimator.covariance_
    between = np.cov(means, rowvar=False, ddof=1)
    within = (within + within.T) / 2
    between = (between + between.T) / 2

    print("Solving generalized between-versus-within eigenproblem", flush=True)
    eigenvalues, eigenvectors = eigh(between, within, check_finite=True)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues[eigenvalues > 1e-12]
    total = positive.sum()
    cumulative = np.cumsum(eigenvalues) / max(total, 1e-12)
    participation_ratio = float(total**2 / np.sum(positive**2))

    spectrum = pd.DataFrame(
        {
            "direction": np.arange(1, len(eigenvalues) + 1),
            "generalized_eigenvalue": eigenvalues,
            "cumulative_discriminative_mass": cumulative,
            "signal_fraction_lambda_over_one_plus_lambda": eigenvalues / (1 + eigenvalues),
        }
    )
    spectrum.to_csv(V2_OUT / "reproducibility_spectrum.csv", index=False)
    loadings = pd.DataFrame(
        eigenvectors[:, :32],
        index=[gene[2:] for gene in genes],
        columns=[f"direction_{index}" for index in range(1, 33)],
    ).reset_index(names="gene_id")
    loadings.to_csv(V2_OUT / "reproducibility_spectrum_top32_loadings.csv", index=False)

    result = {
        "source": "Phase I training compounds only",
        "training_compounds": int(len(names)),
        "training_profiles": int(len(values)),
        "features": int(len(genes)),
        "within_covariance_estimator": "Ledoit-Wolf on within-compound residuals",
        "within_covariance_shrinkage": float(within_estimator.shrinkage_),
        "positive_generalized_directions": int(len(positive)),
        "directions_with_eigenvalue_gt_1": int((eigenvalues > 1).sum()),
        "participation_ratio_effective_dimension": participation_ratio,
        "cumulative_discriminative_mass": {
            "top_8": float(cumulative[7]),
            "top_16": float(cumulative[15]),
            "top_32": float(cumulative[31]),
        },
        "largest_generalized_eigenvalues": eigenvalues[:10].tolist(),
    }
    write_json(V2_OUT / "reproducibility_spectrum_result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
