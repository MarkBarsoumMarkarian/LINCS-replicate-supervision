# Replicate-supervised metric-learning benchmark contract

Frozen before model training or held-out evaluation.

## Objective

Learn an assay-conditioned representation in which independent plate-level
profiles from the same perturbation are close and profiles from different
perturbations are separated. The outcome is held-out perturbation-identity
retrieval, not a claim that different compound identifiers always represent
different biological mechanisms.

## Data

- 2,755 MCF7, 10 micromolar, 6 hour compound-by-plate profiles.
- 168 compounds with at least 12 independent RNA plates.
- 978 directly measured genes.
- Input representation: mean log2 treatment intensity minus same-plate vehicle
  median, identical to the frozen V2 representation.
- One sample is one compound on one RNA plate.

## Locked split

All profiles from a compound stay in one split. The 16 compounds exposed in
earlier pair outcomes or method development are assigned to training only.
The remaining compounds are ordered by SHA-256 of
`20260924|perturbation_identifier` and assigned to produce exactly:

- 100 training compounds total;
- 34 validation compounds;
- 34 untouched test compounds.

No expression value, activity score, mechanism annotation, or model outcome is
used to construct the split. The test split is opened once after model and
hyperparameter selection is complete.

## Preprocessing

Per-gene mean and standard deviation are estimated from training profiles only.
The same frozen transformation is applied to validation and test profiles.

## Models

Primary learned representation:

- encoder: 978 -> 128 -> 32 dimensions;
- GELU activation, layer normalization, 10% dropout, L2-normalized embedding;
- supervised contrastive loss with 16 compounds and 4 independent plates per
  compound in each batch;
- temperature 0.10, AdamW, maximum 120 epochs;
- early stopping uses validation mean average precision only.

A shrinkage linear-discriminant embedding and an equal-weight similarity
ensemble of the neural and linear embeddings are prespecified learned
comparators. The learned representation with the highest validation mean
average precision is selected before the test split is evaluated.

## Non-learned baselines

- cosine similarity on training-standardized gene effects;
- 64-component PCA cosine similarity, fitted on training profiles;
- limma precision-weighted cosine similarity using the previously learned
  development-compound posterior variances;
- ssGSEA Hallmark-score cosine similarity using the frozen pathway scores.

## Evaluation

All retrieval candidates must come from a different RNA plate than the query.
For each query, relevant items are profiles with the same perturbation ID.

Primary metric: mean average precision across test queries.

Secondary metrics: nearest-neighbour accuracy, recall within the top five,
and same-versus-different compound pair AUROC. Uncertainty is estimated by
2,000 compound-cluster bootstrap draws.

The benchmark is successful when the selected learned representation exceeds
the best non-learned baseline in test mean average precision with a positive
95% compound-bootstrap interval for the paired improvement. All results are
reported even when this criterion is not met.

## Leakage controls

- no compound appears in more than one split;
- preprocessing and PCA are fitted on training profiles only;
- neural and linear models see training compound labels only;
- validation selects the model and neural stopping epoch;
- test labels are used only for the single final evaluation;
- same-plate candidates are excluded from retrieval and pairwise metrics.

