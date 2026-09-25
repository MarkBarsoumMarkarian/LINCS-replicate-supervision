# Replicate-supervised metric learning: locked benchmark result

Date: 24 September 2026

## Outcome

The frozen benchmark endpoint **passed**.

The model was trained on 100 compounds, selected using 34 separate validation
compounds, and evaluated once on 34 completely unseen test compounds. All
retrieval comparisons excluded profiles from the query's RNA plate.

The validation-selected representation was an equal-weight similarity ensemble
of a supervised-contrastive neural embedding and a shrinkage linear-discriminant
embedding.

| Method | Test mAP | Nearest-neighbour accuracy | Top-5 hit rate | Pair AUROC |
|---|---:|---:|---:|---:|
| **Replicate-supervised ensemble** | **0.2021** | **0.2779** | 0.4758 | **0.7436** |
| Shrinkage LDA | 0.1978 | 0.2674 | **0.4968** | 0.7263 |
| Contrastive neural encoder | 0.1707 | 0.1874 | 0.3621 | 0.7172 |
| Best non-learned baseline: ssGSEA | 0.1126 | 0.2042 | 0.3916 | 0.5946 |
| Limma-weighted cosine | 0.0867 | 0.1558 | 0.3158 | 0.5555 |
| Raw cosine | 0.0815 | 0.1495 | 0.3053 | 0.5512 |
| PCA cosine | 0.0779 | 0.1621 | 0.3221 | 0.5423 |

Primary mAP improvement over the strongest non-learned baseline:

- absolute improvement: **0.0895**;
- relative improvement: **79.5%**;
- compound-cluster bootstrap 95% interval: **0.0343 to 0.1606**;
- endpoint: **PASS**, because the complete interval is above zero.

The learned ensemble improved compound-level mean average precision for 28 of
34 unseen test compounds. Its nearest-neighbour accuracy was 27.8%, compared
with an exact random-ranking expectation of 2.9% for this test composition.

Validation mAP was 0.2192 and untouched-test mAP was 0.2021, providing no sign
of a large validation-to-test collapse.

## What was learned

The model learned which transcriptional directions remain reproducible across
independent plates. It was never given test compound identities during
training, yet it retrieved replicate profiles of those unseen compounds more
accurately than raw expression, PCA, limma precision weighting, or Hallmark
ssGSEA.

The neural model was not the sole source of improvement. Shrinkage LDA was the
strongest individual learned representation, while the prospectively selected
neural-plus-linear ensemble produced the highest validation and test mAP and
the highest pair AUROC.

## Defensible claim

> Replicate-supervised representation learning improved cross-plate retrieval
> of entirely unseen perturbations in LINCS MCF7 profiles. A validation-selected
> contrastive-neural and shrinkage-linear ensemble increased held-out mean
> average precision by 79.5% relative to the strongest non-learned baseline,
> with a compound-bootstrap confidence interval excluding zero.

This is an assay-reproducibility and perturbation-identity result. It does not
prove shared mechanism, biological equivalence, clinical similarity, or a
validated Confusable branch.

## Reproducibility and leakage controls

- 100/34/34 compound-level train/validation/test split frozen before training;
- all previously exposed compounds forced into training;
- no compound crossed splits;
- preprocessing, PCA, LDA, and neural training used training profiles only;
- validation selected the model and neural stopping epoch;
- the test split was evaluated once after the selection record was written;
- all retrieval candidates came from a different RNA plate than the query;
- 2,000 compound-cluster bootstrap draws quantified improvement;
- a complete independent replay reproduced the result JSON, validation table,
  test table, and per-query table byte-for-byte.

## Independent Phase II validation

The frozen Phase I representation was subsequently evaluated without retraining
in GSE70138. The primary external cohort comprised 81 compounds absent from all
Phase I plate-relative signatures and 569 MCF7 compound-by-plate profiles at 10
micromolar and 24 hours.

The prospectively selected neural-plus-LDA ensemble achieved mAP 0.3931 versus
0.2818 for the strongest non-learned comparator, Hallmark ssGSEA. The paired
improvement was 0.1113 (39.5% relative), with a 95% compound-bootstrap interval
of 0.0749 to 0.1499; the external endpoint therefore passed. Shrinkage LDA alone
was the strongest external method (mAP 0.4446), showing that the linear component
carried most of the cross-phase transfer. The full external report is in
`external_gse70138/RESULTS.md`.
