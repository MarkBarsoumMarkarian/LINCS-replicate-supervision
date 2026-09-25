# Frozen GSE70138 external-validation contract

Frozen on 24 September 2026 before any GSE70138 retrieval result was calculated.

## Purpose

Test whether the representation learned entirely from LINCS Phase I (GSE92742)
improves cross-plate retrieval in the independently generated LINCS Phase II
dataset (GSE70138), without retraining or selecting model parameters on Phase II.

## Frozen model and preprocessing

- Model: the validation-selected equal-weight similarity ensemble of the saved
  supervised-contrastive neural encoder and a shrinkage-LDA embedding refitted
  exactly from the frozen Phase I training split.
- Gene order: the same 978 directly measured genes, with exact identifier and
  order agreement required.
- Phase I training means, standard deviations, PCA basis, LDA fit, limma
  precision weights, and Hallmark-score scaling are applied unchanged.
- No GSE70138 compound labels are used for fitting, tuning, or weighting.

## Phase II representation

For each eligible compound and RNA plate, average log2 Level-2 treatment
intensity and subtract the median log2 vehicle profile from the same plate. This
is the same representation used in Phase I. Retrieval candidates on the query's
own plate are excluded.

## Primary external cohort

- GSE70138 / LINCS Phase II;
- MCF7, compound perturbations, 10 micromolar, 24 hours;
- compound absent from every Phase I plate-relative signature, not merely absent
  from the training split;
- at least six independent Phase II plates per compound.

The eligibility audit, performed without expression values or outcomes, yielded
81 compounds and 569 compound-by-plate profiles.

## Primary endpoint

Mean average precision (mAP) for cross-plate retrieval of perturbation identity.
The external endpoint passes when the frozen learned ensemble exceeds the best
non-learned external baseline and the 95% compound-cluster bootstrap interval
for the paired mAP difference is entirely above zero. Two thousand bootstrap
draws use seed 20260925.

## Prespecified comparators

- training-standardized raw cosine;
- frozen 64-component Phase I PCA cosine;
- Phase I limma precision-weighted cosine;
- frozen Hallmark ssGSEA cosine;
- supervised-contrastive neural encoder;
- shrinkage LDA;
- equal-weight contrastive/LDA similarity ensemble.

## Controls and sensitivity analyses

- replicate-threshold sensitivity at minimum 3, 4, 5, 6, and 7 plates in the
  unseen 24-hour cohort;
- a separate unseen 3-hour, 10-micromolar cohort with at least three plates;
- Phase-II compounds seen anywhere in Phase I as a positive/overlap control;
- 200 label-to-embedding permutations for an empirical chance reference;
- neural-only and LDA-only ablations of the selected ensemble.

The 6-hour Phase II subset is excluded because every eligible compound occurs
on only one plate, leaving no cross-plate positive for the retrieval endpoint.

## Interpretation boundary

This experiment tests reproducible perturbation-identity retrieval across LINCS
phases and time-point shift. It does not establish mechanism equivalence,
therapeutic interchangeability, clinical validity, or biological identity.
