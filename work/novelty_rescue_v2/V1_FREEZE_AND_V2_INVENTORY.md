# Version 1 freeze and Version 2 inventory

Recorded on 24 September 2026.

## Frozen Version 1

The complete Version 1 submission package is preserved at:

`outputs/assay_conditioned_resolution_submission_package/`

The distributable archive is preserved at:

`outputs/Assay_Conditioned_Resolution_Submission_Package.zip`

Version 2 work must not overwrite either path. The frozen result remains a
successful cross-phase perturbation-identity retrieval benchmark. Its central
claim must now cite Filzen et al. and must not describe replicate-supervised
transcriptomic metric learning as new.

## Available local inputs

- GSE92742 Level 2 GCTX and instance metadata.
- GSE70138 Level 2 GCTX and instance metadata.
- 2,755 Phase I MCF7/10 micromolar/6-hour compound-by-plate signatures.
- Frozen compound split, preprocessing values, neural checkpoint, and all
  Version 1 evaluation records.
- GSE70138 MCF7/10 micromolar/24-hour and 3-hour plate signatures.
- Broad Drug Repurposing Hub 24 March 2020 drug and sample annotations.
- Hallmark gene sets and precomputed Phase I ssGSEA scores.

## Metadata-only feasibility findings

After requiring compound treatment, same-cell/same-time vehicle availability,
10 micromolar dose, absence from the Phase I signature collection, and at least
six independent GSE70138 plates, the new 24-hour cell-line cohorts contain:

| Cell line | Eligible unseen compounds |
|---|---:|
| A549 | 102 |
| HA1E | 78 |
| HT29 | 77 |
| A375 | 70 |
| PC3 | 63 |

The already observed MCF7 reference contains 81 eligible unseen compounds.
These counts were obtained without opening any new expression endpoint.

Among the 81-compound MCF7 reference cohort, the archived Repurposing Hub files
match 74 compounds with MoA annotations and 73 with target annotations. Final
eligible-query counts will be determined only by the locked rule requiring at
least one distinct relevant compound.

## Compute and storage

The machine has 8 logical CPUs, approximately 7.5 GiB RAM, 7.5 GiB swap, and
more than 140 GiB free storage at inventory time. The five-context extraction
is therefore designed to stream only selected GCTX rows and to process one
context at a time. No new raw-data download is required.

## Prior-art correction

Filzen et al. trained a Siamese metric-learning network on biological replicate
identity, encoded 978 landmark genes as a 100-dimensional perturbation barcode,
tested target similarity, applied the architecture to GSE70138, and performed
prospective experimental validation. This is a close direct precedent.

The defensible Version 2 distinction is the combination of compound-disjoint
training, same-plate exclusion, no-retraining GSE92742-to-GSE70138 transfer,
multi-cell context testing, direct classical-versus-neural comparison, and an
analysis of why shrinkage LDA can outperform the deep representations.
