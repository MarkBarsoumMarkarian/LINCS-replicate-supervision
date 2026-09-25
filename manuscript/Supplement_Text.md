# Supplementary appendix

## Shrinkage linear replicate supervision recovers perturbation identity and pharmacological relationships across LINCS contexts

Mark Barsoum Markarian

## Supplementary methods

### Source integrity and study chronology

The Phase I matrix was GSE92742_Broad_LINCS_Level2_GEX_epsilon_n1269922x978.gctx. The Phase II matrix was GSE70138_Broad_LINCS_Level2_GEX_n345976x978_2017-03-06.gctx. Metadata and matrix columns were aligned by unique instance identifier, and Phase II landmarks were reordered into the Phase I gene order. The original Phase I test and MCF7 Phase II transfer were complete before the extension. Its contract was frozen before the primary multi-cell, pharmacological-neighbour, reproducibility-spectrum, or scaling outcomes were calculated. Held-out spectrum estimation, conservative cross-context clustering, direct LDA consensus, stronger-comparator inference, longer-schedule neural fits, and the binary-barcode tie audit were later credibility sensitivities and are labelled post-freeze.

### Published perturbation-barcode reconstruction

The Filzen-style network contained 978 inputs and sigmoid layers of 400 and 100 units. During training, independent Gaussian noise with standard deviation 0.5 was added before each sigmoid and dropout probability was 0.5. Positive and negative pairs were sampled at a 1:2 ratio. The loss combined squared distance for positive pairs, a rectified squared-distance margin of 5 for negative pairs, and L1 weight decay 1e-6. RMSprop used learning rate 0.001. Sixteen pair batches were run per epoch, with a maximum of 80 epochs and patience 14. Phase I validation mean average precision selected epoch 48. Inference used deterministic continuous activations or binary activations thresholded at 0.5. The thresholded representation produced only three unique vectors in each Phase I split. A post-freeze audit therefore calculated expected retrieval metrics under uniform random ordering within exact cosine-score ties.

### Task matched MODZ adaptation

Within a candidate compound, each plate profile was ranked across genes. Pairwise Spearman correlations were floored at 0.01. A replicate's weight was its summed correlation to the other replicates divided by the total summed correlation. The weighted gene vector formed the consensus. For every query, profiles from its plate were removed from all candidate consensuses. This is an adaptation of the CMap replicate-collapse rule to the present Level 2 plate-relative profiles, not an official Level 5 file.

### Multi-cell transfer

The five cell lines were selected using metadata alone. Treatment profiles required compound treatment, 10 micromolar dose, 24-hour exposure, a same-cell and same-time vehicle on the plate, absence from the Phase I signature collection, and at least six independent Phase II plates. No context was dropped after expression outcomes were opened. Frozen Phase I means, standard deviations, PCA, limma weights, LDA transformation, neural weights, Filzen weights, and Hallmark scaling were applied unchanged.

### Pharmacological annotations

Broad sample identifiers were reduced to their 13-character perturbation prefix and joined to the Drug Repurposing Hub 24 March 2020 sample and drug tables. Mechanism and target strings were split on vertical bars, stripped, lowercased, and deduplicated. Terms occurring in only one cohort compound were ineligible because they could not define a distinct relevant neighbour. Compounds with missing labels were not treated as negatives.

### Reproducibility spectrum

Training-compound means estimated between-compound covariance. Deviations of each profile from its compound mean estimated within-compound covariance using Ledoit-Wolf shrinkage. The generalized eigenproblem compared between-compound to within-compound covariance. Eigenvalues below zero from numerical precision were truncated to zero. Effective dimensionality was the participation ratio, defined as the squared sum of positive eigenvalues divided by their squared-sum.

### Scaling analysis

Training compounds were ordered by SHA-256 of a fixed seed and identifier. Within each compound, plates were ordered by a second fixed hash. Nested subsets contained 20, 40, 60, 80, or 100 compounds and 3, 6, 9, or 12 plates. Preprocessing was re-estimated from each subset. The complete grid used a computationally bounded maximum of 25 epochs, patience 5, and six batches per epoch. A post-freeze audit repeated the 20-compound row and 100-compound, 12-plate point with the main schedules: maximum 120 epochs and patience 18 with at least 12 batches per epoch for the contrastive encoder, and maximum 80 epochs, patience 14, and 16 batches per epoch for Filzen. The unchanged Phase I validation identities supplied the explanatory endpoint.

## Supplementary results

### Table S1 Study populations

| Cohort | Compounds | Profiles | Cell line | Dose | Time | Status |
|---|---:|---:|---|---|---|---|
| Phase I training | 100 | 1820 | MCF7 | 10 micromolar | 6 hours | Frozen training |
| Phase I validation | 34 | 460 | MCF7 | 10 micromolar | 6 hours | Frozen selection |
| Phase I test | 34 | 475 | MCF7 | 10 micromolar | 6 hours | Locked test |
| Phase II MCF7 | 81 | 569 | MCF7 | 10 micromolar | 24 hours | Previously locked transfer |
| Phase II A375 | 70 | 463 | A375 | 10 micromolar | 24 hours | New locked context |
| Phase II A549 | 102 | 694 | A549 | 10 micromolar | 24 hours | New locked context |
| Phase II HA1E | 78 | 506 | HA1E | 10 micromolar | 24 hours | New locked context |
| Phase II HT29 | 77 | 507 | HT29 | 10 micromolar | 24 hours | New locked context |
| Phase II PC3 | 63 | 423 | PC3 | 10 micromolar | 24 hours | New locked context |

[[PAGEBREAK]]

### Table S2 Core perturbation identity retrieval

| Method | Phase I test mAP | Phase II MCF7 mAP |
|---|---:|---:|
| Shrinkage LDA | 0.1978 | 0.4446 |
| Neural and LDA ensemble | 0.2021 | 0.3931 |
| Contrastive neural encoder | 0.1707 | 0.2620 |
| Hallmark ssGSEA | 0.1126 | 0.2818 |
| Limma weighted cosine | 0.0867 | 0.2190 |
| Raw cosine | 0.0815 | 0.2181 |
| PCA cosine | 0.0779 | 0.1500 |

### Table S3 Direct prior reconstruction

| Split | Representation | mAP | Nearest neighbour | Top 5 hit rate |
|---|---|---:|---:|---:|
| Validation | Filzen continuous | 0.0806 | 0.1174 | 0.2870 |
| Validation | Filzen binary tie-aware | 0.0587 | 0.0479 | 0.1561 |
| Test | Filzen continuous | 0.0730 | 0.1284 | 0.2716 |
| Test | Filzen binary tie-aware | 0.0510 | 0.0457 | 0.1527 |

The frozen stable-order implementation reported identical test nearest-neighbour and top-five rates of 0.0758 because every query had an exact score tie spanning the top ranks. The tie-aware values above replace that arbitrary ordering with the expectation over uniform ordering within ties; the continuous representation is unaffected.

### Table S4 Consensus retrieval

| Cohort | Consensus | Mean reciprocal rank | Top 1 | Top 5 |
|---|---|---:|---:|---:|
| Phase I test | Unweighted mean | 0.2337 | 0.1326 | 0.2947 |
| Phase I test | MODZ weighted | 0.2568 | 0.1537 | 0.3242 |
| Phase I test | Shrinkage LDA mean | 0.4203 | 0.2926 | 0.5516 |
| Phase II MCF7 | Unweighted mean | 0.4747 | 0.3726 | 0.5870 |
| Phase II MCF7 | MODZ weighted | 0.4610 | 0.3497 | 0.5835 |
| Phase II MCF7 | Shrinkage LDA mean | 0.7356 | 0.6309 | 0.8699 |

### Table S5 Multi-cell identity retrieval

| Cell line | Compounds | Profiles | LDA mAP | ssGSEA mAP | Paired difference | 95 percent interval |
|---|---:|---:|---:|---:|---:|---|
| A375 | 70 | 463 | 0.3942 | 0.3441 | 0.0502 | 0.0105 to 0.0930 |
| A549 | 102 | 694 | 0.3441 | 0.2682 | 0.0759 | 0.0503 to 0.1048 |
| HA1E | 78 | 506 | 0.4223 | 0.3722 | 0.0501 | 0.0142 to 0.0863 |
| HT29 | 77 | 507 | 0.3328 | 0.2489 | 0.0839 | 0.0424 to 0.1219 |
| PC3 | 63 | 423 | 0.3924 | 0.3248 | 0.0676 | 0.0244 to 0.1105 |
| Macro result | 390 | 2593 | Not pooled | Not pooled | 0.0655 | 0.0485 to 0.0831 |

[[FIGURE:03_Figures/Figure_S1_All_Multicontext_Methods.png|Figure S1. Complete multi-context method comparison. Every representation was frozen from Phase I and applied without Phase II fitting.|Faceted bar chart showing mean average precision for all methods in five new cell lines.]]

### Table S6 Mechanism of action retrieval

| Method | mAP | Nearest neighbour | Top 5 hit rate | Eligible queries |
|---|---:|---:|---:|---:|
| Shrinkage LDA | 0.4052 | 0.4407 | 0.6949 | 59 |
| Limma weighted cosine | 0.3320 | 0.3729 | 0.6780 | 59 |
| Contrastive neural encoder | 0.3247 | 0.3559 | 0.6271 | 59 |
| Raw cosine | 0.3234 | 0.3729 | 0.6780 | 59 |
| Hallmark ssGSEA | 0.3195 | 0.3559 | 0.6441 | 59 |
| PCA cosine | 0.2710 | 0.2542 | 0.5932 | 59 |
| Filzen continuous | 0.2389 | 0.2203 | 0.5593 | 59 |
| Filzen binary | 0.1594 | 0.1017 | 0.5085 | 59 |

### Table S7 Protein target retrieval

| Method | mAP | Nearest neighbour | Top 5 hit rate | Eligible queries |
|---|---:|---:|---:|---:|
| Shrinkage LDA | 0.3780 | 0.4194 | 0.7419 | 62 |
| Hallmark ssGSEA | 0.3388 | 0.4355 | 0.6935 | 62 |
| Limma weighted cosine | 0.3008 | 0.3226 | 0.6774 | 62 |
| Contrastive neural encoder | 0.2956 | 0.2742 | 0.5806 | 62 |
| Raw cosine | 0.2947 | 0.3226 | 0.6774 | 62 |
| PCA cosine | 0.2701 | 0.2581 | 0.6452 | 62 |
| Filzen continuous | 0.2302 | 0.1935 | 0.5484 | 62 |
| Filzen binary | 0.2081 | 0.1774 | 0.5645 | 62 |

### Table S8 Reproducibility spectrum summary

| Quantity | Value |
|---|---:|
| Training effective dimension | 27.09 |
| Held-out effective dimension | 24.31 |
| Training mass in top 8 | 45.90 percent |
| Training mass in top 16 | 59.94 percent |
| Training mass in top 32 | 73.98 percent |
| Held-out mass in top 32 | 78.59 percent |

An independent spectrum estimated from the 68 validation and test compounds had effective dimension 24.31, with 78.59 percent of discriminative mass in its leading 32 directions. Positive-direction counts equal at most the number of compound means minus one and are not interpreted biologically.

### Table S9 Full scaling grid point

| Method | Validation mAP with 100 compounds and 12 plates |
|---|---:|
| Shrinkage LDA | 0.2043 |
| Contrastive neural encoder | 0.1616 |
| Raw cosine | 0.0892 |
| Filzen continuous | 0.0526 |

Shrinkage LDA ranked first at all 20 reduced-schedule combinations of 20, 40, 60, 80, or 100 training compounds and 3, 6, 9, or 12 plates; Table S11 gives the longer-schedule audit.

### Table S10 Post-freeze conservative sensitivities

| Analysis | Estimate | 95 percent interval |
|---|---:|---|
| Mechanism retrieval LDA minus limma weighted cosine | 0.0732 | 0.0119 to 0.1239 |
| Target retrieval LDA minus raw | 0.0833 | 0.0469 to 0.1215 |
| Target retrieval LDA minus Hallmark ssGSEA | 0.0391 | 0.0026 to 0.0767 |
| Multi-cell macro gain with global compound clusters | 0.0655 | 0.0444 to 0.0888 |
| Held-out spectrum effective dimension | 24.31 | Descriptive |

### Table S11 Main-schedule neural audit

| Training compounds | Plates | LDA mAP | Contrastive mAP | Filzen mAP |
|---:|---:|---:|---:|---:|
| 20 | 3 | 0.1004 | 0.0737 | 0.0498 |
| 20 | 6 | 0.1253 | 0.1018 | 0.0584 |
| 20 | 9 | 0.1415 | 0.1128 | 0.0604 |
| 20 | 12 | 0.1470 | 0.1267 | 0.0650 |
| 100 | 12 | 0.2043 | 0.1684 | 0.0795 |

The main schedules improved the full-point contrastive result from 0.1616 to 0.1684 and the Filzen result from 0.0526 to 0.0795. LDA remained first at the full point and throughout the audited 20-compound row. With all available training plates, the corresponding main-analysis validation values were 0.2141, 0.1854, and 0.0806.

### Table S12 Replicate threshold sensitivity

| Minimum plates | Compounds | Profiles | Ensemble mAP | Shrinkage LDA mAP | Neural mAP | Raw cosine mAP |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1548 | 5027 | 0.1137 | 0.1291 | 0.0590 | 0.0697 |
| 4 | 133 | 782 | 0.3592 | 0.4095 | 0.2226 | 0.2035 |
| 5 | 86 | 594 | 0.3855 | 0.4350 | 0.2552 | 0.2164 |
| 6 | 81 | 569 | 0.3931 | 0.4446 | 0.2620 | 0.2181 |
| 7 | 35 | 293 | 0.4534 | 0.5130 | 0.3238 | 0.2666 |

### Table S13 Independent 3-hour cohort

| Method | mAP | Nearest neighbour | Top 5 hit rate | Pair AUROC |
|---|---:|---:|---:|---:|
| Shrinkage LDA | 0.5936 | 0.5645 | 0.9274 | 0.9105 |
| Neural and LDA ensemble | 0.5298 | 0.5323 | 0.8226 | 0.8812 |
| Contrastive neural encoder | 0.3671 | 0.3145 | 0.6532 | 0.8181 |
| Hallmark ssGSEA | 0.3040 | 0.2742 | 0.5161 | 0.7469 |
| Limma weighted cosine | 0.2230 | 0.1935 | 0.4597 | 0.6796 |
| Raw cosine | 0.2139 | 0.2097 | 0.4355 | 0.6696 |
| PCA cosine | 0.2108 | 0.1935 | 0.4274 | 0.6574 |

### Table S14 Seen-compound overlap control

| Method | mAP | Nearest neighbour | Top 5 hit rate | Pair AUROC |
|---|---:|---:|---:|---:|
| Shrinkage LDA | 0.7745 | 0.8852 | 0.9672 | 0.9443 |
| Neural and LDA ensemble | 0.7208 | 0.8770 | 0.9590 | 0.9181 |
| Contrastive neural encoder | 0.6014 | 0.7295 | 0.8934 | 0.8437 |
| Limma weighted cosine | 0.4324 | 0.6311 | 0.8361 | 0.7468 |
| Raw cosine | 0.4235 | 0.6148 | 0.8443 | 0.7355 |
| PCA cosine | 0.3482 | 0.5902 | 0.8033 | 0.6917 |

### Table S15 Permutation nulls

| Endpoint | Observed mAP | Null mean | Null 95 percent interval | One-sided P |
|---|---:|---:|---|---:|
| Frozen Phase II identity ensemble | 0.3931 | 0.0231 | 0.0208 to 0.0259 | 0.00498 |
| Mechanism retrieval with shrinkage LDA | 0.4052 | 0.1335 | 0.1085 to 0.1659 | 0.00050 |

[[FIGURE:03_Figures/Figure_S2_Scaling_Grid.png|Figure S2. Complete reduced-schedule 20-point scaling grid. Lines show validation mean average precision as compounds and plates increase; Table S11 reports the longer-schedule audit.|Four method-specific scaling curves over training compound and replicate counts.]]

[[FIGURE:03_Figures/Figure_S3_MODZ_and_Filzen.png|Figure S3. Direct comparator diagnostics. Panel A compares mean and MODZ consensus retrieval. Panel B compares the Filzen-style continuous and binary representations with the principal learned models in Phase I.|Comparator chart for consensus collapse and the published perturbation-barcode reconstruction.]]

## Reproducibility inventory

Electronic tables contain all query-level, comparator, spectrum, sensitivity, and scaling results. Code and contracts cover the frozen benchmark and extension. The 25 core result artifacts and two post-freeze sensitivity artifacts each reproduced byte for byte; both verification records include SHA-256 hashes.
