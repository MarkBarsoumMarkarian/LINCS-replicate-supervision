# Independent Phase II validation: GSE70138

Date: 24 September 2026

## Bottom line

The independent external endpoint **passed**.

The representation was developed entirely in LINCS Phase I (GSE92742), then
frozen and transferred without retraining to LINCS Phase II (GSE70138). The
primary external cohort contained 81 compounds never observed in any Phase I
plate-relative signature and 569 independent compound-by-plate profiles from
MCF7 cells treated at 10 micromolar for 24 hours.

The prospectively selected neural-plus-LDA ensemble achieved an external mAP of
**0.3931**, compared with **0.2818** for the strongest non-learned comparator,
Hallmark ssGSEA. The absolute paired improvement was **0.1113** (39.5% relative),
with a compound-cluster bootstrap 95% interval of **0.0749 to 0.1499**. Because
the complete interval was above zero, the frozen external endpoint passed.

## Primary external result

| Method | mAP | Nearest-neighbour accuracy | Top-5 hit rate | Pair AUROC |
|---|---:|---:|---:|---:|
| **Shrinkage LDA** | **0.4446** | **0.6221** | **0.8541** | **0.8946** |
| Frozen neural + LDA ensemble | 0.3931 | 0.5325 | 0.8032 | 0.8645 |
| Hallmark ssGSEA | 0.2818 | 0.4499 | 0.6854 | 0.7648 |
| Contrastive neural encoder | 0.2620 | 0.3585 | 0.5940 | 0.8082 |
| Limma-weighted cosine | 0.2190 | 0.3726 | 0.5940 | 0.7116 |
| Raw cosine | 0.2181 | 0.3761 | 0.5852 | 0.7077 |
| PCA cosine | 0.1500 | 0.2812 | 0.4798 | 0.6527 |

The ensemble was the frozen primary model because it had been selected on the
Phase I validation split before this external dataset was opened. Shrinkage LDA
was a prespecified ablation/comparator and transferred even better: mAP 0.4446,
an absolute improvement of 0.1629 over ssGSEA (bootstrap 95% interval 0.1244 to
0.2033). This cannot be hidden or rewritten as a neural-model victory. The
external result indicates that replicate-supervised learning transfers, but the
shrinkage-linear component carries most of that transfer under the phase and
time-point shift.

## Controls and robustness

- The ensemble improved compound-level mAP for 64 of 81 unseen compounds.
- Its observed mAP of 0.3931 was far above the 200-permutation null mean of
  0.0231 (95% null interval 0.0208 to 0.0259; one-sided empirical p = 0.00498).
- The improvement was not created by the six-plate eligibility cutoff. Ensemble
  mAP exceeded raw cosine at every tested minimum-plate threshold: 3, 4, 5, 6,
  and 7 plates.
- In a separate Phase II 3-hour cohort of 38 unseen compounds and 124 profiles,
  the ensemble reached mAP 0.5298 and shrinkage LDA reached 0.5936, versus 0.3040
  for Hallmark ssGSEA and 0.2139 for raw cosine.
- In the 14-compound Phase-I-overlap positive control, ensemble mAP was 0.7208
  and shrinkage-LDA mAP was 0.7745. The lower performance in fully unseen
  compounds is expected and makes clear that the primary result is genuine
  transfer rather than perfect invariance.
- The Phase II 6-hour slice was not used: every eligible compound appeared on a
  single plate, so the cross-plate retrieval endpoint was mathematically
  unavailable.

## Interpretation

The strongest combined gene-weight signal was enriched for Hallmark TNF-alpha
signalling through NF-kappa-B (FDR 9.45e-7), early estrogen response (FDR
5.19e-6), mTORC1 signalling (FDR 1.45e-4), p53 pathway (FDR 0.00317), and UV
response down (FDR 0.00436). These enrichments show that the representation
prioritizes coherent stress, growth, hormone-response, and checkpoint axes
rather than arbitrary genes. They are interpretive associations, not evidence
that every retrieved compound shares one mechanism.

## Defensible manuscript claim

> Replicate-supervised representation learning improved cross-plate retrieval
> of perturbation identity for compounds unseen during development and measured
> in an independent LINCS phase. The prospectively selected ensemble increased
> external mean average precision by 39.5% relative to the strongest
> non-learned baseline, with a compound-bootstrap interval excluding zero.
> Prespecified ablation showed that shrinkage LDA was the strongest transferable
> component.

This supports an assay-reproducibility and external-transfer claim. It does not
establish shared mechanism, drug equivalence, clinical validity, or therapeutic
interchangeability.

## Reproducibility

- The external-validation contract was frozen before performance was opened.
- Phase II profiles were joined to the GCTX matrix explicitly by instance ID.
- The same 978 genes were reordered explicitly by gene ID into the frozen Phase
  I model order.
- Phase I scaling, PCA, LDA fitting rules, limma weights, Hallmark scoring, and
  the saved neural encoder were applied without Phase II fitting or tuning.
- Same-plate retrieval candidates were excluded.
- Two thousand compound-cluster bootstrap draws quantified the primary paired
  improvement.
- An independent full replay reproduced every hashed CSV, compressed signature
  table, and result JSON exactly.
