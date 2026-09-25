# Version 2 novelty-rescue analysis contract

Frozen on 24 September 2026 before calculation of any endpoint introduced in
this document. The original Phase I test results and the MCF7 Phase II result
had already been observed and are therefore treated as prior evidence, not as
new preregistered confirmation. Version 1 files are immutable.

## Scientific question

The primary question is no longer whether replicate identity can supervise an
L1000 representation; that principle was demonstrated by Filzen et al. (2017).
The new question is whether a simple shrinkage-linear representation trained on
plate-independent replicates transfers more reliably than nonlinear metric
learning and established transcriptomic summaries across compounds, LINCS
phases, cell types, time, and pharmacological relationships.

The intended contribution is a strict transfer benchmark and an empirical
account of when classical shrinkage wins. It is not a claim to have invented
replicate-supervised transcriptomic metric learning.

## Immutable prior experiment

- Training source: GSE92742, MCF7, 10 micromolar, 6 hours.
- Unit: one compound-by-RNA-plate treatment-minus-same-plate-vehicle profile.
- Compound-disjoint split: 100 train, 34 validation, 34 test compounds.
- Frozen transforms: Phase I training mean and standard deviation, PCA,
  limma precision weights, supervised-contrastive encoder, and shrinkage LDA.
- Previously observed Phase II reference: GSE70138, MCF7, 10 micromolar,
  24 hours, compounds absent from the Phase I signature set, at least six
  independent plates.
- The previously observed MCF7 result will be reported but will not count as a
  newly confirmatory Version 2 endpoint.

## Direct-prior comparator: perturbation barcode

A faithful modern reimplementation of Filzen et al. will be trained only on
the frozen Phase I training compounds:

- 978 inputs, hidden layers of 400 and 100 units;
- sigmoid nonlinearities;
- independent Gaussian pre-activation noise with variance 0.25 during
  training and no noise during inference;
- dropout probability 0.5 and L1 weight penalty 1e-6;
- contrastive margin loss on squared Euclidean distance with margin 5;
- RMSprop optimization;
- positive pairs are different plates of the same compound and negatives are
  different compounds, sampled at a fixed 1:2 ratio;
- model selection uses Phase I validation perturbation-identity mAP only;
- both the continuous 100-dimensional activation and the thresholded binary
  barcode are evaluated.

This is called a faithful reimplementation rather than an exact reproduction
because the original proprietary training corpus and complete optimizer
schedule are unavailable.

## CMap consensus comparator

MODZ is evaluated as a task-matched adaptation to the available plate-relative
profiles, not mislabeled as official CMap Level 5 data. For each candidate
compound, replicate weights are proportional to the sum of its pairwise
Spearman correlations, with correlations floored at 0.01 and weights normalized
to sum to one. Each query is ranked against compound consensuses; the query
profile is removed before constructing its own compound consensus. Profiles
from the query plate are excluded from every candidate consensus for that
query.

Because consensus collapse creates one candidate per compound, this endpoint
uses mean reciprocal rank, top-1 accuracy, and top-5 hit rate rather than the
profile-level mAP endpoint.

## New endpoint 1: biological-neighbour retrieval

Primary cohort: the previously fixed 81-compound GSE70138 MCF7 external cohort.
Drug Repurposing Hub annotations are joined by the 13-character Broad compound
identifier. A compound representation is the normalized mean of its frozen
plate-level representation. Exact compound identity is excluded from the
candidate set.

- Primary relevance definition: two distinct compounds share at least one
  normalized mechanism-of-action annotation.
- Secondary relevance definition: they share at least one annotated protein
  target.
- An annotation is eligible only if it occurs in at least two distinct cohort
  compounds; a query is eligible only if at least one other compound is
  relevant.
- Primary metric: query-level mean average precision.
- Secondary metrics: top-1 and top-5 hit rates.
- Comparisons: shrinkage LDA against raw cosine, Hallmark ssGSEA, the original
  supervised-contrastive encoder, and the Filzen-style barcode.
- Uncertainty: 2,000 query-compound bootstrap draws for paired mAP differences.
- Null: 2,000 permutations of whole annotation sets among compounds, preserving
  annotation multiplicity per compound.

The endpoint is successful only if shrinkage LDA exceeds raw cosine for MoA
retrieval with a positive 95% paired-bootstrap interval and exceeds its own
permutation null at one-sided P < 0.05. Target retrieval is reported regardless
of outcome and cannot rescue a failed primary biological endpoint.

## New endpoint 2: multi-cell transfer

The frozen Phase I MCF7/6-hour representations are applied without refitting,
reweighting, or context-specific feature selection to GSE70138 at 10
micromolar and 24 hours. The five new cell lines were selected using metadata
only because they each contain at least 60 Phase-I-unseen compounds with six or
more independent plates:

- A375;
- A549;
- HA1E;
- HT29;
- PC3.

The already observed MCF7 cohort is a reference sixth context. Eligibility,
same-plate vehicle subtraction, and same-plate candidate exclusion match the
frozen external-validation contract.

Primary method: shrinkage LDA. Comparators are raw cosine, frozen PCA,
limma-weighted cosine, Hallmark ssGSEA, the original supervised-contrastive
encoder, its equal-weight LDA ensemble, and the Filzen-style barcode.

The primary statistic is the macro-average across the five new cell lines of
the within-context mAP difference between shrinkage LDA and the strongest
non-learned comparator. A stratified compound bootstrap (2,000 draws within
each cell line) supplies its 95% interval. The panel passes when the interval is
entirely above zero and the point difference is positive in at least four of
five new cell lines. Every context and method is reported.

## New endpoint 3: reproducibility spectrum and scaling

Only Phase I training and validation compounds are used to explain method
behavior. The within-compound covariance is estimated from deviations around
training-compound means with Ledoit-Wolf shrinkage. Between-compound covariance
is estimated from training-compound means. Generalized eigenvalues quantify
between-compound separation relative to within-compound noise. Report:

- the ordered reproducibility spectrum;
- participation-ratio effective dimensionality;
- the number of directions with generalized eigenvalue greater than one;
- cumulative discriminative mass captured by 8, 16, and 32 directions.

Scaling uses deterministic nested subsets of 20, 40, 60, 80, and 100 training
compounds and 3, 6, 9, and 12 plates per compound where available. Subsets are
selected by a fixed SHA-256 ordering. Shrinkage LDA and the two neural methods
are refitted at each point and evaluated only on the unchanged Phase I
validation compounds. This is explanatory, not new external confirmation.

## Multiplicity, reporting, and stop rules

- The endpoint hierarchy above is fixed; secondary analyses cannot relabel a
  failed primary endpoint as a success.
- No context, compound, method, or annotation category is removed because its
  result is unfavorable.
- Missing annotations are reported as missing, not treated as negative labels.
- All eligible-query counts and label prevalences accompany performance.
- Version 1 remains publishable even if every Version 2 endpoint is negative.
- Local expansion stops after these endpoints. Additional thresholds,
  architectures, or plots require a specific reviewer request.

## Collaboration boundary

The local project ends with this multi-context LINCS benchmark. A genuine
next-tier collaboration must add evidence not generated by the same L1000
programme: an independent transcriptomic platform, a cross-modality test such
as Cell Painting, prospective perturbation experiments, or wet-lab validation
of a locked biological-neighbour prediction. Reanalysis of another LINCS slice
alone does not satisfy this boundary.

## Primary sources defining the added comparators

- Filzen TM et al. *PLoS Computational Biology* 2017;13:e1005335.
  doi:10.1371/journal.pcbi.1005335.
- CLUE Connectopedia, "How are replicates collapsed into signatures?" (MODZ).
- Corsello SM et al. *Nature Medicine* 2017;23:405-408.
  doi:10.1038/nm.4306.
