# Shrinkage linear replicate supervision recovers perturbation identity and pharmacological relationships across LINCS contexts

Mark Barsoum Markarian

Independent Researcher, Beirut, Lebanon

ORCID 0009-0006-1240-9534

## Abstract

Replicate identity can supervise denoised perturbational representations, but neural metric learning may not transfer better than classical shrinkage at modest training scale. We benchmarked LINCS L1000 profiles with compound-disjoint splits and query-plate exclusion. In Phase I, a pre-specified neural and shrinkage-linear ensemble achieved test mean average precision (mAP) of 0.2021 versus 0.1126 for Hallmark ssGSEA. Frozen transfer to 81 Phase II MCF7 compounds identified shrinkage linear discriminant analysis as strongest at 0.4446. After that observation, we designated shrinkage LDA as primary in a locked extension across five cell lines containing 390 unseen compound-contexts and 2,593 profiles. It exceeded the strongest non-learned comparator in all five; the macro gain was 0.0655 with a 95% interval of 0.0485 to 0.0831. Among 59 eligible queries from 74 mechanism-annotated compounds, LDA achieved mechanism mAP of 0.4052 versus 0.3234 for the locked raw-cosine comparator (0.3320 for limma weighting): a gain over raw cosine of 0.0818, 95% interval 0.0235 to 0.1355, and permutation P = 0.00050. Among 62 eligible queries from 73 target-annotated compounds, LDA exceeded Hallmark ssGSEA by 0.0391 with a 95% interval of 0.0026 to 0.0767. A reconstructed perturbation-barcode network and supervised-contrastive encoder transferred less reliably. Spectrum effective dimension was 27.1 in training and 24.3 in 68 held-out compounds. LDA led the reduced-schedule 20-point grid and all five longer-schedule points. At this scale, replicate supervision carried transferable identity and pharmacological structure most reliably through regularized linear geometry.

Keywords: LINCS; L1000; perturbational transcriptomics; replicate supervision; shrinkage linear discriminant analysis; metric learning; mechanism of action; reproducibility

## 1 Introduction

Perturbational transcriptomics is often used as a molecular fingerprint. The Connectivity Map established that gene-expression signatures can connect compounds, genetic perturbations, and disease states through pattern matching [1]. LINCS and the L1000 assay expanded that idea to more than one million profiles by directly measuring 978 landmark transcripts [2]. These resources support mechanism studies, drug discovery, and computational benchmarking, but profiles from the same intervention can still differ across plates while unrelated interventions can appear similar because of technical structure or broad stress responses.

Standard similarity analyses begin with a fixed representation. Cosine similarity treats observed gene directions as equally available for retrieval. Principal components compress dominant variance without knowing which directions reproduce. Gene-set aggregation introduces biological priors and can improve robustness, while empirical Bayes and covariance shrinkage stabilize estimates in high-dimensional expression data [3-7]. These approaches solve related but distinct problems.

Replicate-supervised transcriptomic representation learning is not new. Filzen et al. trained a Siamese network to distinguish biological replicates from non-replicates, compressed 978 L1000 measurements into a 100-dimensional perturbation barcode, tested target and structural relationships, applied the architecture to GSE70138, and prospectively validated MAPK-pathway predictions [8]. That work established the core principle that replicate identity can supervise a biologically useful metric. More recent supervised-contrastive methods provide a modern objective for the same broad idea [9].

The unresolved issue is not whether replicate supervision can work, but which implementation transfers under a strict distribution shift. Deep encoders are flexible, yet small numbers of training identities and strong covariance structure may favour regularized linear estimators. Published demonstrations also leave room for a benchmark in which compounds, not profiles, are held out; same-plate candidates are excluded; preprocessing and model parameters are frozen across LINCS phases; classical and neural models are compared directly; and biological relationships are evaluated among compounds unseen during training.

We therefore separated three questions. First, can replicate supervision improve retrieval of unseen perturbation identities across plates and LINCS phases? Second, does the same frozen geometry generalize from MCF7 at 6 hours to multiple Phase II cell types at 24 hours? Third, does it recover shared mechanisms and targets between distinct compounds rather than merely recognizing repeat measurements? We also estimated a reproducibility spectrum to examine why a shrinkage-linear representation might outperform the tested neural alternatives. The contribution is a strict transfer benchmark and a model-behaviour result, not a claim to have invented replicate-supervised transcriptomic metric learning.

[[FIGURE:03_Figures/Figure_1_Study_Design.png|Figure 1. Layered study design. The original compound-disjoint Phase I benchmark and frozen MCF7 Phase II transfer were followed by a separately locked extension comprising direct-prior baselines, five new cell lines, pharmacological-neighbour retrieval, and spectrum and scaling analyses. No Phase II label or expression outcome was used to fit a representation.|Flow diagram of the Phase I training and locked test, the Phase II MCF7 reference, and the locked multi-cell and pharmacology endpoints.]]

## 2 Materials and methods

### 2.1 Data sources and plate-relative signatures

LINCS Phase I data were obtained from GEO accession GSE92742 and Phase II data from GSE70138 [2,10]. Level 2 deconvoluted expression values for the 978 directly measured landmark genes were used. The training context was MCF7, compound treatment at 10 micromolar, and 6 hours. Compounds required at least 12 RNA plates, producing 2,755 compound-by-plate profiles from 168 compounds.

Values were required to be finite and positive and were log2 transformed. Within each RNA plate, treatment wells for a compound were averaged and the gene-wise median across same-plate vehicle wells was subtracted. One sample therefore represented one compound on one plate. Phase II instances and genes were aligned to the GCTX matrix by exact identifiers before the frozen Phase I gene order was applied.

### 2.2 Frozen split and preprocessing

All profiles from a compound remained in one split. Sixteen compounds encountered during development were assigned to training. Remaining compounds were ordered by a fixed SHA-256 rule to produce 100 training, 34 validation, and 34 test compounds. No expression value, activity score, mechanism annotation, or outcome entered allocation. Per-gene means and standard deviations were estimated only from training profiles and applied unchanged to validation, test, and Phase II data.

The first-stage external cohort comprised MCF7 treatments at 10 micromolar and 24 hours. Compounds had to be absent from all Phase I plate-relative signatures and represented on at least six Phase II plates with same-plate vehicle controls. Metadata-only eligibility yielded 81 compounds and 569 profiles. This result was known before the extension and is reported as prior evidence rather than a newly pre-specified endpoint.

### 2.3 Learned representations and fixed baselines

The supervised-contrastive encoder mapped 978 genes through one 128-unit GELU and layer-normalized hidden layer to a 32-dimensional L2-normalized embedding. Balanced batches sampled 16 compounds and four plates per compound. AdamW optimization, temperature 0.10, 10% dropout, maximum 120 epochs, and patience 18 were chosen during exploratory development; the 16 compounds exposed during that work were confined to training, and all values were fixed before the compound-disjoint validation, test, and transfer analyses. Validation mean average precision selected the stopping epoch and the learned representation; test and Phase II outcomes did not tune the network [9].

Shrinkage LDA used the eigenvalue solver, automatic covariance shrinkage, and 32 components. The neural and linear cosine-similarity matrices were also averaged to form the pre-specified first-stage ensemble. Fixed comparators were training-standardized raw cosine, 64-component PCA cosine, limma precision-weighted cosine, and 50-set Hallmark ssGSEA cosine [3-7].

### 2.4 Direct published comparator and MODZ consensus

We reconstructed the Filzen et al. architecture in PyTorch using 978 inputs, noisy sigmoid layers of 400 and 100 units, Gaussian pre-activation noise with variance 0.25, 50% dropout, L1 regularization, squared-Euclidean contrastive margin 5, and RMSprop optimization [8]. Positive pairs came from different plates of one training compound and negatives from different compounds at a 1:2 ratio. Phase I validation retrieval selected the stopping epoch. Continuous activations and binary barcodes thresholded at 0.5 were evaluated. Because the proprietary original corpus and complete optimization schedule are unavailable, this is a faithful modern reimplementation rather than an exact reproduction.

We also implemented a task-matched moderated-z-score consensus. Replicate weights were proportional to the sum of pairwise Spearman correlations after flooring correlations at 0.01 and normalizing weights to one, following the CMap replicate-collapse principle [2]. For each query, its plate was removed from every candidate consensus. This adaptation operates on our plate-relative Level 2 signatures and is not labelled official CMap Level 5 data. Query-to-compound consensus retrieval used mean reciprocal rank, top-one accuracy, and top-five hit rate.

### 2.5 Perturbation-identity retrieval

Every profile served as a query. The query itself and all candidates from its RNA plate were excluded. Relevant candidates shared its perturbation identifier. Mean average precision was the primary profile-level metric; nearest-neighbour accuracy, top-five hit rate, and pair AUROC were secondary. Uncertainty used 2,000 compound-cluster bootstrap draws so profiles from one compound were resampled together [11].

The untouched Phase I test was opened after model selection. Frozen Phase I transformations were then transferred to Phase II without re-estimation. The first-stage external ensemble endpoint and its sensitivity analyses followed the original frozen contract. Shrinkage LDA became the extension's primary method after it was observed to lead the first MCF7 Phase II transfer. The new cell-line and annotation endpoints were then locked before their outcomes were calculated. They therefore confirm transfer of the already observed LDA advantage to new contexts and labels, but they do not prospectively validate the choice of LDA itself.

### 2.6 Multi-cell transfer

The frozen Phase I MCF7/6-hour representations were applied without retraining, reweighting, or context-specific feature selection to five new GSE70138 cell lines at 10 micromolar and 24 hours: A375, A549, HA1E, HT29, and PC3. Each compound was absent from the Phase I signature collection and appeared on at least six Phase II plates. The panel contained 390 compound-contexts and 2,593 profiles: 70 and 463 in A375, 102 and 694 in A549, 78 and 506 in HA1E, 77 and 507 in HT29, and 63 and 423 in PC3.

The primary statistic was the macro-average across cell lines of the within-context mean-average-precision difference between shrinkage LDA and the strongest non-learned comparator. A stratified compound bootstrap sampled compounds within each cell line for 2,000 draws. Because compounds overlap across cell lines, a conservative sensitivity resampled the 158 unique perturbation identifiers as global clusters and retained all of a sampled compound's cell contexts together. The endpoint required a wholly positive 95% interval and positive point estimates in at least four of five contexts.

### 2.7 Mechanism and target retrieval

Drug Repurposing Hub sample identifiers were collapsed to the 13-character Broad perturbation identifier and joined to curated drug-level mechanism-of-action and protein-target annotations from the 24 March 2020 release [12]. Among the 81-compound external MCF7 cohort, 74 compounds had mechanism annotations and 73 had target annotations. Plate embeddings were normalized, averaged within compound, and normalized again. Exact compound identity was excluded from the candidate set.

Two distinct compounds were relevant if they shared at least one normalized mechanism annotation for the primary endpoint or protein target for the secondary endpoint. An annotation term had to occur in at least two cohort compounds, and a query required at least one distinct relevant compound. Missing annotations were excluded rather than treated as negative. The locked mechanism comparison used raw cosine. Post-freeze sensitivities used the strongest non-learned comparator for each endpoint: limma-weighted cosine for mechanisms and Hallmark ssGSEA for targets. Paired method differences used 2,000 query-compound bootstrap draws. The null permuted whole annotation sets among compounds 2,000 times, preserving each set's multiplicity.

### 2.8 Reproducibility spectrum and scaling

Using Phase I training profiles only, within-compound covariance was estimated from deviations around compound means with Ledoit-Wolf shrinkage [6]. Between-compound covariance was estimated from the 100 training-compound means. Generalized eigenvalues of between-compound relative to within-compound covariance defined the reproducibility spectrum. We calculated participation-ratio effective dimension and cumulative discriminative mass. The calculation was repeated independently in the 68 validation and test compounds; positive-direction counts were treated as class-count rank limits rather than biological findings.

Scaling used deterministic nested subsets of 20, 40, 60, 80, and 100 training compounds and 3, 6, 9, and 12 plates per compound. Shrinkage LDA, the supervised-contrastive encoder, and the Filzen-style network were refitted at all 20 grid points and evaluated on the unchanged Phase I validation compounds. To make the complete grid computationally bounded, its neural schedules used maximum 25 epochs, patience 5, and six batches per epoch. Because those schedules were shorter than the main analyses, a post-freeze audit repeated the complete 20-compound row and the 100-compound, 12-plate point. The audit used the main contrastive schedule of maximum 120 epochs, patience 18, and at least 12 batches per epoch, and the main Filzen schedule of maximum 80 epochs, patience 14, and 16 batches per epoch. This analysis describes sample-efficiency behaviour within the observed range and is not an independent external confirmation.

### 2.9 Reproducibility and software

Analyses used Python, NumPy, pandas, SciPy, h5py, scikit-learn, PyTorch, GSEApy, and Matplotlib [13]. Frozen contracts, exact split records, model checkpoints, result tables, source hashes, and executable scripts accompany the manuscript. The original analysis package was preserved unchanged before the extension was created.

The extension contract froze the primary multi-cell, mechanism, spectrum, and scaling endpoints before their outcomes were calculated. Held-out spectrum estimation, the target-comparator interval, the limma mechanism-comparator sensitivity, the global compound-cluster bootstrap, the LDA consensus comparison, and the schedule audit were added later as explicitly post-freeze credibility sensitivities. After the first set was fixed, a complete replay regenerated all 25 core CSV and JSON artifacts byte for byte. The two post-freeze artifacts covering the schedule audit and the added sensitivities were then reproduced byte for byte in a separate replay. Both verification records and per-file SHA-256 hashes accompany the manuscript.

## 3 Results

### 3.1 Compound-disjoint Phase I and MCF7 cross-phase transfer

Validation selected the neural and shrinkage-LDA ensemble before the Phase I test was opened. In 34 unseen test compounds and 475 profiles, ensemble mean average precision was 0.2021 versus 0.1126 for Hallmark ssGSEA. The paired improvement was 0.0895 with a compound-bootstrap 95% interval of 0.0343 to 0.1606. Shrinkage LDA alone reached 0.1978, the neural encoder 0.1707, raw cosine 0.0815, and PCA 0.0779.

In the previously locked MCF7 Phase II cohort, the ensemble transferred at mean average precision 0.3931 versus 0.2818 for Hallmark ssGSEA, with a paired improvement interval of 0.0749 to 0.1499. Shrinkage LDA was strongest at 0.4446, whereas the neural encoder reached 0.2620. The result established that replicate supervision transferred across phases, while the ablation identified shrinkage LDA as the more reliable component.

[[FIGURE:03_Figures/Figure_2_Core_Identity_Retrieval.png|Figure 2. Core perturbation-identity retrieval. Panel A shows the locked Phase I test. Panel B shows frozen transfer to the previously specified GSE70138 MCF7 cohort. Shrinkage LDA nearly matched the selected ensemble internally and was strongest externally.|Two horizontal bar charts comparing method mean average precision in the Phase I test and Phase II MCF7 transfer.]]

### 3.2 Direct-prior and consensus comparators

The faithful Filzen-style continuous representation reached Phase I validation mean average precision 0.0806 and test mean average precision 0.0730. The thresholded barcode collapsed to three unique vectors in each split; a post-freeze tie-aware calculation gave test mean average precision 0.0510. These values were below shrinkage LDA at 0.1978 and the supervised-contrastive encoder at 0.1707. The result does not invalidate the original report, which used a larger and different corpus and endpoint. It shows that the published architecture did not transfer advantageously under this compound-disjoint dataset and preprocessing regime.

MODZ weighting improved Phase I query-to-compound mean reciprocal rank from 0.2337 for an unweighted mean consensus to 0.2568. Shrinkage LDA evaluated on the same query-to-compound endpoint reached 0.4203. In external MCF7, MODZ mean reciprocal rank was 0.4610 versus 0.4747 for the unweighted consensus and 0.7356 for shrinkage LDA. Correlation-weighted replicate collapse was therefore useful internally but did not account for the stronger discriminant geometry.

### 3.3 Frozen transfer across five new cell lines

Shrinkage LDA was the leading identity-retrieval method in every new cell line. Its mean average precision was 0.3942 in A375, 0.3441 in A549, 0.4223 in HA1E, 0.3328 in HT29, and 0.3924 in PC3. Hallmark ssGSEA was the strongest non-learned comparator in all five contexts, with corresponding values of 0.3441, 0.2682, 0.3722, 0.2489, and 0.3248.

The context-specific shrinkage-LDA improvements were 0.0502, 0.0759, 0.0501, 0.0839, and 0.0676. Every context-specific compound-bootstrap interval was positive. The macro improvement was 0.0655 with a stratified 95% interval of 0.0485 to 0.0831, satisfying both locked success criteria. The global compound-cluster sensitivity, which retained repeated compounds across cell lines as one cluster, remained positive at 0.0444 to 0.0888. The neural encoder and the Filzen-style network were below shrinkage LDA in every context.

| Endpoint | Eligible units | LDA mAP | Comparator | Comparator mAP | Paired gain | 95 percent interval |
|---|---:|---:|---|---:|---:|---|
| A375 identity | 70 compounds | 0.3942 | Hallmark ssGSEA | 0.3441 | 0.0502 | 0.0105 to 0.0930 |
| A549 identity | 102 compounds | 0.3441 | Hallmark ssGSEA | 0.2682 | 0.0759 | 0.0503 to 0.1048 |
| HA1E identity | 78 compounds | 0.4223 | Hallmark ssGSEA | 0.3722 | 0.0501 | 0.0142 to 0.0863 |
| HT29 identity | 77 compounds | 0.3328 | Hallmark ssGSEA | 0.2489 | 0.0839 | 0.0424 to 0.1219 |
| PC3 identity | 63 compounds | 0.3924 | Hallmark ssGSEA | 0.3248 | 0.0676 | 0.0244 to 0.1105 |
| Mechanism retrieval | 59 of 74 annotated | 0.4052 | Raw cosine | 0.3234 | 0.0818 | 0.0235 to 0.1355 |
| Mechanism retrieval post-freeze | 59 of 74 annotated | 0.4052 | Limma weighted cosine | 0.3320 | 0.0732 | 0.0119 to 0.1239 |
| Target retrieval | 62 of 73 annotated | 0.3780 | Hallmark ssGSEA | 0.3388 | 0.0391 | 0.0026 to 0.0767 |

Table 1 summarizes the new extension endpoints. Full method rankings, the replicate-threshold and 3-hour analyses, the seen-compound overlap control, the permutation nulls, and the schedule audit are reported in Tables S2-S15 and the electronic tables; query-level annotation definitions are in the Supplementary methods.

[[FIGURE:03_Figures/Figure_3_Multicontext_Transfer.png|Figure 3. Frozen multi-cell transfer. Panel A compares mean average precision across five new GSE70138 cell lines. Panel B shows shrinkage-LDA improvement over the strongest non-learned comparator with compound-bootstrap intervals. No model was refitted on Phase II.|Multi-panel chart of method performance and paired improvements for A375, A549, HA1E, HT29, and PC3.]]

### 3.4 Recovery of pharmacological relationships

Fifty-nine mechanism-annotated compounds had at least one distinct relevant neighbour under the locked rule. Shrinkage LDA achieved mechanism-of-action mean average precision 0.4052, nearest-neighbour accuracy 0.4407, and top-five hit rate 0.6949. Raw cosine reached 0.3234, limma-weighted cosine 0.3320, Hallmark ssGSEA 0.3195, the supervised-contrastive encoder 0.3247, PCA cosine 0.2710, the Filzen continuous representation 0.2389, and its binary barcode 0.1594.

The shrinkage-LDA improvement over raw cosine was 0.0818 with a paired 95% interval of 0.0235 to 0.1355. Its permutation-null mean was 0.1335 with a 95% range of 0.1085 to 0.1659; none of 2,000 permutations reached the observed value after plus-one correction (P = 0.00050). The primary biological endpoint passed. Limma-weighted cosine was the strongest non-learned mechanism comparator, so the post-freeze gain over it was 0.0732 with a paired 95% interval of 0.0119 to 0.1239. The locked comparison used raw cosine.

Target retrieval involved 62 eligible queries among 73 annotated compounds. Shrinkage LDA reached mean average precision 0.3780, compared with 0.3388 for Hallmark ssGSEA, 0.3008 for limma weighting, 0.2956 for the contrastive encoder, and 0.2947 for raw cosine. Its paired gain over the locked raw-cosine comparator was 0.0833 with a 95% interval of 0.0469 to 0.1215. Against Hallmark ssGSEA, the strongest non-learned target comparator, the post-freeze gain was 0.0391 with a 95% interval of 0.0026 to 0.0767. Raw cosine already recovered substantial mechanism signal at 0.3234, so the learned improvement is incremental rather than the source of all pharmacological information. These endpoints show enrichment for shared pharmacology among distinct compounds; they do not establish complete mechanism identity, causal target engagement, or therapeutic interchangeability.

[[FIGURE:03_Figures/Figure_4_Pharmacology_Retrieval.png|Figure 4. Pharmacological-neighbour retrieval among compounds unseen during training. Panel A shows mechanism-of-action mean average precision. Panel B shows target mean average precision. Panel C compares observed shrinkage-LDA MoA retrieval with the annotation-permutation null.|Bar charts for mechanism and target retrieval plus a permutation distribution for the primary mechanism endpoint.]]

### 3.5 Reproducibility spectrum and sample efficiency

The training-compound spectrum had participation-ratio effective dimension 27.1, and its leading 8, 16, and 32 directions captured 45.9%, 59.9%, and 74.0% of discriminative mass. An independent calculation in the 68 validation and test compounds produced effective dimension 24.3 and placed 78.6% of mass in its leading 32 directions. The similarity supports, but does not prove, a low-dimensional explanation for the linear model's performance.

In the computationally bounded 20-point grid, shrinkage LDA ranked first at every point. With only 20 training compounds, its validation mean average precision increased from 0.100 at three plates to 0.147 at 12 plates. With 100 compounds, it increased from 0.186 to 0.204. At the full 100-compound, 12-plate point, the reduced-schedule contrastive and Filzen models reached 0.162 and 0.053. With their main schedules, they improved to 0.168 and 0.080, respectively, while LDA remained at 0.204. LDA also remained first after main-schedule retraining at all four points in the 20-compound row. Using all available training plates and each main schedule, validation mean average precision was 0.214 for LDA, 0.185 for the contrastive encoder, and 0.081 for Filzen. The longer schedules therefore narrowed parts of the neural gap without reversing the linear advantage at any audited point.

[[FIGURE:03_Figures/Figure_5_Spectrum_and_Scaling.png|Figure 5. Reproducibility spectrum and scaling. Panel A compares training and held-out generalized spectra. Panel B shows shrinkage-LDA validation mean average precision across the grid. Panel C compares the full 100-compound, 12-plate point before and after the main-schedule neural audit.|Training and held-out spectrum plot, LDA scaling heatmap, and schedule-audited full-point comparison consistent with a shrinkage-linear advantage.]]

## 4 Discussion

The central result is not that replicate supervision is novel or that a neural encoder wins. Filzen et al. established the former, and our data contradict the latter under this specific benchmark [8]. Under compound-disjoint evaluation, same-plate exclusion, frozen cross-phase transfer, and five new cell contexts, shrinkage LDA was the most reliable representation. It also recovered shared mechanisms and targets between distinct unseen compounds. The contribution is therefore a boundary condition and a practical lesson: with approximately 100 training compounds in this L1000 regime, regularized linear discrimination can outperform the tested neural metrics (a one-hidden-layer contrastive encoder and a two-layer barcode network).

Several observations support that interpretation. Effective dimension was 27.1 in training compounds and 24.3 in held-out compounds, close to the pre-specified 32-dimensional LDA embedding. Shrinkage LDA led the computationally bounded grid and remained first in the longer-schedule audit of the full point and entire 20-compound row; the neural scores improved, so the reduced schedule had understated them, but did not determine the ordering at the audited points. The result is consistent with high-dimensional covariance estimation benefiting from shrinkage when the number of labelled identities is modest [6,7]. It does not establish why LDA wins causally and should not be generalized beyond the tested architectures or 20-100-compound range.

The biological-neighbour endpoint materially extends identity retrieval. Exact compound identity was excluded, annotations were used only after representations were frozen, and the observed mechanism score was well outside its permutation distribution. Nevertheless, annotations are incomplete and polypharmacology is common [12]. Shared labels provide a useful external reference, not exhaustive ground truth. A transcriptional neighbour can reflect convergent downstream biology without sharing every direct target, and shared target annotation does not guarantee equal potency, selectivity, or efficacy.

The multi-cell panel addresses the largest scope limitation of the first-stage study. A representation trained only in MCF7 at 6 hours retained an advantage in A375, A549, HA1E, HT29, and PC3 at 24 hours. These contexts span distinct tissue origins, yet all remain within the LINCS programme and L1000 platform. The experiment demonstrates context transfer inside one assay ecosystem, not platform independence.

The direct-prior comparison also clarifies what the work adds. The Filzen reconstruction performed poorly here, but the comparison is not a historical re-adjudication. The original model used a larger proprietary corpus, a different normalization and pairing regime, and additional biological analyses [8]. Our result instead asks whether that published recipe succeeds under the present locked split and transfer design. Its weakness, alongside the stronger modern contrastive encoder and still stronger LDA, makes the classical-versus-neural finding more credible.

MODZ likewise solves a different problem. Replicate collapse can improve a consensus when multiple profiles already exist, as it did in the Phase I query-to-compound endpoint. It does not by itself learn which multivariate directions distinguish compounds across the training collection, and its external advantage was not consistent. Reporting it prevents the learned representation from being compared only with weak raw-profile baselines.

The study remains limited by retrospective public data, one molecular platform, and no prospective experiment. The Phase I and Phase II cohorts differ in time, context composition, and candidate-set size, so absolute scores should be interpreted within cohort. The biological endpoint uses curated annotations rather than an orthogonal assay. The faithful barcode reconstruction cannot be an exact reproduction without the original proprietary corpus and full training schedule; the released legacy implementation was not run because its Theano and Pylearn2 environment and demonstration preprocessing are not equivalent to the present locked pipeline. Finally, scaling analysis used the Phase I validation set repeatedly and is explanatory rather than confirmatory.

These limits define the next experiment. A genuinely independent extension should freeze this representation and evaluate a different transcriptomic platform, a cross-modality resource such as Cell Painting, or a prospective perturbation panel. Wet-lab testing of a locked pharmacological-neighbour prediction would offer the strongest validation. Additional LINCS thresholds or architectures would add detail but would not provide the same evidentiary step.

## 5 Conclusion

Replicate supervision identified a transferable perturbational geometry, but classical shrinkage was its most reliable implementation at the tested scale. A 32-dimensional shrinkage-LDA representation improved identity retrieval across LINCS phases and five new cell lines, recovered shared mechanisms and targets among distinct unseen compounds, and led the tested neural alternatives in the reduced-schedule grid and longer-schedule audit. The evidence supports reproducibility-aware pharmacological retrieval within L1000. Independent-platform or prospective experimental validation is the appropriate next step before broader biological claims.

## Data availability

Source data are publicly available from NCBI GEO under accessions GSE92742 and GSE70138. Drug mechanism and target annotations are available from the Broad Drug Repurposing Hub (24 March 2020 release). The accompanying package records exact filenames, identifiers, eligibility rules, and hashes. Large source GCTX matrices are not redistributed.

## Code availability

Analysis scripts, frozen contracts, environment specifications, model checkpoints, compact derived signatures, result tables, and artifact hashes accompany this manuscript. A public repository URL and archival DOI should be inserted after repository publication.

## Funding

This research received no external funding.

## Competing interests

The author declares no competing interests.

## Author contributions

Mark Barsoum Markarian: Conceptualization, methodology, software, validation, formal analysis, investigation, data curation, visualization, writing, and project administration.

## Acknowledgements

The author thanks the LINCS Center for Transcriptomics, the Connectivity Map team, and the investigators who deposited GSE92742 and GSE70138 in GEO.

## Declaration of generative AI and AI assisted technologies

During preparation, the author used OpenAI Codex, ChatGPT, Anthropic Claude, and Google Gemini to assist with code drafting, workflow critique, literature discovery, and manuscript language. The author reviewed the generated material, inspected the analysis outputs, revised the text, and takes full responsibility for the manuscript.

## References

1. Lamb J, Crawford ED, Peck D, et al. The Connectivity Map: using gene-expression signatures to connect small molecules, genes, and disease. Science. 2006;313:1929-1935. doi:10.1126/science.1132939.

2. Subramanian A, Narayan R, Corsello SM, et al. A next generation Connectivity Map: L1000 platform and the first 1,000,000 profiles. Cell. 2017;171:1437-1452.e17. doi:10.1016/j.cell.2017.10.049.

3. Hänzelmann S, Castelo R, Guinney J. GSVA: gene set variation analysis for microarray and RNA-seq data. BMC Bioinformatics. 2013;14:7. doi:10.1186/1471-2105-14-7.

4. Liberzon A, Birger C, Thorvaldsdóttir H, Ghandi M, Mesirov JP, Tamayo P. The Molecular Signatures Database Hallmark gene set collection. Cell Systems. 2015;1:417-425. doi:10.1016/j.cels.2015.12.004.

5. Ritchie ME, Phipson B, Wu D, et al. limma powers differential expression analyses for RNA-sequencing and microarray studies. Nucleic Acids Research. 2015;43:e47. doi:10.1093/nar/gkv007.

6. Ledoit O, Wolf M. A well-conditioned estimator for large-dimensional covariance matrices. Journal of Multivariate Analysis. 2004;88:365-411. doi:10.1016/S0047-259X(03)00096-4.

7. Schäfer J, Strimmer K. A shrinkage approach to large-scale covariance matrix estimation and implications for functional genomics. Statistical Applications in Genetics and Molecular Biology. 2005;4:Article 32. doi:10.2202/1544-6115.1175.

8. Filzen TM, Kutchukian PS, Hermes JD, Li J, Tudor M. Representing high throughput expression profiles via perturbation barcodes reveals compound targets. PLoS Computational Biology. 2017;13:e1005335. doi:10.1371/journal.pcbi.1005335.

9. Khosla P, Teterwak P, Wang C, et al. Supervised contrastive learning. Advances in Neural Information Processing Systems. 2020;33:18661-18673.

10. Edgar R, Domrachev M, Lash AE. Gene Expression Omnibus: NCBI gene expression and hybridization array data repository. Nucleic Acids Research. 2002;30:207-210. doi:10.1093/nar/30.1.207.

11. Efron B, Tibshirani RJ. An Introduction to the Bootstrap. New York: Chapman and Hall; 1993. doi:10.1007/978-1-4899-4541-9.

12. Corsello SM, Bittker JA, Liu Z, et al. The Drug Repurposing Hub: a next-generation drug library and information resource. Nature Medicine. 2017;23:405-408. doi:10.1038/nm.4306.

13. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. Journal of Machine Learning Research. 2011;12:2825-2830.
