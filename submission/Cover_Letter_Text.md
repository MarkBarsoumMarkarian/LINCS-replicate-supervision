# Cover letter

Dear Editor,

Please consider my manuscript, **Shrinkage linear replicate supervision recovers perturbation identity and pharmacological relationships across LINCS contexts**, for publication as a methods and benchmarking article.

The study addresses a practical question in perturbational transcriptomics: when replicate identity supervises a retrieval representation, does a neural metric learner or a classical shrinkage estimator transfer more reliably? The core idea of replicate-supervised L1000 representation learning was established by Filzen and colleagues in 2017, which I cite and reconstruct directly. My contribution is a stricter compound-disjoint and no-retraining transfer benchmark that identifies a different practical conclusion.

Using GSE92742 for training and GSE70138 for frozen transfer, shrinkage linear discriminant analysis outperformed raw expression, PCA, limma weighting, Hallmark ssGSEA, a modern supervised-contrastive encoder, and a faithful reconstruction of the published perturbation-barcode network. The result held in five new cell lines comprising 390 unseen compound-contexts and 2,593 plate-level profiles. The same frozen geometry also improved retrieval of shared mechanisms of action and protein targets between distinct unseen compounds. Training and held-out generalized covariance analyses, together with a 20-point scaling grid, were consistent with a sample-efficiency advantage for shrinkage LDA. A longer-schedule audit improved both neural comparators but left LDA first at the full 100-compound point and throughout the 20-compound row.

The manuscript is deliberately bounded. It does not claim invention of replicate supervision, therapeutic equivalence, platform-independent validation, or prospective biological confirmation. It presents an honest benchmark showing that a well-regularized classical model can be more transferable than the tested one-hidden-layer neural alternatives at a scale of approximately 100 training compounds.

All data are public. The submission package contains frozen analysis contracts, exact compound splits, executable code, compact derived data, model checkpoints, query-level results, and artifact hashes. The 25-artifact core replay and a separate two-artifact reviewer-response replay were both byte exact. The work received no external funding, and the author declares no competing interests.

Thank you for your consideration.

Sincerely,

Mark Barsoum Markarian

Independent Researcher

Beirut, Lebanon
