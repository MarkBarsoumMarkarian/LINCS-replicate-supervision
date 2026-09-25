# LINCS replicate supervision benchmark

This repository accompanies the manuscript **“Shrinkage linear replicate supervision recovers perturbation identity and pharmacological relationships across LINCS contexts.”** It evaluates whether replicate identity can supervise representations that transfer across LINCS phases, plates, cell lines, and pharmacological labels.

The main result is methodological rather than therapeutic: at the tested scale of approximately 100 training compounds, shrinkage linear discriminant analysis transferred more reliably than the tested neural encoders. The analysis uses compound-disjoint splits, excludes same-plate retrieval candidates, freezes Phase I transformations before Phase II evaluation, and labels all post-freeze sensitivities explicitly.

## Main results

- Phase I locked test: the pre-specified neural and shrinkage-linear ensemble reached mAP 0.2021 versus 0.1126 for Hallmark ssGSEA.
- Frozen Phase II MCF7 transfer: shrinkage LDA reached mAP 0.4446.
- Five-cell extension: shrinkage LDA exceeded the strongest non-learned comparator in all five contexts; macro gain 0.0655, 95% interval 0.0485 to 0.0831.
- Mechanism retrieval: mAP 0.4052 versus 0.3234 for the locked raw-cosine comparator and 0.3320 for limma weighting. The post-freeze paired gain over limma was 0.0732, 95% interval 0.0119 to 0.1239.
- Target retrieval: mAP 0.3780; gain over Hallmark ssGSEA 0.0391, 95% interval 0.0026 to 0.0767.

These results do not establish therapeutic equivalence, clinical utility, or prospective biological validation.

## Repository map

- `manuscript/`: manuscript and supplementary appendix in DOCX, PDF, and Markdown.
- `figures/`: publication figures in PNG and PDF.
- `work/replicate_metric_learning/`: Phase I benchmark and Phase II validation code and contracts.
- `work/novelty_rescue_v2/`: multi-context, pharmacology, spectrum, scaling, and post-freeze sensitivity code.
- `work/v2_treatment_relative/derived/`: compact Phase I derived inputs.
- `outputs/replicate_metric_learning/`: frozen Phase I and Phase II results.
- `outputs/novelty_rescue_v2/`: extension results, checkpoints, and compact multi-context signatures.
- `submission/`: cover letter, highlights, AI disclosure, and human revision checklist.

## Verify the archived results

The quickest integrity check requires only Python's standard library:

```bash
python work/novelty_rescue_v2/verify_replay.py
python work/novelty_rescue_v2/verify_reviewer_replay.py
```

The first verification covers the 25 core extension artifacts. The second covers the two post-freeze sensitivity artifacts. Both compare the current files with frozen references byte for byte and report SHA-256 hashes.

See [`REPRODUCE.md`](REPRODUCE.md) for environment setup, the full execution order, and the distinction between replaying the analysis from included derived signatures and reconstructing signatures from the original GEO matrices.

## Public source data

- LINCS Phase I: GEO [GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742).
- LINCS Phase II: GEO [GSE70138](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70138).
- Drug mechanism and target annotations: Broad Drug Repurposing Hub, 24 March 2020 release.
- Hallmark gene sets: MSigDB Hallmark collection used by the locked analysis.

The large source GCTX matrices are not redistributed. Compact derived signatures needed for the reported extension are included so the analysis can be replayed without downloading the multi-gigabyte matrices.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). The version 1.0.0 archival release has the reserved DOI [10.5281/zenodo.22956320](https://doi.org/10.5281/zenodo.22956320), which will resolve after the Zenodo record is published.

## License

Code is released under the MIT License. Source datasets and external annotations remain subject to their original providers' terms. The manuscript and figures are provided for scholarly review and citation.
