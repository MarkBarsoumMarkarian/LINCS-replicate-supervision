# Reproducing the LINCS replicate supervision benchmark

## Environment

The frozen environment is listed in `requirements.txt`. A clean installation can be created with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The recorded runs used deterministic random seeds, deterministic PyTorch algorithms, and four CPU threads. Exact byte-for-byte neural replay can still depend on using the pinned package versions and a compatible CPU build.

## Integrity check

Run this before recomputing anything:

```bash
python work/novelty_rescue_v2/verify_replay.py
python work/novelty_rescue_v2/verify_reviewer_replay.py
```

Expected outcome: 25 of 25 core artifacts and 2 of 2 post-freeze artifacts reproduce exactly against the frozen references.

## Replay from included derived signatures

The repository includes the leakage-safe Phase I plate-relative signatures, the fixed compound split, training-only transformations, model checkpoints, the Phase II MCF7 signatures, and five compact Phase II multi-context signature files.

From the repository root, run:

```bash
python work/replicate_metric_learning/train_and_evaluate.py
PYTHONPATH=work/novelty_rescue_v2 python work/novelty_rescue_v2/train_filzen_baseline.py
PYTHONPATH=work/novelty_rescue_v2 python work/novelty_rescue_v2/run_primary_upgrades.py
PYTHONPATH=work/novelty_rescue_v2 python work/novelty_rescue_v2/run_multicontext_transfer.py
PYTHONPATH=work/novelty_rescue_v2 python work/novelty_rescue_v2/run_reproducibility_spectrum.py
PYTHONPATH=work/novelty_rescue_v2 python work/novelty_rescue_v2/run_scaling_analysis.py
PYTHONPATH=work/novelty_rescue_v2 python work/novelty_rescue_v2/run_credibility_sensitivities.py
PYTHONPATH=work/novelty_rescue_v2 python work/novelty_rescue_v2/run_reviewer_responses.py
```

`run_multicontext_transfer.py` uses the archived compact signatures when the raw GSE70138 GCTX and metadata files are absent. If the raw files are present at the locations declared in `work/novelty_rescue_v2/common.py`, it reconstructs those signatures before evaluation.

The scripts write into `outputs/replicate_metric_learning/` and `outputs/novelty_rescue_v2/`. Run them in a clone or clean working tree if you want to preserve the archived outputs unchanged.

## Reconstructing from GEO source matrices

The large public matrices are intentionally excluded. Download the GSE92742 and GSE70138 Level 2 expression matrices and matching instance metadata from GEO, then place or link them at the paths declared near the top of the preparation scripts.

The reconstruction entry points are:

```bash
python work/v2_treatment_relative/prepare_plate_signatures.py
python work/v2_treatment_relative/score_ssgsea.py
python work/replicate_metric_learning/build_split.py
python work/replicate_metric_learning/external_gse70138/download_level2.py
python work/replicate_metric_learning/external_gse70138/run_external_validation.py
```

Read the analysis contracts before reconstruction. They define the cohort, split unit, same-plate exclusion, frozen transformations, primary endpoints, and interpretation limits.

## Bootstrap detail

Mechanism and target comparator intervals use 2,000 paired bootstrap draws over eligible query compounds. For each query, the method-specific average precisions are joined by `pert_id`; the paired difference is then resampled with replacement. Thus both LDA and its comparator vary together in every resample. The limma mechanism comparison uses seed `SEED + 402`.
