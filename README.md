# AQUIRE-Med

Research preprocessing and classical baselines for contemporaneous **MI-pattern vs non-MI-pattern** classification from 10-second, 12-lead ECGs.

The production input is PTB-XL v1.0.3 at 100 Hz (`float32[12,1000]`). PTB-XL+ v1.0.1 is an aligned reference resource, not a second patient cohort. 12SL and Uni-G are benchmark-only because their extractors are commercial; local waveform features form the deployable branch.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[research,test]'
```

Set the dataset roots explicitly:

```bash
export AQUIRE_PTBXL_ROOT=/path/to/ptb-xl/1.0.3
export AQUIRE_PTBXLP_ROOT=/path/to/ptb-xl-plus/1.0.1
export AQUIRE_OUTPUT_ROOT=/path/to/aquire-artifacts
```

## Gate order

1. Run `python run_gates.py --visual-audit` to verify the release, file checksums, all waveform pairs, phenotype, patient folds, cohorts and PTB-XL+ alignment.
2. Run `python run_pipeline.py --role development` to process folds 1–8. The normal path cannot read Fold 10.
3. Calibrate QC detectors from development-fold annotations with `python run_qc_calibration.py --metrics ... --spec configs/qc_detectors.json`.
4. Fit fold-local waveform normalizers from the development HDF5 artifact.
5. Review the G0–G5 artifacts and change every pending gate in [PLAN.md](PLAN.md) only after its acceptance evidence exists.
6. Run `python run_classical_baselines.py --features ... --metadata ... --output ...` only after G0–G5 pass.

The Kaggle entrypoint is only a driver. It imports the same package path as the local CLI and contains no copy of label, QC, filtering or morphology code. Kaggle GPU is disabled for preprocessing.

## Output boundary

Primary and quarantine records are written separately to chunked HDF5. Each row stores the waveform, validity masks, identifiers, target, fold, source checksum, canonical tensor checksum and pipeline version. QC and morphology details are written to separate tables. Interrupted HDF5 runs resume from the `.tmp` file; completed artifacts are never overwritten implicitly.

## Scientific limits

This repository does not establish clinical deployment, future cardiovascular event prediction or quantum advantage. It implements the data and classical-baseline evidence needed before `z4/z8` or quantum experiments are allowed.
