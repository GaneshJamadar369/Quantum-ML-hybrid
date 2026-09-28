# AQUIRE-Med

Research preprocessing and classical baselines for contemporaneous **MI-pattern vs non-MI-pattern** classification from 10-second, 12-lead ECGs.

The production input is PTB-XL v1.0.3 at 100 Hz (`float32[12,1000]`). PTB-XL+ v1.0.1 is an aligned reference resource, not a second patient cohort. 12SL and Uni-G are benchmark-only because their extractors are commercial; local waveform features form the deployable branch.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[research,test]'
```

On Apple Silicon, XGBoost also requires `brew install libomp`.

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
4. Fit fold-local waveform normalizers with `python run_normalization.py --hdf5 "$AQUIRE_OUTPUT_ROOT/primary_development_100hz.h5"`.
5. Review the G0–G5 artifacts and change every pending gate in [PLAN.md](PLAN.md) only after its acceptance evidence exists.
6. Run `python run_classical_baselines.py --features ... --metadata ... --output ...` only after G0–G5 pass.

The Kaggle entrypoint is only a driver. It imports the same package path as the local CLI and contains no copy of label, QC, filtering or morphology code. Kaggle GPU is disabled for preprocessing.

## Output boundary

Primary and quarantine records are written separately to chunked HDF5. Each row stores the waveform, validity masks, identifiers, target, fold, source checksum, canonical tensor checksum and pipeline version. QC and morphology details are written to separate tables. Interrupted HDF5 runs resume from the `.tmp` file; completed artifacts are never overwritten implicitly.

## Scientific limits

This repository does not establish clinical deployment, future cardiovascular event prediction or quantum advantage. It implements the data and classical-baseline evidence needed before `z4/z8` or quantum experiments are allowed.

The later modeling prototype was independently audited on 2026-09-21. Read
[the research code audit](docs/research_code_audit_2026-09-21.md) before using
the classical, fusion, conformal, or QML results. The completed CPU benchmark
and its restricted interpretation are recorded in the
[Phase 6Q-B matched-kernel controls report](docs/phase6q_b_kernel_controls_result.md).
The gated path from the present checkpoint to a one-time holdout and external
validation is in the
[research completion roadmap](docs/research_completion_roadmap.md).

## Hackathon prototype

The implementation-ready UI/backend architecture, API contract, model-bundle
boundary and exact build order are defined in the
[SIH prototype plan](docs/sih_prototype_ui_backend_plan_2026-09-28.md).

The selected fast stack is React/Vite/TypeScript for the interface and FastAPI
for the Python inference service. The plan keeps the fixed parallel q4-VQC and
clinical-HGB routes, their logistic fusion, quality abstention and research
benchmarking intact. A frozen, checksummed inference bundle is the first build
gate; research notebooks are not loaded by the web service.

The platform vertical slice is implemented under `apps/`. It supports CSV,
JSON and paired-WFDB ZIP inspection, twelve-lead visualization, frozen quality
abstention, fixed-route prediction, model and benchmark endpoints, integrity
checks and Docker packaging. The executable bundle was exported and calibrated
on Kaggle, then reproduced locally through eight signed Fold-9 fixtures. Read
the [export and calibration result](docs/fold9_prototype_export_result_2026-09-28.md)
for the exact metrics and limits.

The service becomes ready only after the complete, checksummed bundle loads and
passes its startup golden test. Every accepted ECG executes both the q4-VQC and
clinical-HGB routes followed by the frozen fusion and calibrator. There is no
quantum-only or classical-only serving fallback.

Install and run it locally:

```bash
pip install -e '.[research,prototype]'
export AQUIRE_BUNDLE_ROOT="$PWD/prototype_bundle/current"
python scripts/check_prototype_golden.py "$AQUIRE_BUNDLE_ROOT"
uvicorn apps.api.aquire_api.main:app --port 8000

cd apps/web
npm install
npm run dev
```

`prototype_bundle/current` is intentionally ignored by Git because it contains
model weights. Copy or download the verified bundle there before startup. Or
start the complete containerized stack with `docker compose up -d --build`
after setting `AQUIRE_BUNDLE_ROOT` in a local `.env` file. The UI is served at
`http://localhost:8080` in Compose mode. Stop it with `docker compose down`.

Two de-identified, upload-ready Fold-9 examples are available in
[`demo_samples`](demo_samples): one MI-pattern reference and one non-MI
reference.

For another computer, follow the
[Windows Docker quickstart](docs/windows_docker_quickstart.md). It requires
only Git, Docker Desktop and the separately shared signed model-bundle ZIP.
