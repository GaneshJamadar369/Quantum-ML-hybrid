# SIH 26139 fast prototype: UI and backend implementation plan

**Date:** 2026-09-28  
**Decision:** GO for a hackathon prototype; keep all outputs labelled as
`MI-pattern screening` and all performance labelled as development evidence
until the sealed evaluation is complete.

## 1. Product boundary

The prototype accepts one ten-second, 12-lead ECG and returns a calibrated
MI-pattern screening result. It is a research decision-support demonstration,
not an acute-MI diagnosis, future-event predictor or clinical device.

Every eligible ECG follows both fixed branches:

```text
                         Transformer -> h128 -> PLS-q4 -> 4-qubit VQC -> sQ
ECG -> validation/QC -> <                                                   > -> logistic fusion -> calibration -> P(MI pattern)
                         106 local morphology features -> HGB ----------> sC
```

There is no model-selection gate, residual target or dynamic routing. The
quantum-only and overall-classical models appear only on the research dashboard
as benchmark comparators. The patient-facing result is the hybrid fused output.

## 2. Selected stack

| Layer | Choice | Reason |
|---|---|---|
| Web UI | React 19, Vite, TypeScript | Very fast local development and a polished single-page demo without a server-rendering dependency |
| Styling | Tailwind CSS and a small local component library | Fast responsive layout; no remote UI dependency during judging |
| ECG charts | Plotly.js | Reliable synchronized zoom and twelve-lead waveform plots |
| API | FastAPI, Pydantic v2, Uvicorn | Reuses the Python scientific pipeline and generates an OpenAPI contract automatically |
| Inference | PyTorch plus scikit-learn on CPU | The Transformer and exact four-qubit statevector are small enough for a local demo; no QPU queue in the critical path |
| Metadata | SQLite | Stores non-identifying audit events, model version and timing without operating a database service |
| Heavy files | Ephemeral local directory | Uploaded ECGs are deleted after inference unless an explicit demo-retention flag is enabled |
| Packaging | Docker Compose | One command starts the API and UI; each service remains independently testable |
| Tests | pytest, Vitest and Playwright | Covers scientific adapter, API contract and one browser-level golden path |

Redis, Celery, Kubernetes and a separate feature store are intentionally absent
from the hackathon build. Batch research jobs remain Kaggle/offline workflows.
They can be added after the synchronous single-ECG path is stable.

## 3. Required frozen inference bundle

The web application must never reconstruct a model by reading an experiment
notebook. Export one immutable bundle before connecting the prediction API:

```text
prototype_bundle/v1/
  manifest.json
  preprocessing.json
  transformer.pt
  h128_imputer.joblib
  h128_scaler.joblib
  pls_q4.joblib
  angle_quantiles.npz
  vqc/
    model_01.pt
    model_02.pt
    ...
  morphology_feature_manifest.json
  morphology_conditioner.joblib
  morphology_hgb.joblib
  fusion.joblib
  platt_calibrator.joblib
  decision_threshold.json
  explanation_background.npz
  golden_cases.json
```

`manifest.json` records the Git commit, dataset releases, feature-manifest
hash, training-patient hash, architecture, ensemble members, score-alignment
method, calibration source, decision threshold and SHA-256 checksum of every
file. API startup fails closed if a checksum, feature order or version differs.

The current repository contains research results but does not yet contain this
complete frozen bundle. Fold-9 calibration and threshold fitting are still a
scientific prerequisite for a final probability claim. Until then, the API
must run with `PROTOTYPE_UNCALIBRATED=true` and display `development score`
instead of `calibrated probability`.

## 4. Repository layout to build

```text
apps/
  api/
    aquire_api/
      main.py
      settings.py
      schemas.py
      routes/
        health.py
        model.py
        prediction.py
        benchmarks.py
      services/
        bundle_loader.py
        ecg_parser.py
        inference.py
        explanations.py
        audit.py
    tests/
    Dockerfile
  web/
    src/
      api/
      components/
      pages/
      types/
    tests/
    Dockerfile
prototype_bundle/
  README.md
scripts/
  export_prototype_bundle.py
  verify_prototype_bundle.py
docker-compose.yml
.env.example
```

The API imports `aquire_preprocessing`; it does not copy QC, filtering,
morphology or model code. A single `HybridInferenceService` owns the complete
request path and returns typed results.

## 5. API contract

### Health and model information

- `GET /api/v1/health/live`: process is alive.
- `GET /api/v1/health/ready`: model bundle loaded and golden self-test passed.
- `GET /api/v1/model-card`: task, versions, limitations and allowed claims.
- `GET /api/v1/architecture`: structured nodes and edges for the UI diagram.
- `GET /api/v1/benchmarks`: frozen development metrics and confidence intervals.

### ECG validation and prediction

- `POST /api/v1/ecg/inspect`: parse and validate without running predictors.
- `POST /api/v1/predictions`: validate, preprocess, run both branches, fuse,
  calibrate and explain one ECG.

Accepted demo formats:

1. a ZIP containing a paired WFDB `.hea` and `.dat` record;
2. CSV with exactly 1,000 rows and the canonical twelve lead columns;
3. JSON `float32[12][1000]` for integration tests and device adapters.

Limit uploads to 5 MB, reject archives with path traversal, ignore filenames as
patient identifiers and verify sampling rate, duration and lead order from
content. Failed or indeterminate QC produces an abstention response and does
not run ordinary prediction.

Representative response:

```json
{
  "trace_id": "01J...",
  "model_version": "aquire-hybrid-v1",
  "task": "mi_pattern_vs_non_mi_pattern",
  "input": {"sampling_rate_hz": 100, "shape": [12, 1000]},
  "quality": {"state": "PASS", "messages": []},
  "result": {
    "state": "PREDICTED",
    "mi_pattern_probability": 0.81,
    "threshold": 0.64,
    "label": "MI_PATTERN",
    "review_message": "Research screening output; clinician review required"
  },
  "routes": {
    "quantum_score": 1.42,
    "classical_score": 0.77,
    "fusion_logit": 1.46
  },
  "explanations": {
    "morphology": [],
    "waveform_regions": [],
    "q4_sensitivity": []
  },
  "timing_ms": {}
}
```

Branch scores belong inside an expandable research panel. They must not be
shown as three competing diagnoses.

## 6. Inference execution

1. Parse the upload into a canonical `ValidatedECG`.
2. Apply existing structural checks, masks, QC and morphology safeguards.
3. Abstain on structural failure, QC `FAIL` or `INDETERMINATE`.
4. Produce one frozen `float32[12,1000]` signal.
5. Run the Transformer and morphology extractor concurrently after the common
   preprocessing step.
6. Apply the stored h128 transforms, PLS-q4 and angle map.
7. Execute every registered VQC ensemble member under `torch.inference_mode()`
   and apply the frozen score-alignment rule.
8. Run the 106-feature HGB branch using its exact signed feature order.
9. Apply the frozen logistic fusion, Platt calibrator and threshold.
10. Generate bounded explanations and remove the uploaded temporary files.

Load all artifacts once at API startup. Use a bounded inference semaphore to
prevent concurrent Torch jobs exhausting memory. Cache only public demo cases,
keyed by signal checksum; do not cache user uploads by default.

Latency target on a four-core laptop:

| Stage | Target |
|---|---:|
| Parse, validate and QC | under 1.5 s |
| Transformer and morphology branches | under 2.5 s |
| VQC ensemble, HGB and fusion | under 1.0 s |
| Basic explanations | under 2.0 s |
| Total first response | under 7 s |

Lead/time occlusion may run after the primary response and stream into the
result page. The main prediction must not wait for a hardware QPU.

## 7. UI information architecture

### A. Overview

- one-sentence task and scientific boundary;
- animated architecture showing both branches running for every ECG;
- dataset and development-validation facts;
- clear `Start ECG analysis` action.

### B. Upload and validation

- drag-and-drop WFDB ZIP or CSV;
- file-format example and canonical lead order;
- validation steps for sampling rate, duration, leads and missing values;
- twelve synchronized ECG traces after parsing;
- explicit proceed, abstain or correction status.

### C. Result

- primary fused MI-pattern score and thresholded screening label;
- quality state and a visible research-use notice;
- morphology explanation grouped into rhythm, QRS, amplitude and ST/T;
- lead/time importance heatmap and q4 sensitivity;
- expandable technical panel with `sQ`, `sC`, fusion logit, latency, circuit
  width/depth and model hashes.

### D. Research benchmark

- quantum-only, overall-classical replacement and hybrid-fusion metrics;
- AUPRC, AUROC, Brier score, sensitivity at approximately 90% specificity and
  patient-bootstrap intervals;
- confusion matrices at the same specificity;
- hard-negative analysis;
- wording that the hybrid is comparable to, not superior to, the overall
  classical control.

### E. Model and data card

- PTB-XL/PTB-XL+ roles and licenses;
- patient-safe folds and current evaluation boundary;
- circuit diagram and exact four-qubit role;
- unsupported claims, failure modes and intended use.

## 8. Demo behavior and visual design

Use a clean clinical interface with dark navy text, white surfaces, ECG red for
waveforms and restrained violet for the quantum route. Reserve amber and red
for quality warnings. Avoid decorative quantum imagery that obscures the
signal flow.

Ship three predefined demonstrations referenced by record ID and outcome:

- a correctly detected MI-pattern example;
- a non-MI hard-negative example;
- a QC failure that causes abstention.

Do not bundle redistributable ECG files until their license and attribution are
recorded. The demo can fetch mounted local examples or accept judge-provided
files.

## 9. Verification and acceptance

### Scientific parity

- Golden API fixtures match offline inference within `1e-6` for q4, `sQ`,
  `sC`, fusion logit and probability.
- Feature names and ordering exactly match the signed 106-feature manifest.
- Removing or shuffling either route fails a startup/self-test assertion.
- The API refuses an incomplete, modified or uncalibrated bundle unless the
  explicit prototype flag is set.

### API and security

- malformed WFDB, reordered leads, wrong rate, NaN/Inf, archive traversal,
  oversize upload and concurrent requests have tests;
- temporary uploads are deleted on success and failure;
- OpenAPI examples correspond to real response schemas;
- no patient-identifying value is written to logs.

### UI

- Playwright covers upload, waveform preview, prediction, explanation and
  abstention paths;
- result language always says `MI pattern`, never acute diagnosis or future
  risk;
- mobile projector and 1366x768 judge-laptop layouts are visually checked;
- the app remains usable when explanations fail after a prediction succeeds.

## 10. Exact build order

### Milestone P0 — freeze the executable model

- [ ] Decide whether the prototype deploys one circuit or the documented VQC
  ensemble; never attach ensemble metrics to a single model.
- [ ] Export all upstream transforms, both route models and fusion weights.
- [ ] Fit/freeze Fold-9 calibration and threshold, or enable the prominently
  labelled uncalibrated development-demo mode.
- [ ] Generate and verify bundle checksums and golden cases.

### Milestone P1 — backend vertical slice

- [ ] Add FastAPI scaffolding, settings and typed response schemas.
- [ ] Implement bundle verification and readiness self-test.
- [ ] Implement WFDB/CSV/JSON parsing through existing preprocessing contracts.
- [ ] Implement one end-to-end `/predictions` call and golden parity test.
- [ ] Add abstention, timing and deletion guarantees.

### Milestone P2 — judge-ready UI

- [ ] Add the overview, architecture and upload flow.
- [ ] Render twelve synchronized leads and quality findings.
- [ ] Build the fused-result and technical-detail panels.
- [ ] Add morphology and waveform explanations.
- [ ] Build the benchmark and model-card pages from versioned JSON.

### Milestone P3 — package and rehearse

- [ ] Add Dockerfiles, Compose, `.env.example` and one-command startup.
- [ ] Run pytest, Vitest, Playwright and the golden scientific parity suite.
- [ ] Measure cold start and inference latency on the presentation laptop.
- [ ] Rehearse the clean, hard-negative and abstention demos offline.
- [ ] Tag the exact hackathon release and record its commit in the UI.

## 11. Suggested 72-hour allocation

| Window | Backend/model | Frontend | Research/QA |
|---|---|---|---|
| 0–12 h | Bundle export and parity CLI | Wireframes and component shell | Freeze wording and golden cases |
| 12–30 h | FastAPI vertical slice | Upload and ECG viewer | Adversarial input checks |
| 30–48 h | Explanations and audit metadata | Results and benchmark pages | Scientific parity and metric audit |
| 48–60 h | Containers and performance | Responsive polish | End-to-end tests |
| 60–72 h | Bug fixes | Demo-state polish | Rehearsal, release tag and backup video |

## 12. Hackathon release gate

The prototype is ready to demonstrate when one command starts it, all three
golden paths work without internet, the model bundle passes checksum and
scientific-parity checks, a prediction completes within seven seconds on the
presentation machine, and every performance statement matches the frozen
benchmark report.

The release may state that the hybrid recovered six additional MI-pattern
records at the matched development operating point. It must also state that
the overall classical replacement remained slightly stronger in AUPRC and
that no quantum advantage has been established.
