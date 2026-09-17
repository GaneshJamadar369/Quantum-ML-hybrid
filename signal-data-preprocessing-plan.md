# Robust Signal and Data Preprocessing Plan

**Project:** SIH 26139 — AQUIRE-Med  
**Scope:** Signal and data processing before CNN, SSM, feature extraction, PCA or quantum modelling  
**Primary dataset:** [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)  
**Companion measurements:** [PTB-XL+ v1.0.1](https://physionet.org/content/ptb-xl-plus/1.0.1/)  
**Primary task:** MI-pattern versus non-MI-pattern classification from a 10-second, 12-lead ECG

## Objective

Build a morphology-preserving, quality-aware and reproducible preprocessing pipeline. The pipeline should correct only detected artifacts while preserving the ST segment, QRS complex, T wave and absolute lead amplitudes needed for MI-pattern detection.

PTB-XL provides 500 Hz waveform files, official 100 Hz versions and technical annotations such as baseline drift, static noise, burst noise and electrode problems. These annotations are used to validate the quality-control system, not as disease-prediction inputs.

## Complete pipeline

```text
PTB-XL waveform and metadata
             │
     1. Immutable raw-data vault
             │
     2. Identity and structural validation
             │
     3. Physical-unit and lead harmonization
             │
     4. Duplicate and patient-leakage audit
             │
     5. Lead-level signal-quality analysis
             │
     6. Cross-lead consistency analysis
             │
     7. Artifact-aware processing router
             │
     8. Multiple morphology-preserving views
             │
     9. Processing-distortion validation
             │
    10. Train-only normalization
             │
    11. Training-only corruption augmentation
             │
    12. Versioned tensor + mask + QC output
```

## 1. Preserve the original signal

Keep the original 500 Hz WFDB data immutable:

```text
data/
  raw/
    ptb-xl-1.0.3/
      records500/
      records100/
      ptbxl_database.csv
      scp_statements.csv
      SHA256SUMS.txt
  processed/
    view_100hz_minimal/
    view_100hz_corrected/
    quality_reports/
    manifests/
```

Every processed record must store:

- source checksum;
- dataset version;
- preprocessing version;
- transformations applied;
- transformation parameters;
- quality status;
- repair, warning or exclusion reason.

Processing code must never overwrite the original `.hea` or `.dat` files.

## 2. Structural validation

Before applying any filtering, validate:

- `.hea` and `.dat` pairing;
- unique `ecg_id`;
- valid `patient_id`;
- exactly 12 standard leads;
- lead names and order;
- recording duration;
- sampling frequency;
- voltage units and gain;
- nonfinite or missing samples;
- expected array shape;
- exact duplicate waveforms;
- near-duplicate waveforms;
- agreement between waveform files and metadata.

Canonical lead order:

\[
[I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6]
\]

PTB-XL stores its original waveforms at 500 Hz with 16-bit precision and a resolution of 1 μV/LSB. It also supplies official 100 Hz versions.

## 3. Maintain two sampling-rate representations

### Archival view

\[
X_{500}\in\mathbb{R}^{12\times5000}
\]

Use this view for waveform preservation, high-resolution quality analysis and sampling-rate ablations.

### Main model-ready view

\[
X_{100}\in\mathbb{R}^{12\times1000}
\]

Use this as the initial model input because it is smaller and matches the planned architecture.

For an incoming recording sampled at 250, 500 or 1,000 Hz:

1. validate the stated sampling rate;
2. apply an anti-aliasing low-pass filter;
3. resample to the frozen target rate;
4. record the original rate and resampling parameters;
5. test the result using the morphology-preservation gate.

Never upsample a 100 Hz signal and claim that the missing high-frequency information has been recovered.

## 4. Build a lead-level quality fingerprint

For each lead \(l\), calculate:

\[
q_l=
[q_{flat},q_{clip},q_{missing},q_{baseline},q_{powerline},
q_{high-frequency},q_{QRS-consistency},q_{amplitude}]
\]

The checks include:

- flat or disconnected electrodes;
- repeated saturation values;
- abrupt amplitude jumps;
- missing intervals;
- baseline wander;
- muscular or high-frequency noise;
- power-line interference;
- implausible amplitudes;
- disagreement between independent QRS detectors;
- inconsistent beat timing between leads.

The complete ECG receives one of three quality states:

| State | Meaning | Action |
|---|---|---|
| **PASS** | Signal is suitable for the minimal path | Preserve the minimally processed signal |
| **WARN** | A limited, identifiable artifact is present | Apply targeted correction and retain all QC metadata |
| **FAIL** | Signal cannot support the validated task | Exclude from the primary training cohort or abstain during inference |

Quality rules must never delete a record merely because a model predicts it incorrectly.

## 5. Add cross-lead physics checks

The limb leads should approximately satisfy:

\[
II\approx I+III
\]

\[
aVR\approx-\frac{I+II}{2}
\]

\[
aVL\approx I-\frac{II}{2}
\]

\[
aVF\approx II-\frac{I}{2}
\]

Calculate a consistency residual for each relationship. Large residuals can indicate:

- incorrect lead order;
- lead reversal;
- disconnected electrodes;
- unit or gain errors;
- severe noise.

These checks create warnings. They must not automatically rewrite lead identities without a separately validated reversal-detection method.

## 6. Use an artifact-aware processing router

Do not apply every filter to every ECG.

```text
Quality analysis
      │
      ├── Clean signal
      │      └── Minimal processing
      │
      ├── Baseline drift detected
      │      └── Zero-phase baseline correction
      │
      ├── Power-line interference detected
      │      └── Targeted 50/60 Hz removal
      │
      ├── Short missing interval
      │      └── Limited interpolation + missing mask
      │
      ├── Missing or disconnected lead
      │      └── Lead mask / abstention
      │
      └── Severe corruption
             └── FAIL
```

Every processing decision must be deterministic, versioned and logged in `processing_route`.

## 7. Protect the ST segment

ST-segment morphology is essential for MI-pattern detection. Aggressive high-pass filtering can create or remove apparent ST deviation.

The primary rules are:

- use the minimally processed signal as the primary view;
- apply baseline correction only when a validated detector finds excessive drift;
- use zero-phase processing for offline correction;
- do not use a universal aggressive high-pass filter;
- do not choose a filter only because it increases validation accuracy;
- measure its effect on ST level, T-wave polarity, R-peak timing and QRS width.

The AHA/ACCF/HRS recommendations explain that a 0.5 Hz high-pass cutoff can distort repolarization, while diagnostic ECG filtering traditionally preserves lower-frequency content. See the [ECG standardization recommendations](https://fd.org.ua/wp-content/uploads/2019/03/AHA-ACCF-HRS-Recommendations-for-the-Standardization-and-Interpretation-of-the-Electrocardiogram-Part-I.pdf).

## 8. Produce multiple signal views

### View A — minimally processed

Apply only:

- physical-unit conversion;
- lead reordering;
- duration and shape validation;
- safe resampling;
- validated constant-offset handling.

### View B — diagnostically corrected

Add only when indicated:

- baseline-wander correction;
- targeted interference removal;
- validated zero-phase filtering;
- limited short-gap repair.

### View C — controlled stress-test signal

Create training and robustness-test variants containing known corruption:

- baseline drift;
- electrode-motion artifacts;
- Gaussian or coloured noise;
- amplitude or gain variation;
- temporary lead dropout;
- clipping;
- controlled resampling changes.

View C is never substituted for the original validation or test signal.

## 9. Add a morphology-preservation gate

After any correction, compare the output with the minimally processed reference.

Measure changes in:

- R-peak locations;
- QRS width;
- QRS amplitude;
- ST level;
- T-wave polarity;
- beat-to-beat timing;
- cross-lead consistency;
- energy in clinically relevant frequency bands.

Define preprocessing utility as:

\[
U_P=
\text{artifact reduction}
-\lambda_m\text{morphology distortion}
-\lambda_f\text{failure rate}
-\lambda_t\text{processing time}
\]

A correction is rejected if it reduces noise but violates the prespecified morphology tolerances. The tolerances must be frozen using training data and clinical review before evaluating fold 9 or fold 10.

## 10. Preserve absolute amplitude

Avoid independently forcing every ECG to zero mean and unit variance. Per-record standardization can remove medically meaningful amplitude differences.

Recommended procedure:

1. convert every record to the same physical unit;
2. preserve the unscaled signal view;
3. calculate per-lead robust location and scale statistics using training patients only;
4. serialize those statistics;
5. apply the same transformation to validation, test and new inputs;
6. retain the inverse transformation and physical-unit metadata.

Patient splitting must occur before calculating normalization, clipping or imputation statistics.

## 11. Handle missing values and leads conservatively

For a short isolated missing interval:

- interpolate only within a frozen maximum-gap rule;
- retain a missing-sample mask;
- record the repaired interval and method.

For a long missing interval or a missing lead:

- do not silently synthesize the complete waveform;
- retain a lead mask;
- assign WARN or FAIL;
- abstain if the production model was not validated for that missingness pattern.

Lead reconstruction is a separate experimental component and must not be silently inserted into the primary preprocessing path.

## 12. Use physiologically plausible training augmentation

Permitted training-only augmentations include:

- small amplitude scaling;
- time translation;
- realistic baseline drift;
- muscular noise;
- electrode-motion noise;
- short contiguous masking;
- lead dropout;
- small sampling-rate variation.

Avoid:

- arbitrary lead permutation;
- time reversal;
- large nonlinear time warping;
- transformations that invert ST or T-wave morphology;
- augmentation of validation or test records.

For every augmented sample, record:

- source `ecg_id`;
- corruption type;
- severity;
- affected leads/intervals;
- random seed;
- augmentation version.

## 13. Improve label robustness

Store the following alongside each processed ECG:

- binary MI-superclass label;
- contributing SCP codes;
- annotation likelihoods;
- human-validation status;
- overlapping diagnostic superclasses;
- hard-negative flag;
- infarction stage for audit only;
- label-mapping version.

Do not use report text, `infarction_stage`, SCP diagnostic statements or other label-derived metadata as predictive inputs.

Run two label analyses:

1. **Full clinical cohort:** all valid MI and non-MI records.
2. **High-confidence cohort:** records meeting frozen annotation-confidence and human-validation requirements.

The high-confidence analysis helps distinguish model failure from annotation ambiguity. It does not replace the full-cohort result.

## 14. Patient and duplicate safety

- Preserve PTB-XL's official patient-aware `strat_fold` assignment.
- Use folds 1–8 for training/development, fold 9 for validation/calibration and fold 10 for the locked test.
- Confirm zero `patient_id` overlap between these roles.
- Hash waveforms to detect exact duplicates.
- Use high-correlation or distance screening to detect near duplicates.
- Keep all records belonging to one patient in the same role.
- Bootstrap and report uncertainty at the patient level.

PTB-XL's official documentation states that the folds respect patient assignments and that folds 9 and 10 have particularly strong human annotation quality.

## 15. Preprocessing output contract

Each ECG produces a versioned object equivalent to:

```python
{
    "ecg_id": int,
    "patient_id": int,
    "signal_minimal": "float32[12, 1000]",
    "signal_corrected": "float32[12, 1000]",
    "lead_mask": "bool[12]",
    "sample_mask": "bool[12, 1000]",
    "qc_status": "PASS | WARN | FAIL",
    "qc_metrics": {
        "per_lead": {},
        "cross_lead": {},
        "morphology_preservation": {}
    },
    "processing_route": [],
    "mi_label": "0 | 1",
    "label_confidence": float,
    "hard_negative": bool,
    "strat_fold": int,
    "source_checksum": str,
    "dataset_version": "1.0.3",
    "pipeline_version": str
}
```

## 16. Required validation tests

### File and structural tests

- [ ] Valid `.hea`/`.dat` pair loads successfully.
- [ ] Missing header or signal file fails with a deterministic error.
- [ ] Wrong lead order is detected and safely canonicalized.
- [ ] Unknown lead names fail rather than being guessed.
- [ ] Sampling rate, duration, units and shape are verified.
- [ ] Exact and near duplicates are reported.

### Quality tests

- [ ] Flatline lead is detected.
- [ ] Saturation/clipping is detected.
- [ ] Baseline drift is detected.
- [ ] High-frequency noise is detected.
- [ ] Short and long missing intervals follow different policies.
- [ ] Lead inconsistency creates a warning without automatic relabelling.

### Morphology-preservation tests

- [ ] Synthetic baseline drift is reduced without unacceptable ST displacement.
- [ ] R-peak timing remains inside the frozen tolerance.
- [ ] QRS width and amplitude remain inside the frozen tolerance.
- [ ] T-wave polarity is preserved.
- [ ] Clean signals are not unnecessarily filtered.

### Leakage tests

- [ ] Normalization statistics use training patients only.
- [ ] QC and routing thresholds are fixed without fold 9 or fold 10.
- [ ] Augmentation never runs on validation or test records.
- [ ] Diagnostic metadata is excluded from predictive tensors.
- [ ] No patient crosses dataset roles.

## 17. New project contributions

The preprocessing layer introduces the following project-level ideas:

1. **Artifact-aware routing:** clean only the artifact actually detected.
2. **Minimal and corrected dual views:** preserve the original morphology while allowing controlled correction.
3. **Preprocessing utility gate:** balance artifact removal against morphology distortion, failure rate and latency.
4. **Cross-lead physics validation:** use known limb-lead relationships to detect structural problems.
5. **PASS/WARN/FAIL routing:** retain difficult but usable data while supporting safe abstention.
6. **Versioned corruption library:** test robustness using reproducible, physiologically plausible artifacts.
7. **Full and high-confidence label analyses:** quantify the effect of annotation ambiguity.
8. **Quality metadata separation:** use PTB-XL artifact fields to validate QC without leaking them into disease prediction.

## References

- [PTB-XL v1.0.3 official dataset page](https://physionet.org/content/ptb-xl/1.0.3/)
- [PTB-XL dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7248071/)
- [PTB-XL+ v1.0.1 official dataset page](https://physionet.org/content/ptb-xl-plus/1.0.1/)
- [AHA/ACCF/HRS ECG standardization recommendations](https://fd.org.ua/wp-content/uploads/2019/03/AHA-ACCF-HRS-Recommendations-for-the-Standardization-and-Interpretation-of-the-Electrocardiogram-Part-I.pdf)
- [Signal quality indices and data fusion for clinical ECG acceptability](https://pubmed.ncbi.nlm.nih.gov/22902749/)
- [AQUIRE-Med master implementation plan](plan.md)
- [AQUIRE-Med architecture](output/aquire-med-system-architecture.svg)

