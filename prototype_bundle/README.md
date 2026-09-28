# Prototype model bundle

The API loads one immutable bundle from `AQUIRE_BUNDLE_ROOT`. Model weights and
patient-derived transformation artifacts are not committed to Git.

Run the verifier before starting the API:

```bash
python scripts/verify_prototype_bundle.py /path/to/prototype_bundle/v1
```

After exporting every artifact, create the content registry with:

```bash
python scripts/build_prototype_manifest.py /path/to/prototype_bundle/v1 \
  --model-version aquire-hybrid-v1 \
  --calibration-state development_uncalibrated \
  --training-patient-sha256 SHA256 \
  --feature-manifest-sha256 SHA256 \
  --artifact transformer=transformer.pt \
  --artifact waveform_normalizer=waveform_normalizer.json \
  --artifact h128_imputer=h128_imputer.joblib \
  --artifact h128_scaler=h128_scaler.joblib \
  --artifact pls_q4=pls_q4.joblib \
  --artifact angle_quantiles=angle_quantiles.joblib \
  --artifact vqc_model=vqc/model_01.pt \
  --artifact vqc_score_alignment=vqc_score_alignment.joblib \
  --artifact morphology_feature_manifest=morphology_feature_manifest.json \
  --artifact morphology_conditioner=morphology_conditioner.joblib \
  --artifact morphology_hgb=morphology_hgb.joblib \
  --artifact fusion=fusion.joblib
```

An uncalibrated development bundle is accepted only when both the verifier and
API receive the explicit prototype override. It must never be described as a
calibrated disease probability.

See [`docs/sih_prototype_ui_backend_plan_2026-09-28.md`](../docs/sih_prototype_ui_backend_plan_2026-09-28.md)
for the required files and scientific boundary.
