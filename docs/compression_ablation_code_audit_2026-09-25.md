# Compression-ablation prototype audit

**Verdict: do not submit the current prototype.**

The proposed PLS/NCA/SupConAE/full-reupload screen addresses a relevant
question, but its present implementation cannot support a research claim.

## Blocking defects

1. It filters `meta["fold"]`, while the validated production contract uses
   `strat_fold`.
2. It constructs an outer-fold training matrix by concatenating validation
   embeddings produced by different fitted Transformer encoders. Those latent
   coordinate systems are not guaranteed to be aligned.
3. It then pairs the concatenated matrix with `labels[~is_val]` by position
   rather than joining every embedding to its `ecg_id`. This can silently pair
   a waveform representation with the wrong label.
4. The Kaggle runner clones the current repository head rather than fetching a
   frozen commit and uses unversioned notebook inputs.
5. The quantum head returns one `Z0` expectation with no learned linear
   multi-observable readout, making it weaker than the retained Z/ZZ VQC.
6. The 128-feature re-upload circuit performs 32 sequential upload blocks. It
   is much deeper and more expensive than the four-dimensional arms, so it is
   neither a controlled compression ablation nor a realistic NISQ candidate.
7. The experiment uses one seed, no nested inner validation, no
   patient-cluster bootstrap, no entanglement ablation and no fold-safe score
   calibration.
8. The supervised autoencoder includes a label-prediction head. Its result
   would need a matched classical-head ablation and careful wording because a
   supervised classical encoder may already perform much of the MI decision.

## Corrected contract

- Load `train_embeddings`, `val_embeddings`, record IDs, labels and patient IDs
  from each saved outer-fold representation file. Never mix latent vectors
  from independently fitted outer encoders.
- Assert ID, label and patient alignment against the immutable manifest.
- Fit each compression method on outer-training patients only.
- Give every representation the same retained VQC, training budget, restart
  count and identical-input logistic/MLP controls.
- Treat full 128-feature re-uploading as a separate depth/resource experiment,
  not a matched compression arm.
- Use a one-seed prespecified screen followed by five seeds only when the
  representation improves both VQC AUPRC and its quantum-versus-matched-control
  delta.
- Keep folds 9 and 10 sealed during development.

The validated information-loss trace already shows that q4 retains most
discrimination despite losing local geometry. Score alignment and training
stability therefore have higher priority than another broad representation
search.
