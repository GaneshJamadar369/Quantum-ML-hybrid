# G6Q-CAL — train-reference score alignment follow-up

## Trigger

This follow-up is permitted only after G6Q-OPT1 finishes all five prespecified
seeds. It addresses a specific diagnostic observed in the first two completed
seeds:

- pooled VQC minus q4-logistic AUPRC was negative;
- macro within-fold AUPRC differences were only `+0.00024` and `-0.00107`;
- held-out-fold score standardization recovered much of the pooled loss;
- 15/16 inner selections chose the narrow-bandwidth JS arm and 10/16 selected
  epochs hit the 30-epoch cap.

Held-out-fold standardization is transductive and is **diagnostic only**. It
must never appear in the deployable pipeline.

## Frozen hypothesis

The retained VQC loses pooled OOF ordering partly because independently fitted
outer-fold models emit scores on different scales. A mapping derived only from
each model's outer-training score distribution may align those scales without
using held-out labels or held-out score distributions.

## Experiment

For each outer fold:

1. Fit the existing fold-local robust-scaling, PLS-q4 and angle transform on
   outer-training patients only.
2. Train the fixed four-qubit, two-layer `narrow_js` circuit. Do not search a
   new circuit family on outer outcomes.
3. Extend the training budget to a fixed 60 epochs with a prespecified cosine
   schedule. Save the full learning curve; do not select an epoch on outer
   outcomes.
4. Evaluate the fitted circuit on the full outer-training representation to
   form a train-reference score distribution.
5. Produce three held-out scores:
   - raw VQC logit;
   - train-reference z score;
   - train-reference empirical-CDF score.
6. Repeat with three fixed initializations. Report every restart and a mean
   score ensemble. Apply identical restart accounting to the matched q4 MLP.
7. Compare against tuned q4 logistic, q4 MLP and calibrated RBF controls on the
   same q4 coordinates and patient sample.
8. Save raw logits, aligned scores, q4 coordinates, local Z/ZZ observables,
   learned feature scales, rotations, interactions, gradients and checksums.

The empirical CDF for a new patient is computed against stored training
scores. No batch of future patients and no validation/test distribution is
needed at inference.

## Statistical protocol

- Primary metric: patient-pooled OOF AUPRC.
- Secondary: macro fold AUPRC, AUROC, Brier after train-only calibration,
  sensitivity at 90% specificity, hard-negative FPR and inference cost.
- Uncertainty: paired patient-cluster bootstrap.
- Stability: five fixed seeds and per-seed wins.
- Ablations: raw versus z/CDF alignment; one versus three restarts;
  entangled versus no-entanglement.

This is an exploratory development experiment because its hypothesis was
identified from folds 1–8. If it passes its development gate, freeze the entire
pipeline and inspect fold 9 exactly once. Fold 10 remains sealed for final
evaluation.

## Gates

Advance only if:

1. the train-reference mapping improves raw VQC by at least `0.003` AUPRC;
2. the aligned VQC beats the strongest identical-q4 control with a positive
   patient-bootstrap interval;
3. the improvement appears in at least four of five seeds;
4. the quantum-plus-clinical fusion exceeds `0.83802` AUPRC;
5. entanglement removal causes a reproducible decline;
6. hard-negative FPR does not materially worsen.

If these gates fail, stop VQC score-alignment work. Do not continue selecting
transformations on folds 1–8.

## Compression decision

The current 128-to-4 trace shows a real geometry loss, but it is not the largest
remaining bottleneck. A separate compression experiment must use the saved
outer-fold `train_embeddings`/`val_embeddings` and align records by `ecg_id`.
The existing prototype compression notebook must not run because it combines
embeddings from different fitted encoders, assumes positional label alignment,
uses an unpinned repository head and lacks nested validation.
