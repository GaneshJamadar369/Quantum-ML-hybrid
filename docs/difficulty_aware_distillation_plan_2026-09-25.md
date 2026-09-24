# G6Q-KD4 — Difficulty-aware distillation plan

## Question

Can a training-only difficulty curriculum transfer more of the q4 MLP
assistant's decision boundary into the retained four-qubit VQC than uniform JS
distillation, without changing the input representation, circuit, sample
budget, inference graph or sealed-fold policy?

This is the final bounded KD screen. It adapts the difficulty-aware principle
to binary ECG classification; it does not copy autoregressive token-generation
losses from language models.

## Frozen experiment

- 17,348 primary development ECGs and official folds 1–8
- folds 9 and 10 inaccessible
- fold-coherent supervised Transformer h128
- fold-local supervised PLS q4 and quantile angle mapping
- one record per patient, 2,000 patients per class in each outer-fold fit
- four-qubit, two-layer ring VQC with 45 inference parameters
- 30 epochs, Adam, fixed initialization and minibatch order
- inner-official-fold OOF q4 MLP teacher plus training-only sigmoid calibration
- identical-q4 logistic and sample-matched MLP controls
- clinical HistGB expert used only for the secondary cross-fitted fusion analysis

No record is filtered. Every record retains an equally weighted hard-label
loss. Difficulty changes only the JS teacher term.

## Per-record evidence

At training epoch `e`, compute without gradient:

- calibration reliability from the inner-OOF teacher;
- teacher agreement with the known outer-training label;
- current student hard-label error;
- absolute teacher/student probability disagreement;
- prespecified hard-negative status (STTC/CD/HYP non-MI cohort).

The dynamic soft-loss weight is proportional to:

`reliability × teacher-label agreement × mean(student error, disagreement)`.

Weights are normalized within each minibatch and clipped to `[0.25, 2.0]`.
The hard-negative arm multiplies the raw soft weight by `1.5` for prespecified
hard negatives. The curriculum arm transitions linearly from uniform JS in the
first epoch to the dynamic rule in the final epoch.

## Arms

1. hard-label VQC
2. uniform JS VQC
3. calibration/label-agreement weighted JS
4. dynamic difficulty JS
5. uniform-to-dynamic curriculum JS
6. dynamic JS with hard-negative emphasis

## Evaluation and stop rule

Primary: pooled patient-safe OOF AUPRC. Secondary: AUROC, Brier, log loss,
ECE-10, hard-negative behavior and clinical+quantum fusion. Save complete OOF
predictions, teacher audits, weight ranges, gradients, fusion coefficients and
2,000-replicate paired patient-cluster bootstraps.

Advance only if the best difficulty-aware VQC exceeds hard VQC by at least
`0.005` AUPRC with a 95% paired interval above zero. A standalone quantum win
also requires a positive interval against the strongest identical-q4 classical
control. System-level quantum value requires the difficulty-aware fusion to
beat the clinical+q4-MLP fusion with a positive interval.

If these gates fail, stop KD development and preserve the result as a negative
ablation. Do not tune clipping, multipliers or curricula on pooled outer-fold
labels.
