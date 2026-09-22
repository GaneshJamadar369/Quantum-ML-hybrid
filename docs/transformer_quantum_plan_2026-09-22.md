# Compact ECG Transformer → quantum-head experiment

## Scientific question

Does replacing the supervised 1D ResNet waveform encoder with a compact
attention encoder improve held-out MI-pattern ranking **after** compression
to four quantum input coordinates? A better Transformer classification score
alone is insufficient: the quantum and classical heads must receive the same
fold-local coordinates and be compared on the same patients.

This is a development-fold experiment. The Transformer is MI-supervised, so
its embedding is **not** evidence that the quantum head independently
discovered the disease pattern. A later label-free encoder is a separate
ablation. Folds 9 and 10 remain sealed.

## Frozen layer configuration (first screen)

| Stage | Configuration | Reason and check |
|---|---|---|
| Input | 12 leads × 1,000 samples, 100 Hz, 10 s | Same accepted-signal view and fold-local normalizers as ResNet |
| Tokenizer | Non-overlapping 10-sample (100 ms) patches; flatten 12 × 10 and linearly project to 96 | 100 tokens retain fine waveform samples while limiting quadratic attention cost; no pre-blurring CNN stem |
| Positions | Learned 100 × 96 positional embedding | Preserves time order; input shape and finite-value assertions |
| Attention stack | 3 pre-norm Transformer encoder layers, width 96, 4 heads (24 dimensions/head), feed-forward width 192, GELU | Constrained capacity; residual/pre-norm training stability; inspect parameter count and gradients |
| Regularization | Attention/feed-forward dropout 0.15; token dropout 0.10; embedding-head dropout 0.25; AdamW weight decay 0.05 | Reduce memorization; verify train/eval behavior |
| Pooling | Concatenate mean and max across 100 time tokens | Represents repeated patterns and focal morphology without a high-capacity classification head |
| Representation | 192 → 128 linear, GELU, LayerNorm; linear MI logit head | Compatible with the existing fold-coherent 128-dimensional downstream contract |

The exact PyTorch encoder arguments are `batch_first=True`, `norm_first=True`,
`activation="gelu"`; see the [official layer API](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html).
This is a patch Transformer, not DeepAR. The CNN baseline stays in the study.

## Patient-safe anti-overfit protocol

1. For each held-out outer fold 1–8, load the saved fold-specific robust
   12-lead normalizer. No fold 9/10 input or labels enter any stage.
2. Split only the seven outer-training folds' patients into 90% optimization
   and 10% inner validation, stratified by whether each patient has any MI
   ECG. Keep all ECGs from a patient in one role. Fit the model and choose the
   best epoch by inner-validation AUPRC, with at least six epochs, a maximum
   of 20 and patience four. Never use outer labels for stopping.
3. Retrain a fresh model on **all seven** outer-training folds for the chosen
   epoch count. This restores the training-data budget before the outer
   comparison. Set deterministic seeds, log the chosen epoch, inner curve,
   train loss and outer score, and store model/normalizer hashes.
4. Apply only mild training-time amplitude jitter (within ±5%) and additive
   normalized noise (σ=0.01). No time warping, beat deletion or aggressive
   masking that could erase QRS/ST/T evidence. Retain patient-balanced and
   hard-negative-weighted focal loss, gradient clipping at 1, and mixed
   precision on GPU. The inner/outer inference paths apply no augmentation.
5. Export train and held-out 128-dimensional embeddings from the **same
   retrained fold model**. Validate finite values, unique ECG IDs, patient
   isolation, exact 17,348-record OOF coverage and distinct fold checksums.

The inner validation split and retraining prevent choosing a training length
from outer OOF performance. They do not guarantee absence of overfitting;
generalization must be judged by the gap between inner and outer results,
fold variance and comparison with the CNN.

## Quantum-head comparison after export

- Apply the existing fold-local, patient-unique sample of 500 MI and 500
  non-MI ECGs and the same supervised PLS/quantile mapping to `q4`. This
  deliberately holds head inputs and budgets constant with the CNN experiment.
- Run the direct q4 VQC, matched-input MLP and RBF-SVC on the Transformer
  embeddings. Use identical optimizer and 20-epoch screen settings as Phase
  6Q-E, save eight OOF score files and paired patient-cluster bootstrap CIs.
- Compare **Transformer versus CNN for each head** on aligned held-out ECGs.
  Report both the Transformer encoder's own OOF AUPRC and the compressed-head
  AUPRC. A gain shared by MLP and VQC belongs to the representation. A
  quantum-specific claim additionally requires VQC to beat the strongest
  classical head on the same Transformer `q4`, with uncertainty above zero.
- A Transformer result below the CNN or showing severe train/inner/outer
  generalization gaps is a negative result. Do not expand model width/depth
  or tune on outer OOF labels. Follow-up label-free and q6/q8 experiments
  require a new inner-fold protocol.

## Execution gates

- **T0:** synthetic shape/gradient/parameter-count checks and patient-split
  tests pass; CPU one-fold fixture runs.
- **T1:** private Kaggle GPU encoder job completes eight folds with audited
  checkpoints and zero fold 9/10 access.
- **T2:** private Kaggle CPU q4 head job completes with matched controls and
  patient bootstrap. No clinical probability or quantum-hardware claim.
