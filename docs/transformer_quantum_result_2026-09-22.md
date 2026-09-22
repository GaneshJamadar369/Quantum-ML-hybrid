# Compact ECG Transformer and q4 quantum-head result — 2026-09-22

## Outcome

**GO for the compact Transformer as a promising representation candidate;
MODIFY before selecting a quantum champion.** The Transformer substantially
improved all compressed heads over their CNN-input versions. The direct q4
VQC became the highest-scoring head in this one-seed three-head screen, but
its 0.0035 AUPRC lead over the same-input MLP is not statistically resolved.
No quantum advantage, calibrated disease probability or clinical-use claim
follows from this result.

Both private Kaggle jobs completed:

- [Transformer GPU encoder export](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-compact-ecg-transformer-representation), local output `kaggle_outputs/transformer_representation_20260922/transformer-representation-v1/`.
- [Transformer-input q4 VQC, MLP and RBF screen](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-phase-6q-h-transformer-vqc), local output `kaggle_outputs/transformer_quantum_20260922/transformer-quantum-q4-screen/`.

The jobs checked out pinned code revision `9678e1f`. All reported numbers
come from 17,348 held-out ECGs of 14,958 patients on PTB-XL development
folds 1–8. No patient crossed training and validation roles. Folds 9 and 10
remained sealed. Every fold produced finite 128-dimensional training and
validation embeddings from **one common fold-specific encoder**; the 17,348
held-out ECG IDs were unique and all eight encoder hashes were distinct.

## Encoder and overfitting audit

The exact frozen layer configuration is in
[`transformer_quantum_plan_2026-09-22.md`](transformer_quantum_plan_2026-09-22.md):
12×1000 input; 100 non-overlapping 100-ms patches; width 96; four attention
heads; three pre-norm encoder layers; feed-forward width 192; mean-plus-max
pooling; 128-dimensional representation; 271,041 parameters. The preceding
1D ResNet has 2,309,953 parameters. Only small amplitude/noise augmentation
was applied in training. Patient-separated inner validation chose the number
of epochs (11–19 across folds); a fresh model was then trained on all seven
outer-training folds for that number of epochs. The outer fold did not set
the stopping epoch.

| Supervised waveform encoder | Raw OOF AUPRC | Raw OOF AUROC | Mean train minus outer AUPRC gap |
|---|---:|---:|---:|
| Existing 1D ResNet, fixed 20 epochs | 0.7878 | 0.8826 | 0.2109 |
| Compact Transformer, inner-selected epochs | **0.8325** | **0.9232** | **0.0684** |

Transformer versus CNN raw encoder delta AUPRC was approximately +0.0445
with a 95% patient-cluster bootstrap interval of [0.0349, 0.0549]. The
Transformer's per-fold train/outer AUPRC gaps were 0.041–0.087, so
**overfitting is reduced but not absent**. Inner-validation AUPRC was
0.826–0.873 and outer-fold AUPRC was 0.814–0.859; individual inner-minus-
outer gaps ranged from -0.025 to +0.033. The two encoders had different
training schedules and augmentation, so the observed gain is attributable to
the **whole architecture-plus-training recipe**, not attention alone. A
same-protocol repeated-seed CNN/Transformer experiment is still needed.

## Quantum bottleneck and matched heads

For each outer fold, the same 128-dimensional Transformer vectors were
reduced to four train-only PLS/quantile angles. A patient-unique sample of
500 MI and 500 non-MI ECGs trained each head; all heads scored that fold's
held-out patients. The q4 VQC had two data re-uploading layers, sparse
Ising-ZZ entanglement and a linear readout (45 trainable parameters). It ran
on a CPU statevector simulator, not quantum hardware.

| Four-coordinate head | CNN input AUPRC | Transformer input AUPRC | Transformer AUROC |
|---|---:|---:|---:|
| Direct q4 VQC | 0.7325 | **0.8148** | **0.9172** |
| Small MLP | 0.7583 | 0.8112 | 0.9158 |
| RBF-SVC | 0.7246 | 0.7747 | 0.9035 |

The Transformer minus CNN VQC delta AUPRC was +0.0819 [0.0683, 0.0958]
under paired patient bootstrap. The corresponding MLP gain was +0.0527
[0.0406, 0.0654]; RBF gained +0.0500 [0.0347, 0.0649]. The exploratory
difference in VQC-versus-MLP *uplift* was +0.0291 [0.0183, 0.0405]. This
shows that the new representation closed the prior VQC deficit; it does not
show an absolute quantum win. Multiple candidate designs have already been
tried, and these intervals do not correct for that selection.

Within the Transformer representation, VQC minus MLP delta AUPRC was
+0.0036 [-0.0020, 0.0088], which crosses zero. VQC minus RBF was +0.0399
[0.0324, 0.0480], but RBF was not the strongest classical head. VQC
training loss ended around 0.229–0.290 across folds, with finite, nonzero
gradients. Full Transformer encoder AUPRC 0.8325 versus compressed VQC
0.8148 suggests only a modest four-dimensional bottleneck loss in this
configuration; the CNN analogue lost much more (0.7878 to 0.7325).

## Interpretation and next gate

The Transformer was trained on MI labels, so its embedding already contains
MI-discriminative information. The quantum circuit is a genuine trained
predictive head on that representation, but the result does not prove the
circuit is indispensable. Scores are uncalibrated monotonic transforms, not
patient risk probabilities. Commercial PTB-XL+ measurements were not used
in this waveform screen.

Before choosing a champion, run at least five prespecified seeds for the
Transformer q4 VQC and tuned parameter-count-matched MLP, plus strong RBF
and Laplacian controls on **identical patient selections and fold-local
angles**. Use inner folds for learning rate, circuit bandwidth and stopping
choices; do not tune on the pooled outer OOF labels. Include a label-free
Transformer representation arm to test the problem statement's quantum-core
claim more cleanly. Keep q6/q8 as prespecified follow-ups only if q4 remains
stable and the quantum head beats the strongest classical control with a
patient-bootstrap interval entirely above zero. Clinical deployment,
external validation, probability calibration and Fold-10 evaluation remain
outside this development screen.
