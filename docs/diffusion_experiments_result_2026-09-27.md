# G6Q-DM and G6Q-DPE results — diffusion representation experiments

## Execution validity

Both Kaggle GPU jobs completed successfully.

- Cohort: 17,348 ECGs from 14,958 patients
- Development folds: 1–8 only
- Fold 9/10 access: none
- Outer-fold patient overlap: zero
- Diffusion-pretrained encoders: eight distinct EMA checkpoint checksums
- All predictions: finite and complete OOF

These are valid negative experiments rather than execution failures.

## Diffusion Map q4

| Predictor | AUPRC | AUROC | Brier |
|---|---:|---:|---:|
| Retained PLS-q4 VQC, paired seed 42 | **0.82938** | **0.92180** | **0.09254** |
| Diffusion-Map q4 VQC | 0.81350 | 0.91540 | 0.09584 |
| Diffusion-Map q4 MLP | 0.81040 | 0.91462 | **0.09525** |
| Diffusion-Map q4 logistic | 0.79865 | 0.91272 | 0.09591 |
| Diffusion-Map q4 VQC without entanglement | 0.80676 | 0.91355 | 0.09676 |

The Diffusion-Map VQC lost `0.01589` AUPRC versus paired PLS-q4 VQC;
patient-bootstrap 95% CI `[-0.02184, -0.01064]`. It exceeded the matched MLP
by `0.00307`, but the interval `[-0.00075, +0.00665]` crossed zero. Therefore
there is no established identical-input quantum win.

Entanglement helped within this representation: VQC minus the
no-entanglement circuit was `+0.00658`, CI `[+0.00344, +0.01001]`. This is a
valid circuit ablation, but it does not rescue the weaker representation or
establish system-level quantum value.

### Fusion

| Fusion | AUPRC | Brier |
|---|---:|---:|
| Retained PLS-q4 VQC fusion | **0.83546** | **0.08972** |
| Diffusion-Map VQC fusion | 0.82608 | 0.09157 |
| Diffusion-Map MLP fusion | 0.82590 | 0.09103 |
| Frozen all-classical ceiling | **0.83802** | **0.08895** |

Diffusion-Map fusion lost `0.00936` AUPRC versus retained VQC fusion, CI
`[-0.01300, -0.00596]`, and remained `0.01194` below the classical ceiling.

### Geometry diagnosis

- Mean pairwise-distance Spearman correlation with h128: `0.6980`
- Mean 15-neighbour overlap: `0.1500`
- Mean validation h128 effective rank: `20.44`
- Mean Diffusion-Map q4 effective rank: `2.15`

The unsupervised manifold coordinates retained broad distance structure but
collapsed most four-dimensional variance into roughly two effective axes.
Those axes represent dominant ECG variation, not necessarily MI-discriminative
variation. Supervised PLS remains the better q4 compressor for this target.

## Diffusion-pretrained ECG encoder

Pretraining itself behaved correctly. Across all outer folds, min-SNR noise
loss declined from approximately `0.818–0.828` to `0.724–0.733`. Validation
representation effective rank ranged from `29.35` to `37.52`, with finite
features and nonzero variance. The weak prediction is therefore not numerical
collapse.

| Predictor | AUPRC | AUROC | Brier |
|---|---:|---:|---:|
| Supervised Transformer h128 logistic probe | **0.83253** | **0.92356** | — |
| Diffusion h128 logistic probe | 0.53241 | 0.76999 | — |
| Diffusion h128→PLS-q4 MLP | **0.50418** | **0.74525** | **0.16015** |
| Diffusion h128→PLS-q4 VQC | 0.50265 | 0.74454 | 0.16142 |
| Diffusion h128→PLS-q4 logistic | 0.49962 | 0.74278 | 0.16071 |

The full h128 diffusion representation lost `0.30008` AUPRC versus the
supervised Transformer probe, patient-bootstrap CI
`[-0.31659, -0.28384]`. Thus most loss occurs in the pretrained encoder before
PLS or the VQC. The denoising objective learned general voltage and waveform
statistics but did not emphasize MI-specific ST/T/Q-wave morphology.

The diffusion-input VQC also failed its matched control: VQC minus q4 MLP was
`-0.00146`, CI `[-0.00476, +0.00172]`. Its entanglement comparison was
unresolved.

### Fusion

| Fusion | AUPRC |
|---|---:|
| Retained PLS-q4 VQC fusion | **0.83546** |
| Diffusion-encoder q4 MLP fusion | 0.71546 |
| Diffusion-encoder VQC fusion | 0.71467 |
| Clinical teacher alone | 0.71162 |

Diffusion-encoder VQC fusion lost `0.12079` AUPRC versus retained VQC fusion,
CI `[-0.13078, -0.11120]`, and remained `0.12334` below the classical ceiling.

## Decision

**STOP both diffusion branches.** Neither improves prediction, calibration or
system score. Do not spend compute on confirmation seeds, larger landmark
graphs, longer DDPM training or generative augmentation under the current
target. Preserve the retained supervised Transformer → fold-local PLS-q4 →
train-aligned four-qubit VQC as the best quantum path. These experiments remain
useful negative ablations and evidence that label-aware compression is critical
for the MI-pattern task.

