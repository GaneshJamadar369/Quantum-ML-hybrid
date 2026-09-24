# Advanced Classical-to-Quantum Input Fusion Protocol

Status: **running screen; folds 9 and 10 sealed**  
Frozen source implementation: `dfc416219affb9e6dbff3b4aa3547772510ec601`

## Question

Can a stronger classical representation front end improve the four values
presented to a four-qubit VQC enough for the VQC to outperform classical heads
trained on the identical four values?

This is an accuracy experiment on an exact quantum simulator. It cannot prove
computational quantum advantage.

## Patient-safe path

```text
12 x 1000 PTB-XL waveform
  -> frozen outer-fold ECG Patch Transformer
  -> 100 x 96 waveform tokens

106 deployable waveform-derived measurements + observed-value mask
  -> eight prespecified clinical group tokens

waveform tokens + clinical group tokens
  -> one frozen fusion arm
  -> h32
  -> outer-training-only robust scale + supervised PLS
  -> q4 in [-pi/2, pi/2]
  -> q4 VQC or an identical-input classical control
```

The eight clinical groups cover the approved manifest exactly:

| Group | Features |
|---|---:|
| Rhythm | 4 |
| Limb QRS | 29 |
| Precordial QRS | 27 |
| Inferior ST/T | 10 |
| High-lateral ST/T | 8 |
| Lateral ST/T | 8 |
| Anterior ST/T | 12 |
| Global/spatial | 8 |

## Parallel arms

1. **FiLM:** the clinical summary generates bounded scale and shift terms for
   every waveform patch.
2. **Low-rank bilinear fusion:** rank-four multiplicative interactions retain
   cross-modal effects without constructing a full outer product.
3. **Clinical-query cross-attention:** the eight clinical group tokens query
   the 100 unpooled waveform patches, followed by a gated residual path.

The Transformer stays frozen. Fusion training uses feature dropout, patch
dropout, AdamW, gradient clipping, a fixed epoch budget, and patient-unique
balanced samples. Imputation and scaling are fitted using only the outer
training records. Missingness masks survive imputation.

## Matched q4 controls

Each arm compares these heads on the same q4 coordinates and the same 500
patient-unique records per class:

- exact four-qubit, two-layer statevector VQC;
- the same VQC with entangling angles fixed to zero;
- logistic regression;
- a 43-parameter MLP, matched to the 45-parameter VQC;
- RBF-SVC.

The representation stage also records the frozen Transformer score, its
supervised fusion probe, and zero-clinical and zero-waveform ablations.

## Promotion rules

The representation passes only if:

1. fusion improves OOF AUPRC over the frozen Transformer by at least `0.005`;
2. the paired patient-bootstrap 95% interval is entirely above zero; and
3. removing either modality reduces AUPRC.

The VQC passes only if:

1. it beats the strongest identical-q4 classical head by at least `0.005`;
2. the paired patient-bootstrap 95% interval is entirely above zero; and
3. entanglement removal does not perform as well.

Only a passing arm is eligible for the five-seed confirmation. Parallel arms
are a prespecified screen; their comparisons must receive multiplicity
correction before any confirmatory claim.
