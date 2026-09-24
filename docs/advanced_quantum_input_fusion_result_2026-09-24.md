# Advanced Classical-to-Quantum Input Experiments — Final Result

Date: 2026-09-24  
Development cohort: 17,348 ECGs / 14,958 patients / official folds 1–8  
Fold 9 accessed: **no**  
Fold 10 accessed: **no**

## Verdict

**STOP the learned fusion and angle-adapter branches. KEEP the frozen
Transformer → fold-local PLS-q4 → two-layer ring VQC as the quantum prototype
candidate. Do not claim quantum predictive advantage.**

The larger patient-unique training budget improved the unchanged q4 VQC to
`0.82284` AUPRC. On the exact same q4 coordinates, logistic regression reached
`0.82731`. The paired patient-bootstrap interval for VQC minus logistic was
`[-0.00681, -0.00206]`, so logistic remains the statistically supported winner.
The VQC was statistically indistinguishable from the 43-parameter MLP on AUPRC
(`0.82426`; interval `[-0.00437, 0.00157]`) and beat RBF-SVC (`0.77560`) by
`0.04722 [0.03928, 0.05591]`.

## Frozen input-fusion screen

Every arm used eight prespecified clinical groups covering all 106 approved
features, fold-local missing-value handling, frozen 100×96 Transformer patch
tokens, an h32 bottleneck and the same PLS-q4/head controls.

| Classical front end | Fusion probe | q4 VQC | Best identical-q4 control | Decision |
|---|---:|---:|---:|---|
| FiLM conditioning | 0.79163 | 0.74142 | Logistic 0.78804 | Stop |
| Rank-4 low-rank bilinear | 0.79266 | 0.72238 | Logistic 0.78154 | Stop |
| Clinical-query patch attention | 0.79002 | 0.72870 | Logistic 0.78531 | Stop |
| Cross-attention + bilinear | 0.71338 | 0.63822 | Logistic 0.70347 | Stop |

The frozen Transformer reference was `0.83250`. Every learned fusion encoder
replaced too much of that representation. Modality removal showed that the
models did use both inputs; usage did not translate into better generalization.

## Classical guidance experiments

| Experiment | Quantum AUPRC | Relevant control | Finding |
|---|---:|---:|---|
| Knowledge-distilled q4 VQC | 0.78592 | q4 logistic 0.82362 | Soft labels harmed VQC |
| h128 residual angle adapter | 0.77000 | unchanged VQC 0.82284 | Adapter overfit |
| h128 + clinical residual adapter | 0.76366 | unchanged VQC 0.82284 | Clinical correction overfit |

The residual adapters began as the identity and altered validation angles by
only about 0.07 radians, yet produced a large performance decline. The correct
classical contribution is therefore the stable supervised Transformer and
fold-local PLS reducer, rather than a trainable correction after PLS.

## Constrained geometry/circuit screen

A six-parameter orthogonal mixer rotated q4 in unconstrained tanh-latent space.
It preserved latent norm and was initialized to the identity.

| Circuit | Mixed VQC | Same-q4 logistic | Result |
|---|---:|---:|---|
| 3-layer ring | 0.80937 | 0.82856 | Stop |
| 3-layer ladder | 0.81062 | 0.83060 | Stop |

The mixer improved the corresponding three-layer ring base by about `0.0041`,
but the interval included zero and both three-layer variants remained below the
simpler two-layer VQC. Extra depth and extra ladder edges did not help.

## Best retained quantum architecture

```text
12-lead ECG [12,1000]
  → validated 100 Hz preprocessing
  → frozen fold-specific ECG Patch Transformer
  → h128
  → outer-training-only robust scaling + supervised PLS
  → q4 angles in [-π/2, π/2]
  → 4-qubit, 2-layer ring VQC
     RY data upload + trainable RZ-RY-RZ
     nearest-neighbour IsingZZ entanglement
     local Z and adjacent ZZ measurements
     linear quantum readout
  → classical calibration and operating threshold
  → MI-pattern probability
```

The development training screen used 2,000 patient-unique records per class,
30 epochs and an exact batched statevector simulator. The corresponding VQC
score was `0.82284` AUPRC, `0.92012` AUROC and `0.11026` Brier score.

## Research interpretation

The work found a useful near-parity quantum model, not a quantum winner. The
VQC learned the q4 representation much better than an RBF kernel and about as
well as a parameter-matched MLP, but linear logistic regression still used the
same supervised PLS coordinates more effectively. Because PLS is linear and
label-supervised, that result is plausible rather than surprising.

For the hackathon, present the platform as an honest model laboratory:

1. the QML core is fully trained and executable, rather than used only at
   inference;
2. every comparison uses the same patient folds and coordinates;
3. the platform automatically rejects expensive quantum configurations that
   do not pass a prespecified utility gate; and
4. the retained VQC is compact, near-term compatible and near the strongest
   identical-input classical control.

Do not report the VQC as beating classical ML. If the final product must use a
quantum core, deploy the retained two-layer VQC configuration and disclose the
`0.00447` AUPRC gap. Keep logistic as a visible scientific benchmark and safety
fallback.

## Next authorized modeling work

1. Freeze the retained q4 VQC and q4 logistic control before touching fold 9.
2. Run the promised five-seed confirmation for only those two heads.
3. Select probability calibration and the clinical threshold on fold 9 once.
4. Evaluate fold 10 once after every artifact and decision rule is frozen.
5. Build the prototype around the frozen VQC, calibration layer, abstention,
   explanation view and classical comparison dashboard.

No additional fusion architecture, qubit-width expansion or pooled-development
circuit search is justified by the current evidence.
