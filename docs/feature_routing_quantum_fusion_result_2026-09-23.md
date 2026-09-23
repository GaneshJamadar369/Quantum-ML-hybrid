# Evidence-routed residual q4 fusion result — 2026-09-23

## Outcome

**STOP the residual-fusion branch; it is not a quantum winner.** The eight-fold
Kaggle GPU study completed successfully on 17,348 ECGs from 14,958 patients.
All outer validation patients were isolated, inner classical predictions were
OOF, the approved 106-feature manifest was enforced, and folds 9 and 10 were
not accessed.

Cloud job: [AQUIRE-Med Residual q4 Feature Routing](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-residual-q4-feature-routing)

The frozen protocol and research basis are documented in
[`feature_routing_quantum_fusion_plan_2026-09-23.md`](feature_routing_quantum_fusion_plan_2026-09-23.md).

## Why the representation was changed

The preliminary OOF junction audit found that the previous supervised-PLS q4
VQC and Transformer scores were nearly redundant (Spearman rho 0.9510). A
diagnostic Transformer+clinical stack had AUPRC 0.84102; adding the old q4 VQC
changed it only to 0.84111, and the mean VQC meta-coefficient was 0.026.

The new experiment therefore trained q4 inputs from directions that covaried
with the errors of a properly nested classical expert. Three source banks were
tested: h128 waveform embeddings, 106 approved clinical measurements, and
their combined space. Each produced four fold-local residual PLS coordinates.
Every VQC had an identical-input, parameter-count-matched MLP control.

## Main result

The properly nested classical expert reached AUPRC **0.837996** and AUROC
**0.926493**. At the prespecified primary fusion weight 0.25:

| Residual source | q4 VQC fusion AUPRC | Δ vs classical | 95% patient-bootstrap CI | Δ vs matched-MLP fusion | 95% CI |
|---|---:|---:|---:|---:|---:|
| h128 | 0.837933 | -0.000061 | [-0.000338, 0.000202] | -0.000155 | [-0.000419, 0.000111] |
| clinical 106 | 0.837174 | -0.000828 | [-0.001716, 0.000174] | -0.000747 | [-0.001564, 0.000141] |
| h128 + clinical | 0.837544 | -0.000441 | [-0.000992, 0.000087] | -0.000026 | [-0.000393, 0.000354] |

No interval is entirely above zero, no point estimate exceeds the classical
expert, and every result is far below the prespecified +0.005 AUPRC minimum.

The q4 heads alone also failed their same-input controls:

| Residual representation | q4 VQC AUPRC | Matched MLP AUPRC |
|---|---:|---:|
| h128 | 0.747556 | **0.753982** |
| clinical 106 | 0.335657 | **0.380783** |
| h128 + clinical | 0.447571 | **0.498149** |

The strongest exploratory fusion was classical, not quantum:
clinical-residual MLP at weight 0.10 reached AUPRC 0.838123, only +0.000127
above classical-only. It is below the effect-size gate and is not promoted.

## Fold and clinical operating-point checks

The primary h128 VQC fusion improved AUPRC in four of eight folds, with fold
deltas ranging from -0.000465 to +0.000418. Clinical-residual and combined-
residual VQC fusion improved only three folds each. The small pooled changes
therefore do not hide a stable cross-fold gain.

At the threshold giving 90% specificity:

| Model | Sensitivity | Hard-negative FPR |
|---|---:|---:|
| Classical expert | **0.775183** | 0.167278 |
| h128 VQC fusion, weight 0.10 | 0.774496 | 0.166911 |
| h128 VQC fusion, weight 0.25 | 0.774954 | **0.166728** |
| clinical VQC fusion, weight 0.25 | 0.774954 | 0.168012 |
| combined VQC fusion, weight 0.25 | 0.774725 | 0.167278 |

The largest hard-negative FPR decrease was 0.00055 and came with lower
sensitivity. This is not a clinically meaningful trade.

## Training validity

This is a trained negative result:

- every fold and representation completed with finite outputs;
- VQC median gradient norms remained nonzero, approximately 0.058–0.199;
- h128 VQC weighted loss declined in every fold, for example 0.738→0.446 in
  fold 1 and 0.697→0.437 in fold 8;
- patient overlap was zero and each fold stored source/training hashes;
- 1,000 patient-unique ECGs trained every quantum and matched-MLP head;
- the same q4 circuit, epochs, weights and source coordinates were used in the
  matched comparison.

The failure is attributable to lack of unique generalizing information in the
tested residual quantum mappings, not vanishing gradients or an interrupted
job.

## Feature-routing interpretation

The statistical audit remains useful for understanding feature roles, but it
does not justify a hard classical/quantum partition.

- **Classical core:** validated lead-level QRS amplitude/R:S balance, ST60,
  T polarity, regional ST/QRS/T composites, rhythm measurements, and h128.
  Their strong main effects are handled efficiently by the classical experts.
- **Nested-ablation hypotheses:** aVL R amplitude/R:S ratio, lead-I range,
  V1 R amplitude/R:S ratio, V2/V4 ST60, aVF range, and high-lateral R/R:S
  composites showed stable pooled associations with classical residuals and
  quantum/classical loss differences.
- **Excluded from both branches:** the 20 frozen manifest exclusions,
  including invalid 100 Hz interval/QTc fields, V2/V3 R amplitudes that failed
  the ECGDeli error gate, and constant valid-fraction fields.

The full nested experiment allowed all approved measurements to form residual
clinical coordinates and still found no quantum gain. Therefore the pooled
candidate list should remain an explanatory audit, not become a new selected
quantum feature set.

## Decision

Do not spend fold 9, fold 10, additional seeds or learned fusion weights on
this branch. The frozen rule allowed those steps only after a positive fixed-
weight result. A learned gate on near-zero residual value would add selection
variance and could manufacture an apparent gain.

For the SIH prototype, the defensible configuration remains:

1. classical ECG preprocessing and compact Transformer representation;
2. the four-qubit VQC as the required quantum-core demonstration;
3. q4 logistic and matched MLP as mandatory scientific controls;
4. an explicit statement that quantum predictive advantage has not been
   established on PTB-XL;
5. classical calibration and thresholding only after the final model is frozen.

The best research result from this phase is the tested conclusion that
feature splitting and residual score fusion do not make the current q4 VQC a
winner. The model-selection record remains complete and falsifiable.

