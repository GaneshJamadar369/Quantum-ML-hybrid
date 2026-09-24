# Independent dual-route fusion result

**Run date:** 2026-09-24
**Kaggle kernel:** `swayamjeetbhagat4/aquire-med-independent-dual-route-fusion`, Version 4
**Source commit executed:** `656a3e83c3445113d2bfad6ea8b624821cb02fcf`
**Protocol:** [`independent_dual_route_fusion_plan_2026-09-24.md`](independent_dual_route_fusion_plan_2026-09-24.md)
**Verdict:** **AVOID promotion as a quantum-beneficial architecture**

## Executive finding

The independent-route idea produced complementary predictions, but the
benefit came from the different input representation rather than from quantum
processing.

Route A combined the 106-feature clinical HistGradientBoosting expert with a
Transformer-h128 → PLS-q4 → four-qubit VQC. Its fusion AUPRC was 0.8010,
compared with 0.7125 for the clinical expert alone. However, logistic
regression on the identical Transformer q4 reached 0.8236, and the matched
classical fusion reached 0.8212. The VQC fusion therefore lost by 0.0201 AUPRC
to its matched all-classical fusion, with a paired 95% patient-bootstrap
interval of [-0.0252, -0.0149].

Route B also failed its value gate. QRS/rhythm HistGradientBoosting plus the
ST/T VQC improved over QRS/rhythm alone by 0.0151 AUPRC [0.0098, 0.0206], but
lost to the all-feature classical oracle by 0.0334 [-0.0404, -0.0263] and to
the same-route matched MLP fusion by 0.0071 [-0.0103, -0.0039].

The correct interpretation is: **independent representations help, but these
experiments do not show that the quantum circuit adds predictive value over
strong classical learners on the same coordinates.**

## Execution integrity

- Real-data preflight passed for 17,348 eligible ECGs from 14,958 patients.
- Only official development folds 1–8 were accessed; folds 9 and 10 remained
  sealed.
- All eight fold-specific Transformer archives passed record-ID, label,
  patient-separation, shape and OOF-coverage checks.
- Route B was the frozen disjoint partition: 60 QRS/rhythm fields and 46 ST/T
  fields from the signed 106-feature manifest.
- PLS, robust scaling and quantile angle mapping were fit within each outer
  training fold.
- VQC and every matched q4 control used the same patient-unique balanced
  training subset.
- The q4 VQC was an exact Torch statevector circuit with 45 trainable
  parameters. All eight Route A folds had finite gradients and declining
  loss.
- The fusion layer used meta-fold cross-fitting and nonnegative classical and
  quantum coefficients.
- Paired uncertainty used 2,000 patient-cluster bootstrap replicates.
- Kaggle completed in about 7.2 minutes after preflight. Local regression
  verification reported 108 passed tests and five environment-dependent
  skips.

## Primary results

| Model | Input | AUPRC | AUROC | Brier | Log loss |
|---|---|---:|---:|---:|---:|
| A2 Logistic | Transformer PLS-q4 | **0.8236** | 0.9210 | 0.1092 | 0.3445 |
| A4 matched classical fusion | clinical 106 + q4 MLP | 0.8212 | **0.9233** | **0.0924** | **0.2965** |
| A3 quantum fusion | clinical 106 + q4 VQC | 0.8010 | 0.9159 | 0.0960 | 0.3101 |
| A2 matched MLP | Transformer PLS-q4 | 0.8017 | 0.9154 | 0.1109 | 0.3505 |
| A1 VQC | Transformer PLS-q4 | 0.7559 | 0.8960 | 0.1290 | 0.4135 |
| A0 clinical expert | approved 106 features | 0.7125 | 0.8696 | 0.1195 | 0.3774 |
| B5 matched classical split | QRS/rhythm HistGB + ST/T q4 MLP | **0.6863** | **0.8592** | **0.1245** | **0.3900** |
| B2 clinical-family quantum fusion | QRS/rhythm HistGB + ST/T q4 VQC | 0.6792 | 0.8543 | 0.1262 | 0.3953 |
| B0 QRS/rhythm expert | 60 QRS/rhythm features | 0.6640 | 0.8412 | 0.1304 | 0.4089 |
| B1 ST/T VQC | ST/T PLS-q4 | 0.4501 | 0.7138 | 0.2138 | 0.6192 |

The raw Kaggle metric label `A2_*_h128` means the model consumed the q4
coordinates derived from h128; it did not consume all 128 coordinates. The
driver is corrected for future runs to name these artifacts
`A2_*_transformer_q4`.

## Paired patient-bootstrap conclusions

| Comparison, first minus second | Delta AUPRC | 95% interval | Result |
|---|---:|---:|---|
| A3 quantum fusion − A0 clinical expert | +0.0884 | [0.0794, 0.0975] | complementary waveform representation helps |
| A3 quantum fusion − A4 matched classical fusion | **−0.0201** | **[-0.0252, -0.0149]** | quantum fusion loses |
| A3 quantum fusion − shuffled quantum score | +0.0887 | [0.0798, 0.0978] | unshuffled second branch contains signal |
| A1 VQC − identical-q4 logistic | **−0.0676** | **[-0.0767, -0.0583]** | VQC loses |
| A1 VQC − identical-q4 MLP | **−0.0457** | **[-0.0546, -0.0367]** | VQC loses |
| A1 VQC − identical-q4 RBF-SVC | **−0.0216** | **[-0.0323, -0.0106]** | VQC loses |
| A1 VQC − identical-q4 Laplacian-SVC | +0.0057 | [-0.0057, 0.0173] | no established difference |
| B2 quantum fusion − B0 QRS/rhythm expert | +0.0151 | [0.0098, 0.0206] | second feature family helps |
| B2 quantum fusion − B3 all-feature oracle | **−0.0334** | **[-0.0404, -0.0263]** | forced routing loses |
| B2 quantum fusion − B5 matched MLP fusion | **−0.0071** | **[-0.0103, -0.0039]** | VQC fusion loses |
| B1 VQC − identical-q4 MLP | **−0.0340** | **[-0.0419, -0.0264]** | VQC loses |
| B1 VQC − identical-q4 logistic | **−0.0301** | **[-0.0386, -0.0217]** | VQC loses |
| B1 VQC − identical-q4 RBF-SVC | **−0.0154** | **[-0.0290, -0.0026]** | VQC loses |
| B1 VQC − identical-q4 Laplacian-SVC | **−0.0227** | **[-0.0346, -0.0107]** | VQC loses |

The extra matched-control intervals were computed after the Kaggle run from
the immutable downloaded OOF table. No model was retrained and no folds were
opened.

## Complementarity and operating point

Route A's classical and VQC logits had Spearman rho 0.6114 and Pearson r
0.6414. Route B had Spearman rho 0.3574 and Pearson r 0.3684. This confirms
that the branches were not redundant, but low correlation alone does not prove
that the weaker learner is useful.

At approximately 90% specificity:

| Model | Sensitivity | Hard-negative FPR | Hard-cohort AUPRC |
|---|---:|---:|---:|
| A3 quantum fusion | 0.7479 | 0.1742 | 0.8425 |
| A4 matched classical fusion | **0.7592** | 0.1726 | **0.8593** |
| B2 clinical-family quantum fusion | 0.5694 | **0.1831** | 0.7365 |
| B5 matched classical split | **0.5840** | 0.1897 | **0.7377** |

There is no operating-point result that reverses the primary AUPRC decision.
The thresholds above are descriptive development-OOF thresholds, not final
clinical thresholds. Fold 9 was not opened for calibration.

## Fusion coefficient audit

Route A's standardized nonnegative quantum coefficient was positive in all
eight meta-folds: median 1.444, range 1.409–1.526. Route B's was also positive
in every fold: median 0.464, range 0.452–0.475. The coefficients show stable
conditional association. They do not establish quantum benefit because the
matched classical branches exploit the same information more effectively.

## Frozen-gate decision

### R0 — data and route contract: pass for the screen

IDs, patients, folds, feature manifest and disjoint Route B feature lists were
validated before training. Full serialized transform hashes promised by the
larger protocol were not emitted and remain an engineering gap.

### R1 — representation health: partial pass

Coordinates, losses and gradients were finite; every fold trained. Observable
variance and quantum-geometry artifacts were not saved, so the complete R1
artifact contract is not satisfied.

### R2 — direct quantum prediction: fail the competitive-control criterion

The VQC trained successfully in all folds, but lost to strong identical-input
classical controls in both routes.

### R3 — independent fusion value: fail

Quantum removal and score shuffling hurt, and quantum coefficients were
positive and stable. Promotion still fails because:

1. both VQC fusions lost to their matched all-classical fusions;
2. Route B lost to the all-feature classical oracle;
3. each VQC branch lost to the strongest identical-input controls; and
4. the conditional five-seed stage is not justified after these negative
   matched-control results.

### R4 — claim language: negative result

The system may be called a **quantum-containing demonstrator**. It must not be
called quantum-beneficial or quantum-advantaged.

### R5 — sealed validation: stop before Fold 9

The architecture does not advance to calibration, threshold selection or
Fold-10 evaluation.

## What changes in the project

1. Stop the independent feature-routing strategy as an optimization path.
2. Retain Route A as an explainable ablation showing that learned waveform and
   clinical representations are complementary.
3. If the SIH prototype must contain a quantum core, present the q4 VQC as the
   mandated core demonstrator and show its matched classical controls in the
   research dashboard.
4. Do not claim that periodic or nonperiodic feature families determine
   quantum suitability; Route B directly contradicts that hypothesis.
5. Do not run q6, deeper circuits, noise studies or five-seed expansion for
   this route. The prespecified gate permits those costs only after an initial
   matched-control win.
6. Resume the main research roadmap with corrected repeated-seed waveform and
   classical-fusion validation. Select the best predictive system separately
   from the quantum-compliant demonstrator.

## Artifacts

Downloaded run artifacts are under
`kaggle_outputs/independent_dual_route_20260924/independent-dual-route-v1/`:

- `branch_oof_predictions.csv`
- `all_model_metrics.csv`
- `route_a_fusion_coefficients.json`
- `route_b_fusion_coefficients.json`
- `quantum_training_audit.json`
- `bootstrap_comparisons.json`
- `additional_patient_bootstrap.json`
- `operating_point_and_calibration.csv`
- `fold_metrics.csv`
- `complementarity_report.json`
- `additional_complementarity.json`
- `preflight.json`

The complete Kaggle execution log is stored at
`kaggle_outputs/independent_dual_route_20260924/aquire-med-independent-dual-route-fusion.log`.
