# Calibrated divergence-distillation result

**Run date:** 2026-09-25  
**Kaggle kernel:** `swayamjeetbhagat4/aquire-med-divergence-kd-q4`, Version 1  
**Executed source:** `9d128ad185fbf8779ed730612e3ac551b8afb894`  
**Protocol:** [`divergence_distillation_protocol_2026-09-25.md`](divergence_distillation_protocol_2026-09-25.md)  
**Verdict:** **STOP divergence search; retain the hard-label VQC and calibrate later**

## Executive result

Pure reverse KL did not improve the primary discrimination endpoint. Its VQC
reached 0.82538 AUPRC versus 0.82567 for the hard-label VQC. Forward KL,
25%-reverse mixed KL and symmetric KL were also essentially tied with or
slightly below the hard-label control.

Jensen-Shannon was the best distillation objective at 0.82747 AUPRC. It
improved the hard VQC by 0.00180, but the paired patient-bootstrap interval was
[-0.00014, 0.00358]. It therefore missed both prespecified requirements: a
gain of at least 0.005 and an interval entirely above zero.

The JS VQC was numerically above identical-q4 logistic regression by 0.00058,
but its paired interval was [-0.00189, 0.00314]. This is parity, not evidence
of quantum superiority. The all-classical clinical-plus-q4-MLP fusion remained
the strongest end-to-end system at 0.83798 AUPRC.

## Standalone models

| Model | AUPRC | AUROC | Brier | Log loss | ECE-10 |
|---|---:|---:|---:|---:|---:|
| JS-distilled VQC | **0.82747** | 0.92164 | 0.10253 | 0.32554 | 0.07289 |
| q4 logistic | 0.82686 | **0.92187** | 0.11034 | 0.35012 | 0.08434 |
| q4 MLP | 0.82641 | 0.92156 | 0.10993 | 0.35081 | 0.08151 |
| Hard-label VQC | 0.82567 | 0.92046 | 0.11202 | 0.35405 | 0.08518 |
| Forward-KL VQC | 0.82565 | 0.92137 | 0.09763 | 0.32096 | 0.06690 |
| 25%-reverse mixed-KL VQC | 0.82551 | 0.92131 | 0.09739 | 0.32001 | 0.06503 |
| Pure reverse-KL VQC | 0.82538 | 0.92113 | **0.09663** | **0.31693** | **0.05878** |
| Symmetric-KL VQC | 0.82538 | 0.92125 | 0.09717 | 0.31906 | 0.06300 |
| Calibrated clinical teacher | 0.71624 | 0.87082 | 0.11871 | 0.37475 | 0.00617 |

Reverse KL produced the best uncalibrated VQC probability metrics, reducing
Brier by 0.01538 versus hard-label training with paired interval
[-0.01709, -0.01366]. That does not justify replacing the primary classifier:
it did not improve AUPRC, it slightly reduced sensitivity at the descriptive
90%-specificity operating point, and fold-9 probability calibration remains
the correct later stage for a frozen model.

## Primary paired comparisons

| First model minus second | Delta AUPRC | 95% patient-bootstrap interval | Decision |
|---|---:|---:|---|
| JS VQC − hard VQC | +0.00180 | [-0.00014, 0.00358] | no established gain |
| JS VQC − q4 logistic | +0.00058 | [-0.00189, 0.00314] | parity |
| JS VQC − q4 MLP | +0.00100 | [-0.00164, 0.00368] | parity |
| Forward-KL VQC − hard VQC | -0.00004 | [-0.00322, 0.00306] | parity |
| 25%-reverse VQC − hard VQC | -0.00019 | [-0.00338, 0.00292] | parity |
| Symmetric-KL VQC − hard VQC | -0.00032 | [-0.00352, 0.00275] | parity |
| Reverse-KL VQC − hard VQC | -0.00032 | [-0.00341, 0.00271] | parity |

JS improved AUPRC in six of eight folds, while each KL variant improved only
three. The JS improvement was still too small and uncertain for promotion.

## Meta-fusion

| Fusion | AUPRC | Brier | ECE-10 |
|---|---:|---:|---:|
| Clinical + q4 MLP | **0.83798** | **0.08875** | **0.01033** |
| Clinical + hard VQC | 0.83619 | 0.08919 | 0.01518 |
| Clinical + JS VQC | 0.83563 | 0.08963 | 0.01702 |
| Clinical + forward-KL VQC | 0.83351 | 0.09014 | 0.01498 |
| Clinical + pure reverse-KL VQC | 0.83334 | 0.09018 | 0.01491 |

JS quantum fusion lost to the matched all-classical fusion by 0.00239 AUPRC,
with paired interval [-0.00442, -0.00042]. Distillation increased agreement
with the clinical teacher and therefore reduced the complementary information
available to the fusion layer. This explains why calibrated standalone scores
did not translate into better fusion.

## Descriptive 90%-specificity operating point

| Model | Sensitivity | Hard-negative FPR |
|---|---:|---:|
| Hard VQC | **0.7697** | **0.1570** |
| JS VQC | 0.7676 | 0.1581 |
| Reverse-KL VQC | 0.7637 | 0.1605 |
| q4 logistic | 0.7621 | **0.1570** |
| Clinical + q4 MLP | **0.7809** | 0.1704 |
| Clinical + hard VQC | 0.7795 | 0.1706 |

These thresholds were derived descriptively from development OOF results and
are not final clinical thresholds.

## Execution integrity

- 17,348 ECGs from 14,958 patients across folds 1–8.
- Eight coherent Transformer archives and the signed 106-feature manifest.
- Inner-fold OOF clinical-teacher probabilities with outer-training sigmoid
  calibration.
- Same 2,000-per-class records, initialization, minibatch ordering, circuit
  and 30-epoch budget for every objective.
- All 48 VQC fold/objective runs had finite gradients and declining loss.
- Folds 9 and 10 remained sealed.
- Kaggle runtime was approximately 41 minutes on a Tesla T4.

## Decision

1. Do not promote forward, reverse, mixed, symmetric or JS distillation to the
   final VQC.
2. Do not run the conditional five-seed divergence confirmation because no
   objective passed the prespecified +0.005 AUPRC gate.
3. Retain the hard-label four-qubit VQC as the quantum-compliant core.
4. Perform probability calibration only after the complete model is frozen,
   using fold 9 as planned.
5. Report JS as a promising but inconclusive ablation and reverse KL as a
   calibration-improving, discrimination-neutral result.
6. Do not claim quantum predictive or computational advantage.

Raw artifacts are stored under
`kaggle_outputs/divergence_distillation_20260925_v1/divergence-distillation-q4-v1/`.
