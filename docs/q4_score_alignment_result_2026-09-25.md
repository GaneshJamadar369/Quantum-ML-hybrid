# G6Q-CAL result — train-reference quantum score alignment

## Verdict

**Matched-input quantum gate: PASS. System and expansion gates: STOP.**

The frozen two-seed screen completed for 17,348 ECGs and 14,958 patients on
official folds 1–8. Folds 9 and 10 remained sealed. Each outer model used the
fixed narrow-bandwidth JS circuit, 60 cosine-scheduled epochs, three entangled
restarts, a no-entanglement control and score mappings derived exclusively from
the outer-training score distribution.

## Two-seed ensemble

| Model | AUPRC | AUROC | Brier |
|---|---:|---:|---:|
| stabilized raw VQC | 0.82913 | 0.92199 | 0.09205 |
| train-z VQC | 0.82948 | 0.92200 | **0.09203** |
| train-CDF VQC | **0.82980** | **0.92202** | 0.09247 |
| no-entanglement train-CDF | 0.82932 | 0.92172 | 0.09263 |
| q4 logistic | 0.82606 | 0.92195 | 0.09205 |
| q4 MLP ensemble | 0.82676 | 0.92188 | 0.09231 |
| clinical + train-CDF VQC | 0.83600 | 0.92659 | 0.08961 |
| clinical + q4 MLP | **0.83653** | **0.92736** | **0.08940** |

Train-CDF VQC minus the strongest identical-q4 control was `+0.00307` AUPRC,
with paired patient-bootstrap 95% interval `[+0.00107, +0.00506]`. Both screen
seeds showed a positive delta (`+0.00391` and `+0.00208`). This is evidence that
the stabilized quantum head learned a better ranking than the tested classical
heads on the same four coordinates.

This is a predictive comparison on a classical simulator. It is not evidence
of computational quantum advantage.

## Why the branch still stops

The frozen expansion rule required alignment, matched quantum and system gates
to pass together.

- Train-CDF minus stabilized raw VQC was only `+0.00067`, 95% interval
  `[-0.00007, +0.00143]`. The alignment-specific `+0.003` gate failed. Most of
  the improvement over the prior VQC comes from longer scheduled training and
  restart averaging, not CDF mapping alone.
- Train-CDF minus no-entanglement was `+0.00048`, interval
  `[-0.00075, +0.00178]`. Entanglement value remains unproven.
- Quantum fusion minus classical fusion was `-0.00052`, interval
  `[-0.00185, +0.00077]`.
- Quantum fusion `0.83600` remains below the frozen all-classical ceiling
  `0.83802`.

Therefore the protocol does not expand to five seeds and does not open fold 9.
Doing so would continue adapting to the same development outcomes without a
credible path to the system target.

## Clinical operating point

At approximately 90% specificity:

- train-CDF VQC sensitivity: `0.76763`; hard-negative FPR: `0.15664`;
- q4 MLP sensitivity: `0.76442`; hard-negative FPR: `0.15517`;
- quantum-fusion sensitivity: `0.77793`; hard-negative FPR: `0.17095`;
- classical-fusion sensitivity: `0.77656`; hard-negative FPR: `0.16893`.

The quantum route has a small sensitivity gain, but also a higher hard-negative
false-positive rate and lower system AUPRC/AUROC/calibration.

## Research interpretation

The 128-to-4 compression is not the whole limitation: a stabilized VQC can
beat classical heads given identical q4 coordinates. The remaining system
bottleneck is complementarity. The clinical branch and quantum branch make
nearly the same decisions, so fusion does not add enough independent evidence
to pass the full classical system.
