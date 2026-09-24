# G6Q-KD3 — Clinical concept and teacher-assistant distillation plan

## Decision question

Can training-only, waveform-derived clinical concepts or a capacity-matched
teacher assistant improve the retained four-qubit VQC without changing its
inference architecture or accessing folds 9–10?

This experiment translates “distill rationales” into measurable ECG concepts.
It does **not** generate natural-language explanations, use diagnostic reports,
or expose the student to the MI label through a concept target.

## Frozen data and prediction path

The study uses the same 17,348 primary development ECGs, official folds 1–8,
fold-coherent Transformer h128 representations, outer-fold PLS q4 map,
patient-unique 2,000-per-class training sample, two-layer four-qubit ring VQC,
30 epochs, and exact statevector simulator as G6Q-AF and G6Q-KD2.

The deployed path remains:

`ECG -> frozen fold-specific Transformer -> PLS q4 -> 4-qubit VQC -> MI logit`

The auxiliary concept head is discarded after training. Removing the VQC still
breaks the prediction path.

## Clinical concept vector

Ten prespecified, deployable v0.4 measurements are predicted from the VQC's
four Z and four ring-ZZ expectation values:

1. RR coefficient of variation
2. global ST RMS
3. global positive-ST lead count
4. global negative-ST lead count
5. global T-inversion lead count
6. inferior ST mean
7. high-lateral ST mean
8. anterior ST mean
9. precordial transition lead
10. frontal-axis proxy

These values are produced locally from a submitted waveform. MI codes,
diagnostic statements, report text, infarct stage, PTB-XL+ commercial outputs,
and failed interval families are excluded. Missing masks are retained. Median,
center and IQR statistics are fit separately inside each outer training fold;
targets are bounded with `tanh`.

## Teacher assistant

A seven-hidden-unit q4 MLP acts as the intermediate-capacity assistant. Its
student soft targets are inner-official-fold OOF predictions, then sigmoid
calibrated on those outer-training OOF scores. A full outer-training assistant
predicts the outer validation fold. No outer validation label enters teacher
training, calibration, concept scaling, PLS fitting, or VQC optimization.

The reliability arm weights only the soft JS term. Weight combines assistant
confidence with bin-level calibration agreement, preventing confidence alone
from rewarding confidently wrong answers. The hard-label term remains
unweighted.

## Prespecified arms

| Arm | Epochs 1–10 | Epochs 11–30 |
|---|---|---|
| hard | hard labels (all 30 epochs) | hard labels |
| assistant_js_full | 0.5 hard + 0.5 assistant JS (all 30) | same |
| assistant_two_stage | 0.5 hard + 0.5 assistant JS | hard labels |
| concept_two_stage | 0.5 hard + 0.5 concept | hard labels |
| concept_assistant_two_stage | 0.4 hard + 0.3 JS + 0.3 concept | hard labels |
| reliable_concept_assistant | same, calibration-aware JS weights | hard labels |

All arms receive identical q4 inputs, initialization seed, shuffle seed,
optimizer, batches, circuit, MI readout, training sample and epoch budget.
Identical-q4 logistic and matched MLP controls are retained.

## Outputs and gates

Save complete OOF probabilities, MI metrics, concept MAE/correlation/coverage,
fold audits, gradients, assistant calibration, and 2,000-replicate paired
patient-cluster bootstraps.

Advance the best distillation arm to five prespecified seeds only when:

- AUPRC improves by at least 0.005 over the hard VQC;
- the paired 95% interval is entirely above zero; and
- no fold, patient, concept-transform, or teacher-target leakage check fails.

Report a quantum predictive win only if the paired interval versus the best
identical-q4 classical control is also above zero. Even then, this establishes
predictive evidence under simulation, not computational quantum advantage.

If the primary gate fails, stop concept/assistant tuning, retain the hard VQC,
and move to frozen fold-9 calibration after the final model-selection decision.
