# Scientific task contract — AQUIRE-Med v0.2.0

## Intended task

For one 10-second, 12-lead ECG, estimate whether the waveform contains an MI-superclass pattern (`MI=1`) or no MI-superclass pattern (`MI=0`) under the PTB-XL v1.0.3 annotation ontology. This is contemporaneous pattern classification, not prediction of a future myocardial infarction.

## Population and target

- Dataset: PTB-XL v1.0.3, exactly 21,799 ECG records.
- Positive target: at least one SCP code whose `diagnostic_class` is `MI`.
- Negative target: no MI-superclass SCP code; overlapping non-MI diagnoses remain attached.
- Hard cohort: MI positives versus abnormal non-MI records with STTC, CD or HYP diagnostic classes.
- Annotation likelihood is evidence metadata, not a patient risk probability.

## Inputs and deployability

The primary input is the complete official 100 Hz waveform in canonical lead order, shape `12×1000`, physical unit mV. Any inference feature must be reproducible locally from a newly submitted waveform. PTB-XL+ 12SL and Uni-G features are oracle/reference ablations. ECGDeli is an open reference used to validate local measurements.

## Splits

Folds 1–8 are development folds. Fold 9 is reserved for later calibration/validation. Fold 10 is locked until final evaluation. No threshold, feature, architecture or model choice may use folds 9 or 10. Patient overlap across roles must be zero.

## Primary evidence

The primary model-selection metric is patient-pooled OOF AUPRC. Secondary evidence is AUROC, sensitivity at 90% specificity, specificity, F1, log loss, Brier score, calibration error, stability, latency and cost. QC thresholds are selected against technical annotations, never downstream classification accuracy.

## Output

A calibrated MI-pattern probability, binary decision at a frozen threshold, QC state and possible abstention. It is not a diagnosis or a statement that a patient has an 87% chance of disease.

## Claims excluded at this stage

No clinical-deployment, future-event-prediction or quantum-advantage claim is permitted. Quantum experiments begin only after G0–G5 and a reproducible strong classical champion.
