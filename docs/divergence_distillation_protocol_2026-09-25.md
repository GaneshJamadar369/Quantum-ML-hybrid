# Bidirectional-divergence distillation protocol

**Frozen:** 2026-09-25  
**Phase:** G6Q-KD2  
**Purpose:** determine whether a calibrated classical teacher improves the
retained four-qubit VQC, and whether limited reverse-KL pressure adds value
beyond ordinary forward-KL distillation.

## Decision being tested

The earlier knowledge-distillation screen improved a small-budget VQC from
0.7559 to 0.7859 AUPRC and its clinical fusion from 0.8010 to 0.8150. That run
used in-sample, uncalibrated clinical-teacher probabilities and only 500
examples per class for 20 epochs. It cannot establish whether distillation
improves the retained strong VQC, which reached 0.82284 using 2,000 examples
per class and 30 epochs.

This protocol changes the training loss while freezing the strongest verified
representation and circuit. It does not open folds 9 or 10.

## Frozen prediction path

```text
10 s × 12 lead ECG
  -> validated 100 Hz preprocessing
  -> outer-fold ECG Patch Transformer
  -> h128
  -> outer-training robust scale + supervised PLS
  -> q4 in [-pi/2, pi/2]
  -> 4-qubit, 2-layer ring VQC
  -> MI-pattern score
```

The clinical teacher consumes the signed 106-feature deployable manifest. It
does not replace the quantum input or run inside the VQC.

## Leakage boundary and teacher construction

For outer held-out fold `k`:

1. Load only that fold's coherent Transformer training and validation
   representations.
2. Fit the q4 transform on the outer-training representation and labels.
3. For every outer-training record, obtain a clinical-teacher prediction from
   a HistGradientBoosting model trained without that record's official fold.
4. Fit a two-parameter sigmoid calibrator to these outer-training OOF logits.
5. Use calibrated OOF probabilities as student soft targets.
6. Retrain the clinical teacher on the complete outer-training partition,
   score outer validation, and apply the training-only calibrator.

Thus neither teacher targets, q4 transforms nor calibrators use the outer
validation labels. Patients remain confined to their official PTB-XL fold.

## Frozen loss screen

For teacher distribution `pT`, VQC distribution `pQ`, hard label `y`,
distillation weight `alpha=0.5`, and temperature `T=2`:

```text
L = (1-alpha) BCE(y, q-logit) + alpha T^2 Lsoft
```

| Name | Soft objective | Purpose |
|---|---|---|
| `hard` | none | exact retained VQC control |
| `forward_t2` | `KL(pT || pQ)` | stable conventional distillation |
| `mixed_rkl25_t2` | `0.75 KL(pT || pQ) + 0.25 KL(pQ || pT)` | limited mode-seeking pressure |
| `symmetric_kl_t2` | equal forward and reverse KL | symmetry ablation |
| `js_t2` | Jensen-Shannon divergence | bounded symmetric alternative |
| `reverse_t2` | `KL(pQ || pT)` | pure reverse-KL stress test |

Every objective receives the same 2,000 patient-unique MI and 2,000 non-MI
records, initialization, minibatch order, 30 epochs, learning rate, circuit
and exact statevector simulator. Divergence is the intended difference.

## Controls and outputs

The experiment saves complete OOF probabilities for:

- all six VQC objectives;
- identical-q4 logistic regression and 43-parameter MLP;
- the calibrated clinical teacher;
- cross-fitted nonnegative logistic fusion of the teacher with every VQC;
- the corresponding clinical-plus-q4-MLP fusion.

Primary endpoint is patient-pooled AUPRC. Secondary endpoints are AUROC,
Brier score, log loss and 10-bin expected calibration error. All important
comparisons use paired patient-cluster bootstrap intervals.

## Promotion gates

This first pass is a prespecified, single-seed screen.

1. **Distillation gate:** the best soft objective must improve the hard-label
   VQC by at least 0.005 AUPRC and its paired 95% interval must be above zero.
2. **Quantum predictive gate:** comparison with the best identical-q4
   logistic/MLP control is reported separately. Only an interval above zero
   supports a predictive quantum-benefit claim.
3. **Calibration gate:** a candidate with worse Brier and ECE is not promoted
   solely because of a very small AUPRC gain.
4. **Confirmation:** a candidate passing Gate 1 is rerun with five
   prespecified seeds. Gate 2 is not required to call distillation helpful,
   but it is required for a quantum predictive-advantage claim.
5. **Stop rule:** if no objective passes Gate 1, retain the hard-label VQC and
   stop divergence search. Fold 9 remains closed.

No conclusion from this simulator experiment establishes computational
quantum advantage or clinical readiness.

## Exact build order

1. Implement and unit-test forward KL, reverse KL, mixed KL and JS for binary
   logits.
2. Validate all eight Transformer archives and the 106-feature manifest.
3. Generate inner-OOF calibrated teacher targets inside each outer fold.
4. Train the six objectives under matched seeds and budgets.
5. Train identical-q4 classical controls.
6. Cross-fit the optional clinical/quantum fusion layer.
7. Produce patient-bootstrap comparisons and the frozen verdict.
8. Run five seeds only if the distillation gate passes.

Reverse-KL findings from large-vocabulary or generative models are not assumed
to transfer to binary ECG classification. Reverse KL has documented
mode-seeking behavior and can also create difficult optimization behavior;
this is why it is tested as a controlled component rather than adopted as the
default. See [Probability Distillation: A Caveat and
Alternatives](https://proceedings.mlr.press/v115/huang20c.html).
