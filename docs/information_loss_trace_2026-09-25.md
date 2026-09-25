# Current architecture information-loss trace

## Scope

The audit aligns 17,348 OOF ECGs from 14,958 patients across official folds
1–8. Folds 9 and 10 remain sealed. Every representation probe is fitted on the
other seven outer folds.

The reported 0–100 score is:

```text
100 × (stage AUPRC − MI prevalence) /
      (current all-classical fusion AUPRC − MI prevalence)
```

MI prevalence is `0.251787`; the reference all-classical fusion is `0.838015`.
This is a relative discrimination index, not a literal percentage of Shannon
information retained.

## Predictive trace

| Stage | AUPRC | AUROC | Log loss | Relative discrimination |
|---|---:|---:|---:|---:|
| Transformer waveform head | 0.832500 | 0.923201 | 0.350062 | 99.06 |
| h128 full-data logistic probe | **0.833282** | **0.923980** | 0.344678 | 99.19 |
| PLS-q4 full-data logistic probe | 0.826981 | 0.922090 | 0.349220 | 98.12 |
| q4 full-data MLP assistant | 0.830849 | 0.922312 | **0.308334** | 98.78 |
| q4 sample-matched MLP | 0.828841 | 0.921824 | 0.350071 | 98.44 |
| hard VQC | 0.825671 | 0.920458 | 0.354051 | 97.89 |
| JS-distilled VQC | 0.827469 | 0.921635 | 0.325544 | 98.20 |
| curriculum VQC | 0.827239 | 0.921172 | 0.331910 | 98.16 |
| clinical + curriculum VQC | 0.836973 | 0.927067 | 0.289741 | 99.82 |
| clinical + q4 MLP | **0.838015** | **0.927516** | **0.288245** | 100.00 |

Patient-cluster comparisons:

- h128 logistic minus q4 logistic: `+0.00627` AUPRC,
  95% CI `[+0.00343,+0.00938]`;
- sample-matched q4 MLP minus JS VQC: `+0.00134`,
  CI `[-0.00076,+0.00345]`;
- curriculum VQC minus hard VQC: `+0.00157`,
  CI `[+0.00041,+0.00273]`;
- all-classical fusion minus quantum fusion: `+0.00104`,
  CI `[-0.00010,+0.00215]`.

## Geometry trace

Across folds, the validation embeddings had:

| Statistic | Mean |
|---|---:|
| h128 effective covariance rank | 13.69 |
| q4 effective covariance rank | 3.79 |
| h128/q4 pairwise-distance Spearman correlation | 0.703 |
| preserved 15-nearest-neighbour overlap | 0.161 |

Therefore PLS-q4 discards much of the local/global geometry while preserving
most label-aligned separation. The full-data q4 logistic deficit is real but
small, and a nonlinear q4 MLP recovers most of it. Expanding generic qubit
count is poorly motivated because prior q8/q12/q16 VQCs degraded strongly.

## Redundancy trace

Selected score correlations were:

| Scores | Spearman rho |
|---|---:|
| Transformer vs h128 logistic | 0.9842 |
| h128 logistic vs q4 logistic | 0.9801 |
| q4 MLP vs JS VQC | 0.9782 |
| quantum fusion vs classical fusion | 0.9898 |

The quantum path is not mainly missing the MI decision boundary. It is
reproducing nearly the same ordering slightly less accurately and adding very
little complementary evidence to the clinical expert.

## Research decision

The next justified experiment is a nested optimization of the shallow q4 VQC:
trainable input bandwidth, clinical-teacher JS, an AUPRC-aligned pairwise
ranking term, inner-selected learning rate and epoch, repeated over five
seeds. The experiment must export the eight quantum observables so loss can be
localized between q4, quantum measurement and classical readout.

Another encoder, larger generic latent vector, deeper circuit or additional
fusion architecture has lower prior probability because those branches have
already failed controlled gates.

Raw audit artifacts are stored in
`artifacts/information-loss-trace-v1/` and can be reproduced with
`run_information_loss_trace.py`.
