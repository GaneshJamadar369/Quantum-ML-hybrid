# G6Q-DPE — diffusion-pretrained ECG encoder

## Question

Can label-free denoising-diffusion pretraining learn a more useful ECG `h128`
representation for the retained quantum core than the previous label-free JEPA
encoder, without allowing a classical diagnostic head to make the final MI
prediction?

## Architecture

```text
PTB-XL waveform float32[12,1000]
  -> existing fold-local normalization
  -> Gaussian forward diffusion at random timestep t
  -> forced-bottleneck temporal encoder/decoder predicts epsilon
  -> EMA diffusion encoder on the clean ECG exports h128
  -> existing fold-local supervised PLS q4
  -> existing 4-qubit, 2-layer ring VQC with Z/ZZ observables
  -> train-reference CDF alignment
  -> cross-fitted probability calibration and clinical fusion
```

The diffusion decoder has no waveform skip connections, forcing denoising
information through its 96-channel temporal bottleneck. The exported `h128`
combines mean and maximum temporal pooling of that bottleneck. The quantum
classifier, not the diffusion model, produces the MI/non-MI score.

## Leakage controls

- Eight separate diffusion encoders, one per outer fold.
- Encoder training receives only normalized outer-training waveforms, random
  timesteps and generated Gaussian noise.
- `mi_label`, hard-negative state and clinical features are absent from the
  pretraining function signature.
- Epoch count and hyperparameters are frozen before execution; outer-validation
  diffusion loss does not select checkpoints.
- EMA encoder exports training and outer-validation coordinates from the same
  fold-specific checkpoint.
- Folds 9 and 10 remain sealed.

## Frozen configuration

- cosine 1,000-step forward diffusion schedule
- epsilon prediction with min-SNR weighting, gamma 5
- compact temporal channels 32/48/64/96 and a 128-dimensional projection
- AdamW, learning rate `2e-4`, weight decay `0.02`
- 8 epochs per outer fold, batch size 128
- EMA decay `0.999`
- existing VQC recipe: PLS-q4, 2,000 patient-unique samples per class, 60
  epochs, three restarts, JS objective and train-CDF score alignment

## Required evidence

- finite decreasing diffusion loss and gradients in every outer fold
- non-collapsed representations: finite values, nonzero minimum feature
  standard deviation and effective-rank report
- exact 17,348-record OOF coverage and zero patient overlap
- identical-q4 logistic, MLP and no-entanglement controls
- paired patient-cluster bootstrap against the retained supervised-Transformer
  VQC and its all-classical fusion

## Decision gates

The branch is promoted only if the diffusion-input VQC:

1. reaches at least the retained train-aligned VQC result (`0.82980` AUPRC),
2. beats its strongest identical-q4 classical control with a positive paired
   interval, and
3. produces fusion above the frozen all-classical ceiling (`0.838015` AUPRC).

Failure stops diffusion-pretraining expansion. Passing all gates permits a
confirmation seed; it does not open folds 9 or 10 automatically.

