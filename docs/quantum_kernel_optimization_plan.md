# Quantum-kernel optimization plan

## Objective and claim boundary

The objective is to determine whether a quantum feature map adds predictive
information beyond the strongest classical kernel on the **same representation,
patients, labels, sample budget, tuning budget and calibration protocol**. The
experiment cannot guarantee a quantum win. It must be able to conclude that no
utility was demonstrated.

The current IQP fidelity kernel is a baseline, not an optimized quantum model.
Its AUPRC was 0.4357, compared with 0.4933 for the matched Laplacian SVC. Its
per-fold Gram-matrix condition numbers were approximately `8.9e7–3.7e10`, so
improving kernel geometry is the first problem.

## Why the current feature map underperformed

1. `z8` was unsupervised PCA over tabular measurements rather than a
   disease-oriented ECG representation.
2. `StandardScaler → tanh → π` was fixed, saturating some features and providing
   no learned feature-wise bandwidth.
3. The IQP circuit used a generic dense interaction pattern rather than ECG or
   data-derived feature relationships.
4. Global state fidelity can concentrate and produce a nearly singular or
   identity-like Gram matrix.
5. Circuit depth, repeats, SVC `C`, encoding scale and kernel regularization were
   not selected jointly by nested development validation.
6. Only 500 examples per class trained each fold, with no learning-curve or
   hard-negative sampling study.

## Representation track

Run each representation as a separate prespecified experiment. Every classical
and quantum head receives the identical representation.

### Q-R1 — Clinically structured tabular representation

- Begin with the signed 106-feature deployable manifest.
- Group features by rhythm/RR, QRS, R amplitude, ST/T morphology, lead ratios,
  reciprocal-lead relationships and QC.
- Learn a fold-local supervised projection with sparse PLS, supervised
  contrastive projection or a small regularized encoder.
- Produce `z4`, `z6` and `z8`; record label-free PCA as the control.
- Order latent coordinates by clinical group or learned mutual information so
  circuit edges have a defined meaning.

### Q-R2 — Frozen waveform-encoder representation

- Train/freeze the best development ECG encoder first.
- Extract patient-safe OOF embeddings; never extract training representations
  from a model that was trained on the same held-out patient.
- Learn a fold-local bottleneck from 128/384 dimensions to `z4/z6/z8` using
  supervised contrastive loss plus reconstruction or variance preservation.
- Give the same bottleneck output to all classical and quantum heads.

This is the most credible route to higher end-to-end performance because the
waveform encoder retains continuous ST/T/QRS morphology that tabular PCA lost.

### Q-R3 — OOF residual/hard-negative representation

- Generate development OOF logits from the frozen classical champion.
- Train a second-stage model only on OOF information: encoder embedding,
  champion logit, uncertainty and hard-negative indicators available at
  inference.
- Compare direct classification with residual correction and a hard-negative
  specialist.
- Match each quantum residual model with linear, MLP, RBF and Laplacian residual
  controls.

## Quantum feature-map family

For a shallow problem-shaped IQP map, test

\[
U_{\theta,\alpha}(z)=\prod_{\ell=1}^{L}
R(\theta_\ell)\exp\left[i\sum_j \alpha_j z_j Z_j
+i\sum_{(j,k)\in E}\beta_{jk}g_j(z_j)g_k(z_k)Z_jZ_k\right]H^{\otimes n}.
\]

- `n ∈ {4, 6, 8}` qubits.
- `L ∈ {1, 2, 3}` data-reuploading layers; deeper circuits require explicit
  evidence that kernel concentration does not worsen.
- Trainable feature scales `alpha_j` replace a single fixed `π` scale.
- Sparse `E`: ring, clinically specified pairs and training-only correlation or
  mutual-information graph.
- `g_j`: identity, clipped robust rank or low-order nonlinear transform.
- Trainable rotations and interaction weights are optimized with centered
  kernel-target alignment inside inner training folds only.
- Penalize alignment solutions with collapsed off-diagonal variance or very low
  effective rank.

## Projected quantum kernel: primary candidate

Global fidelity is retained as a baseline. The primary candidate measures local
observables after the quantum map:

\[
q(z)=[\langle X_j\rangle,\langle Y_j\rangle,\langle Z_j\rangle,
\langle Z_jZ_k\rangle]_{j,(j,k)\in E},
\]

then constructs

\[
K_{PQ}(z,z')=\exp[-\gamma\|q(z)-q(z')\|_2^2].
\]

Tune `gamma` inside the training fold. Compare one-qubit observables, local
two-qubit correlations and their concatenation. This preserves local quantum
geometry and is less dependent on a global overlap measurement.

## Kernel health gate

For every inner-fold candidate, record:

- centered kernel-target alignment;
- off-diagonal mean, variance and quantiles;
- effective rank and eigenvalue spectrum;
- condition number before and after ridge regularization;
- train/validation margin distribution;
- support-vector fraction;
- sensitivity to input perturbations, shot noise and device-noise simulation.

Reject kernels that are approximately constant, approximately identity,
numerically indefinite after expected shot noise, or unstable across folds.
Center and normalize the Gram matrix using training statistics. Tune SVC `C`
and ridge/eigenvalue regularization jointly with the circuit.

## Data and scaling study

- Replace random negative sampling with stratified sampling of MI, hard-negative
  STTC/CD/HYP cases and normal controls.
- Run learning curves at 250, 500, 1,000, 2,000 and 4,000 examples per class
  where memory allows.
- Use Nyström landmarks for the larger kernel matrices; select landmarks inside
  training folds and compare random, k-means and leverage-score selection.
- Report exact statevector cost, estimated shot cost and wall-clock latency.

## Required matched controls

Every representation/circuit experiment must include:

- linear and logistic heads;
- small parameter-matched MLP;
- RBF, Laplacian, polynomial and product-cosine SVCs;
- classical RBF on the measured projected-quantum feature vector `q(z)`;
- Nyström/random-feature approximations with the same landmark budget;
- the frozen end-to-end classical champion.

The classical RBF on `q(z)` separates the value of the quantum representation
from the choice of the outer kernel. Classical controls receive the same
supervised projection and label budget used to optimize the circuit.

## Nested selection protocol

1. Outer development folds 1–8 produce patient-safe OOF predictions.
2. Inner folds select representation, qubits, depth, topology, bandwidth,
   alignment parameters, SVC `C` and regularization.
3. No outer validation labels participate in representation or circuit tuning.
4. Run at least five seeds for trainable embeddings.
5. Use patient-cluster bootstrap for paired delta AUPRC/AUROC/Brier and subgroup
   deltas.
6. Keep Fold 9 and Fold 10 sealed.

Use successive halving:

1. kernel-health screen on 128–256 training records;
2. inner-fold AUPRC screen on 1,000 records;
3. three finalists at 2,000–4,000 records;
4. full eight-fold OOF only for the winning prespecified configuration.

## Promotion gates

The optimized quantum kernel advances only if all are true:

1. Its paired delta AUPRC versus the best same-representation classical head has
   a 95% patient-bootstrap interval entirely above zero.
2. It improves or preserves sensitivity at the prespecified high-specificity
   operating point and does not materially worsen hard-negative FPR.
3. The gain persists across at least four outer folds and five seeds rather than
   depending on one partition.
4. Kernel health remains stable under finite-shot and realistic-noise analysis.
5. The gain remains after comparison with classical approximations of the
   quantum kernel.

Beating the full classical system is a second, harder gate: the quantum head or
residual system must exceed the frozen waveform/fusion champion end to end. A
quantum head that only beats another `z8` head is a representation experiment,
not the project champion.

## Exact build order

1. Implement reusable fold-local representation objects for PCA, supervised
   projection and frozen-encoder bottlenecks.
2. Implement bandwidth-scaled, sparse-topology IQP circuits.
3. Implement projected observable extraction and projected kernels.
4. Implement kernel centering, normalization, ridge correction and health
   diagnostics.
5. Implement train-only kernel-target alignment.
6. Add the complete matched-control registry.
7. Run the small kernel-health screen.
8. Run nested learning-curve experiments.
9. Run eight-fold OOF for the selected quantum and classical heads.
10. Decide `PROMOTE`, `RETAIN_AS_ABLATION` or `STOP` before any Fold-9 access.

## References

- [Power of data in quantum machine learning](https://www.nature.com/articles/s41467-021-22539-9)
- [Exponential concentration in quantum kernel methods](https://www.nature.com/articles/s41467-024-49287-w)
- [Training Quantum Embedding Kernels on Near-Term Quantum Computers](https://arxiv.org/abs/2105.02276)
- [Training embedding quantum kernels with data re-uploading QNNs](https://arxiv.org/abs/2401.04642)
- [Quantum-Efficient Kernel Target Alignment](https://arxiv.org/abs/2502.08225)
