# Observer Foundation Calibration — 2026-07-19

This is the first target-model evidence produced after Observer's foundation
rebuild. It uses only independent same-context branchpoint forks and matched
clean/perturbed recovery traces.

## Design

- Model: Qwen3-1.7B on MPS in float32
- Prompts: sourdough instructions and water-cycle description
- Seeds: 0, 1, 2
- Sampling: temperature 0.8, top-p 1.0
- M1R: layer 27, 48 decisions, relative additive magnitudes 0, 0.15, 0.30
- MHR: layer 13, 20 exposure decisions, 16 recovery decisions, relative
  magnitudes 0, 0.10, 0.20, 0.30
- Total: 42 runs

All 42 configuration hashes reproduced from their exact saved configurations.
All run directories were unique. Every branchpoint context matched, every
hysteresis recovery pair matched, and every zero control was an exact no-op.

## M1R — local same-context flippability

| Magnitude | Prompt | Argmax flips | Flip rate | Unique held-out splits | AUROC mean (range) | Precision | Recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.00 | Sourdough | 0 / 141 | 0.000 | — | — | — | — |
| 0.00 | Water cycle | 0 / 141 | 0.000 | — | — | — | — |
| 0.15 | Sourdough | 10 / 141 | 0.071 | 3 | 0.790 (0.656–0.886) | 0.000 | 0.000 |
| 0.15 | Water cycle | 6 / 141 | 0.043 | 3 | 0.889 (0.811–1.000) | 0.778 | 0.611 |
| 0.30 | Sourdough | 18 / 141 | 0.128 | 3 | **0.831** (0.773–0.886) | 0.233 | 0.178 |
| 0.30 | Water cycle | 14 / 141 | 0.099 | 3 | **0.886** (0.855–0.913) | 0.689 | 0.444 |

At relative magnitude 0.30, both prompt slices clear M1R's prespecified
held-out AUROC threshold of 0.80. The recurring clean predictors are top-1
margin, logit entropy, layer stiffness, and spectral structure.

This is a scoped positive result, not a universal branchpoint detector.
Fixed-threshold precision and recall remain uneven, especially for sourdough.
Each prompt has only three unique leave-one-run-out splits; duplicate group
assignments are not counted as additional evidence.

## MHR — matched exposure and recovery

| Magnitude | Prompt | Initial effect mean | Propagated mean (range) | Residual mean (range) |
|---:|---|---:|---:|---:|
| 0.00 | Sourdough | 0.000000 | 0.000000 (0–0) | 0.000000 (0–0) |
| 0.00 | Water cycle | 0.000000 | 0.000000 (0–0) | 0.000000 (0–0) |
| 0.10 | Sourdough | 0.100000 | 0.000105 (0.000034–0.000211) | 0.000032 (0.000002–0.000061) |
| 0.10 | Water cycle | 0.100000 | 0.315899 (0.000001–0.945486) | 0.393869 (0.000559–1.177891) |
| 0.20 | Sourdough | 0.200000 | 0.231319 (0.000232–0.693069) | 0.303214 (0.000109–0.909285) |
| 0.20 | Water cycle | 0.200000 | 0.753098 (0.007991–1.305816) | 0.778865 (0.012198–1.177891) |
| 0.30 | Sourdough | 0.300000 | 0.544829 (0.000987–0.940433) | 0.534163 (0.000185–0.909285) |
| 0.30 | Water cycle | 0.300000 | 1.046424 (0.887970–1.305816) | 1.107808 (0.999028–1.177891) |

The important finding is the range, not the regime label: the same magnitude
can leave almost no endpoint separation on one seed and produce an order-one
cascade on another. That is consistent with thresholded, basin-like
propagation, but this experiment does not yet identify a safe bounded-cascade
layer.

The first active paired decision is reported as the initial intervention
distance. The active-window peak is separate because later active decisions
can include accumulated consequences after an earlier token flip.

## Claim boundary

M1R is complete and Q1 has a scoped positive answer at the calibrated
Qwen3-1.7B setting. MHR is complete as a valid endpoint calibration.

**Controller remains paused.** The controller-return precision criterion is
not robust across both prompts, no bounded-cascade layer has been identified,
and Q3 outcome directionality remains open.

The next foundation task is Q2: measure clean-vs-perturbed delta propagation
at every downstream layer rather than infer it from endpoint distances.

## Evidence

The durable local evidence directory is:

`/Users/jmalone/Documents/Artifacts/Observer/foundation-calibration-2026-07-19`

It contains the 42 run directories, exact configs, manifest, six unique-split
predictor analyses, `calibration_report.json`, and `CALIBRATION_REPORT.md`.
