# Observer Foundations

Observer is a runtime research instrument for asking causal questions about
language-model generation. Its foundation is:

**Observe → Perturb → Compare → Prove**

`Act` is deliberately downstream. A controller may eventually consume evidence
from Observer, but controller behavior is not part of the foundation and is
paused until the observation and perturbation signals earn that use.

## 1. Observe

Observer records one causal decision timeline at token time. The terms are
literal:

1. The cache represents the context before the current forward pass.
2. The **consumed token** is fed into that context.
3. Observer captures the selected layer before and after any intervention.
4. The model produces logits and a **predicted next token**.
5. That prediction becomes the consumed token for the next decision.

`RuntimeEvent` keeps the consumed token and predicted next token separate.
Legacy `token_id` and `token_text` fields remain compatibility aliases for the
consumed token; they must not be interpreted as the prediction produced by the
same event.

Canonical passive diagnostics include:

- `hidden_velocity` and `hidden_acceleration`, simple token-to-token baselines;
- `logit_entropy` and `top1_margin`, direct output-distribution baselines;
- `local_prediction_error`, the descriptive name for the sliding VAR(1)
  prediction error;
- `divergence`, retained as a compatibility alias for
  `local_prediction_error`;
- token-time spectral, windowed SVD, and optional multi-layer measurements.

These are measurements, not automatic semantic conclusions. In particular,
none is a proven hallucination, correctness, danger, or instability detector.

## 2. Perturb

Observer supports two distinct perturbation questions.

### Local branchpoint counterfactual

At each clean decision boundary:

- clone the same cache into clean and perturbed forks;
- verify both cache fingerprints and sequence lengths match;
- use the same consumed token and paired sampling RNG;
- apply the intervention to the perturbed fork only;
- compare the one-step predictions;
- discard the perturbed fork and advance only the clean trajectory.

This answers the local causal question: **would this intervention change the
next decision under the same context?** It does not confuse a downstream
cascade after an earlier flip with a new local flip.

### Exposure and propagation

A stress run can instead continue clean and perturbed branches across an
intervention window. This measures how an injected difference propagates
through subsequent decisions. Once tokens differ, later rows are
trajectory-level consequences, not independent local branchpoint labels.

Prompt contamination remains a valid context-sensitivity experiment, but it is
not labeled an internal hidden-state perturbation or an internal hysteresis
experiment.

## 3. Compare

Comparisons must match the question being asked.

- **Local sensitivity:** argmax flip, paired sampled-token flip, logit KL/JS,
  hidden cosine distance, and relative hidden L2 at one shared context.
- **Propagation:** aligned clean and perturbed trajectories across and after an
  intervention window.
- **Matched recovery:** two branches begin from the same cache, experience
  equal exposure instructions, and then both continue for the same requested
  recovery length with no intervention. Recovery is classified only when the
  exposure endpoint proves propagation above the configured floor and both
  recovery traces remain valid and aligned.

The historical `BASE → PERTURB → REASK` experiment changed the context and
continued only the perturbed branch. It can preserve useful observations about
prompt contamination or perturbation effects, but its recovery ratios and
regime labels are not evidence of internal matched recovery.

## 4. Prove

Every scientific claim should be traceable to evidence that can reject it:

- authoritative loaded-model identity, device, dtype, and backend metadata;
- exact final configuration plus SHA-256 config hash;
- unique run directory and structured events;
- seed-cache and local-context fingerprints where matching is claimed;
- explicit protocol-validity fields;
- run-grouped analysis splits with preprocessing fitted on training runs only;
- zero-perturbation controls;
- deterministic synthetic diagnostic validation;
- multi-seed and multi-prompt calibration before promoting a result.

The diagnostic validator is:

```bash
python scripts/validate_diagnostics.py \
  --json-out /tmp/observer-diagnostic-validation.json
```

It passes only if the actual diagnostic stack behaves sensibly on constant,
linear-drift, oscillatory, and shock trajectories.

## Current claim boundary

Verified in code and automated tests:

- canonical token/event semantics;
- exact run provenance and unique artifact directories;
- loaded-model-aware semantic layer resolution;
- local one-step branchpoint forks with matched contexts and paired RNG;
- matched hidden-state exposure and recovery mechanics;
- simple diagnostic baselines and deterministic synthetic validation.

Preserved but reopened:

- **F31 / Q1:** historical controller-paired AUROCs do not establish local
  flippability. After the first token difference, the active and shadow runs
  have different contexts, so later labels are shifted or cascading
  consequences. The numbers remain in the research record as an audit lesson.
- **Historical hysteresis regimes:** perturbations did produce observable
  effects, but recovery and regime interpretations from the unmatched re-ask
  protocol are provisional.

Not claimed:

- that a diagnostic has established semantic or safety meaning;
- that local branchpoint flippability is already predictable out of sample;
- that matched recovery regimes have already been calibrated on the target
  research model;
- that the controller is validated or ready to resume.

## Next evidence-producing experiment

After release verification, run a small calibration suite using only the new
protocols:

1. local branchpoints at zero and nonzero magnitude across at least three seeds
   and two prompts;
2. matched hysteresis at zero and several nonzero magnitudes over the same
   seeds and prompts;
3. analysis grouped by whole run, reporting AUROC, precision, recall, direct
   perturbation effect, propagation, persistence, and recovery validity;
4. no controller-mode rows used as local branchpoint labels.

That is the shortest honest route from a repaired instrument to new Observer
findings.
