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

### Same-context per-layer propagation

At the same independent decision boundary, propagation mode keeps the clean
and perturbed forks local but captures both at every downstream transformer
block and at final normalization. It reports:

- absolute delta L2, which is sensitive to the residual stream's changing
  scale;
- delta L2 relative to the clean state;
- cosine distance, which isolates angular/structural change;
- amplification relative to the injected delta;
- logit KL/JS and local token flips.

These quantities must be interpreted together. Raw L2 growth is not by itself
evidence of destructive cascade, and final normalization can remove radial
scaling while preserving directional changes.

### Causal branch continuation

Basin mode extends a selected local fork without confusing selection with
outcome:

- pass one maps every independent same-context fork and advances only the clean
  path;
- an outcome-blind early/middle/late rule selects a bounded set of episodes;
- pass two resets and replays the exact clean path, rejecting any context,
  fingerprint, or decision mismatch;
- clean and perturbed branches continue for equal lengths with the
  intervention disabled and deterministic paired sampling RNG at every
  position;
- a declared deterministic prompt rubric scores both texts without receiving
  branch identity.

This answers a trajectory-level causal question: **what outcome follows from
the selected local intervention-driven branch?** A higher rubric score is a
task-specific operational outcome, not a universal semantic, preference, or
safety judgment. Ties, no-flips, and exact no-effects remain part of coverage.

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
  intervention window, or independent same-context layer-by-layer paired
  deltas for local propagation.
- **Basin outcome:** exact replay plus paired intervention-free continuation
  from an outcome-blind selected local fork. Report local-flip coverage,
  no-flips, ties, rubric class balance, and whole-prompt-held-out prediction.
- **Matched recovery:** two branches begin from the same cache, experience
  equal exposure instructions, and then both continue for the same requested
  recovery length with no intervention. Recovery is classified only when the
  exposure endpoint proves propagation above the configured floor and both
  recovery traces remain valid and aligned.

Matched hysteresis keeps two exposure measurements distinct:

- **initial intervention distance** is the first active decision, before an
  earlier intervention-driven token difference can contaminate the comparison;
- **active-window peak** is the largest paired distance while intervention is
  active and can include accumulated trajectory consequences after a flip.

The propagated endpoint and recovery residual are trajectory-level distances,
not independent local causal effects.

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
- same-context downstream-block and final-normalization propagation capture;
- exact clean replay and paired intervention-free branch continuation;
- deterministic prompt-specific rubrics with branch identity hidden from the
  scorer;
- simple diagnostic baselines and deterministic synthetic validation.

Calibrated but scoped:

- **F32 / Q1:** clean pre-intervention features cleared the local
  flippability AUROC gate on two Qwen3-1.7B prompt slices at one calibrated
  setting. Precision/recall did not become a controller-ready trigger.
- **F33:** matched hysteresis produced valid endpoint propagation and recovery
  measurements, with strong seed dependence.
- **F34 / Q2:** same-context per-layer traces separated raw scale growth from
  relative/angular disturbance. No prompt-robust bounded controller layer was
  established.
- **F35 / Q3:** causal branch continuations produced rubric improvements,
  degradations, and ties, but the clean pre-flip predictor missed its
  whole-prompt-held-out AUROC gate.

Preserved historical corrections:

- **F31 / Q1:** historical controller-paired AUROCs do not establish local
  flippability. After the first token difference, the active and shadow runs
  have different contexts, so later labels are shifted or cascading
  consequences. The numbers remain in the research record as an audit lesson.
- **Historical hysteresis regimes:** perturbations did produce observable
  effects, but recovery and regime interpretations from the unmatched re-ask
  protocol are provisional.

Not claimed:

- that a diagnostic has established semantic or safety meaning;
- that the scoped Q1 predictor generalizes beyond its calibrated setting;
- that matched recovery defines universal dynamical regimes;
- that deterministic basin rubrics measure universal output quality;
- that the procedural F35 slice will replicate on unseen prompts;
- that the controller is validated or ready to resume.

## Next evidence-producing experiment

M1R, MHR, Q2, and M3R have now calibrated all four foundation comparisons on
Qwen3-1.7B. If basin mapping continues, the next evidence-producing experiment
is a preregistered prompt-breadth replication:

1. freeze improved outcome rubrics before generating new data, especially for
   code continuations;
2. add independent prompts with enough improve and degrade outcomes to support
   whole-prompt holdout;
3. preserve identity controls, exact clean replay, outcome-blind selection,
   ties/no-flips, and the pre-flip-only predictor boundary;
4. test the scoped procedural pattern without treating it as a replacement for
   the failed aggregate Q3 gate.

No controller rows should be used as causal branchpoint or basin-outcome
labels, and no controller redesign should resume from the present evidence.
