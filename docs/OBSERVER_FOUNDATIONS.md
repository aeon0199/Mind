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
- **F36 / Q3 replication:** 15 fresh prompts, higher-resolution frozen
  rubrics, and a compact predictor produced 13 valid holdouts but only 0.596
  mean AUROC versus the unchanged 0.70 gate. The earlier procedural direction
  did not replicate.

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
- that any observed prompt-class outcome direction is universal;
- that the controller is validated or ready to resume.

## Next evidence-producing experiment

M1R, MHR, Q2, M3R, and M3R-2 have calibrated and replicated the four
foundation comparisons on Qwen3-1.7B. Q3's compact clean pre-flip predictor
missed its fixed held-out gate twice, and the original procedural direction
did not replicate. The next experiment should not be post-hoc feature tuning.
If basin mapping continues, it should begin from a new preregistered
mechanistic question with a stated interpretation boundary and rejection
criterion.

No controller rows should be used as causal branchpoint or basin-outcome
labels, and no controller redesign should resume from the present evidence.

### Completed M3R-2 preregistration

The prompt-breadth replication was frozen as
`observer-m3r2-basin-replication-v1` before any replication output was
generated. It used 15 prompts that were entirely new relative to M3R:

- factual: `seasons_tilt`, `rock_cycle`, `electric_circuit`;
- procedural: `rice_steps`, `flat_tire_steps`, `tomato_transplant_steps`;
- creative: `glass_violin`, `mars_orchard`, `undersea_library`;
- reasoning: `museum_tickets`, `rectangle_garden`, `discount_tax`;
- code: `dedupe_order_code`, `clamp_code`, `word_frequencies_code`.

The fixed grid was three seeds (`0`, `1`, `2`) by magnitudes `0` and `0.30`,
for 90 runs on Qwen3-1.7B. Each run maps 64 tokens at layer 27 and continues
at most five outcome-blind episodes for 96 intervention-free tokens with
temperature 0.8, top-p 1.0, top-k 0, and paired per-position randomness.
Identity controls and clean replay must be exact.

The predictor used whole-prompt holdout, L2 regularization `0.05`, a mean
AUROC threshold of `0.70`, and at least five valid held-out prompts. Its input
is limited to this frozen 23-field tuple:

```text
clean_top1_margin
clean_hidden_norm
normalized_position
context_sequence_length
prefix_repetition_score
prefix_unique_token_ratio
consumed_token.length
consumed_token.leading_space
clean_diagnostics.local_prediction_error
clean_diagnostics.logit_entropy
clean_diagnostics.hidden_velocity
clean_diagnostics.hidden_acceleration
clean_diagnostics.spectral.spectral_entropy
clean_diagnostics.spectral.permutation_change
clean_diagnostics.svd.effective_rank
clean_diagnostics.svd.top1_energy_frac
clean_diagnostics.layer_stiffness.27.stiffness
clean_diagnostics.layer_stiffness.27.stiffness_trend
prompt_class.code
prompt_class.creative
prompt_class.factual
prompt_class.procedural
prompt_class.reasoning
```

The interpretation boundary is unchanged: improve and degrade mean higher and
lower scores under a declared deterministic task rubric, not universal
semantic quality. Ties, no-flips, invalid holdouts, and exact controls remain
part of the reported evidence. All 90 runs and 45 identity controls validated.
The 13 valid held-out prompts reached mean AUROC 0.596, below 0.70, and the
fresh procedural slice produced 16 improve / 19 degrade / 9 tie. The frozen
design was not changed in response to those outcomes.
