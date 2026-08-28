# Observer Research Lab — Foundation-First Mapping Program

_Active living document. Observer's current work begins with the repaired
Observe → Perturb → Compare → Prove foundation. The controller arc is archived
in [RESEARCH_CONTROLLER.md](../../archive/controller/RESEARCH_CONTROLLER.md)._

Last updated: **2026-07-19** · Foundation rebuild started: **2026-07-19**

> **2026-07-19 claim correction.** The foundation audit found that the
> historical branchpoint analyzer labeled controller shadow/active token
> differences as if every row were an independent same-context local
> counterfactual. After the first flip, those branches have different contexts;
> later labels are shifted or cascading consequences. The reported AUROCs are
> preserved below, but they do not establish local flippability. F31 and Q1 are
> reopened. The old hysteresis protocol also changed context and continued only
> the perturbed branch; its perturbation effects remain observations, while its
> recovery ratios and regime interpretations are provisional.

---

## 0. How to use this doc (for the next LLM)

**Start of session**
1. Read §1 (Current state) to know the active mapping question.
2. Skim §2 (Carry-over facts) — you don't need to re-derive these.
3. Open an experiment from §4 (Mapping backlog). Check what's unchecked with highest priority.
4. If the user redirects, follow the redirect explicitly and update §1.

**Historical context**
- This doc tracks the mapping program that started 2026-04-19. It is forward-looking by design.
- For *why* this program exists — the complete controller arc, findings F1–F29, experiments E1–E8.7 — see [RESEARCH_CONTROLLER.md](../../archive/controller/RESEARCH_CONTROLLER.md).
- Don't re-run experiments already recorded there. Evidence in the carry-over section below is a summary; the full record is in the archive.

**Discipline**
- Full protocol: [docs/RESEARCH_WORKFLOW.md](../workflow/RESEARCH_WORKFLOW.md)
- Evidence standard: strong findings need ≥3 seeds AND ≥2 prompts AND ≥2 models. Single-seed or single-model results are labeled as provisional.
- Before trusting any result, check provenance (config.json, events.jsonl, applied-count).
- Negative results are progress. Don't retroactively adjust success criteria.

**During the session**
- Mark experiments in-progress with today's date.
- Log run_ids under Outcome lines so future agents can re-inspect raw data.
- Stop when the stop condition is met — don't keep probing after the question is answered.

**End of session**
- Update §2 if a finding meets the evidence standard.
- Update §4 with outcomes on completed experiments, and new follow-ups.
- Update §5 with a one-line session entry.
- Update §1's "next recommended action" so the next agent can start cleanly.

---

## 1. Current state

- **North star**: *Observer maps trajectory sensitivity, propagation,
  persistence, and recovery through matched causal experiments.* The first
  calibration target remains Qwen3-1.7B; cross-model generalization is later.
- **Foundation contract**: [docs/OBSERVER_FOUNDATIONS.md](../foundations/OBSERVER_FOUNDATIONS.md).
  Active work follows Observe → Perturb → Compare → Prove. Act is downstream.
- **Program structure**: three mapping questions, Q1–Q3 in §3. **Q1 status: scoped positive**;
  **Q2 status: scoped mechanistic answer**; **Q3 status: measured negative**
  and independently replicated against its prespecified prompt-held-out
  prediction gate.
- **Controller status**: **paused**, not dead. §4 defines the evidence that would bring controller research back to active status.
- **Instrument status**: foundation mechanics, local branchpoints, matched
  recovery, same-context per-layer propagation, causal paired branch
  continuation, and branch-blind declared outcome evaluation are implemented
  and calibrated on Qwen3-1.7B. The completed suites contain 472 target-model
  runs (42 M1R/MHR + 280 Q2 + 60 M3R + 90 M3R-2).
- **Next recommended action**: keep the controller paused. M3R-2 completed the
  preregistered Q3 replication, confirmed the negative prediction gate, and
  retired the original procedural directional hypothesis. Do not tune the
  frozen predictor after seeing this result. Any further basin work should
  begin from a new, explicitly preregistered mechanistic question.

---

## 2a. Mapping-program findings

- **F36 — M3R-2 PROMPT-BREADTH REPLICATION** — **A larger fresh-prompt
  replication confirmed Observer's causal basin measurement and confirmed the
  negative Q3 prediction gate.** M3R-2 ran 90 Qwen3-1.7B cells across 15 new
  prompts, five classes, three seeds, and relative additive magnitudes
  {0, 0.30} at layer 27.
  - The registry, v2 rubrics, 23-field clean pre-flip predictor, L2
    regularization 0.05, whole-prompt split rule, five-split minimum, and AUROC
    threshold 0.70 were frozen before any replication output was generated.
  - All 90 hashes and run directories validated. All 45 identity controls
    were exact. The protocol mapped 5,670 independent local rows and retained
    217 active episodes: 195 sampled-token flips and 22 no-flips.
  - Sampled flips produced 67 improve, 49 degrade, and 79 tie outcomes. Across
    all active episodes the counts were 67 / 49 / 101. Improve/degrade means
    higher/lower frozen task-rubric score, not universal semantic quality.
  - The compact predictor had 116 eligible rows and 13 valid whole-prompt
    holdouts. Mean held-out AUROC was **0.596** (range 0.000–1.000), below
    0.70. Two prompts lacking the negative class were reported as invalid.
  - The original procedural slice did not replicate: three fresh procedural
    prompts produced 16 improve, 19 degrade, and 9 tie outcomes, versus M3R's
    12 / 0 / 6.
  - **Verdict:** M3R-2 confirmed the negative Q3 gate under a broader,
    higher-resolution preregistered design. The controller-return criterion is
    not met, and the controller remains paused.
  - Evidence: `docs/results/Q3_M3R2_REPLICATION_2026-07-19.md`.
  - Confidence: **high** in causal mechanics, replay, controls, artifacts, and
    the negative result for this frozen design; **low** that a universal
    prompt-class direction exists.

- **F35 — CAUSAL BASIN MAPPING CALIBRATED** — **Intervention-driven token
  flips can improve, degrade, or leave a declared task score unchanged, but
  the tested clean pre-flip features do not predict that outcome robustly
  across unseen prompts.** M3R ran 60 Qwen3-1.7B cells: 10 prompts spanning
  factual, procedural, creative, reasoning, and code; 3 seeds; and relative
  additive magnitudes {0, 0.30} at layer 27.
  - The two-pass protocol first mapped 2,820 independent same-context local
    forks, selected at most one early/middle/late episode without reading the
    outcome, then reset and replayed the exact clean path before continuing
    clean and perturbed branches for 48 tokens with no further intervention.
  - All 60 run hashes validated. All 30 identity-control runs were exact:
    90/90 selected control episodes had identical tokens and tied rubric
    scores.
  - The 30 active runs contributed 90 selected episodes: 69 sampled-token
    flips and 21 no-flips. Sampled flips produced 31 improve, 11 degrade, and
    27 tie outcomes under the declared deterministic prompt rubrics.
  - Ties and no-flips stayed in coverage. The binary predictor used only 42
    non-tie sampled-flip rows (31 improve / 11 degrade), clean pre-flip/context
    features, and prompt class; each test fold held out a complete prompt.
  - Five prompts had both labels. Mean held-out AUROC was **0.571** (range
    0.000–0.889), below the prespecified 0.70 stop condition. Near-perfect
    training AUROC with unstable holdout results indicates overfit.
  - The procedural slice produced 12 improve, 0 degrade, and 6 ties across two
    prompts and three seeds. This is a scoped replication target, not a
    controller gate: both procedural prompt folds lacked the negative class.
  - **Verdict:** Q3's measurement rebuild is complete, but its predictive stop
    condition is not met. “Improve” means higher score under a predeclared
    deterministic prompt rubric, not universal semantic quality. The
    controller-return criterion is not met.
  - Evidence: `docs/results/Q3_BASIN_MAPPING_2026-07-19.md`.
  - Confidence: **high** in protocol validity, replay, controls, and recorded
    rubric outcomes; **low-medium** in prompt-class interpretation because the
    eligible binary sample is small and class balance is uneven.

- **F34 — PER-LAYER PROPAGATION CALIBRATED** — **Raw L2 amplification is not
  a sufficient definition of destructive cascade.** Q2 ran 280
  Qwen3-1.7B same-context propagation cells: 5 seeds × 2 prompts × 4
  injection layers × additive {0, 0.10, 0.20, 0.30} or scaling
  {1.0, 0.5, 1.5}. Each of 4,200 paired decisions captured every downstream
  transformer block and final RMSNorm.
  - All 80 identity-control runs were exact no-ops across 1,200 paired
    decisions. All 280 config hashes and run directories were unique.
  - At additive magnitude 0.30, raw L2 amplification to layer 27 fell from
    8.662× for layer-6 injection to 1× for layer-27 injection. Final relative
    disturbance moved the other way: 0.0699 at layer 6 versus 0.2800 at layer
    27. Earlier raw growth largely reflects residual-stream scale.
  - Final-layer scale 0.5 was removed exactly by RMSNorm; scale 1.5 was reduced
    to relative delta 7.54e-8 and mean KL 1.33e-8. Both produced 0/150 argmax
    flips. Directional additive magnitude 0.30 survived with relative delta
    0.2800, mean KL 0.0861, and 18/150 flips.
  - Scaling down was much more disruptive than scaling up despite equal direct
    relative magnitude. At layer 6, mean KL was 0.1947 for scale 0.5 and
    0.0447 for scale 1.5.
  - Layer 13 had the lowest aggregate mean logit KL at every tested
    non-control setting, but did not uniformly minimize prompt-slice argmax
    flips. It is a mapping candidate, not a controller-ready bounded layer.
  - **Verdict:** Q2 has a scoped mechanistic answer. Historical F27's “earlier
    layer = worse” is not a universal local law and likely includes repeated
    exposure/context cascade. The Q2 controller-return criterion is not met.
  - Evidence: `docs/results/Q2_PROPAGATION_CALIBRATION_2026-07-19.md`.
  - Confidence: **high** in protocol validity and measured curves; **medium**
    in layer-13 candidacy because prompt diversity remains narrow.

- **F32 — RECALIBRATED** — **Clean pre-intervention features predict local
  same-context argmax flippability at the calibrated Qwen3-1.7B setting.**
  M1R ran layer-27 independent one-step forks across 3 seeds × 2 prompts ×
  magnitudes {0, 0.15, 0.30}, 48 decisions per run, temperature 0.8.
  - All six zero-magnitude runs were exact no-ops across 282 local decisions.
  - At magnitude 0.30, sourdough produced 18/141 flips and water cycle 14/141.
  - Unique leave-one-run-out prompt-slice AUROC was **0.831**
    (range 0.773–0.886) on sourdough and **0.886** (0.855–0.913) on water
    cycle. Both clear Q1's prespecified AUROC ≥0.80 stop condition.
  - Recurring clean predictors were top-1 margin, logit entropy, layer
    stiffness, and spectral structure.
  - Fixed-threshold precision/recall remained uneven: 0.233/0.178 on
    sourdough and 0.689/0.444 on water cycle at magnitude 0.30.
  - **Verdict:** Q1 has a scoped positive answer for this model, layer,
    intervention, magnitude, prompts, and seeds. It is not yet a robust
    controller trigger or a cross-prompt/model universal detector.
  - Evidence:
    `docs/results/FOUNDATION_CALIBRATION_2026-07-19.md` and the durable local artifact
    directory recorded there.
  - Confidence: **medium-high** in the scoped AUROC result; only three unique
    prompt-slice holdouts exist.

- **F33 — MATCHED HYSTERESIS CALIBRATED** — **Propagation is thresholded and
  strongly trajectory-dependent at Qwen3-1.7B layer 13.** MHR ran the same
  prompts/seeds at magnitudes {0, 0.10, 0.20, 0.30}, with 20 matched exposure
  decisions and 16 intervention-free recovery decisions.
  - All six zero controls had exactly zero initial, propagated, and residual
    distance. All 24 runs had `matched_recovery=true`.
  - At magnitude 0.30, propagated endpoint distance ranged from 0.000987 to
    0.940433 on sourdough but from 0.887970 to 1.305816 on water cycle.
  - The same intervention magnitude can therefore be nearly absorbed on one
    sampled trajectory and cause an order-one basin switch on another.
  - Absolute initial, propagated, and residual distances are primary.
    Recovery ratios/regime names near tiny absolute distances are not treated
    as strong scientific categories.
  - **Verdict:** MHR's validity stop condition is met, but Q2 is not answered.
    Endpoint distances do not reveal which downstream layer amplifies or
    absorbs the injected delta, and no bounded-cascade controller layer has
    been identified.
  - Confidence: **high** in protocol validity and endpoint measurements;
    **medium** in the thresholded/basin interpretation.

- **F31 — REOPENED** — **The historical controller-paired AUROCs are invalid
  for the claimed local-flippability question.** The original M1.2 ran 10
  additional controller runs on a second prompt ("Describe the water cycle in
  a few sentences") and analyzed shadow/active token differences:
  - **Sourdough alone** (12 pairs / 576 rows): held-out AUROC **0.82**, top features `spectral.permutation_change` (+), `layer_stiffness.-1.elasticity` (+), `svd.top1_energy_frac` (+)
  - **Water cycle alone** (5 pairs / 240 rows): held-out AUROC **0.86**, top feature `step_idx` alone reaches AUROC 0.80; other features include `svd.effective_rank` (+), `spectral.permutation_change` (**negative** — flips sign vs sourdough)
  - **Combined** (17 pairs / 816 rows): held-out AUROC 0.99 — likely inflated by pair-level random split landing easy watercycle pairs in test; per-prompt numbers above are the trustworthy ones
  - **Universal predictor across prompts**: `step_idx` (later tokens more flippable in both). Mechanistically sensible: baseline divergence accumulates with length, branchpoint density rises.
  - **Prompt-class dependence**: `spectral.permutation_change` is a POSITIVE predictor on sourdough (how-to-style) and NEGATIVE on watercycle (descriptive). This means different prompt classes drive Qwen3 into different trajectory structures, and what "looks like" a branchpoint geometrically is prompt-type-dependent.
  - **Why invalid for Q1:** the active branch continued after intervention. Once
    a token differed, all later active tokens were conditioned on a different
    context. Those rows label cascaded trajectory divergence, not whether a
    perturbation at the same clean context flips the next decision.
  - **Additional risk:** sequence position can predict accumulated divergence,
    making the strong `step_idx` result evidence of shifted labels rather than a
    local causal boundary.
  - **What remains useful:** the saved outputs establish that interventions can
    alter continued trajectories, and the AUROCs are a valuable audit example
    of why causal labels need same-context forks.
  - **Q1 verdict at audit time:** **REOPENED.** Rerun with
    `runtime_lab.cli.main branchpoints`, whole-run splits, clean
    pre-intervention features, and precision/recall as well as AUROC.
    F32 records the later repaired-protocol recalibration.
  - Confidence in this correction: **high**, from the protocol semantics.

- **F30** — **(historical M1.1 controller-paired analysis; invalidated for
  local Q1 together with F31).** Using existing Qwen3-1.7B controller runs with
  pair-level train/test split and shadow-features-only produced held-out AUROC
  **0.82** across 12 pairs / 576 rows / 4 seeds.
  - Stop condition (§3 Q1): AUROC ≥ 0.80. ✓ met on this slice.
  - Evidence standard (§3 Q1): ≥3 seeds AND ≥2 prompts within Qwen3-1.7B. Seeds ✓ (4). Prompts ✗ (only sourdough has valid shadow/active pairs in the archive).
  - Top predictive features (shadow trajectory only, clean): `spectral.permutation_change` (trajectory-axis FFT response), `spectral.total_power` (higher = less flippable), `layer_stiffness.-1.elasticity` (low final-layer velocity = flippable), `svd.top1_energy_frac` (concentrated trajectory = flippable), `step_idx` (later tokens more flippable).
  - **Interpretation**: within Qwen3-1.7B on this prompt, hidden-state geometry features from the clean trajectory predict flippability at 0.82 AUROC. Mechanistically coherent: flippable steps are low-velocity, spectrally-broad-but-low-power, concentrated in a dominant direction, and later in the sequence.
  - **Correction:** shadow features removed direct feature leakage, but did not
    repair the shifted/cascading label. This analysis cannot close Q1.
  - Artifacts: analyzer at `analysis/analyze_branchpoints.py`, commit `1ec2c14`.
  - Confidence: **medium** (1 prompt, 4 seeds, mechanistically consistent features, but prompt-breadth evidence standard not yet met).

## 2. Carry-over facts (established in controller arc)

_These are motivating evidence for the mapping program. Full records in archive/controller/RESEARCH_CONTROLLER.md §2._

- **F13** (cited): reproducible drift operating point at (Qwen3-1.7B, L27, noise_mag=1.0, factual, temp=0.8) — drift 1.96 ± 0.37, DSR=5.36 across 5 seeds. The instrument can produce reproducible perturbation effects.
- **F17 / F22**: scaling at final layer is absorbed by the model's final RMSNorm. Zero effect on logits. Don't use scaling as an intervention class at L=-1.
- **F18**: additive at L=-1 with mag=1.0 produces reproducible decision-distribution shifts (logit_kl=10.40, DSR=3.73, 85% token flip rate on factual prompts).
- **F25 Part A (corrected boundary)**: continued final-layer additive
  interventions produced token/output changes in both Qwen3 and TinyLlama
  runs. Whether the same-context local flip mechanism generalizes is reopened
  for the independent branchpoint protocol.
- **F25 Part B (historical observation)**: changed Qwen3 sourdough outputs
  sometimes landed in lower local-prediction-error continuations, while changed
  TinyLlama outputs degraded. Basin direction is not established as a general
  law.
- **F27**: acting 1–2 layers back from the final layer produces WORSE control, not better. Mid-stack interventions cascade destructively. Layer-move hypothesis is falsified.
- **F28** (the big one): the divergence signal measures token-level prose surprise (word-starts, semantic transitions, punctuation) — not dynamical instability. The original controller trigger wasn't what we thought it was.

---

## 3. Mapping questions

Three questions, each a deliberate research program with a stop condition. Work serially on one question at a time; don't split attention.

### Q1 — Branchpoint geometry: when are tokens flippable?

**The claim we're trying to establish.** Given a token position during generation, predict whether a small additive perturbation at L=-1 will flip the argmax. Build a feature-based classifier from step features already in `events.jsonl`:
- top-2 logit margin
- step entropy
- position in sequence
- token type (word-start vs mid-word vs punctuation)
- hidden-state delta norm from prior step
- predicted next-token divergence

**Evidence standard** (Qwen3-1.7B scope): ≥3 seeds, ≥2 prompts within
Qwen3-1.7B. Labels must come from independent one-step forks with a verified
shared context. Train/test splits hold out whole runs, and all preprocessing is
fit on training runs only. Cross-model generalization is later.

**Stop condition**: a feature-based rule achieves precision ≥0.7 AND recall ≥0.5 on held-out runs. If no single rule gets there, we can train a simple logistic regression / decision tree — still counts if AUROC ≥ 0.8.

**Why it matters**
- Clean interpretability finding on its own: "we can predict when perturbations will flip tokens."
- It's the direct path back to a smarter controller — predictable branchpoints = designable trigger.
- Matches where mechanistic interpretability research is going.

**Current status**: **scoped positive after M1R/F32.** At relative additive
magnitude 0.30 and layer 27, both prompt slices clear AUROC 0.80 under unique
whole-run holdouts. Precision/recall are not robust enough to satisfy the
controller-return trigger criterion, and broader prompts/seeds/models remain
open.

---

### Q2 — Perturbation propagation: how does an injected delta evolve?

**The claim we're trying to establish.** Given an additive injection of size `s` at layer `L`, measure the delta's L2 norm at every downstream layer up to L=-1. Does it decay, amplify, or get absorbed by specific norm layers?

**Evidence standard** (Qwen3-1.7B scope): ≥5 seed replicates per `(injection_layer, intervention_type, magnitude)` cell within Qwen3-1.7B. Cross-model claims are out of scope for this mapping program.

**Stop condition**: per-layer propagation curve that cleanly explains F27's "earlier layer = worse" finding. Specifically — the curve should predict which layers produce "bounded" cascade (delta stays bounded at each downstream layer) vs "destructive" cascade (delta amplifies or destroys hidden-state structure).

**Why it matters**
- Explains F27 mechanistically instead of treating it as an empirical fact.
- If any layer has bounded propagation, that's the principled layer for a future controller.
- Useful for any hidden-state intervention work (safety, steering, interpretability).

**Current status**: **scoped mechanistic answer after M2R/F34.** The repaired
instrument maps paired deltas through every downstream block and final norm.
Raw L2 amplification, relative/angular structure, and output disruption do not
share one monotonic depth law. Layer 13 is the lowest aggregate-KL candidate,
but the controller-return requirement for a prompt-robust bounded layer has not
landed.

---

### Q3 — Basin structure: when does a flip improve vs. degrade output?

**The claim we're trying to establish.** Given a token flip at branchpoint T, predict whether the resulting trajectory lands in a better or worse basin. F29 showed this is model-specific; we want to characterize the model-dependent features that determine it.

**Evidence standard** (Qwen3-1.7B scope): ≥5 prompts × ≥3 seeds per (prompt, config) cell, all within Qwen3-1.7B. Cross-model basin comparison is out of scope here (see archive/controller/RESEARCH_CONTROLLER.md F29 for the one cross-model data point we already have, to be revisited later).

**Stop condition**: a feature of the pre-flip Qwen3-1.7B generation that predicts improve-vs-degrade with AUROC ≥ 0.7. Candidate features:
- Baseline output repetition score (is the model in a degenerate loop?)
- Baseline avg_divergence (how unstable is the trajectory anyway?)
- Prompt class (factual/procedural/creative/reasoning/code)
- Token position where flip lands (early in generation may matter more)
- Baseline top-2 logit margin at the flip token (once logged)

**Why it matters**
- Resolves the F25/F29 split — tells us when branchpoint-hijacking actually helps.
- If improvement predictors exist, the controller gains a meaningful trigger: "act only when expected-improvement > threshold."
- Either way: a complete interpretability story about perturbation-induced basin-hopping.

**Current status**: **measured negative after M3R/F35 and independently
replicated by M3R-2/F36.** The fresh 15-prompt replication retained the same
causal protocol while freezing higher-resolution rubrics, a compact predictor,
and stronger minimum-split requirement before data generation. Mean held-out
AUROC was 0.596 across 13 valid prompts, below 0.70. The earlier procedural
directional pattern did not replicate. Q3 does not justify reopening the
controller.

---

## 4. Controller-return criteria

The controller thesis is paused, not dead. It returns to active status when **any two** of the following become true during mapping work:

1. **Q1 produces a branchpoint predictor with precision ≥ 0.7.** Means we have a real trigger (predictable flippable tokens) instead of blind `raw_div > threshold`.
2. **Q2 identifies a specific layer with bounded-cascade propagation.** Means we know where to act without destructive cascade (F27 problem solved).
3. **Q3 identifies a predictable class of prompts or models where flips tend to improve.** Means we know when firing the controller helps rather than harms.

If any two land, a controller redesign experiment is warranted. If none land during the mapping program, the interpretability findings stand alone and the controller stays on ice.

**What does NOT justify re-opening controller work**: more tuning of the existing `raw_div → scaling | random additive | EMA opposition` design. That entire search space is exhausted (archive/controller/RESEARCH_CONTROLLER.md F17–F27).

---

## 5. Mapping backlog

Order: M1R, MHR, M2R, M3R, and M3R-2 are complete. The repaired foundation now
maps local sensitivity, matched persistence/recovery, downstream propagation,
and paired basin outcomes. Do not return to controller work: the
controller-specific precision, prompt-robust bounded-layer, and
expected-improvement criteria have not landed.

### [x] M1R — Recalibrate local branchpoint flippability

- **Question**: Can clean pre-intervention features predict a one-step argmax
  flip under an independently forked, same-context additive perturbation?
- **Design**: at least 3 seeds × 2 prompts × zero/nonzero magnitudes using
  `runtime_lab.cli.main branchpoints`. Analyze only `branchpoint_run_*`
  artifacts with unique whole-run splits.
- **Stop condition**: precision ≥0.7 and recall ≥0.5, or AUROC ≥0.8, with
  per-split min/max and both prompt slices reported. Zero magnitude must produce
  no local flips.
- **Outcome**: complete 2026-07-19. Magnitude 0.30 cleared mean held-out AUROC
  0.80 on both prompt slices (0.831 sourdough, 0.886 water cycle). Zero controls
  were exact. See F32 and
  `docs/results/FOUNDATION_CALIBRATION_2026-07-19.md`.

### [x] MHR — Calibrate matched hysteresis

- **Question**: After a perturbation propagates beyond its active window, how
  much clean-vs-perturbed separation persists during equal,
  intervention-free continuations?
- **Design**: same prompts/seeds as M1R, zero plus several nonzero magnitudes,
  sufficient exposure for at least one post-intervention decision, fixed
  recovery length.
- **Stop condition**: every classified run has
  `matched_recovery=true`; zero magnitude reports no propagation; report direct,
  propagated, and residual distance rather than regime counts alone.
- **Outcome**: complete 2026-07-19. All recovery traces matched and zero
  controls were exact. Nonzero endpoint propagation ranged from near-zero to
  order-one at the same magnitude, so a bounded layer has not been identified.
  See F33.

### [!] Historical M1.1 — Offline analysis from controller runs *(invalid for local Q1)*
- **Outcome**: the historical analyzer reported cross-model AUROC 0.77 and
  same-model AUROC 0.82 after a direct active-feature leak was removed.
- **Correction**: the shadow/active token-difference label still became shifted
  after the first divergent token. See F30/F31 — REOPENED.
- **Archived task entry** (preserving original design for context):

### [ ] M1.1 — Offline branchpoint analysis from existing control runs (original task)
- **Question**: From our ~150 existing control runs, can we characterize which step-level features predict whether a token gets flipped?
- **Design**:
  - Iterate over every shadow/active pair we have in `runs/`
  - For each step, compute `flipped = shadow_token_text != active_token_text`
  - Extract per-step features from events.jsonl (raw_div, avg_score, hidden_post_norm, pre_post_delta_norm, ...)
  - Fit a simple classifier (logistic regression or decision tree, feature importances visible)
  - Hold out 20% of runs as test set
- **Stop condition**: AUROC ≥ 0.8 on held-out runs using features already in events.jsonl. If that fails, AUROC ≥ 0.7 with one or two additional re-extracted features (top-2 logit margin computed offline from saved logits if available).
- **Expected runtime**: no GPU, all offline. Writing the analyzer is ~1 hour.
- **Expected payoff**: either the Q1 predictor (if features are sufficient) or a clear specification of what per-step data we need to log in future runs for M1.2 to work.
- **Runs**: (no new runs — purely offline analysis)
- **Outcome**: _(filled when complete)_
- **Follow-ups**: _(filled when complete)_

### [!] Historical M1.2 — Second controller-run prompt *(completed, claim invalidated)*
- **Outcome**: ran 10 control runs on "Describe the water cycle in a few sentences." with same config as M1.1's passing slice. All active cells fired (8-10 interventions per seed). Rerun of `analysis/analyze_branchpoints.py` on Qwen3-only opposing-anchor corpus:
  - Sourdough: AUROC 0.82 ✓
  - Water cycle: AUROC 0.86 ✓
  - Both prompts clear the 0.80 threshold independently
  - Top features differ between prompts: sourdough predictors are trajectory-geometry features (spectral.permutation_change, elasticity); water cycle's strongest single predictor is `step_idx` alone (AUROC 0.80)
- **Corrected verdict**: did not answer local Q1. See F31 — REOPENED.
- **Follow-ups**:
  - Interesting side-finding worth its own write-up: `spectral.permutation_change` flips sign between prompt classes. Suggests Qwen3 uses different trajectory structures for how-to vs descriptive generation. Could be explored later as a small mapping sub-question.

### [~] Historical M1.2 original task entry (preserved)
- **Status**: queued 2026-04-19. Runs-only task (Claude).
- **Question**: Does the M1.1 classifier (shadow-trajectory features → flip/no-flip prediction) generalize to a second Qwen3-1.7B prompt? Needed to meet Q1's ≥2 prompts evidence standard.
- **Design**:
  - Pick one non-sourdough prompt already in the registry (e.g., "Describe the water cycle in a few sentences." or "Explain how airplanes fly in a clear, accurate way.").
  - Run 5 seeds × 2 cells (shadow + opposing-anchor additive at L=-1, mag=0.3/0.6, temp=0.8, max_tokens=48). 10 runs.
  - Run `analysis/analyze_branchpoints.py` on the combined corpus (sourdough + new prompt).
- **Stop condition** (Q1 closure): within-Qwen3 held-out AUROC ≥ 0.80 across both prompts.
- **Historical expected payoff**: this was expected to settle Q1. The
  2026-07-19 audit established that the label did not match the causal question.
- **Expected runtime**: ~2 min with daemon.

### [x] M1.3 — Add direct diagnostic baselines *(foundation rebuild)*
- **Status**: complete 2026-07-19.
- **Outcome**: canonical diagnostics now include top-1 margin, logit entropy,
  hidden velocity, and hidden acceleration. The active local branchpoint format
  records clean top-1 margin and clean diagnostics.

### [!] M2.1 — Historical offline propagation proposal *(insufficient)*
- **Correction**: `layer_stiffness` is within-trajectory hidden velocity, not a
  paired clean-vs-perturbed layer delta. Existing stress rows also become
  context-shifted after a token difference. They cannot reconstruct Q2's local
  propagation curve.
- **Outcome**: rejected as a measurement mismatch before claim generation.

### [x] M2R / historical M2.2 — Same-context synthetic propagation sweep
- **Question**: At one verified decision context, how does an injected delta
  change through every downstream block and final normalization?
- **Design**: 280 Qwen3-1.7B runs across 5 seeds, 2 prompts, layers
  {6, 13, 20, 27}, additive and scaling interventions, exact type-specific
  controls, 15 paired decisions per run.
- **Stop condition**: valid per-layer curves for at least two intervention
  types and three injection layers, with normalized/angular metrics and prompt
  slices as well as raw L2.
- **Outcome**: complete 2026-07-19. See F34 and
  `docs/results/Q2_PROPAGATION_CALIBRATION_2026-07-19.md`.

### [x] M3R — Causal prompt-class basin mapping on Qwen3-1.7B
- **Question**: When an independent branchpoint fork changes a decision and
  both branches are then continued, which pre-flip/context features predict a
  better versus worse outcome?
- **Protocol rebuild required**:
  - no controller or shadow/active pair;
  - select branchpoints from verified same-context forks;
  - continue both clean and perturbed branches under an explicit matched
    continuation policy;
  - evaluate outcome with declared prompt-specific checks and blinded paired
    scoring;
  - preserve no-flip and no-effect cells instead of silently dropping them.
- **Evidence standard**: at least 5 prompt classes × 3 seeds per configuration,
  all on Qwen3-1.7B.
- **Stop condition**: a pre-flip feature or declared prompt-class feature
  predicts improve versus degrade with held-out AUROC ≥0.70. Report coverage,
  ties/no-effect, class balance, and prompt-held-out results.
- **Outcome**: complete 2026-07-19. All 60 runs and 30 identity controls
  validated. Among 69 selected sampled-token flips, declared rubric outcomes
  were 31 improve / 11 degrade / 27 tie. The clean pre-flip prompt-held-out
  predictor reached mean AUROC 0.571 across five valid splits, below 0.70.
  Procedural prompts produced a scoped 12 improve / 0 degrade / 6 tie pattern
  that was subsequently tested and did not replicate in M3R-2/F36. See F35,
  F36, `docs/results/Q3_BASIN_MAPPING_2026-07-19.md`, and
  `docs/results/Q3_M3R2_REPLICATION_2026-07-19.md`.

### [x] M3R-2 — Fresh-prompt causal basin replication

- **Question**: Does M3R's outcome-prediction result or its scoped procedural
  direction replicate on entirely new prompts under higher-resolution frozen
  rubrics and a compact preregistered predictor?
- **Design**: 15 disjoint prompts, three per class; 3 seeds; magnitudes
  {0, 0.30}; layer 27; 64 mapped tokens; 96 continuation tokens; at most five
  outcome-blind episodes; exact controls/replay; 23 frozen predictor fields;
  L2 0.05; whole-prompt holdout.
- **Stop condition**: mean held-out AUROC at least 0.70 across at least five
  valid prompt splits. Preserve invalid splits, ties, no-flips, and exact
  controls.
- **Outcome**: complete 2026-07-19. All 90 runs and 45 controls validated.
  Sampled-token flips produced 67 improve / 49 degrade / 79 tie outcomes.
  Thirteen valid prompt holdouts reached mean AUROC 0.596, below 0.70. The
  procedural slice produced 16 improve / 19 degrade / 9 tie, so the M3R
  directional pattern did not replicate. See F36 and
  `docs/results/Q3_M3R2_REPLICATION_2026-07-19.md`.

---

## 6. Sessions log (Mapping phase)

- **2026-04-19 · Program defined.** Pivoted from controller-focused to mapping-focused. Three questions (Q1 branchpoint geometry, Q2 propagation, Q3 basin structure) with stop conditions. Controller-return criteria specified. Next: M1.1.
- **2026-04-19 · M1.1 ran, Qwen3 AUROC 0.82.** Codex wrote `analysis/analyze_branchpoints.py`. Claude caught + fixed a circular-feature leak (features from active events included intervention_applied, scale_used, etc. — tautological). With honest shadow-trajectory features on the Qwen3-1.7B sourdough slice, held-out AUROC is **0.82** across 12 pairs / 4 seeds — clears Q1's 0.80 threshold. F30 added. Remaining Q1 gap: ≥2 prompts evidence standard. M1.2 queued: run a second Qwen3 prompt suite.
- **2026-04-19 · Scope correction.** Human redirected: mapping program is Qwen3-1.7B, not cross-model. Earlier M1.2 spec (add logit-margin logging for cross-model generalization) deferred to M1.3 / later work. Current M1.2 is a simple 10-run experiment on a second Qwen3 prompt.
- **2026-04-19 · Historical M1.2 verdict (invalidated 2026-07-19).** Ten
  water-cycle controller runs produced historical AUROCs of 0.82 and 0.86 and
  were initially recorded as closing Q1. The later foundation audit found the
  shifted/cascading-label defect; see the next entry and F31 — REOPENED.
- **2026-07-19 · Foundation audit; F31/Q1 reopened.** Repaired event
  semantics, provenance, loaded-model layer resolution, independent local
  branchpoints, run-grouped analyzer splits, matched recovery, and simple
  diagnostic baselines. The historical controller-paired AUROCs used
  shifted/cascading labels after the first divergent token and cannot prove
  local flippability. Historical perturbation effects remain observations;
  unmatched recovery/regime claims are provisional. Next: verify the release,
  then run M1R + MHR.
- **2026-07-19 · M1R + MHR calibrated; F32/F33.** Completed 42 Qwen3-1.7B
  repaired-protocol runs across 3 seeds and 2 prompts. All provenance and zero
  controls passed. Local flippability cleared AUROC 0.80 on both prompt slices
  at magnitude 0.30. Matched hysteresis exposed seed-dependent near-zero versus
  order-one propagation at the same magnitude. Q1 is scoped positive; Q2
  per-layer paired-delta tracing is next; controller remains paused.
- **2026-07-19 · Q2 propagation calibrated; F34.** Completed and revalidated
  280 same-context per-layer runs / 4,200 paired decisions across two
  intervention families and four injection depths. Raw L2 grew toward late
  layers while relative/angular disturbance usually contracted; final RMSNorm
  erased radial final-layer scaling but preserved directional additive change.
  Layer 13 is the lowest aggregate-KL candidate, not a controller-ready answer.
  Q3's historical controller-dependent task was replaced with M3R, a causal
  branch-continuation protocol rebuild.
- **2026-07-19 · Q3 causal basin mapping calibrated; F35.** Completed and
  independently revalidated 60 Qwen3-1.7B runs across ten prompts, five
  classes, three seeds, and exact zero controls. Sampled-token flips produced
  improvement, degradation, and ties under frozen branch-blind rubrics, but the
  clean pre-flip predictor reached only 0.571 mean prompt-held-out AUROC. The
  prespecified stop condition did not pass. Procedural outcomes are a scoped
  replication target; controller work remains paused.
- **2026-07-19 · M3R-2 prompt-breadth replication; F36.** Froze 15 new
  prompts, higher-resolution v2 rubrics, a 23-field predictor, L2 0.05, and
  the original 0.70 gate before running 90 Qwen3-1.7B cells. All controls and
  artifacts validated. Mean prompt-held-out AUROC was 0.596 across 13 valid
  prompts, still below threshold, and the procedural 12 / 0 / 6 direction
  became 16 / 19 / 9 on fresh prompts. The negative Q3 result is replicated;
  controller work remains paused.

_(For sessions covering the controller arc 2026-02 through 2026-04-19, see [RESEARCH_CONTROLLER.md](../../archive/controller/RESEARCH_CONTROLLER.md) §5.)_

---

## 7. Quick-start snippets

**Orient before starting**
```bash
# see the status of both docs
head -40 docs/research/RESEARCH.md archive/controller/RESEARCH_CONTROLLER.md

# current state via API
curl -s http://127.0.0.1:8899/api/capabilities | jq '.modes | keys'
curl -s http://127.0.0.1:8899/api/runs | jq '.runs[0:5] | .[] | {id, mode, headline}'
```

**Run the daemon (required for any active experiment)**
```bash
cd ~/observer
source .venv/bin/activate
python tools/observer_daemon.py --model qwen3-1.7b
```

**Orchestrator pattern for a new mapping experiment**
See `/tmp/run_e87_phase2.py` or `/tmp/run_f25_scope_check.py` in the controller arc — same pattern works:
  1. spawn daemon as subprocess (stdin=PIPE, stdout=PIPE)
  2. read "ready" line from stdout
  3. send JSON-line request, read JSON-line ack
  4. record run_id + summary metrics + event-level data
  5. aggregate per-cell, print verdict table with stop-condition check

**Offline analysis pattern for M1.1 / M2.1**
Read `runs/*/events.jsonl`, extract per-step features, pair shadow vs active by matching seed + config, aggregate into a DataFrame / dict-of-lists, fit a scikit-learn classifier or hand-compute thresholds. No daemon needed.

---

_End of document. Update before ending the session. The next LLM is counting on you._
