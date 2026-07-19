# Observer Research Lab — Foundation-First Mapping Program

_Active living document. Observer's current work begins with the repaired
Observe → Perturb → Compare → Prove foundation. The controller arc is archived
in [RESEARCH_CONTROLLER.md](RESEARCH_CONTROLLER.md)._

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
- For *why* this program exists — the complete controller arc, findings F1–F29, experiments E1–E8.7 — see [RESEARCH_CONTROLLER.md](RESEARCH_CONTROLLER.md).
- Don't re-run experiments already recorded there. Evidence in the carry-over section below is a summary; the full record is in the archive.

**Discipline**
- Full protocol: [docs/RESEARCH_WORKFLOW.md](docs/RESEARCH_WORKFLOW.md)
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
- **Foundation contract**: [docs/OBSERVER_FOUNDATIONS.md](docs/OBSERVER_FOUNDATIONS.md).
  Active work follows Observe → Perturb → Compare → Prove. Act is downstream.
- **Program structure**: three mapping questions, Q1–Q3 in §3. **Q1 status: scoped positive**
  after independent local branchpoint calibration; Q2 is the active question.
- **Controller status**: **paused**, not dead. §4 defines the evidence that would bring controller research back to active status.
- **Instrument status**: foundation mechanics are repaired and the first
  42-cell Qwen3-1.7B calibration is complete. All zero controls, context
  matching, recovery matching, config hashes, and run-directory uniqueness
  checks passed.
- **Next recommended action**: begin Q2 with a true per-layer paired-delta
  propagation instrument. Endpoint hysteresis distances cannot identify where
  a delta is amplified or absorbed inside the residual stream.

---

## 2a. Mapping-program findings

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
    `docs/FOUNDATION_CALIBRATION_2026-07-19.md` and the durable local artifact
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
  - Artifacts: analyzer at `scripts/analyze_branchpoints.py`, commit `1ec2c14`.
  - Confidence: **medium** (1 prompt, 4 seeds, mechanistically consistent features, but prompt-breadth evidence standard not yet met).

## 2. Carry-over facts (established in controller arc)

_These are motivating evidence for the mapping program. Full records in RESEARCH_CONTROLLER.md §2._

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

**Current status**: **active after MHR/F33.** Matched endpoint measurements now
show thresholded, seed-dependent propagation, but `layer_stiffness` is a
within-trajectory velocity diagnostic and cannot reconstruct the paired
clean-vs-perturbed delta at each downstream layer. Q2 therefore needs an
explicit per-layer paired-delta trace.

---

### Q3 — Basin structure: when does a flip improve vs. degrade output?

**The claim we're trying to establish.** Given a token flip at branchpoint T, predict whether the resulting trajectory lands in a better or worse basin. F29 showed this is model-specific; we want to characterize the model-dependent features that determine it.

**Evidence standard** (Qwen3-1.7B scope): ≥5 prompts × ≥3 seeds per (prompt, config) cell, all within Qwen3-1.7B. Cross-model basin comparison is out of scope here (see RESEARCH_CONTROLLER.md F29 for the one cross-model data point we already have, to be revisited later).

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

**Current status**: not started. F29 is a single data point (TinyLlama). Needs at least 2 more models.

---

## 4. Controller-return criteria

The controller thesis is paused, not dead. It returns to active status when **any two** of the following become true during mapping work:

1. **Q1 produces a branchpoint predictor with precision ≥ 0.7.** Means we have a real trigger (predictable flippable tokens) instead of blind `raw_div > threshold`.
2. **Q2 identifies a specific layer with bounded-cascade propagation.** Means we know where to act without destructive cascade (F27 problem solved).
3. **Q3 identifies a predictable class of prompts or models where flips tend to improve.** Means we know when firing the controller helps rather than harms.

If any two land, a controller redesign experiment is warranted. If none land during the mapping program, the interpretability findings stand alone and the controller stays on ice.

**What does NOT justify re-opening controller work**: more tuning of the existing `raw_div → scaling | random additive | EMA opposition` design. That entire search space is exhausted (RESEARCH_CONTROLLER.md F17–F27).

---

## 5. Mapping backlog

Order: M1R and MHR are complete. Q2 per-layer paired-delta propagation is next.
Do not return to controller work: only the mapping AUROC stop condition landed;
the controller-specific precision, bounded-cascade, and directionality
criteria have not.

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
  `docs/FOUNDATION_CALIBRATION_2026-07-19.md`.

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
- **Outcome**: ran 10 control runs on "Describe the water cycle in a few sentences." with same config as M1.1's passing slice. All active cells fired (8-10 interventions per seed). Rerun of `scripts/analyze_branchpoints.py` on Qwen3-only opposing-anchor corpus:
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
  - Run `scripts/analyze_branchpoints.py` on the combined corpus (sourdough + new prompt).
- **Stop condition** (Q1 closure): within-Qwen3 held-out AUROC ≥ 0.80 across both prompts.
- **Historical expected payoff**: this was expected to settle Q1. The
  2026-07-19 audit established that the label did not match the causal question.
- **Expected runtime**: ~2 min with daemon.

### [x] M1.3 — Add direct diagnostic baselines *(foundation rebuild)*
- **Status**: complete 2026-07-19.
- **Outcome**: canonical diagnostics now include top-1 margin, logit entropy,
  hidden velocity, and hidden acceleration. The active local branchpoint format
  records clean top-1 margin and clean diagnostics.

### [ ] M2.1 — Per-layer propagation measurement from existing data
- **Question**: For the stress runs we already have at varying `intervention_layer`, extract the delta L2 norm at each probe_layer downstream of the injection. Build the propagation curve from existing events.jsonl layer_stiffness fields.
- **Design**: pure offline analysis of all existing stress runs.
- **Stop condition**: per-layer propagation curve for at least 2 intervention types at 3 injection layers on Qwen3-1.7B. If clean monotonic pattern, claim; otherwise needs M2.2.
- **Expected runtime**: ~30 min analysis.
- **Outcome**: _(filled when complete)_

### [ ] M2.2 — Synthetic-injection propagation sweep (only if M2.1 inconclusive)
- **Question**: If events.jsonl doesn't have enough probe_layer coverage, run a controlled observe suite with synthetic injection: seed an additive delta at chosen layer, hook-capture hidden states at every subsequent layer.
- **Design**: needs a small code change in the observe runner to allow "inject at L, capture at all downstream" mode. 3 injection layers × 3 magnitudes × 3 seeds × 2 models.
- **Code**: Codex territory. ~40 lines in observe/runner.py + a new orchestrator.
- **Blocked on**: M2.1 being insufficient + Codex code change.

### [ ] M3.1 — Prompt-class basin mapping on Qwen3-1.7B
- **Question**: Within Qwen3-1.7B, does the branchpoint-hijack outcome (improve vs degrade vs no-effect) systematically depend on prompt class? E.g., does the controller help on procedural prompts more than on creative ones?
- **Design**: 5 prompt classes (factual / procedural / creative / reasoning / code) × 3 seeds × 2 cells (shadow + opposing-anchor additive at L=-1, mag=0.3/0.6) = 30 control runs.
- **Stop condition**: one prompt-class feature that predicts improve-vs-degrade with AUROC ≥ 0.7 within Qwen3-1.7B.
- **Expected runtime**: ~5 min with daemon.
- **Outcome**: _(filled when complete)_

---

## 6. Sessions log (Mapping phase)

- **2026-04-19 · Program defined.** Pivoted from controller-focused to mapping-focused. Three questions (Q1 branchpoint geometry, Q2 propagation, Q3 basin structure) with stop conditions. Controller-return criteria specified. Next: M1.1.
- **2026-04-19 · M1.1 ran, Qwen3 AUROC 0.82.** Codex wrote `scripts/analyze_branchpoints.py`. Claude caught + fixed a circular-feature leak (features from active events included intervention_applied, scale_used, etc. — tautological). With honest shadow-trajectory features on the Qwen3-1.7B sourdough slice, held-out AUROC is **0.82** across 12 pairs / 4 seeds — clears Q1's 0.80 threshold. F30 added. Remaining Q1 gap: ≥2 prompts evidence standard. M1.2 queued: run a second Qwen3 prompt suite.
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

_(For sessions covering the controller arc 2026-02 through 2026-04-19, see [RESEARCH_CONTROLLER.md](RESEARCH_CONTROLLER.md) §5.)_

---

## 7. Quick-start snippets

**Orient before starting**
```bash
# see the status of both docs
head -40 RESEARCH.md RESEARCH_CONTROLLER.md

# current state via API
curl -s http://127.0.0.1:8899/api/capabilities | jq '.modes | keys'
curl -s http://127.0.0.1:8899/api/runs | jq '.runs[0:5] | .[] | {id, mode, headline}'
```

**Run the daemon (required for any active experiment)**
```bash
cd ~/observer
source .venv/bin/activate
python scripts/observer_daemon.py --model qwen3-1.7b
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
