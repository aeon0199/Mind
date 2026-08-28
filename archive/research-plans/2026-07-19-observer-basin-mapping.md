# Observer M3R Causal Basin Mapping Implementation Plan

> **Execution note:** Follow the repository's review → repair → validate loop.
> The controller remains out of scope. Each task lands as a small verified
> commit before the target-model calibration begins.

**Goal:** Determine whether clean, pre-intervention context features predict
whether a verified local token flip improves or degrades a task-specific
continuation on Qwen3-1.7B.

**Architecture:** Add a controller-free `basins` protocol. Pass one maps every
independent same-context local fork along a clean reference trajectory. Pass
two deterministically replays that reference, re-verifies selected contexts,
and continues clean and perturbed branches with the intervention disabled and
paired sampling randomness. A prompt registry scores each completed output
independently with a declared deterministic rubric. A separate analyzer uses
only pre-flip clean features and prompt metadata, with whole-prompt holdout.

**Evidence interpretation:**

- “better” means a higher normalized score under the prompt's declared rubric;
  it does not mean universal semantic quality;
- output scoring is branch-blind and occurs before paired comparison;
- score deltas within the declared tolerance are ties;
- the binary predictor uses sampled-token-flip episodes labeled improve or
  degrade; ties, no-flips, latent effects, and exact no-effects remain in
  coverage accounting;
- the Q3 stop condition is met only when mean prompt-held-out AUROC is at least
  0.70 across at least three valid held-out prompts.

**Calibration scope:** Qwen3-1.7B, five prompt classes, two independently
specified prompts per class, three generation seeds, additive magnitude 0.30
at layer 27, plus exact magnitude-0 controls. This exceeds the five-prompt
minimum while allowing prompt-class information to survive a held-out prompt.

---

## Task 1: Declare and validate outcome evaluators

**Files**

- Create: `src/runtime_lab/basins/__init__.py`
- Create: `src/runtime_lab/basins/evaluators.py`
- Test: `tests/test_basin_evaluators.py`

**Acceptance**

- Ten prompt specifications cover factual, procedural, creative, reasoning,
  and code tasks.
- Every rubric is serialized, versioned, and composed only from known check
  types.
- Every prompt has fixed good/degraded fixtures; good scores materially higher.
- Pair comparison is branch-order symmetric and honors the tie tolerance.
- Generated code is parsed but never executed.

## Task 2: Implement the causal two-pass basin protocol

**Files**

- Modify: `src/runtime_lab/config/schemas.py`
- Create: `src/runtime_lab/basins/experiment.py`
- Test: `tests/test_basin_experiment.py`

**Pass 1**

1. Start from one prompt-pass seed cache.
2. At every clean decision, fork identical caches.
3. Run one clean and one intervened decision with paired RNG.
4. Record clean-only diagnostics and features, local flip/effect metrics, and
   context fingerprints.
5. Advance only the clean reference.

**Selection**

- Partition decisions into early, middle, and late bands.
- In each band select the earliest sampled flip.
- If a band has no sampled flip, select its earliest measurable latent effect.
- For identity controls, select deterministic band anchors.
- Never inspect continuation outcomes during selection.

**Pass 2**

1. Reset sampling to the original seed and replay the clean reference.
2. Re-verify selected context fingerprints and clean decisions.
3. Continue clean and perturbed branches with no further intervention.
4. Pair the per-position random draw even after branch contexts diverge.
5. Restore the clean reference RNG and cache after every episode.
6. Score each full output independently, then compare.

**Acceptance**

- Zero intervention yields exact local rows and exact continuation episodes.
- A perturbed episode cannot alter the replayed clean reference.
- Selected contexts exactly match pass 1.
- Every continued episode records branch outputs, causal classification,
  cross-context trajectory separation, scores, and provenance.
- No continuation outcome is available to the selector.

## Task 3: Expose the protocol through Observer interfaces

**Files**

- Create: `src/runtime_lab/cli/basins.py`
- Modify: `src/runtime_lab/cli/main.py`
- Modify: `scripts/observer_daemon.py`
- Modify: `scripts/observer_console.py`
- Modify: `scripts/observer_console/index.html`
- Modify: `scripts/observer_console/app.js`
- Modify: `scripts/observer_console/styles.css` only if required
- Test: `tests/test_daemon_contract.py`
- Test: `tests/test_basin_cli.py`

**Acceptance**

- CLI supports registered prompts and a generic custom-prompt fallback.
- Daemon and console capability documents describe the causal protocol without
  controller language.
- Console launch/archive/detail views recognize `basins`.
- Existing modes remain unchanged.

## Task 4: Add a resumable warm-model M3R runner

**Files**

- Create: `scripts/run_basin_calibration.py`
- Test: `tests/test_basin_calibration.py`

**Acceptance**

- Deterministic grid: 10 prompts × 3 seeds × {0.00, 0.30}.
- One loaded Qwen3-1.7B backend is reused.
- Every completed cell is config-hash validated before resume.
- Controls must be exact; active cells may legitimately have no sampled flips.
- Manifest records complete, failed, resumed, and run provenance atomically.

## Task 5: Add prompt-held-out analysis

**Files**

- Create: `scripts/analyze_basins.py`
- Test: `tests/test_basin_analyzer.py`

**Acceptance**

- Re-hash and validate every run listed in the manifest.
- Load predictor inputs exclusively from `pre_flip_features` plus declared
  prompt class.
- Train preprocessing on training prompts only.
- Leave one prompt out per split; never split episodes from one prompt across
  train and test.
- Report valid/invalid splits, AUROC distribution, coverage, ties, no-flips,
  exact no-effects, class balance, prompt slices, and top feature weights.
- Fail closed if fewer than three prompt-held-out splits contain both labels.

## Task 6: Run calibration and record the result

**Artifact root**

`/Users/jmalone/Documents/Artifacts/Observer/m3r-basin-mapping-2026-07-19`

**Run sequence**

1. Validate evaluator fixtures.
2. Run one fake-model smoke.
3. Run one real Qwen3-1.7B smoke with a zero and active cell.
4. Run the resumable 60-cell grid.
5. Re-run the analyzer independently and compare timestamp-free output.
6. Inspect representative paired outputs manually without changing rubrics.

**Files**

- Create: `docs/Q3_BASIN_MAPPING_2026-07-19.md`
- Modify: `RESEARCH.md`
- Modify: `README.md`
- Modify: `REPRODUCIBILITY.md`
- Modify: `docs/OBSERVER_FOUNDATIONS.md`
- Modify: `docs/observer_paper.html`
- Test: `tests/test_claim_status.py`

**Claim rule**

- Report the stop condition as met only if the predeclared analyzer clears it.
- Otherwise report the actual limiting factor: insufficient flips, insufficient
  improve/degrade balance, weak held-out prediction, or evaluator sensitivity.
- The controller remains paused unless Q3 and one other return criterion are
  both genuinely satisfied.

## Task 7: Release verification and local integration

1. Run full pytest.
2. Run active-surface Ruff and compile checks.
3. Run diagnostic validation.
4. Build wheel and source distribution.
5. Parse the paper HTML and syntax-check console JavaScript.
6. Browser-QA the new mode at desktop and narrow viewport sizes.
7. Fast-forward the verified branch into local `main`; do not push.

