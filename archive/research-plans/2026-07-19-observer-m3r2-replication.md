# Observer M3R-2 Basin Replication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a preregistered, fresh-prompt replication of Observer's causal
basin experiment with higher-resolution code rubrics and a compact,
predeclared pre-flip predictor.

**Architecture:** Preserve `causal-branch-continuation-v1`, exact controls,
outcome-blind selection, exact clean replay, paired intervention-free
continuation, and whole-prompt holdout. Add a disjoint 15-prompt M3R-2 registry,
a non-executing code extractor/evaluator, manifest-declared predictor features
and regularization, and backward-compatible runner/analyzer support. Freeze and
commit all protocol code before generating calibration data.

**Tech Stack:** Python 3.12, PyTorch, Transformers, NumPy, pytest, Ruff, Git,
Qwen3-1.7B on Apple MPS.

## Global Constraints

- Do not use controller or historical shadow/active rows.
- Do not execute generated code.
- M3R-2 prompts must be disjoint from the ten M3R prompts.
- Keep the AUROC threshold at 0.70; require at least five valid whole-prompt
  holdouts.
- Fit preprocessing and the predictor on training prompts only.
- Use only the predeclared pre-flip/context features plus prompt class.
- Preserve ties, no-flips, invalid holdouts, and exact controls in coverage.
- Do not change prompts, rubrics, predictor features, regularization, or stop
  conditions after the full run begins.
- Full grid: 15 prompts × 3 seeds × magnitudes {0, 0.30} = 90 runs.
- Generation: 64 mapped tokens, 96 continuation tokens, at most 5
  outcome-blind episodes, layer 27, temperature 0.8, top-p 1.0.
- The codebase's original M3R prompts, evaluator behavior, manifests, and
  analysis must remain reproducible.

---

### Task 1: Freeze the fresh prompt and evaluator registry

**Files:**
- Modify: `src/runtime_lab/basins/evaluators.py`
- Modify: `src/runtime_lab/basins/__init__.py`
- Modify: `tests/test_basin_evaluators.py`

**Interfaces:**
- Produces: `M3R2_PROMPT_SPECS: tuple[BasinPromptSpec, ...]`
- Produces: `ALL_PROMPT_SPECS: tuple[BasinPromptSpec, ...]`
- Produces: `EVALUATOR_V2_ID = "deterministic-prompt-rubric-v2"`
- Produces: `validate_prompt_specs(prompt_specs=...)`
- Preserves: `PROMPT_SPECS` as the original ten-prompt M3R registry.

- [ ] **Step 1: Write failing registry and evaluator tests**

Add tests that require:

```python
counts = Counter(spec.prompt_class for spec in M3R2_PROMPT_SPECS)
assert counts == {
    "factual": 3,
    "procedural": 3,
    "creative": 3,
    "reasoning": 3,
    "code": 3,
}
assert set(spec.key for spec in PROMPT_SPECS).isdisjoint(
    spec.key for spec in M3R2_PROMPT_SPECS
)
assert all(
    spec.evaluator_id == "deterministic-prompt-rubric-v2"
    for spec in M3R2_PROMPT_SPECS
)
```

Add a code-evaluator test whose output contains prose and a fenced, valid
function. The v2 contract component must parse the extracted function while
the separate no-fence component fails. Add a malicious extracted function and
prove scoring never creates its marker file.

- [ ] **Step 2: Run the new evaluator tests and verify RED**

Run:

```bash
PYTHONPATH=src /Users/jmalone/Code/Observer/.venv/bin/pytest \
  tests/test_basin_evaluators.py -q
```

Expected: import or assertion failures because the M3R-2 registry and v2
contract do not exist.

- [ ] **Step 3: Implement the v2 evaluator without changing v1 behavior**

Add a `python_ast_contract_v2` check. It must:

1. locate the requested top-level `def` or `async def`;
2. try successively shorter suffixes until a syntactically complete module
   containing that function parses;
3. score the same explicit signature, annotation, source-pattern, return, and
   raise criteria as v1 against only the extracted source;
4. return extraction metadata in evidence;
5. use `ast.parse` only and never import or execute generated text.

Support both evaluator IDs in `score_output`. Keep v1 scoring protocol strings
for old specs and emit `branch-blind-deterministic-paired-scoring-v2` for v2
specs.

Declare exactly these disjoint prompt keys, three per class:

| Class | Keys | Required content |
|---|---|---|
| factual | `seasons_tilt`, `rock_cycle`, `electric_circuit` | axial tilt/hemispheres; complete rock transformations; closed/open circuit mechanism |
| procedural | `rice_steps`, `flat_tire_steps`, `tomato_transplant_steps` | numbered steps, quantities/times, ordered task constraints |
| creative | `glass_violin`, `mars_orchard`, `undersea_library` | 80-150 words, fixed story elements, exact phrase and ending constraint |
| reasoning | `museum_tickets`, `rectangle_garden`, `discount_tax` | intermediate values, correct derivation, exact final answer |
| code | `dedupe_order_code`, `clamp_code`, `word_frequencies_code` | requested signature, AST/source contract, no markdown fence |

Every spec gets a complete good fixture and a clearly degraded fixture.

- [ ] **Step 4: Verify GREEN and fixture margins**

Run:

```bash
PYTHONPATH=src /Users/jmalone/Code/Observer/.venv/bin/pytest \
  tests/test_basin_evaluators.py -q
```

Expected: every original and M3R-2 fixture passes, each good score is at least
0.70, each good-minus-degraded margin is at least 0.20, and malicious code is
not executed.

- [ ] **Step 5: Commit the frozen registry**

Commit:

```text
feat: freeze M3R2 prompt rubrics
```

---

### Task 2: Preregister the compact predictor and replication design

**Files:**
- Create: `src/runtime_lab/basins/replication.py`
- Modify: `scripts/run_basin_calibration.py`
- Modify: `scripts/analyze_basins.py`
- Modify: `tests/test_basin_calibration.py`
- Modify: `tests/test_basin_analyzer.py`

**Interfaces:**
- Produces: `M3R2_CALIBRATION_PROTOCOL`
- Produces: `M3R2_ANALYSIS_PROTOCOL`
- Produces: `M3R2_PREDICTOR_FEATURES`
- Produces: runner flag `--prompt-set m3r2`
- Consumes: `M3R2_PROMPT_SPECS`

- [ ] **Step 1: Write failing runner-design tests**

Require `--prompt-set m3r2` to select all 15 fresh prompts and build 90 unique
cells for seeds 0/1/2 and magnitudes 0/0.30. Require the design record to
contain:

```python
{
    "protocol": "observer-m3r2-basin-replication-v1",
    "prompt_set": "m3r2",
    "max_tokens": 64,
    "continuation_tokens": 96,
    "max_episodes": 5,
    "predictor_feature_set": list(M3R2_PREDICTOR_FEATURES),
    "predictor_l2_regularization": 0.05,
    "minimum_valid_prompt_splits": 5,
    "auroc_threshold": 0.70,
}
```

Also assert that the default `m3r` path still selects the original ten prompts
and keeps `observer-m3r-basin-calibration-v1`.

- [ ] **Step 2: Write failing analyzer-boundary tests**

Create a synthetic M3R-2 manifest containing a strong leaked feature outside
the allowlist. Assert that:

- the manifest protocol is accepted;
- only `M3R2_PREDICTOR_FEATURES` present in the data appear in
  `predictor.feature_names`;
- the leaked feature is absent;
- `l2_regularization == 0.05`;
- `minimum_valid_splits == 5`;
- whole-prompt holdout remains intact.

- [ ] **Step 3: Run runner/analyzer tests and verify RED**

Run:

```bash
PYTHONPATH=src /Users/jmalone/Code/Observer/.venv/bin/pytest \
  tests/test_basin_calibration.py tests/test_basin_analyzer.py -q
```

Expected: failures for missing protocol constants, prompt-set selection, and
manifest-driven predictor boundaries.

- [ ] **Step 4: Implement replication constants**

Declare this exact feature tuple in `replication.py`:

```python
M3R2_PREDICTOR_FEATURES = (
    "clean_top1_margin",
    "clean_hidden_norm",
    "normalized_position",
    "context_sequence_length",
    "prefix_repetition_score",
    "prefix_unique_token_ratio",
    "consumed_token.length",
    "consumed_token.leading_space",
    "clean_diagnostics.local_prediction_error",
    "clean_diagnostics.logit_entropy",
    "clean_diagnostics.hidden_velocity",
    "clean_diagnostics.hidden_acceleration",
    "clean_diagnostics.spectral.spectral_entropy",
    "clean_diagnostics.spectral.permutation_change",
    "clean_diagnostics.svd.effective_rank",
    "clean_diagnostics.svd.top1_energy_frac",
    "clean_diagnostics.layer_stiffness.27.stiffness",
    "clean_diagnostics.layer_stiffness.27.stiffness_trend",
    "prompt_class.code",
    "prompt_class.creative",
    "prompt_class.factual",
    "prompt_class.procedural",
    "prompt_class.reasoning",
)
```

Set M3R-2 regularization to `0.05`, threshold to `0.70`, and minimum valid
prompt splits to `5`.

- [ ] **Step 5: Make runner selection backward-compatible**

Add `--prompt-set {m3r,m3r2}` with default `m3r`. Resolve `"all"` inside the
selected set, validate only selected fixtures, write the appropriate manifest
protocol, and record the exact analysis boundary in `design`. Reject prompt
keys outside the selected set.

- [ ] **Step 6: Make analyzer behavior manifest-driven**

Accept both manifest protocols. For M3R-2:

- use the manifest's feature allowlist exactly;
- pass `l2_regularization=0.05` into the existing deterministic logistic
  fitter;
- use manifest threshold/minimum-split values unless explicitly overridden;
- report feature-set source, regularization, train AUROC, held-out AUROC, and
  invalid prompts;
- use `observer-m3r2-prompt-held-out-analysis-v1`.

For historical M3R artifacts, preserve the existing all-pre-flip-feature
behavior and analysis protocol.

- [ ] **Step 7: Verify GREEN and backward compatibility**

Run:

```bash
PYTHONPATH=src /Users/jmalone/Code/Observer/.venv/bin/pytest \
  tests/test_basin_calibration.py tests/test_basin_analyzer.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit the preregistered design**

Commit:

```text
feat: preregister M3R2 basin replication
```

---

### Task 3: Expose and document the frozen replication

**Files:**
- Modify: `scripts/observer_console.py`
- Modify: `tests/test_daemon_contract.py`
- Modify: `REPRODUCIBILITY.md`
- Modify: `docs/OBSERVER_FOUNDATIONS.md`

**Interfaces:**
- Console/daemon registered-prompt recipes must include M3R and M3R-2 keys.
- Reproducibility documentation must name the frozen M3R-2 grid and analysis
  boundary before data generation.

- [ ] **Step 1: Write failing console contract assertions**

Assert the console recipe/capabilities include `seasons_tilt`,
`tomato_transplant_steps`, and `clamp_code`, while custom prompts remain
explicitly exploratory.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src /Users/jmalone/Code/Observer/.venv/bin/pytest \
  tests/test_daemon_contract.py -q
```

Expected: M3R-2 keys are absent.

- [ ] **Step 3: Expose both frozen registries and write preregistration docs**

Update console recipes to enumerate `ALL_PROMPT_SPECS`. Add an M3R-2
preregistration section to the foundation and reproducibility documents with
the exact 15 prompts, 3 seeds, two magnitudes, 64/96 token lengths, five
episodes, compact feature tuple, regularization 0.05, whole-prompt holdout,
AUROC 0.70, minimum five valid splits, and interpretation boundary.

- [ ] **Step 4: Verify GREEN and full protocol suite**

Run:

```bash
PYTHONPATH=src /Users/jmalone/Code/Observer/.venv/bin/pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the complete protocol freeze**

Commit:

```text
docs: freeze M3R2 replication protocol
```

After this commit, do not change the protocol in response to outcomes.

---

### Task 4: Run a fail-closed smoke experiment

**Files:**
- No repository changes unless a mechanical defect is first reproduced by a
  failing test.
- Create artifacts under:
  `/Users/jmalone/Documents/Artifacts/Observer/m3r2-basin-smoke-2026-07-19`

**Interfaces:**
- Consumes: frozen M3R-2 runner and analyzer.
- Produces: a two-prompt, one-seed, zero/active smoke manifest and analysis.

- [ ] **Step 1: Run fixture validation and CLI help**

Run the selected evaluator validator and inspect:

```bash
PYTHONPATH=src /Users/jmalone/Code/Observer/.venv/bin/python \
  scripts/run_basin_calibration.py --help
```

- [ ] **Step 2: Run two fresh prompt controls and active cells**

Use `rice_steps,clamp_code`, seed 0, magnitudes 0/0.30, 64 mapped tokens, 96
continuation tokens, max episodes 5, layer late, temperature 0.8.

- [ ] **Step 3: Analyze the smoke root**

Run `scripts/analyze_basins.py` on the smoke root. Expected: every config hash
and identity control validates; the predictor may report insufficient valid
splits and must not be promoted.

- [ ] **Step 4: Mechanical gate**

If smoke fails because of implementation mechanics, write a failing regression
test, fix it, commit the fix, and restart smoke in a new root. Do not alter
prompts, rubrics, feature set, regularization, or stop conditions based on
smoke outcomes.

---

### Task 5: Execute and analyze the full 90-run M3R-2 calibration

**Files:**
- Create durable artifacts under:
  `/Users/jmalone/Documents/Artifacts/Observer/m3r2-basin-replication-2026-07-19`

**Interfaces:**
- Produces: `basin_manifest.json`, `basin_analysis.json`,
  `BASIN_ANALYSIS.md`, 90 validated run directories, and per-episode text.

- [ ] **Step 1: Start the resumable warm-model grid**

Run:

```bash
PYTHONPATH=src /Users/jmalone/Code/Observer/.venv/bin/python \
  scripts/run_basin_calibration.py \
  --root /Users/jmalone/Documents/Artifacts/Observer/m3r2-basin-replication-2026-07-19 \
  --prompt-set m3r2 \
  --prompt-keys all \
  --seeds 0,1,2 \
  --magnitudes 0,0.3 \
  --max-tokens 64 \
  --continuation-tokens 96 \
  --max-episodes 5 \
  --layer late \
  --temperature 0.8 \
  --top-p 1.0 \
  --top-k 0 \
  --intervention-seed 42 \
  --tie-tolerance 0.025
```

- [ ] **Step 2: Validate completion before analysis**

Require 90/90 complete cells, 90 unique run directories, 90 unique config
hashes, and exact identity controls. Count all local rows, selected episodes,
sampled flips, no-flips, and outcome classes.

- [ ] **Step 3: Run the frozen analyzer**

Generate JSON and Markdown in the artifact root. Do not tune the threshold,
features, regularization, or minimum valid splits.

- [ ] **Step 4: Independently reproduce analysis**

Write a second result to `/tmp`, remove only `generated_at`, and require a
zero diff from the durable JSON.

- [ ] **Step 5: Manually audit representative artifacts**

Inspect bounded excerpts for at least:

- one improve;
- one degrade;
- one sampled-flip tie;
- one no-flip exact continuation;
- one v2 code-contract parse.

Record evaluator-resolution limitations without rescoring the run.

---

### Task 6: Record the outcome, verify, integrate, and publish

**Files:**
- Create: `docs/Q3_M3R2_REPLICATION_2026-07-19.md`
- Modify: `RESEARCH.md`
- Modify: `README.md`
- Modify: `docs/observer_paper.html`
- Modify: `tests/test_claim_status.py`

**Interfaces:**
- Consumes: frozen M3R-2 analysis.
- Produces: an exact positive, negative, or insufficient finding without
  changing the success criterion.

- [ ] **Step 1: Write failing claim-status tests from the actual result**

Tests must lock run count, controls, coverage, class outcomes, valid/invalid
prompt counts, observed held-out AUROC, stop status, scoped procedural result,
interpretation boundary, and controller status.

- [ ] **Step 2: Verify RED**

Run `tests/test_claim_status.py`; expected failures because the report and
ledger entry do not yet exist.

- [ ] **Step 3: Write the evidence report and update claims**

State exactly one of:

- threshold met;
- below threshold;
- insufficient valid prompt splits.

Do not call task-rubric improvement universal quality. Compare M3R and M3R-2,
state whether the procedural pattern replicated, and keep the controller
paused unless the predeclared return criteria genuinely change.

- [ ] **Step 4: Run complete release verification**

Run:

```bash
PYTHONPATH=src /Users/jmalone/Code/Observer/.venv/bin/pytest -q
PYTHONPATH=src /Users/jmalone/Code/Observer/.venv/bin/ruff check src scripts tests
PYTHONPATH=src /Users/jmalone/Code/Observer/.venv/bin/python -m compileall -q src scripts tests
PYTHONPATH=src /Users/jmalone/Code/Observer/.venv/bin/python scripts/validate_diagnostics.py --json-out /tmp/observer-m3r2-diagnostics.json
node --check scripts/observer_console/app.js
uv build --out-dir /tmp/observer-m3r2-dist-20260719
```

Parse `docs/observer_paper.html`, run `git diff --check`, and confirm the full
artifact root is unchanged by independent reanalysis.

- [ ] **Step 5: Commit and finish the branch**

Commit the result, fast-forward into local `main`, rerun the full test suite on
the merged tree, remove the owned worktree, delete the feature branch, push
`main`, and verify GitHub points to the final commit.

