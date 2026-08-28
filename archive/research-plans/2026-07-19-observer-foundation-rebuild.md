# Observer Foundation Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Observer as a trustworthy trajectory-measurement instrument before any controller research resumes.

**Architecture:** Observer's foundation is split into four contracts: observe a canonical token-time event, perturb an identical cached context locally, compare matched clean and perturbed trajectories, and preserve enough provenance to reproduce every result. Controller policy remains downstream and is intentionally excluded except where its existing event artifacts must conform to the shared runtime contract.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, NumPy, pytest/unittest, stdlib JSON/HTTP tooling.

## Global Constraints

- Preserve `SeedCache` deterministic prompt-pass branching and verify zero-intervention equality.
- Treat event `t` as one causal decision: consume the already-selected token, measure/intervene during that forward pass, then record the newly predicted next token.
- Never train a branchpoint predictor on post-divergence continuation mismatches.
- Compute hysteresis recovery only between matched clean and perturbed continuations with equal phase instructions and aligned decision positions.
- Record actual backend model identity separately from any requested alias.
- Hash the final resolved configuration written to disk.
- Use collision-resistant run IDs and refuse artifact-directory reuse.
- Keep controller policy, cooldown tuning, dashboard redesign, and controller claims outside this rebuild.
- Keep NNsight remote execution explicitly unavailable until a real trace-based backend exists; do not advertise a broken path.
- Follow red-green-refactor for every production behavior change.

---

## File Structure

- `src/runtime_lab/core/runtime/events.py`: canonical event serialization contract.
- `src/runtime_lab/core/runtime/engine.py`: one-token causal forward execution.
- `src/runtime_lab/core/model/cache_utils.py`: cache cloning, fingerprinting, and sequence-length inspection.
- `src/runtime_lab/core/io/run_artifacts.py`: unique run-directory allocation and final configuration records.
- `src/runtime_lab/core/backend/identity.py`: requested-versus-loaded model identity validation.
- `src/runtime_lab/core/diagnostics/baselines.py`: simple baseline signals used to interpret Observer diagnostics.
- `src/runtime_lab/core/trajectory/generation.py`: reusable traced generation and continuation state.
- `src/runtime_lab/branchpoints/experiment.py`: independent local counterfactual branchpoint protocol.
- `src/runtime_lab/hysteresis/runner.py`: matched clean/perturbed exposure and recovery protocol.
- `scripts/analyze_branchpoints.py`: analysis of local counterfactual rows only.
- `scripts/analyze_branchpoints_legacy.py`: preserved invalid historical control-pair analyzer with an explicit warning.
- `scripts/validate_diagnostics.py`: synthetic diagnostic acceptance suite and optional run analysis.
- `tests/fakes.py`: deterministic tiny causal model and tokenizer with no network dependency.
- `docs/OBSERVER_FOUNDATIONS.md`: scientific and artifact contracts.

---

### Task 1: Restore installation, version, and CI truth

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `.github/workflows/ci.yml`
- Test: `tests/test_installation_contract.py`

**Interfaces:**
- Consumes: existing `runtime_lab.cli.main:main`.
- Produces: an editable install that makes `python -m runtime_lab.cli.main --help` and the test suite work in a clean environment.

- [ ] **Step 1: Write failing installation-contract tests**

```python
def test_ci_installs_the_src_layout_package():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "pip install -e ." in workflow


def test_public_version_is_consistent():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    citation = (ROOT / "CITATION.cff").read_text()
    version = pyproject["project"]["version"]
    assert version == "0.2.0"
    assert f'version: "{version}"' in citation
```

- [ ] **Step 2: Run the tests and verify they fail for the missing editable install and `0.1.0` package version**

Run: `.venv/bin/python -m pytest tests/test_installation_contract.py -q`

Expected: two assertion failures identifying the CI install command and version mismatch.

- [ ] **Step 3: Align package metadata and setup instructions**

Set `project.version = "0.2.0"`, change CI installation to `pip install -e .`, run pytest in CI, and make README/CONTRIBUTING setup use `python -m pip install -e .`.

- [ ] **Step 4: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_installation_contract.py -q`

Expected: all focused tests pass.

Run: `.venv/bin/python -m pytest -q`

Expected: the complete suite passes.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml README.md CONTRIBUTING.md .github/workflows/ci.yml tests/test_installation_contract.py
git commit -m "fix: restore install and CI contract"
```

---

### Task 2: Establish the canonical token-time runtime contract

**Files:**
- Modify: `src/runtime_lab/core/runtime/events.py`
- Modify: `src/runtime_lab/core/runtime/engine.py`
- Modify: `src/runtime_lab/core/model/cache_utils.py`
- Modify: `src/runtime_lab/observe/runner.py`
- Modify: `src/runtime_lab/control/adaptive_runner.py`
- Modify: `src/runtime_lab/core/diagnostics/manager.py`
- Modify: `src/runtime_lab/core/sampling.py`
- Modify: `src/runtime_lab/core/advisory.py`
- Modify: `src/runtime_lab/stress/experiment.py`
- Test: `tests/test_runtime_contract.py`
- Test: `tests/test_core_regressions.py`

**Interfaces:**
- Produces: `runtime_event_to_record(event: RuntimeEvent) -> dict[str, Any]`.
- Produces: `cache_sequence_length(past_key_values: Any) -> int | None`.
- Preserves: legacy `token_id` and `token_text` event aliases while adding explicit consumed/predicted fields.

- [ ] **Step 1: Write failing event and attention-mask tests**

```python
def test_runtime_event_record_preserves_both_sides_of_the_decision():
    record = runtime_event_to_record(sample_runtime_event())
    assert record["consumed_token_id"] == 7
    assert record["predicted_next_token_id"] == 11
    assert record["token_id"] == 7
    assert record["measure_resolved_layer_idx"] == 2
    assert record["act_resolved_layer_idx"] == 4


def test_step_attention_mask_uses_cache_length_plus_input_token(fake_runtime):
    fake_runtime.engine.step(
        t=99,
        consumed_token_id=3,
        prompt_len=2,
        past_key_values=fake_runtime.cache_with_length(5),
    )
    assert fake_runtime.model.last_attention_mask.shape[-1] == 6
```

- [ ] **Step 2: Verify both tests fail for missing serialization and the `prompt_len + t + 1` mask**

Run: `.venv/bin/python -m pytest tests/test_runtime_contract.py -q`

Expected: failure for the missing serializer and incorrect mask length.

- [ ] **Step 3: Implement canonical serialization and cache-derived mask length**

Add `runtime_event_to_record`, add `cache_sequence_length`, and make `RuntimeEngine.step` prefer `cache_sequence_length(cache) + 1`. Update observe and control to serialize the same event contract, including exact measure/act layer indices.

- [ ] **Step 4: Write failing regression tests for health, nucleus sampling, seed zero, and advisory semantics**

```python
def test_empty_diagnostics_are_not_healthy():
    assert summarize_diagnostics_health([])["ok"] is False


def test_top_p_keeps_the_first_token_crossing_the_cutoff():
    kept = nucleus_keep_mask(torch.tensor([0.4, 0.3, 0.2, 0.1]), top_p=0.5)
    assert kept.tolist() == [True, True, False, False]


def test_seed_zero_is_not_replaced_by_forty_two():
    assert resolve_intervention_seed(0) == 0


def test_first_token_divergence_zero_means_immediate_flip():
    advisory = advise_stress(stress_summary(first_token_divergence=0))
    assert any("differed immediately" in text for text in advisory.observations)
```

- [ ] **Step 5: Verify the regression tests fail for the audited behavior**

Run: `.venv/bin/python -m pytest tests/test_core_regressions.py -q`

Expected: four failures matching the known regressions.

- [ ] **Step 6: Implement the minimum fixes**

Make health require at least one non-degraded valid step, expose a tested nucleus keep-mask that includes the crossing token, preserve integer seed zero, and correct the advisory wording.

- [ ] **Step 7: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_runtime_contract.py tests/test_core_regressions.py -q`

Expected: focused tests pass.

Run: `.venv/bin/python -m pytest -q`

Expected: full suite passes.

- [ ] **Step 8: Commit**

```bash
git add src/runtime_lab/core src/runtime_lab/observe/runner.py src/runtime_lab/control/adaptive_runner.py src/runtime_lab/stress/experiment.py tests
git commit -m "fix: define canonical token decision contract"
```

---

### Task 3: Repair artifact and model provenance

**Files:**
- Create: `src/runtime_lab/core/io/run_artifacts.py`
- Create: `src/runtime_lab/core/backend/identity.py`
- Modify: `src/runtime_lab/observe/runner.py`
- Modify: `src/runtime_lab/stress/experiment.py`
- Modify: `src/runtime_lab/hysteresis/runner.py`
- Modify: `src/runtime_lab/control/adaptive_runner.py`
- Modify: `scripts/observer_daemon.py`
- Modify: `scripts/observer_console.py`
- Modify: `src/runtime_lab/core/backend/loader.py`
- Test: `tests/test_artifact_contract.py`
- Test: `tests/test_daemon_contract.py`

**Interfaces:**
- Produces: `create_run_dir(runs_root: str | Path, mode: str) -> tuple[str, Path]`.
- Produces: `resolved_model_identity(requested: str | None, backend_result: BackendLoadResult) -> dict[str, str | None]`.
- Produces: `write_config_record(run_dir: Path, config: dict[str, Any]) -> tuple[str, Path]`.

- [ ] **Step 1: Write failing unique-ID and final-hash tests**

```python
def test_run_directories_never_collide(tmp_path):
    first_id, first = create_run_dir(tmp_path, "stress")
    second_id, second = create_run_dir(tmp_path, "stress")
    assert first_id != second_id
    assert first != second


def test_written_hash_matches_the_exact_saved_config(tmp_path):
    config_hash, path = write_config_record(tmp_path, {"resolved_layer": 13})
    saved = json.loads(path.read_text())
    assert saved["config_hash"] == config_hash
    assert hash_config(saved["config"]) == config_hash
```

- [ ] **Step 2: Verify the artifact tests fail because the module does not exist**

Run: `.venv/bin/python -m pytest tests/test_artifact_contract.py -q`

Expected: import failure for `run_artifacts`.

- [ ] **Step 3: Implement collision-resistant directories and final configuration records**

Use `YYYYMMDD_HHMMSS_microseconds_<8-char UUID>` IDs, create directories with `exist_ok=False`, and hash only the final JSON-safe resolved configuration.

- [ ] **Step 4: Write failing backend-identity and daemon tests**

```python
def test_prebuilt_backend_rejects_a_different_requested_model(fake_backend):
    with pytest.raises(ValueError, match="requested model"):
        resolved_model_identity("other-model", fake_backend)


def test_daemon_preserves_control_threshold_overrides():
    cfg = _build_config_control({"prompt": "p", "threshold_warn": 0.2, "hold_warn": 9})
    assert cfg.threshold_warn == 0.2
    assert cfg.hold_warn == 9
```

- [ ] **Step 5: Verify the identity tests fail for model mislabeling and ignored daemon fields**

Run: `.venv/bin/python -m pytest tests/test_daemon_contract.py -q`

Expected: failures showing no identity guard and defaulted control values.

- [ ] **Step 6: Apply provenance helpers across all runners**

Record `model` as the loaded backend key, preserve `model_requested`, serialize resolved layers, write `config.json` for every mode, and make daemon requests reject a different model. Accept both flat request fields and the documented nested `config` object. Use `sys.executable` in the Console and preserve seed zero.

- [ ] **Step 7: Make NNsight remote fail honestly**

Reject `nnsight_remote=True` with a message explaining that a trace-based backend is required, and update public capability text so unsupported remote execution is not advertised.

- [ ] **Step 8: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_artifact_contract.py tests/test_daemon_contract.py -q`

Expected: focused tests pass.

Run: `.venv/bin/python -m pytest -q`

Expected: full suite passes.

- [ ] **Step 9: Commit**

```bash
git add src scripts tests
git commit -m "fix: make experiment provenance authoritative"
```

---

### Task 4: Resolve model-depth-dependent layers after model load

**Files:**
- Modify: `src/runtime_lab/cli/_common.py`
- Modify: `src/runtime_lab/cli/observe.py`
- Modify: `src/runtime_lab/cli/stress.py`
- Modify: `src/runtime_lab/cli/hysteresis.py`
- Modify: `src/runtime_lab/cli/control.py`
- Modify: `scripts/observer_daemon.py`
- Create: `tests/fixtures/models.json`
- Test: `tests/test_layer_resolution_contract.py`

**Interfaces:**
- Produces: `backend_num_layers(backend_result: BackendLoadResult) -> int`.
- Requires: semantic aliases and `auto` probes are resolved with a concrete positive model depth.

- [ ] **Step 1: Write failing small-model layer tests**

```python
def test_mid_is_not_resolved_without_model_depth():
    with pytest.raises(ValueError, match="num_layers"):
        resolve_semantic_layer("mid", None)


def test_auto_probes_for_five_layers_are_all_valid():
    assert resolve_probe_layers("auto", 5) == [0, 1, 2, 4]
```

- [ ] **Step 2: Verify the first test fails because `mid` silently becomes `-14`**

Run: `.venv/bin/python -m pytest tests/test_layer_resolution_contract.py -q`

Expected: failure showing `-14` was returned.

- [ ] **Step 3: Load once in each CLI, then resolve layers**

Have CLI entry points load the backend, derive `num_layers`, resolve semantic intervention/probe selections, and pass the prebuilt backend to the runner. Have the daemon perform the same resolution against its already-loaded backend.
Add an `audit-tiny` registry entry for `hf-internal-testing/tiny-random-gpt2` so the smoke command below is self-contained and repeatable.

- [ ] **Step 4: Run focused, full, and tiny-model smoke tests**

Run: `.venv/bin/python -m pytest tests/test_layer_resolution_contract.py -q`

Expected: focused tests pass.

Run: `.venv/bin/python -m pytest -q`

Expected: full suite passes.

Run: `.venv/bin/python -m runtime_lab.cli.main observe --model audit-tiny --registry-path tests/fixtures/models.json --max-tokens 2 --probe-layers auto --runs-dir /tmp/observer-layer-smoke`

Expected: command exits successfully with every resolved layer in range.

- [ ] **Step 5: Commit**

```bash
git add src/runtime_lab/cli scripts/observer_daemon.py tests/fixtures/models.json tests/test_layer_resolution_contract.py
git commit -m "fix: resolve semantic layers from loaded models"
```

---

### Task 5: Build the local counterfactual branchpoint protocol

**Files:**
- Create: `src/runtime_lab/branchpoints/__init__.py`
- Create: `src/runtime_lab/branchpoints/experiment.py`
- Create: `src/runtime_lab/core/trajectory/generation.py`
- Create: `src/runtime_lab/cli/branchpoints.py`
- Modify: `src/runtime_lab/cli/main.py`
- Modify: `src/runtime_lab/config/schemas.py`
- Create: `tests/fakes.py`
- Test: `tests/test_branchpoint_experiment.py`

**Interfaces:**
- Produces: `BranchpointConfig`.
- Produces: `run_branchpoint_experiment(config, ..., prebuilt_backend=None) -> dict[str, Any]`.
- Produces one `events.jsonl` row per independent local fork with `argmax_flip`, `sampled_flip`, clean diagnostics, clean margin, logit KL/JS, context fingerprint, and intervention delta.
- Continues generation only along the clean branch; every perturbed branch is discarded after one decision.

- [ ] **Step 1: Write a failing local-causality test using the deterministic fake model**

```python
def test_each_branchpoint_uses_the_same_clean_context(fake_backend, tmp_path):
    summary = run_branchpoint_experiment(branchpoint_config(), runs_dir=tmp_path, prebuilt_backend=fake_backend)
    rows = read_jsonl(Path(summary["artifacts"]["events_path"]))
    assert all(row["contexts_matched"] for row in rows)
    assert [row["decision_index"] for row in rows] == list(range(1, len(rows) + 1))
    assert all("clean_predicted_token_id" in row for row in rows)
    assert all("perturbed_predicted_token_id" in row for row in rows)


def test_one_flip_does_not_label_later_clean_steps_positive(fake_backend, tmp_path):
    rows = run_single_flip_branchpoint_fixture(fake_backend, tmp_path)
    assert [row["argmax_flip"] for row in rows] == [True, False, False]
```

- [ ] **Step 2: Verify the tests fail because the branchpoint package does not exist**

Run: `.venv/bin/python -m pytest tests/test_branchpoint_experiment.py -q`

Expected: import failure for `runtime_lab.branchpoints`.

- [ ] **Step 3: Implement reusable traced continuation state**

`generation.py` must retain the clean cache, pending token, generated token count, per-step hidden vector, and logits so a caller can fork before any decision and continue only the selected branch.

- [ ] **Step 4: Implement independent one-step forks**

For each decision after the prompt-selected first token, clone the same clean cache twice, snapshot the CPU sampling RNG, run clean and perturbed steps, restore the clean post-step RNG state, emit the comparison row, and continue from the clean result only.

- [ ] **Step 5: Add the `branchpoints` CLI**

Expose model, prompt, max tokens, seed, sampling, layer, intervention type, magnitude, and output directory. Resolve semantic layers after backend load.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_branchpoint_experiment.py -q`

Expected: local-causality tests pass.

Run: `.venv/bin/python -m pytest -q`

Expected: full suite passes.

- [ ] **Step 7: Commit**

```bash
git add src/runtime_lab/branchpoints src/runtime_lab/core/trajectory/generation.py src/runtime_lab/cli src/runtime_lab/config/schemas.py tests
git commit -m "feat: add local counterfactual branchpoint mapping"
```

---

### Task 6: Rebuild branchpoint analysis around causal rows

**Files:**
- Create: `scripts/analyze_branchpoints_legacy.py`
- Replace: `scripts/analyze_branchpoints.py`
- Test: `tests/test_branchpoint_analyzer.py`

**Interfaces:**
- Consumes: `branchpoint_run_*/config.json`, `summary.json`, and `events.jsonl`.
- Produces: repeated run-grouped evaluation with AUROC, precision, recall, split counts, and uncertainty range.
- Rejects: control-run token mismatches as causal branchpoint labels.

- [ ] **Step 1: Preserve the historical analyzer with an invalidity warning**

Move its current contents to `analyze_branchpoints_legacy.py` and make its entry point exit unless `--acknowledge-invalid-labels` is supplied.

- [ ] **Step 2: Write failing analyzer-contract tests**

```python
def test_analyzer_reads_only_local_counterfactual_rows(tmp_path):
    write_control_pair_fixture(tmp_path)
    write_branchpoint_fixture(tmp_path)
    rows = load_branchpoint_rows(tmp_path, label="argmax_flip")
    assert len(rows) == 3
    assert {row["run_id"] for row in rows} == {"branchpoint_run_fixture"}


def test_evaluation_splits_by_whole_run(tmp_path):
    result = analyze_fixture_runs(tmp_path)
    assert all(not set(split["train_runs"]) & set(split["test_runs"]) for split in result["splits"])
    assert "precision" in result["aggregate"]
    assert "recall" in result["aggregate"]
```

- [ ] **Step 3: Verify the tests fail because the current analyzer reads controller pairs**

Run: `.venv/bin/python -m pytest tests/test_branchpoint_analyzer.py -q`

Expected: failures showing the old control-pair contract.

- [ ] **Step 4: Implement local-row loading and repeated grouped evaluation**

Flatten only clean diagnostics and pre-intervention clean fields. Perform repeated run-grouped splits, train preprocessing on training rows only, and report mean/median/min/max AUROC plus precision and recall.

- [ ] **Step 5: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_branchpoint_analyzer.py -q`

Expected: analyzer tests pass.

Run: `.venv/bin/python -m pytest -q`

Expected: full suite passes.

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_branchpoints.py scripts/analyze_branchpoints_legacy.py tests/test_branchpoint_analyzer.py
git commit -m "fix: analyze only causal branchpoint forks"
```

---

### Task 7: Rebuild hysteresis as a matched recovery experiment

**Files:**
- Modify: `src/runtime_lab/config/schemas.py`
- Replace: `src/runtime_lab/hysteresis/runner.py`
- Test: `tests/test_hysteresis_protocol.py`

**Interfaces:**
- Produces four primary traces: `clean_exposure`, `perturbed_exposure`, `clean_recovery`, `perturbed_recovery`.
- Both recovery traces have equal requested length, no active intervention, and aligned decision indices.
- Computes drift from paired exposure distances and residual/recovery from paired recovery distances.
- Uses trajectory hidden cosine/L2 and logit JS/KL rather than a one-row SVD that duplicates hidden norm.

- [ ] **Step 1: Write failing matched-protocol tests**

```python
def test_recovery_compares_matched_clean_and_perturbed_continuations(fake_backend, tmp_path):
    summary = run_hysteresis_experiment(hysteresis_config(), runs_dir=tmp_path, prebuilt_backend=fake_backend)
    protocol = summary["protocol_validity"]
    assert protocol["matched_recovery"] is True
    assert protocol["clean_recovery_steps"] == protocol["perturbed_recovery_steps"]
    assert summary["metrics"]["distance_curve_recovery"]


def test_zero_perturbation_reports_no_propagation(fake_backend, tmp_path):
    summary = run_hysteresis_experiment(zero_noise_config(), runs_dir=tmp_path, prebuilt_backend=fake_backend)
    assert summary["metrics"]["perturbation_did_not_propagate"] is True
    assert summary["metrics"]["recovery"] is None
```

- [ ] **Step 2: Verify tests fail for the three-endpoint unmatched protocol**

Run: `.venv/bin/python -m pytest tests/test_hysteresis_protocol.py -q`

Expected: failures for missing matched recovery traces and validity metadata.

- [ ] **Step 3: Implement matched exposure and recovery traces**

Fork clean and perturbed exposure from identical `SeedCache` clones. Continue both resulting states for `recovery_tokens` with intervention disabled. If EOS causes unequal lengths, mark the recovery comparison invalid and do not classify a regime.

- [ ] **Step 4: Compute trajectory-level distances and validity-aware regimes**

Emit aligned per-decision distance curves. Compute recovery against the propagated exposure distance only when the exposure crossed the propagation floor and matched recovery is valid.

- [ ] **Step 5: Write phase-paired artifacts**

Write `events.jsonl`, `config.json`, `summary.json`, and output text for all four traces. Preserve legacy frame filenames as compatibility views, but label them as endpoint summaries rather than the source of the recovery metric.

- [ ] **Step 6: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_hysteresis_protocol.py -q`

Expected: matched-protocol tests pass.

Run: `.venv/bin/python -m pytest -q`

Expected: full suite passes.

- [ ] **Step 7: Commit**

```bash
git add src/runtime_lab/hysteresis src/runtime_lab/config/schemas.py tests/test_hysteresis_protocol.py
git commit -m "fix: measure hysteresis with matched recovery branches"
```

---

### Task 8: Add diagnostic meaning baselines and synthetic validation

**Files:**
- Create: `src/runtime_lab/core/diagnostics/baselines.py`
- Modify: `src/runtime_lab/core/diagnostics/manager.py`
- Create: `scripts/validate_diagnostics.py`
- Test: `tests/test_diagnostic_validation.py`

**Interfaces:**
- Produces simple `hidden_velocity`, `hidden_acceleration`, `logit_entropy`, and `top1_margin` baselines beside Observer's VAR/spectral/SVD signals.
- Produces a deterministic synthetic validation report for constant, linear-drift, oscillatory, and shock trajectories.

- [ ] **Step 1: Write failing baseline and synthetic-ordering tests**

```python
def test_baseline_probe_distinguishes_constant_and_shock_sequences():
    constant = run_baseline_probe([torch.ones(8) for _ in range(8)])
    shock = run_baseline_probe([torch.ones(8) for _ in range(7)] + [torch.ones(8) * 10])
    assert constant[-1]["hidden_velocity"] == 0.0
    assert shock[-1]["hidden_velocity"] > 0.0


def test_synthetic_validation_has_explicit_acceptance_checks():
    report = validate_synthetic_diagnostics()
    assert report["checks"]["constant_is_quiet"]["passed"] is True
    assert report["checks"]["shock_exceeds_constant"]["passed"] is True
    assert report["all_passed"] is True
```

- [ ] **Step 2: Verify tests fail because the baseline module and validator do not exist**

Run: `.venv/bin/python -m pytest tests/test_diagnostic_validation.py -q`

Expected: import failures for the new diagnostic modules.

- [ ] **Step 3: Implement simple baselines and manager integration**

Keep names descriptive rather than causal: VAR output is `local_prediction_error`; retain `divergence` as a compatibility alias. Emit baselines in every canonical diagnostics record.

- [ ] **Step 4: Implement the synthetic validator**

Feed known trajectories through the actual probes, serialize metrics and explicit pass/fail checks, and return a nonzero process exit code when an invariant fails.

- [ ] **Step 5: Run focused, validator, and full tests**

Run: `.venv/bin/python -m pytest tests/test_diagnostic_validation.py -q`

Expected: focused tests pass.

Run: `.venv/bin/python scripts/validate_diagnostics.py --json-out /tmp/observer-diagnostic-validation.json`

Expected: `all_passed=true` and exit code zero.

Run: `.venv/bin/python -m pytest -q`

Expected: full suite passes.

- [ ] **Step 6: Commit**

```bash
git add src/runtime_lab/core/diagnostics scripts/validate_diagnostics.py tests/test_diagnostic_validation.py
git commit -m "feat: ground diagnostics with simple baselines"
```

---

### Task 9: Reset claims and document Observer's foundation

**Files:**
- Create: `docs/OBSERVER_FOUNDATIONS.md`
- Modify: `RESEARCH.md`
- Modify: `RESEARCH_CONTROLLER.md`
- Modify: `REPRODUCIBILITY.md`
- Modify: `README.md`
- Modify: `docs/observer_paper.html`
- Modify: `docs/RESEARCH_WORKFLOW.md`
- Test: `tests/test_claim_status.py`

**Interfaces:**
- Declares F31 reopened.
- Declares historical hysteresis recovery/regime interpretations provisional while preserving observed perturbation effects.
- Defines Observer as Observe → Perturb → Compare → Prove, with Act/controller downstream.

- [ ] **Step 1: Write failing claim-status tests**

```python
def test_research_reopens_f31():
    text = (ROOT / "RESEARCH.md").read_text()
    assert "F31 — REOPENED" in text
    assert "Q1 is closed" not in text


def test_foundation_contract_defines_causal_event_timeline():
    text = (ROOT / "docs/OBSERVER_FOUNDATIONS.md").read_text()
    assert "consumed token" in text
    assert "predicted next token" in text
    assert "matched recovery" in text
```

- [ ] **Step 2: Verify tests fail against current claims and the missing foundation document**

Run: `.venv/bin/python -m pytest tests/test_claim_status.py -q`

Expected: failures for `Q1 is closed` and missing `OBSERVER_FOUNDATIONS.md`.

- [ ] **Step 3: Update the research record without erasing history**

Mark the prior AUROCs as invalid for local flippability because of shifted/cascading labels. Preserve them as an audit lesson. Mark hysteresis perturbation effects as observed while recovery/regime interpretation awaits matched reruns.

- [ ] **Step 4: Document the rebuilt contracts and next experiment**

Make the next research action a small local-branchpoint and matched-hysteresis calibration suite after code verification. Keep controller research paused.

- [ ] **Step 5: Run focused and full tests**

Run: `.venv/bin/python -m pytest tests/test_claim_status.py -q`

Expected: claim-status tests pass.

Run: `.venv/bin/python -m pytest -q`

Expected: full suite passes.

- [ ] **Step 6: Commit**

```bash
git add README.md RESEARCH.md RESEARCH_CONTROLLER.md REPRODUCIBILITY.md docs tests/test_claim_status.py
git commit -m "docs: restore Observer's foundation-first research program"
```

---

### Task 10: Full scientific and release verification

**Files:**
- Modify only if verification exposes a tested defect.

**Interfaces:**
- Verifies installation, tests, lint, package build, local counterfactual invariants, matched hysteresis invariants, artifact hashes, and a real tiny-model smoke.

- [ ] **Step 1: Run all automated tests**

Run: `.venv/bin/python -m pytest -q`

Expected: zero failures.

- [ ] **Step 2: Run unittest compatibility and compilation**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: zero failures.

Run: `.venv/bin/python -m compileall -q src scripts tests`

Expected: exit code zero.

- [ ] **Step 3: Run lint and package build**

Run: `.venv/bin/ruff check src scripts tests`

Expected: zero findings.

Run: `uv build`

Expected: wheel and source distribution build successfully.

- [ ] **Step 4: Run diagnostic acceptance**

Run: `.venv/bin/python scripts/validate_diagnostics.py --json-out /tmp/observer-diagnostic-validation.json`

Expected: all acceptance checks pass.

- [ ] **Step 5: Run real tiny-model smoke protocols**

Run observe, stress with zero magnitude, branchpoints with zero and nonzero magnitude, and matched hysteresis with zero and nonzero noise against `hf-internal-testing/tiny-random-gpt2`. Store outputs outside the repository.

Expected:
- zero stress magnitude has identical logits and tokens;
- zero branchpoint magnitude has zero local argmax flips;
- nonzero branchpoint rows all report matched contexts;
- zero hysteresis magnitude reports no propagation;
- every saved `config_hash` equals the hash of its exact saved config;
- all run directories are unique.

- [ ] **Step 6: Verify repository and commit history**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git status --short`

Expected: no uncommitted files after generated build artifacts are removed.

Run: `git log --oneline --decorate -12`

Expected: the foundation rebuild commits appear on `codex/observer-foundation-rebuild`.

- [ ] **Step 7: Perform plan self-review**

Confirm every global constraint has an implementation, test, or explicit deferred boundary. Confirm no controller policy behavior was changed.
