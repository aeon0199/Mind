# Observer Repository Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the active Observer measurement system immediately legible by separating experiment runners, offline analysis, operational tools, current research documentation, and historical implementations without changing runtime behavior.

**Architecture:** Preserve the existing `src/runtime_lab` Python package and its public `runtime_lab.cli.main` entrypoint. Move reusable calibration runners into `experiments/`, offline analyzers into `analysis/`, daemon/validation/console support into `tools/`, and historical controller/prototype code into `archive/`. Keep the hosted paper at `docs/observer_paper.html` so the existing GitHub Pages URL remains stable.

**Tech Stack:** Python 3.10+, setuptools, PyTorch, Transformers, pytest, GitHub Actions, static HTML/CSS/JavaScript console.

**Spec:** Repository-organization request from the user, based on the active Observer structure described in the session.

## Global Constraints

- Do not alter experiment semantics, model behavior, metrics, artifact formats, or the `runtime_lab` package name.
- Preserve every historical implementation and research record; move them, do not delete them.
- Keep `docs/observer_paper.html` and `docs/assets/observer_paper_cover.png` at their current hosted paths.
- Update all imports, commands, tests, documentation links, and launch configuration affected by moves.
- Compile and test active code paths; historical archive code is preserved but excluded from the active CI compile target.

---

### Task 1: Save the layout plan

**Files:**
- Create: `docs/superpowers/plans/2026-08-28-observer-repository-layout.md`

- [ ] **Step 1: Record the target boundaries and invariants**

Document the final directory responsibilities, the files being moved, and the compatibility constraints above.

- [ ] **Step 2: Commit the plan**

Create one commit containing only this plan so the implementation has a durable handoff record.

---

### Task 2: Separate active runners, analysis, and tools

**Files:**
- Move: `scripts/run_basin_calibration.py` → `experiments/run_basin_calibration.py`
- Move: `scripts/run_foundation_calibration.py` → `experiments/run_foundation_calibration.py`
- Move: `scripts/run_propagation_calibration.py` → `experiments/run_propagation_calibration.py`
- Move: `scripts/analyze_basins.py` → `analysis/analyze_basins.py`
- Move: `scripts/analyze_branchpoints.py` → `analysis/analyze_branchpoints.py`
- Move: `scripts/analyze_propagation.py` → `analysis/analyze_propagation.py`
- Move: `scripts/observer_daemon.py` → `tools/observer_daemon.py`
- Move: `scripts/validate_diagnostics.py` → `tools/validate_diagnostics.py`
- Move: `scripts/observer_console.py` → `tools/console/server.py`
- Move: `scripts/observer_console/app.js` → `tools/console/static/app.js`
- Move: `scripts/observer_console/index.html` → `tools/console/static/index.html`
- Move: `scripts/observer_console/styles.css` → `tools/console/static/styles.css`
- Create: `experiments/__init__.py`
- Create: `analysis/__init__.py`
- Create: `tools/__init__.py`
- Create: `tools/console/__init__.py`

- [ ] **Step 1: Preserve module contents while moving active files**

Move each active file to its responsibility-based destination. Keep the existing `runtime_lab` imports and command-line interfaces unchanged.

- [ ] **Step 2: Repair console path resolution**

In `tools/console/server.py`, set the repository root to `Path(__file__).resolve().parents[2]` and static root to `Path(__file__).resolve().parent / "static"`. Update its usage example to `python tools/console/server.py --port 8899`.

- [ ] **Step 3: Repair the cross-module analyzer import**

In `analysis/analyze_propagation.py`, import the calibration helpers from `experiments.run_propagation_calibration` with the existing script-execution fallback preserved.

- [ ] **Step 4: Commit the active-tooling move**

Create one commit containing the active runner, analysis, daemon, validation, and console reorganization.

---

### Task 3: Move historical code behind an explicit archive boundary

**Files:**
- Move: `scripts/analyze_branchpoints_legacy.py` → `archive/analysis/analyze_branchpoints_legacy.py`
- Move: `adaptive_controller_system4/` → `archive/controller/adaptive_controller_system4/`
- Move: `baseline_hysteresis_v1/` → `archive/prototypes/baseline_hysteresis_v1/`
- Move: `intervention_engine_v1.5_v2/` → `archive/prototypes/intervention_engine_v1.5_v2/`
- Move: `v1.5/` → `archive/prototypes/v1.5/`
- Move: `RESEARCH_CONTROLLER.md` → `archive/controller/RESEARCH_CONTROLLER.md`
- Create: `archive/__init__.py`
- Create: `archive/analysis/__init__.py`

- [ ] **Step 1: Move historical implementations without editing their contents**

Retain the original filenames and internal structure under `archive/` so old experiments remain inspectable and reproducible.

- [ ] **Step 2: Add archive boundary markers**

Add minimal package markers only where tests need imports; do not turn archived code into active package code or change its semantics.

- [ ] **Step 3: Commit the archive move**

Create one commit containing only historical relocation and package markers.

---

### Task 4: Organize current research documentation

**Files:**
- Move: `RESEARCH.md` → `docs/research/RESEARCH.md`
- Move: `REPRODUCIBILITY.md` → `docs/research/REPRODUCIBILITY.md`
- Move: `docs/OBSERVER_FOUNDATIONS.md` → `docs/foundations/OBSERVER_FOUNDATIONS.md`
- Move: `docs/RESEARCH_WORKFLOW.md` → `docs/workflow/RESEARCH_WORKFLOW.md`
- Move: `docs/FOUNDATION_CALIBRATION_2026-07-19.md` → `docs/results/FOUNDATION_CALIBRATION_2026-07-19.md`
- Move: `docs/Q2_PROPAGATION_CALIBRATION_2026-07-19.md` → `docs/results/Q2_PROPAGATION_CALIBRATION_2026-07-19.md`
- Move: `docs/Q3_BASIN_MAPPING_2026-07-19.md` → `docs/results/Q3_BASIN_MAPPING_2026-07-19.md`
- Move: `docs/Q3_M3R2_REPLICATION_2026-07-19.md` → `docs/results/Q3_M3R2_REPLICATION_2026-07-19.md`

- [ ] **Step 1: Move current documents by purpose**

Group the scientific contract, workflow, research ledger, reproducibility checklist, and calibration reports without modifying their findings.

- [ ] **Step 2: Update relative links inside moved documents**

Make every foundation, workflow, result, analyzer, runner, daemon, and archived-controller reference resolve from its new location.

- [ ] **Step 3: Keep the hosted paper path stable**

Leave `docs/observer_paper.html` and its asset directory in place, while updating its displayed source/reference paths to the organized documentation locations.

- [ ] **Step 4: Commit the documentation move**

Create one commit containing current-document relocation and link corrections.

---

### Task 5: Update repository surfaces and tests

**Files:**
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `.claude/launch.json`
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_repo_hygiene.py`
- Modify: `tests/test_claim_status.py`
- Modify: `tests/test_branchpoint_analyzer.py`
- Modify: `tests/test_basin_analyzer.py`
- Modify: `tests/test_basin_calibration.py`
- Modify: `tests/test_daemon_contract.py`
- Modify: `tests/test_diagnostic_validation.py`
- Modify: `tests/test_foundation_calibration.py`
- Modify: `tests/test_hysteresis_protocol.py`
- Modify: `tests/test_layer_resolution_contract.py`
- Modify: `tests/test_propagation_analyzer.py`
- Modify: `tests/test_propagation_calibration.py`

- [ ] **Step 1: Rewrite the README repository map**

Show `src/runtime_lab/` as the active package, `experiments/`, `analysis/`, `tools/`, `tests/`, organized `docs/`, the stable paper source, and `archive/` as the historical boundary.

- [ ] **Step 2: Update commands and links**

Point quick checks, calibration commands, source links, research references, console launch configuration, and CI compile targets at their new paths.

- [ ] **Step 3: Update test imports and fixture paths**

Replace `scripts.*` imports with `analysis.*`, `experiments.*`, `tools.*`, and `tools.console.server`; point claim/status and console-static assertions at their organized locations.

- [ ] **Step 4: Commit the public-surface update**

Create one commit containing README, contributor guidance, launch/CI configuration, and test path updates.

---

### Task 6: Verify the reorganized repository

**Files:**
- Verify: complete Git tree, imports, documentation links, CI commands, and test suite

- [ ] **Step 1: Verify no stale active paths remain**

Search tracked text for `scripts/`, old root research paths, and old legacy top-level paths; allow only intentional historical prose or URL references.

- [ ] **Step 2: Verify active Python compilation**

Run `python -m compileall -q src analysis experiments tools` and confirm exit code 0.

- [ ] **Step 3: Verify the unit/contract suite**

Run `python -m pytest -q` and record the exact pass/fail result.

- [ ] **Step 4: Verify the final tree**

Confirm active code is outside `archive/`, all historical files are preserved under `archive/`, the paper remains at `docs/observer_paper.html`, and all tests refer to the new locations.

- [ ] **Step 5: Commit any verification-only corrections**

If verification exposes a stale path, fix it in the smallest logical commit and rerun the affected checks.
