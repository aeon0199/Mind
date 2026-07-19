from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_research_preserves_f31_and_records_scoped_q1_recalibration():
    text = _read("RESEARCH.md")

    assert "F31 — REOPENED" in text
    assert "F32 — RECALIBRATED" in text
    assert "F34 — PER-LAYER PROPAGATION CALIBRATED" in text
    assert "Q1 status: scoped positive" in text
    assert "Q2 status: scoped mechanistic answer" in text
    assert "q1 is closed" not in text.lower()
    assert "historical controller-paired AUROCs" in text


def test_foundation_contract_defines_causal_event_timeline():
    text = _read("docs/OBSERVER_FOUNDATIONS.md")

    assert "Observe → Perturb → Compare → Prove" in text
    assert "consumed token" in text
    assert "predicted next token" in text
    assert "matched recovery" in text
    assert "initial intervention distance" in text
    assert "active-window peak" in text
    assert "Same-context per-layer propagation" in text
    assert "Act" in text


def test_readme_presents_foundation_before_controller():
    text = _read("README.md")

    assert "Observe → Perturb → Compare → Prove" in text
    assert "local branchpoint" in text
    assert "first target-model calibration is complete" in text
    assert "BASE → PERTURB → REASK" not in text
    assert text.index("Observe → Perturb → Compare → Prove") < text.index(
        "controller"
    )


def test_archived_claims_are_preserved_with_corrections():
    controller = _read("RESEARCH_CONTROLLER.md")
    reproducibility = _read("REPRODUCIBILITY.md")
    paper = _read("docs/observer_paper.html")

    assert "FOUNDATION REBUILD CORRECTION" in controller
    assert "matched-exposure-recovery-v2" in reproducibility
    assert "Foundation rebuild status" in paper
    assert "F32 — RECALIBRATED" in paper
    assert "F34 — Per-layer propagation" in paper


def test_repository_calibration_report_states_scope_and_controller_boundary():
    text = _read("docs/FOUNDATION_CALIBRATION_2026-07-19.md")

    assert "42" in text
    assert "0.831" in text
    assert "0.886" in text
    assert "Controller remains paused" in text


def test_q2_report_records_validated_scope_and_normalization_result():
    text = _read("docs/Q2_PROPAGATION_CALIBRATION_2026-07-19.md")

    assert "280 runs" in text
    assert "4,200 paired decisions" in text
    assert "Layer 13" in text
    assert "7.54e-8" in text
    assert "Controller remains paused" in text


def test_console_capabilities_describe_active_protocols():
    text = _read("scripts/observer_console.py")

    assert '"branchpoints": {' in text
    assert '"propagation": {' in text
    assert "same-context-layer-propagation" in text
    assert "matched-exposure-recovery-v2" in text
    assert "prompt | noise" not in text
