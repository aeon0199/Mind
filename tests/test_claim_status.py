from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_research_preserves_f31_and_records_scoped_q1_recalibration():
    text = _read("RESEARCH.md")

    assert "F31 — REOPENED" in text
    assert "F32 — RECALIBRATED" in text
    assert "F34 — PER-LAYER PROPAGATION CALIBRATED" in text
    assert "F35 — CAUSAL BASIN MAPPING CALIBRATED" in text
    assert "Q1 status: scoped positive" in text
    assert "Q2 status: scoped mechanistic answer" in text
    assert "Q3 status: measured negative" in text
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
    assert "Causal branch continuation" in text
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
    assert "F35 — Causal basin mapping" in paper


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


def test_q3_report_records_negative_gate_without_erasing_scoped_signal():
    text = _read("docs/Q3_BASIN_MAPPING_2026-07-19.md")

    assert "60/60" in text
    assert "2,820" in text
    assert "31 / 11 / 27" in text
    assert "0.571" in text
    assert "12 improve, 0 degrade, and 6 ties" in text
    assert "Stop condition: **NOT MET**" in text
    assert "Controller remains paused" in text
    assert "not a universal semantic-quality judgment" in text


def test_m3r2_report_locks_the_negative_replication_and_controller_boundary():
    text = _read("docs/Q3_M3R2_REPLICATION_2026-07-19.md")

    assert "90/90" in text
    assert "5,670" in text
    assert "45/45" in text
    assert "217" in text
    assert "195" in text
    assert "22" in text
    assert "67 / 49 / 79" in text
    assert "13 valid" in text
    assert "2 invalid" in text
    assert "0.596" in text
    assert "below threshold" in text
    assert "16 improve, 19 degrade, and 9 ties" in text
    assert "procedural pattern did not replicate" in text
    assert "not a universal semantic-quality judgment" in text
    assert "Controller remains paused" in text


def test_public_claim_surfaces_record_m3r2_without_moving_the_gate():
    research = _read("RESEARCH.md")
    readme = _read("README.md")
    paper = _read("docs/observer_paper.html")

    assert "F36 — M3R-2 PROMPT-BREADTH REPLICATION" in research
    assert "M3R-2 confirmed the negative Q3 gate" in research
    assert "90-run M3R-2 replication" in readme
    assert "docs/Q3_M3R2_REPLICATION_2026-07-19.md" in readme
    assert "F36 — M3R-2 prompt-breadth replication" in paper
    assert "procedural pattern did not replicate" in paper
    assert "0.596" in readme
    assert "0.596" in paper


def test_console_capabilities_describe_active_protocols():
    text = _read("scripts/observer_console.py")

    assert '"branchpoints": {' in text
    assert '"propagation": {' in text
    assert "same-context-layer-propagation" in text
    assert "matched-exposure-recovery-v2" in text
    assert "prompt | noise" not in text
