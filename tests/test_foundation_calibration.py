from pathlib import Path

import pytest

from scripts.run_foundation_calibration import (
    DEFAULT_PROMPTS,
    build_calibration_cells,
    magnitude_slug,
    validate_cell_summary,
)


def test_calibration_grid_covers_prompts_seeds_and_separate_magnitudes(tmp_path):
    cells = build_calibration_cells(
        root=tmp_path,
        prompts=DEFAULT_PROMPTS,
        seeds=[0, 1, 2],
        branchpoint_magnitudes=[0.0, 0.15, 0.3],
        hysteresis_magnitudes=[0.0, 0.1, 0.2, 0.3],
    )

    branchpoints = [cell for cell in cells if cell["mode"] == "branchpoints"]
    hysteresis = [cell for cell in cells if cell["mode"] == "hysteresis"]

    assert len(branchpoints) == 18
    assert len(hysteresis) == 24
    assert {cell["prompt_key"] for cell in cells} == {"sourdough", "water_cycle"}
    assert {cell["seed"] for cell in cells} == {0, 1, 2}
    assert len({cell["cell_id"] for cell in cells}) == len(cells)
    assert all(
        Path(cell["runs_dir"]).name == magnitude_slug(cell["magnitude"])
        for cell in cells
    )


def test_zero_controls_are_not_mixed_with_nonzero_predictor_artifacts(tmp_path):
    cells = build_calibration_cells(
        root=tmp_path,
        prompts=DEFAULT_PROMPTS,
        seeds=[0],
        branchpoint_magnitudes=[0.0, 0.3],
        hysteresis_magnitudes=[0.0, 0.3],
    )

    branch_roots = {
        cell["magnitude"]: Path(cell["runs_dir"])
        for cell in cells
        if cell["mode"] == "branchpoints"
    }

    assert branch_roots[0.0] != branch_roots[0.3]
    assert branch_roots[0.0].name == "magnitude_0p000"
    assert branch_roots[0.3].name == "magnitude_0p300"


def test_zero_branchpoint_control_must_be_an_exact_no_op():
    cell = {"mode": "branchpoints", "magnitude": 0.0}
    summary = {
        "protocol_validity": {
            "all_contexts_matched": True,
            "rows": 1,
        },
        "metrics": {
            "argmax_flips": 1,
            "sampled_flips": 0,
            "mean_logit_kl": 0.0,
            "mean_logit_js": 0.0,
        },
    }

    with pytest.raises(AssertionError, match="zero branchpoint"):
        validate_cell_summary(cell, summary)


def test_zero_hysteresis_control_requires_matched_no_propagation():
    cell = {"mode": "hysteresis", "magnitude": 0.0}
    summary = {
        "protocol_validity": {
            "matched_exposure": True,
            "matched_recovery": True,
        },
        "metrics": {
            "perturbation_did_not_propagate": True,
            "direct_effect_peak": 0.0,
            "propagated_distance": 0.0,
            "residual_distance": 0.0,
        },
    }

    validate_cell_summary(cell, summary)
