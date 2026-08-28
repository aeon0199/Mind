from pathlib import Path

import pytest

from experiments.run_foundation_calibration import DEFAULT_PROMPTS
from experiments.run_propagation_calibration import (
    build_propagation_cells,
    control_magnitude,
    is_control_cell,
    magnitude_slug,
    validate_cell_summary,
)


def _summary(
    *,
    layer: int = 6,
    layers: list[int] | None = None,
    max_delta: float = 0.0,
    injection_delta: float = 0.0,
    terminal_delta: float = 0.0,
    logit_kl: float = 0.0,
    logit_js: float = 0.0,
) -> dict:
    layers = layers or [layer, layer + 1]
    return {
        "capture_layers": layers,
        "terminal_capture": {
            "name": "model.norm",
            "available": True,
        },
        "protocol_validity": {
            "protocol": "same-context-layer-propagation-v1",
            "independent_one_step_forks": True,
            "continued_clean_only": True,
            "all_contexts_matched": True,
            "all_layers_captured": True,
            "terminal_capture_complete": True,
            "rows": 4,
        },
        "metrics": {
            "argmax_flips": 0,
            "sampled_flips": 0,
            "mean_logit_kl": logit_kl,
            "mean_logit_js": logit_js,
            "max_delta_l2": max_delta,
            "layer_summary": {
                str(layer): {
                    "delta_l2": {
                        "mean": injection_delta,
                        "min": injection_delta,
                        "max": injection_delta,
                    }
                }
            },
            "terminal_summary": {
                "final_norm": {
                    "delta_l2": {
                        "mean": terminal_delta,
                        "min": terminal_delta,
                        "max": terminal_delta,
                    }
                }
            },
        },
    }


def test_grid_covers_two_types_four_layers_five_seeds_and_prompts(tmp_path):
    cells = build_propagation_cells(
        root=tmp_path,
        prompts=DEFAULT_PROMPTS,
        seeds=range(5),
        injection_layers=[6, 13, 20, 27],
        magnitudes_by_type={
            "additive": [0.0, 0.1, 0.2, 0.3],
            "scaling": [1.0, 0.5, 1.5],
        },
    )

    assert len(cells) == 280
    assert {cell["intervention_type"] for cell in cells} == {
        "additive",
        "scaling",
    }
    assert {cell["injection_layer"] for cell in cells} == {6, 13, 20, 27}
    assert {cell["seed"] for cell in cells} == set(range(5))
    assert {cell["prompt_key"] for cell in cells} == {
        "sourdough",
        "water_cycle",
    }
    assert len({cell["cell_id"] for cell in cells}) == len(cells)
    assert all(
        Path(cell["runs_dir"]).parts[-3:]
        == (
            cell["intervention_type"],
            magnitude_slug(cell["magnitude"]),
            f"layer_{cell['injection_layer']:02d}",
        )
        for cell in cells
    )


def test_controls_are_type_specific_and_live_in_separate_roots(tmp_path):
    cells = build_propagation_cells(
        root=tmp_path,
        prompts=DEFAULT_PROMPTS[:1],
        seeds=[0],
        injection_layers=[6],
        magnitudes_by_type={
            "additive": [0.0, 1.0],
            "scaling": [0.0, 1.0],
        },
    )
    lookup = {
        (cell["intervention_type"], cell["magnitude"]): cell
        for cell in cells
    }

    assert control_magnitude("additive") == 0.0
    assert control_magnitude("scaling") == 1.0
    assert is_control_cell(lookup[("additive", 0.0)]) is True
    assert is_control_cell(lookup[("additive", 1.0)]) is False
    assert is_control_cell(lookup[("scaling", 1.0)]) is True
    assert is_control_cell(lookup[("scaling", 0.0)]) is False
    assert (
        lookup[("additive", 1.0)]["runs_dir"]
        != lookup[("scaling", 1.0)]["runs_dir"]
    )


@pytest.mark.parametrize(
    ("intervention_type", "magnitude"),
    [("additive", 0.0), ("scaling", 1.0)],
)
def test_zero_effect_controls_must_be_exact(
    intervention_type,
    magnitude,
):
    cell = {
        "intervention_type": intervention_type,
        "magnitude": magnitude,
        "injection_layer": 6,
    }

    validate_cell_summary(cell, _summary())

    with pytest.raises(AssertionError, match="control"):
        validate_cell_summary(
            cell,
            _summary(
                max_delta=1e-4,
                injection_delta=1e-4,
                logit_kl=1e-6,
            ),
        )


def test_noncontrol_requires_a_direct_injection_but_not_a_logit_effect():
    cell = {
        "intervention_type": "scaling",
        "magnitude": 0.5,
        "injection_layer": 27,
    }
    absorbed_by_final_norm = _summary(
        layer=27,
        layers=[27],
        max_delta=1200.0,
        injection_delta=1200.0,
        terminal_delta=0.0,
        logit_kl=0.0,
        logit_js=0.0,
    )

    validate_cell_summary(cell, absorbed_by_final_norm)

    with pytest.raises(AssertionError, match="direct effect"):
        validate_cell_summary(cell, _summary(layer=27, layers=[27]))


def test_validation_rejects_missing_terminal_normalization_capture():
    cell = {
        "intervention_type": "additive",
        "magnitude": 0.2,
        "injection_layer": 6,
    }
    summary = _summary(max_delta=1.0, injection_delta=1.0)
    summary["terminal_capture"]["available"] = False
    summary["protocol_validity"]["terminal_capture_complete"] = False

    with pytest.raises(AssertionError, match="final normalization"):
        validate_cell_summary(cell, summary)
