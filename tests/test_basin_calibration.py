import argparse
import json
from pathlib import Path

import pytest

from runtime_lab.basins.evaluators import PROMPT_SPECS
from scripts.run_basin_calibration import (
    _build_design,
    _find_completed_summary,
    _run_cell,
    build_basin_cells,
    validate_cell_summary,
)
from tests.fakes import make_fake_backend


def _args():
    return argparse.Namespace(
        model="audit-fake",
        registry_path="models.json",
        prompts=list(PROMPT_SPECS),
        seeds=[0, 1, 2],
        magnitudes=[0.0, 0.3],
        max_tokens=4,
        continuation_tokens=2,
        max_episodes=3,
        layer="late",
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        intervention_seed=42,
        tie_tolerance=0.025,
    )


def test_calibration_grid_has_ten_prompts_three_seeds_and_two_configs(tmp_path):
    cells = build_basin_cells(
        root=tmp_path,
        prompts=PROMPT_SPECS,
        seeds=[0, 1, 2],
        magnitudes=[0.0, 0.3],
    )

    assert len(cells) == 60
    assert len({cell["cell_id"] for cell in cells}) == 60
    assert {cell["prompt_class"] for cell in cells} == {
        "factual",
        "procedural",
        "creative",
        "reasoning",
        "code",
    }
    assert {
        (cell["prompt_key"], cell["seed"], cell["magnitude"])
        for cell in cells
    } == {
        (spec.key, seed, magnitude)
        for spec in PROMPT_SPECS
        for seed in (0, 1, 2)
        for magnitude in (0.0, 0.3)
    }


def test_zero_cell_runs_and_passes_fail_closed_validation(tmp_path):
    backend = make_fake_backend()
    design = _build_design(_args(), backend_result=backend)
    cell = build_basin_cells(
        root=tmp_path,
        prompts=[PROMPT_SPECS[0]],
        seeds=[0],
        magnitudes=[0.0],
    )[0]

    summary = _run_cell(cell, design=design, backend_result=backend)
    validate_cell_summary(cell, summary)

    assert summary["metrics"]["identity_control_exact"] is True
    assert summary["protocol_validity"]["selection_outcome_blind"] is True


def test_active_cell_may_have_no_flip_but_must_have_direct_effect(tmp_path):
    backend = make_fake_backend()
    design = _build_design(_args(), backend_result=backend)
    cell = build_basin_cells(
        root=tmp_path,
        prompts=[PROMPT_SPECS[0]],
        seeds=[0],
        magnitudes=[0.3],
    )[0]

    summary = _run_cell(cell, design=design, backend_result=backend)
    summary["metrics"]["sampled_flips"] = 0
    validate_cell_summary(cell, summary)

    assert summary["metrics"]["local_effect_rows"] > 0


def test_zero_cell_rejects_nonexact_control():
    cell = {
        "magnitude": 0.0,
        "prompt_key": "water_cycle",
        "prompt_class": "factual",
    }
    summary = {
        "prompt_key": "water_cycle",
        "prompt_class": "factual",
        "protocol_validity": {
            "protocol": "causal-branch-continuation-v1",
            "pass1_all_contexts_matched": True,
            "selection_outcome_blind": True,
            "replay_all_contexts_verified": True,
            "clean_reference_replay_exact": True,
            "continuation_intervention_disabled": True,
            "continuation_sampling_rng_paired": True,
            "branchpoint_rows": 3,
            "selected_episodes": 3,
        },
        "metrics": {
            "identity_control": True,
            "identity_control_exact": False,
            "argmax_flips": 1,
            "sampled_flips": 1,
            "local_effect_rows": 1,
        },
    }

    with pytest.raises(AssertionError, match="identity control"):
        validate_cell_summary(cell, summary)


def test_completed_cell_is_resumable_only_when_hash_and_design_match(tmp_path):
    backend = make_fake_backend()
    design = _build_design(_args(), backend_result=backend)
    cell = build_basin_cells(
        root=tmp_path,
        prompts=[PROMPT_SPECS[0]],
        seeds=[0],
        magnitudes=[0.0],
    )[0]
    summary = _run_cell(cell, design=design, backend_result=backend)

    completed = _find_completed_summary(cell, design=design)
    assert completed is not None
    loaded, summary_path = completed
    assert loaded["config_hash"] == summary["config_hash"]
    assert summary_path == Path(summary["artifacts"]["summary_path"])

    config_path = Path(summary["artifacts"]["config_path"])
    record = json.loads(config_path.read_text())
    record["config"]["continuation_tokens"] = 99
    config_path.write_text(json.dumps(record))

    assert _find_completed_summary(cell, design=design) is None
