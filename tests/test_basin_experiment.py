import json
from pathlib import Path

import pytest

from runtime_lab.basins.evaluators import get_prompt_spec
from runtime_lab.basins.experiment import (
    _select_branchpoints,
    run_basin_experiment,
)
from runtime_lab.config.schemas import BasinConfig
from runtime_lab.core.io.hashing import hash_config
from tests.fakes import make_fake_backend


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _config(*, intervention_type="additive", magnitude=0.0, temperature=0.0):
    spec = get_prompt_spec("water_cycle")
    return BasinConfig(
        prompt=spec.prompt,
        prompt_key=spec.key,
        prompt_class=spec.prompt_class,
        evaluator_id=spec.evaluator_id,
        model_key="audit-fake",
        max_new_tokens=4,
        continuation_tokens=2,
        max_episodes=3,
        seed=0,
        temperature=temperature,
        intervention_layer=-1,
        intervention_type=intervention_type,
        intervention_magnitude=magnitude,
        intervention_seed=42,
        tie_tolerance=0.025,
    )


def test_selector_uses_predeclared_band_priority_without_outcomes():
    rows = [
        {
            "decision_index": 1,
            "sampled_flip": False,
            "logit_js": 0.1,
            "intervention_delta_norm": 1.0,
        },
        {
            "decision_index": 2,
            "sampled_flip": True,
            "logit_js": 0.2,
            "intervention_delta_norm": 1.0,
        },
        {
            "decision_index": 4,
            "sampled_flip": False,
            "logit_js": 0.1,
            "intervention_delta_norm": 1.0,
        },
        {
            "decision_index": 5,
            "sampled_flip": True,
            "logit_js": 0.2,
            "intervention_delta_norm": 1.0,
        },
        {
            "decision_index": 7,
            "sampled_flip": False,
            "logit_js": 0.0,
            "intervention_delta_norm": 0.0,
        },
    ]

    selected = _select_branchpoints(
        rows,
        max_new_tokens=8,
        max_episodes=3,
    )

    assert [
        (row["decision_index"], row["selection_basis"])
        for row in selected
    ] == [
        (2, "earliest_sampled_flip"),
        (5, "earliest_sampled_flip"),
        (7, "exact_no_effect_anchor"),
    ]
    assert all("outcome" not in row for row in selected)


def test_zero_intervention_is_exact_through_local_map_and_continuations(tmp_path):
    summary = run_basin_experiment(
        _config(),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    branchpoints = _read_jsonl(summary["artifacts"]["branchpoints_path"])
    episodes = _read_jsonl(summary["artifacts"]["episodes_path"])

    assert len(branchpoints) == 3
    assert len(episodes) == 3
    assert all(row["contexts_matched"] for row in branchpoints)
    assert all(row["argmax_flip"] is False for row in branchpoints)
    assert all(row["sampled_flip"] is False for row in branchpoints)
    assert all(row["logit_kl"] == 0.0 for row in branchpoints)
    assert all(row["logit_js"] == 0.0 for row in branchpoints)

    assert all(row["context_replay_verified"] for row in episodes)
    assert all(row["clean_decision_replay_verified"] for row in episodes)
    assert all(row["exact_token_match"] for row in episodes)
    assert all(row["score_comparison"]["outcome"] == "tie" for row in episodes)
    assert all(row["causal_classification"] == "exact_no_effect" for row in episodes)
    assert summary["protocol_validity"]["clean_reference_replay_exact"] is True
    assert summary["metrics"]["identity_control_exact"] is True


def test_local_flip_continues_both_branches_without_changing_reference(tmp_path):
    active = run_basin_experiment(
        _config(intervention_type="scaling", magnitude=0.1),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    identity = run_basin_experiment(
        _config(intervention_type="scaling", magnitude=1.0),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    episodes = _read_jsonl(active["artifacts"]["episodes_path"])
    flipped = next(row for row in episodes if row["decision_index"] == 1)

    assert flipped["local_counterfactual"]["sampled_flip"] is True
    assert flipped["clean_branch_token_ids"][0] != (
        flipped["perturbed_branch_token_ids"][0]
    )
    assert flipped["causal_classification"] == "local_token_flip"
    assert flipped["continuation_intervention_disabled"] is True
    assert active["clean_reference_token_ids"] == identity["clean_reference_token_ids"]


def test_continuation_sampling_is_paired_and_zero_control_stays_exact(tmp_path):
    summary = run_basin_experiment(
        _config(magnitude=0.0, temperature=1.0),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    trajectory = _read_jsonl(summary["artifacts"]["trajectory_path"])

    assert trajectory
    assert all(row["sampling_seed_paired"] is True for row in trajectory)
    assert all(row["intervention_active_clean"] is False for row in trajectory)
    assert all(row["intervention_active_perturbed"] is False for row in trajectory)
    assert summary["metrics"]["identity_control_exact"] is True


def test_basin_artifacts_hash_the_final_resolved_protocol(tmp_path):
    summary = run_basin_experiment(
        _config(),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    saved = json.loads(Path(summary["artifacts"]["config_path"]).read_text())

    assert saved["config"]["protocol"] == "causal-branch-continuation-v1"
    assert saved["config"]["intervention_layer_resolved"] == 2
    assert saved["config"]["prompt_spec"]["key"] == "water_cycle"
    assert saved["config_hash"] == hash_config(saved["config"])
    assert summary["config_hash"] == saved["config_hash"]


def test_prompt_spec_mismatch_fails_closed(tmp_path):
    config = _config()
    config.prompt = "A different prompt"

    with pytest.raises(ValueError, match="does not match registered prompt"):
        run_basin_experiment(
            config,
            runs_dir=tmp_path,
            prebuilt_backend=make_fake_backend(),
        )
