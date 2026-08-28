import argparse
import json
from pathlib import Path

import pytest
import torch

from runtime_lab.config.schemas import HysteresisConfig
from runtime_lab.core.io.hashing import hash_config
from runtime_lab.hysteresis.runner import _hidden_distances, run_hysteresis_experiment
from runtime_lab.cli.hysteresis import add_hysteresis_args
from tools.console import server as observer_console\nfrom tools import observer_daemon
from tests.fakes import make_fake_backend


def _config(*, magnitude=0.5):
    return HysteresisConfig(
        prompt="P",
        model_key="audit-fake",
        max_new_tokens=4,
        seed=0,
        perturbation_mode="noise",
        noise_layer=-1,
        noise_magnitude=magnitude,
        noise_start=1,
        noise_duration=1,
        noise_seed=7,
        recovery_tokens=3,
    )


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def test_recovery_compares_matched_clean_and_perturbed_continuations(tmp_path):
    summary = run_hysteresis_experiment(
        _config(),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    protocol = summary["protocol_validity"]

    assert protocol["matched_exposure"] is True
    assert protocol["matched_recovery"] is True
    assert protocol["clean_recovery_steps"] == 3
    assert protocol["perturbed_recovery_steps"] == 3
    assert summary["metrics"]["distance_curve_exposure"]
    assert len(summary["metrics"]["distance_curve_recovery"]) == 3


def test_recovery_has_no_active_intervention_on_either_branch(tmp_path):
    summary = run_hysteresis_experiment(
        _config(),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    rows = _read_jsonl(summary["artifacts"]["events_path"])
    recovery_rows = [row for row in rows if row["phase"] == "recovery"]

    assert {row["branch"] for row in recovery_rows} == {"clean", "perturbed"}
    assert all(row["intervention_active"] is False for row in recovery_rows)


def test_zero_perturbation_reports_no_propagation(tmp_path):
    summary = run_hysteresis_experiment(
        _config(magnitude=0.0),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )

    assert summary["metrics"]["perturbation_did_not_propagate"] is True
    assert summary["metrics"]["recovery"] is None
    assert all(
        point["distance"] == 0.0
        for point in summary["metrics"]["distance_curve_exposure"]
    )
    assert all(
        point["distance"] == 0.0
        for point in summary["metrics"]["distance_curve_recovery"]
    )
    assert "no-propagation" in summary["advisory"]["flags"]
    assert "REASK" not in json.dumps(summary["advisory"])


def test_hysteresis_separates_initial_effect_from_active_window_peak(tmp_path):
    config = _config(magnitude=0.5)
    config.noise_duration = 2
    summary = run_hysteresis_experiment(
        config,
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    active_points = [
        point
        for point in summary["metrics"]["distance_curve_exposure"]
        if point["intervention_active"]
    ]

    assert summary["metrics"]["initial_intervention_distance"] == (
        active_points[0]["distance"]
    )
    assert summary["metrics"]["active_window_peak_distance"] == max(
        point["distance"] for point in active_points
    )
    assert summary["metrics"]["direct_effect_peak"] == (
        summary["metrics"]["active_window_peak_distance"]
    )


def test_identical_hidden_vectors_have_exact_zero_distance():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1)
    hidden = torch.randn(32, generator=generator)

    assert _hidden_distances(hidden, hidden.clone()) == (0.0, 0.0)


def test_hysteresis_artifacts_use_final_config_and_four_primary_traces(tmp_path):
    summary = run_hysteresis_experiment(
        _config(),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    saved = json.loads(Path(summary["artifacts"]["config_path"]).read_text())

    assert saved["config_hash"] == hash_config(saved["config"])
    assert saved["config_hash"] == summary["config_hash"]
    assert set(summary["traces"]) == {
        "clean_exposure",
        "perturbed_exposure",
        "clean_recovery",
        "perturbed_recovery",
    }
    for key in (
        "output_clean_exposure",
        "output_perturbed_exposure",
        "output_clean_recovery",
        "output_perturbed_recovery",
    ):
        assert Path(summary["artifacts"][key]).exists()


def test_perturbation_never_changes_the_clean_branch(tmp_path):
    perturbed = run_hysteresis_experiment(
        _config(magnitude=0.5),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    zero = run_hysteresis_experiment(
        _config(magnitude=0.0),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )

    assert perturbed["clean_token_ids"] == zero["clean_token_ids"]


def test_all_entrypoints_default_to_the_matched_noise_protocol():
    parser = argparse.ArgumentParser()
    add_hysteresis_args(parser)
    args = parser.parse_args([])
    daemon_config = observer_daemon._build_config_hysteresis(
        {"prompt": "P"},
        num_layers=3,
    )
    console_command = observer_console.build_cli_command(
        {"mode": "hysteresis", "prompt": "P"}
    )

    assert args.perturbation_mode == "noise"
    assert args.recovery_tokens == 32
    assert daemon_config.perturbation_mode == "noise"
    assert daemon_config.recovery_tokens == 32
    mode_index = console_command.index("--perturbation-mode")
    assert console_command[mode_index + 1] == "noise"


def test_prompt_contamination_is_rejected_as_internal_hysteresis(tmp_path):
    config = _config()
    config.perturbation_mode = "prompt"

    with pytest.raises(ValueError, match="prompt contamination"):
        run_hysteresis_experiment(
            config,
            runs_dir=tmp_path,
            prebuilt_backend=make_fake_backend(),
        )
