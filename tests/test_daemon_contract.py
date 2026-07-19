import sys
from pathlib import Path

import pytest

from scripts import observer_console, observer_daemon
from runtime_lab.core.backend.loader import load_model_with_backend


ROOT = Path(__file__).resolve().parents[1]


def test_daemon_preserves_control_threshold_overrides():
    cfg = observer_daemon._build_config_control(
        {
            "prompt": "p",
            "threshold_warn": 0.2,
            "threshold_crit": 0.4,
            "hold_warn": 9,
            "hold_crit": 12,
            "ma_window": 7,
            "scale_warn": 0.8,
            "scale_crit": 0.6,
        }
    )

    assert cfg.threshold_warn == 0.2
    assert cfg.threshold_crit == 0.4
    assert cfg.hold_warn == 9
    assert cfg.hold_crit == 12
    assert cfg.ma_window == 7
    assert cfg.scale_warn == 0.8
    assert cfg.scale_crit == 0.6


def test_daemon_accepts_documented_nested_config():
    merged = observer_daemon._merged_request_config(
        {
            "mode": "control",
            "config": {"prompt": "nested", "threshold_warn": 0.25},
            "threshold_warn": 0.3,
        }
    )

    assert merged["prompt"] == "nested"
    assert merged["threshold_warn"] == 0.3


def test_console_uses_current_python_and_preserves_seed_zero():
    command = observer_console.build_cli_command(
        {"mode": "observe", "prompt": "p", "seed": 0}
    )

    assert command[0] == sys.executable
    assert command[command.index("--seed") + 1] == "0"


def test_daemon_and_console_expose_local_branchpoints():
    cfg = observer_daemon._build_config_branchpoints(
        {
            "prompt": "p",
            "layer": "mid",
            "magnitude": 0.2,
            "seed": 0,
        },
        num_layers=6,
    )
    command = observer_console.build_cli_command(
        {
            "mode": "branchpoints",
            "prompt": "p",
            "layer": 2,
            "magnitude": 0.2,
            "seed": 0,
        }
    )

    assert cfg.intervention_layer == 2
    assert cfg.intervention_magnitude == 0.2
    assert cfg.seed == 0
    assert "branchpoints" in command
    assert command[command.index("--magnitude") + 1] == "0.2"


def test_nnsight_remote_fails_before_loading_with_an_honest_boundary():
    with pytest.raises(RuntimeError, match="trace-based backend"):
        load_model_with_backend(
            model_key="anything",
            registry_path="does-not-need-to-exist.json",
            backend="nnsight",
            nnsight_remote=True,
        )


def test_readme_does_not_advertise_remote_nnsight_support():
    readme = (ROOT / "README.md").read_text()

    assert "NNsight remote execution is intentionally unavailable" in readme
    assert "NNsight backend (remote execution support)" not in readme
