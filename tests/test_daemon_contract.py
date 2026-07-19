import sys
from pathlib import Path

import pytest

from scripts import observer_console, observer_daemon
from runtime_lab.core.backend.loader import load_model_with_backend
from tests.fakes import make_fake_backend


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


def test_daemon_and_console_expose_per_layer_propagation():
    cfg = observer_daemon._build_config_propagation(
        {
            "prompt": "p",
            "layer": "late",
            "intervention_type": "scaling",
            "magnitude": 0.5,
            "seed": 0,
        },
        num_layers=6,
    )
    command = observer_console.build_cli_command(
        {
            "mode": "propagation",
            "prompt": "p",
            "layer": 4,
            "intervention_type": "scaling",
            "magnitude": 0.5,
            "seed": 0,
        }
    )

    assert cfg.intervention_layer == 5
    assert cfg.intervention_type == "scaling"
    assert cfg.intervention_magnitude == 0.5
    assert cfg.seed == 0
    assert "propagation" in command
    assert command[command.index("--layer") + 1] == "4"
    assert "--probe-layers" not in command
    assert observer_console.infer_mode("propagation_run_abc") == "propagation"
    assert "propagation" in observer_console._MODES_DOC


def test_console_static_ui_exposes_propagation_mode():
    html = (
        ROOT / "scripts" / "observer_console" / "index.html"
    ).read_text()
    javascript = (
        ROOT / "scripts" / "observer_console" / "app.js"
    ).read_text()
    stylesheet = (
        ROOT / "scripts" / "observer_console" / "styles.css"
    ).read_text()

    assert 'data-mode="propagation"' in html
    assert 'data-filter="propagation"' in html
    assert html.count('data-only-for="stress"') == 2
    assert "state.mode === 'propagation'" in javascript
    assert "$$('[data-only-for]')" in javascript
    assert "[hidden]" in stylesheet
    assert "display: none !important" in stylesheet


def test_daemon_dispatches_propagation_with_the_warm_backend(tmp_path):
    result = observer_daemon._run_request(
        {
            "mode": "propagation",
            "prompt": "p",
            "layer": 0,
            "magnitude": 0.2,
            "max_tokens": 3,
            "runs_dir": str(tmp_path),
        },
        make_fake_backend(),
    )

    assert result["ok"] is True
    assert result["mode"] == "propagation"
    assert result["summary"]["protocol_validity"][
        "all_contexts_matched"
    ] is True
    assert result["summary"]["protocol_validity"][
        "terminal_capture_complete"
    ] is True


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
