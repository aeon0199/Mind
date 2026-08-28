import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime_lab.cli._common import (
    backend_num_layers,
    resolve_probe_layers,
    resolve_semantic_layer,
)
from tools import observer_daemon


ROOT = Path(__file__).resolve().parents[1]


def _backend_with_layers(count):
    model = SimpleNamespace(
        transformer=SimpleNamespace(h=[object() for _ in range(count)])
    )
    return SimpleNamespace(model=model)


def test_mid_is_not_resolved_without_model_depth():
    with pytest.raises(ValueError, match="num_layers"):
        resolve_semantic_layer("mid", None)


def test_auto_probes_for_five_layers_are_all_valid():
    assert resolve_probe_layers("auto", 5) == [0, 1, 2, 4]


def test_explicit_negative_layers_become_concrete_indices():
    assert resolve_semantic_layer(-1, 5) == 4
    assert resolve_probe_layers("0,-1", 5) == [0, 4]


def test_backend_depth_is_derived_from_the_loaded_model():
    assert backend_num_layers(_backend_with_layers(5)) == 5


def test_daemon_resolves_semantic_layers_against_its_loaded_backend():
    stress = observer_daemon._build_config_stress(
        {"prompt": "p", "layer": "mid"},
        num_layers=5,
    )
    hysteresis = observer_daemon._build_config_hysteresis(
        {"prompt": "p", "noise_layer": "late"},
        num_layers=5,
    )

    assert stress.intervention_layer == 1
    assert hysteresis.noise_layer == 4


def test_each_cli_loads_once_and_passes_the_backend_to_its_runner():
    for name in ("observe", "stress", "hysteresis", "control"):
        source = (ROOT / "src" / "runtime_lab" / "cli" / f"{name}.py").read_text()
        assert "load_backend_from_args(args)" in source, name
        assert "prebuilt_backend=backend_result" in source, name


def test_tiny_model_registry_fixture_is_self_contained():
    registry = json.loads((ROOT / "tests" / "fixtures" / "models.json").read_text())

    assert registry["default_model"] == "audit-tiny"
    assert registry["models"]["audit-tiny"]["hf_id"] == "hf-internal-testing/tiny-random-gpt2"
