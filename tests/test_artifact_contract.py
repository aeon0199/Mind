import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime_lab.core.backend.identity import resolved_model_identity
from runtime_lab.core.io.hashing import hash_config
from runtime_lab.core.io.run_artifacts import create_run_dir, write_config_record


ROOT = Path(__file__).resolve().parents[1]


def _fake_backend(key="loaded-model", hf_id="org/loaded-model"):
    return SimpleNamespace(
        config=SimpleNamespace(key=key, hf_id=hf_id),
        backend="hf",
    )


def test_run_directories_never_collide(tmp_path):
    first_id, first = create_run_dir(tmp_path, "stress")
    second_id, second = create_run_dir(tmp_path, "stress")

    assert first_id != second_id
    assert first != second
    assert first.is_dir()
    assert second.is_dir()


def test_written_hash_matches_the_exact_saved_config(tmp_path):
    config_hash, path = write_config_record(tmp_path, {"resolved_layer": 13})
    saved = json.loads(path.read_text())

    assert saved["config_hash"] == config_hash
    assert hash_config(saved["config"]) == config_hash


def test_prebuilt_backend_rejects_a_different_requested_model():
    with pytest.raises(ValueError, match="requested model"):
        resolved_model_identity("other-model", _fake_backend())


def test_loaded_model_identity_is_authoritative():
    identity = resolved_model_identity(None, _fake_backend())

    assert identity == {
        "model": "loaded-model",
        "model_requested": None,
        "model_hf_id": "org/loaded-model",
    }


def test_all_experiment_runners_use_authoritative_artifact_helpers():
    runner_paths = [
        "src/runtime_lab/observe/runner.py",
        "src/runtime_lab/stress/experiment.py",
        "src/runtime_lab/hysteresis/runner.py",
        "src/runtime_lab/control/adaptive_runner.py",
    ]

    for relative_path in runner_paths:
        source = (ROOT / relative_path).read_text()
        assert "resolved_model_identity(" in source, relative_path
        assert "create_run_dir(" in source, relative_path
        assert "write_config_record(" in source, relative_path
