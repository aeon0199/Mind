import json
import argparse
from pathlib import Path

import pytest

from runtime_lab.config.schemas import PropagationConfig
from runtime_lab.cli.propagation import add_propagation_args
from runtime_lab.propagation.experiment import run_propagation_experiment
from tests.fakes import make_fake_backend


def _config(*, magnitude=0.2):
    return PropagationConfig(
        prompt="P",
        model_key="audit-fake",
        max_new_tokens=4,
        seed=0,
        intervention_layer=0,
        intervention_magnitude=magnitude,
        intervention_seed=7,
    )


def _rows(summary):
    return [
        json.loads(line)
        for line in Path(summary["artifacts"]["events_path"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def test_propagation_measures_every_downstream_layer_at_one_shared_context(
    tmp_path,
):
    summary = run_propagation_experiment(
        _config(),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    rows = _rows(summary)

    assert rows
    assert summary["protocol_validity"]["all_contexts_matched"] is True
    assert summary["protocol_validity"]["continued_clean_only"] is True
    assert summary["capture_layers"] == [0, 1, 2]
    assert summary["terminal_capture"]["available"] is True
    assert summary["terminal_capture"]["name"] == "transformer.ln_f"
    assert all(set(row["layer_deltas"]) == {"0", "1", "2"} for row in rows)
    assert all(
        row["terminal_deltas"]["final_norm"]["delta_l2"] > 0.0
        for row in rows
    )
    assert all(row["contexts_matched"] is True for row in rows)
    assert all(row["sampling_rng_matched"] is True for row in rows)


def test_identity_downstream_layers_preserve_the_injected_delta(tmp_path):
    summary = run_propagation_experiment(
        _config(),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )

    for row in _rows(summary):
        injected = row["layer_deltas"]["0"]["delta_l2"]
        assert injected > 0.0
        for layer in ("1", "2"):
            assert row["layer_deltas"][layer]["delta_l2"] == pytest.approx(
                injected,
                rel=1e-5,
            )
            assert row["layer_deltas"][layer][
                "amplification_vs_injection"
            ] == pytest.approx(1.0, rel=1e-5)


def test_zero_propagation_control_is_exact_at_every_layer(tmp_path):
    summary = run_propagation_experiment(
        _config(magnitude=0.0),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )

    assert summary["metrics"]["max_delta_l2"] == 0.0
    assert summary["metrics"]["final_layer_argmax_flips"] == 0
    for row in _rows(summary):
        assert row["logit_kl"] == 0.0
        assert row["logit_js"] == 0.0
        assert all(
            layer["delta_l2"] == 0.0
            for layer in row["layer_deltas"].values()
        )
        assert row["terminal_deltas"]["final_norm"]["delta_l2"] == 0.0


def test_perturbed_forks_never_change_the_clean_generation(tmp_path):
    active = run_propagation_experiment(
        _config(magnitude=0.3),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    zero = run_propagation_experiment(
        _config(magnitude=0.0),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )

    assert active["clean_token_ids"] == zero["clean_token_ids"]


def test_propagation_cli_defaults_to_same_context_additive_mapping():
    parser = argparse.ArgumentParser()
    add_propagation_args(parser)
    args = parser.parse_args([])

    assert args.type == "additive"
    assert args.layer == "mid"
    assert args.magnitude == 0.15
