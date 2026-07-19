import json
from pathlib import Path

import pytest

from scripts.analyze_propagation import (
    aggregate_runs,
    load_propagation_runs,
    render_markdown,
)


def _summary(
    *,
    injection_layer=6,
    rows=4,
    layer_values=None,
    terminal=(0.1, 0.2, 0.3, 0.4),
    logit_kl=0.01,
    argmax_flips=1,
):
    layer_values = layer_values or {
        injection_layer: (1.0, 0.1, 0.01, 1.0),
        injection_layer + 1: (2.0, 0.2, 0.02, 2.0),
    }

    def metric_block(values):
        delta, relative, cosine, amplification = values
        return {
            "delta_l2": {"mean": delta, "min": delta, "max": delta},
            "delta_relative_clean": {
                "mean": relative,
                "min": relative,
                "max": relative,
            },
            "cosine_distance": {
                "mean": cosine,
                "min": cosine,
                "max": cosine,
            },
            "amplification_vs_injection": {
                "mean": amplification,
                "min": amplification,
                "max": amplification,
            },
        }

    return {
        "config_hash": "hash",
        "capture_layers": list(layer_values),
        "terminal_capture": {"name": "model.norm", "available": True},
        "protocol_validity": {
            "protocol": "same-context-layer-propagation-v1",
            "independent_one_step_forks": True,
            "continued_clean_only": True,
            "all_contexts_matched": True,
            "all_layers_captured": True,
            "terminal_capture_complete": True,
            "rows": rows,
        },
        "metrics": {
            "argmax_flips": argmax_flips,
            "sampled_flips": argmax_flips,
            "mean_logit_kl": logit_kl,
            "mean_logit_js": logit_kl / 2.0,
            "max_delta_l2": max(
                [values[0] for values in layer_values.values()]
                + [terminal[0]]
            ),
            "layer_summary": {
                str(layer): metric_block(values)
                for layer, values in layer_values.items()
            },
            "terminal_summary": {
                "final_norm": metric_block(terminal),
            },
        },
    }


def _record(
    *,
    summary,
    intervention_type="additive",
    magnitude=0.2,
    seed=0,
    prompt_key="p",
):
    return {
        "intervention_type": intervention_type,
        "magnitude": magnitude,
        "injection_layer": summary["capture_layers"][0],
        "prompt_key": prompt_key,
        "seed": seed,
        "summary": summary,
    }


def test_aggregation_weights_runs_equally_not_decisions():
    first = _summary(
        rows=100,
        layer_values={
            6: (1.0, 0.1, 0.01, 1.0),
            7: (1.0, 0.1, 0.01, 1.0),
        },
        logit_kl=0.1,
        argmax_flips=10,
    )
    second = _summary(
        rows=1,
        layer_values={
            6: (3.0, 0.3, 0.03, 1.0),
            7: (9.0, 0.9, 0.09, 3.0),
        },
        logit_kl=0.3,
        argmax_flips=1,
    )

    report = aggregate_runs(
        [
            _record(summary=first, seed=0),
            _record(summary=second, seed=1),
        ]
    )
    group = report["groups"][0]
    downstream = group["curve"][1]

    assert group["run_count"] == 2
    assert group["decision_count"] == 101
    assert downstream["delta_l2"]["mean"] == pytest.approx(5.0)
    assert downstream["amplification_vs_injection"]["mean"] == pytest.approx(
        2.0
    )
    assert group["output"]["mean_logit_kl"]["mean"] == pytest.approx(0.2)
    assert group["output"]["argmax_flip_rate"] == pytest.approx(11 / 101)
    assert group["prompt_slices"]["p"]["run_count"] == 2
    assert group["prompt_slices"]["p"]["mean_logit_kl"]["mean"] == (
        pytest.approx(0.2)
    )


def test_peak_layer_and_final_norm_absorption_are_derived_from_curve():
    summary = _summary(
        injection_layer=27,
        layer_values={27: (1000.0, 0.5, 0.0, 1.0)},
        terminal=(1e-6, 2e-8, 0.0, 1e-9),
        logit_kl=2e-8,
        argmax_flips=0,
    )

    report = aggregate_runs(
        [
            _record(
                summary=summary,
                intervention_type="scaling",
                magnitude=0.5,
            )
        ]
    )
    group = report["groups"][0]

    assert group["peak_transformer_layer"] == 27
    assert group["peak_transformer_amplification"] == pytest.approx(1.0)
    assert group["final_norm_near_numeric_floor"] is True
    assert group["control"] is False


def test_loader_refuses_an_incomplete_manifest(tmp_path):
    manifest_path = Path(tmp_path) / "propagation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "running",
                "cell_count": 1,
                "completed_cells": 0,
                "design": {},
                "cells": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not complete"):
        load_propagation_runs(tmp_path)


def test_markdown_exposes_run_level_aggregation_and_terminal_norm():
    report = aggregate_runs([_record(summary=_summary())])
    markdown = render_markdown(report, artifact_root="/tmp/q2")

    assert "equal-weight run means" in markdown
    assert "Final norm" in markdown
    assert "same-context-layer-propagation-v1" in markdown
