#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from runtime_lab.config.schemas import DiagnosticsConfig
from runtime_lab.core.diagnostics.manager import DiagnosticsManager
from runtime_lab.core.io.json import json_safe


def _known_trajectories(length: int = 16, dimensions: int = 8):
    base = torch.ones(dimensions, dtype=torch.float32)
    direction = torch.linspace(
        0.25,
        1.0,
        dimensions,
        dtype=torch.float32,
    )
    return {
        "constant": [base.clone() for _ in range(length)],
        "linear_drift": [
            base + float(step) * direction for step in range(length)
        ],
        "oscillatory": [
            base + direction * (1.0 if step % 2 == 0 else -1.0)
            for step in range(length)
        ],
        "shock": [
            base.clone() if step < length - 1 else base + 10.0 * direction
            for step in range(length)
        ],
    }


def _synthetic_logits(hidden: torch.Tensor) -> torch.Tensor:
    vector = hidden.detach().float().reshape(-1)
    return torch.stack(
        (
            vector.mean(),
            -vector.mean(),
            vector.std(unbiased=False),
        )
    )


def _run_trajectory(states: list[torch.Tensor]) -> list[dict[str, Any]]:
    manager = DiagnosticsManager(
        DiagnosticsConfig(
            predictor_window=8,
            predictor_proj_dim=8,
            svd_window=16,
            spectral_window=16,
        )
    )
    return [
        manager.step(hidden, logits=_synthetic_logits(hidden))
        for hidden in states
    ]


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "steps": len(records),
        "hidden_velocity": [
            float(record["hidden_velocity"]) for record in records
        ],
        "hidden_acceleration": [
            float(record["hidden_acceleration"]) for record in records
        ],
        "local_prediction_error": [
            float(record["local_prediction_error"]) for record in records
        ],
        "logit_entropy": [
            float(record["logit_entropy"]) for record in records
        ],
        "top1_margin": [
            float(record["top1_margin"]) for record in records
        ],
        "spectral_total_power": [
            float(record["spectral"]["total_power"]) for record in records
        ],
        "spectral_high_fraction": [
            float(record["spectral"]["high_frac"]) for record in records
        ],
        "svd_effective_rank": [
            float(record["svd"]["effective_rank"]) for record in records
        ],
        "final": records[-1],
    }


def _check(
    *,
    passed: bool,
    expectation: str,
    observed: dict[str, Any],
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "expectation": expectation,
        "observed": observed,
    }


def validate_synthetic_diagnostics() -> dict[str, Any]:
    trajectories = {
        name: _summary(_run_trajectory(states))
        for name, states in _known_trajectories().items()
    }
    constant = trajectories["constant"]
    linear = trajectories["linear_drift"]
    oscillatory = trajectories["oscillatory"]
    shock = trajectories["shock"]

    constant_max_velocity = max(constant["hidden_velocity"])
    linear_steady_velocity = linear["hidden_velocity"][2:]
    linear_velocity_spread = max(linear_steady_velocity) - min(
        linear_steady_velocity
    )
    linear_max_steady_acceleration = max(linear["hidden_acceleration"][2:])
    oscillatory_power = oscillatory["spectral_total_power"][-1]
    constant_power = constant["spectral_total_power"][-1]
    shock_velocity = shock["hidden_velocity"][-1]

    checks = {
        "constant_is_quiet": _check(
            passed=constant_max_velocity == 0.0,
            expectation="A constant hidden trajectory has zero velocity.",
            observed={"max_hidden_velocity": constant_max_velocity},
        ),
        "linear_has_steady_velocity": _check(
            passed=(
                linear_steady_velocity[0] > 0.0
                and linear_velocity_spread <= 1e-5
                and linear_max_steady_acceleration <= 1e-5
            ),
            expectation=(
                "A linear trajectory has nonzero steady velocity and near-zero "
                "steady acceleration."
            ),
            observed={
                "steady_velocity": linear_steady_velocity[-1],
                "velocity_spread": linear_velocity_spread,
                "max_steady_acceleration": linear_max_steady_acceleration,
            },
        ),
        "oscillation_has_temporal_power": _check(
            passed=oscillatory_power > constant_power,
            expectation=(
                "An alternating trajectory has more token-time spectral power "
                "than a constant trajectory."
            ),
            observed={
                "oscillatory_total_power": oscillatory_power,
                "constant_total_power": constant_power,
                "spectral_axis": oscillatory["final"]["spectral"]["axis"],
            },
        ),
        "shock_exceeds_constant": _check(
            passed=shock_velocity > constant_max_velocity,
            expectation=(
                "A terminal shock has greater hidden velocity than a constant "
                "trajectory."
            ),
            observed={
                "shock_hidden_velocity": shock_velocity,
                "constant_max_hidden_velocity": constant_max_velocity,
            },
        ),
        "advanced_probes_are_finite": _check(
            passed=all(
                torch.isfinite(
                    torch.tensor(
                        summary["local_prediction_error"]
                        + summary["spectral_total_power"]
                        + summary["svd_effective_rank"],
                        dtype=torch.float64,
                    )
                ).all()
                for summary in trajectories.values()
            ),
            expectation=(
                "The actual VAR, temporal spectral, and windowed SVD probes "
                "remain finite on every known trajectory."
            ),
            observed={"trajectory_count": len(trajectories)},
        ),
    }
    return json_safe(
        {
            "schema_version": 1,
            "trajectories": trajectories,
            "checks": checks,
            "all_passed": all(check["passed"] for check in checks.values()),
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Observer diagnostics against known trajectories."
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = validate_synthetic_diagnostics()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
