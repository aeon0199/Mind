import torch

from runtime_lab.config.schemas import DiagnosticsConfig
from runtime_lab.core.diagnostics.baselines import (
    BaselineProbe,
    run_baseline_probe,
)
from runtime_lab.core.diagnostics.manager import DiagnosticsManager
from scripts.validate_diagnostics import validate_synthetic_diagnostics


def test_baseline_probe_distinguishes_constant_and_shock_sequences():
    constant = run_baseline_probe([torch.ones(8) for _ in range(8)])
    shock = run_baseline_probe(
        [torch.ones(8) for _ in range(7)] + [torch.ones(8) * 10]
    )

    assert constant[-1]["hidden_velocity"] == 0.0
    assert constant[-1]["hidden_acceleration"] == 0.0
    assert shock[-1]["hidden_velocity"] > 0.0
    assert shock[-1]["hidden_acceleration"] > 0.0


def test_baseline_probe_reports_logit_entropy_and_top1_margin():
    probe = BaselineProbe()
    metrics = probe.step(
        torch.ones(4),
        logits=torch.tensor([4.0, 1.0, -2.0]),
    )

    assert metrics["logit_entropy"] > 0.0
    assert metrics["top1_margin"] == 3.0


def test_diagnostics_manager_emits_baselines_and_prediction_error_alias():
    manager = DiagnosticsManager(
        DiagnosticsConfig(
            predictor_proj_dim=4,
            spectral_window=8,
        )
    )
    diagnostics = manager.step(
        torch.ones(4),
        logits=torch.tensor([4.0, 1.0, -2.0]),
    )

    assert diagnostics["hidden_velocity"] == 0.0
    assert diagnostics["hidden_acceleration"] == 0.0
    assert diagnostics["logit_entropy"] > 0.0
    assert diagnostics["top1_margin"] == 3.0
    assert diagnostics["local_prediction_error"] == diagnostics["divergence"]


def test_synthetic_validation_has_explicit_acceptance_checks():
    report = validate_synthetic_diagnostics()

    assert report["checks"]["constant_is_quiet"]["passed"] is True
    assert report["checks"]["linear_has_steady_velocity"]["passed"] is True
    assert report["checks"]["oscillation_has_temporal_power"]["passed"] is True
    assert report["checks"]["shock_exceeds_constant"]["passed"] is True
    assert report["all_passed"] is True
