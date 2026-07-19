import torch

from runtime_lab.core.advisory import advise_stress
from runtime_lab.core.diagnostics.manager import summarize_diagnostics_health
from runtime_lab.core.interventions.factory import build_intervention
from runtime_lab.core.sampling import nucleus_keep_mask
from runtime_lab.stress.experiment import resolve_intervention_seed


def test_empty_diagnostics_are_not_healthy():
    assert summarize_diagnostics_health([])["ok"] is False


def test_all_degraded_diagnostics_are_not_healthy():
    records = [{"health": {"valid": True, "degraded": True, "issues": []}}]

    assert summarize_diagnostics_health(records)["ok"] is False


def test_top_p_keeps_the_first_token_crossing_the_cutoff():
    kept = nucleus_keep_mask(torch.tensor([0.4, 0.3, 0.2, 0.1]), top_p=0.5)

    assert kept.tolist() == [True, True, False, False]


def test_seed_zero_is_not_replaced_by_forty_two():
    assert resolve_intervention_seed(0) == 0
    assert resolve_intervention_seed(None) == 42


def test_first_token_divergence_zero_means_immediate_flip():
    advisory = advise_stress(
        {
            "metrics": {
                "regime": "PLASTIC",
                "token_match_rate": 0.0,
                "first_token_divergence": 0,
            },
            "config": {},
        }
    )

    assert any("differed immediately" in text for text in advisory.observations)


def test_fractional_projection_default_never_becomes_zero_dimensions():
    intervention = build_intervention("projection", magnitude=0.15)

    assert intervention.subspace_dim == 1
