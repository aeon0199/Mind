from collections import Counter
from pathlib import Path

import pytest

from runtime_lab.basins.evaluators import (
    PROMPT_SPECS,
    compare_outputs,
    get_prompt_spec,
    score_output,
    validate_prompt_specs,
)


def test_registry_has_two_prompts_in_each_required_class():
    counts = Counter(spec.prompt_class for spec in PROMPT_SPECS)

    assert counts == {
        "factual": 2,
        "procedural": 2,
        "creative": 2,
        "reasoning": 2,
        "code": 2,
    }
    assert len({spec.key for spec in PROMPT_SPECS}) == len(PROMPT_SPECS)


@pytest.mark.parametrize("spec", PROMPT_SPECS, ids=lambda spec: spec.key)
def test_declared_good_fixture_beats_degraded_fixture(spec):
    good = score_output(spec, spec.good_fixture)
    degraded = score_output(spec, spec.degraded_fixture)

    assert good["evaluator_id"] == "deterministic-prompt-rubric-v1"
    assert good["total_score"] >= 0.70
    assert good["total_score"] - degraded["total_score"] >= 0.20
    assert good["rubric_hash"] == degraded["rubric_hash"]
    assert good["components"]


def test_registry_validator_reports_every_prompt():
    report = validate_prompt_specs()

    assert report["all_passed"] is True
    assert report["prompt_count"] == 10
    assert set(report["prompts"]) == {spec.key for spec in PROMPT_SPECS}


def test_pair_comparison_is_branch_order_symmetric():
    spec = get_prompt_spec("water_cycle")
    forward = compare_outputs(
        spec,
        clean_text=spec.degraded_fixture,
        perturbed_text=spec.good_fixture,
        tie_tolerance=0.025,
    )
    reverse = compare_outputs(
        spec,
        clean_text=spec.good_fixture,
        perturbed_text=spec.degraded_fixture,
        tie_tolerance=0.025,
    )

    assert forward["outcome"] == "improve"
    assert reverse["outcome"] == "degrade"
    assert forward["score_delta"] == pytest.approx(-reverse["score_delta"])
    assert forward["clean"]["total_score"] == reverse["perturbed"]["total_score"]
    assert forward["perturbed"]["total_score"] == reverse["clean"]["total_score"]
    assert forward["branch_identity_hidden_from_scorer"] is True


def test_pair_comparison_preserves_ties():
    spec = get_prompt_spec("lighthouse_future")

    result = compare_outputs(
        spec,
        clean_text=spec.good_fixture,
        perturbed_text=spec.good_fixture,
        tie_tolerance=0.025,
    )

    assert result["outcome"] == "tie"
    assert result["score_delta"] == 0.0


def test_python_evaluator_never_executes_generated_code(tmp_path):
    marker = tmp_path / "must-not-exist"
    malicious = (
        "def is_palindrome(text: str) -> bool:\n"
        f"    __import__('pathlib').Path({str(marker)!r}).write_text('bad')\n"
        "    cleaned = ''.join(c.lower() for c in text if c.isalnum())\n"
        "    return cleaned == cleaned[::-1]\n"
    )

    result = score_output(get_prompt_spec("palindrome_code"), malicious)

    assert result["total_score"] > 0.0
    assert not Path(marker).exists()


def test_unknown_prompt_key_fails_closed():
    with pytest.raises(KeyError, match="Unknown basin prompt"):
        get_prompt_spec("made_up")
