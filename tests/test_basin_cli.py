import argparse

from runtime_lab.basins.evaluators import (
    build_generic_prompt_spec,
    get_prompt_spec,
)
from runtime_lab.cli.basins import (
    _resolve_prompt_spec,
    _run_one,
    add_basin_args,
)
from tests.fakes import make_fake_backend


def _parse(*values):
    parser = argparse.ArgumentParser()
    add_basin_args(parser)
    return parser.parse_args(list(values))


def test_registered_prompt_uses_its_predeclared_rubric():
    args = _parse("--prompt-key", "water_cycle")

    spec = _resolve_prompt_spec(args)

    assert spec == get_prompt_spec("water_cycle")
    assert spec.prompt_class == "factual"


def test_custom_prompt_uses_explicit_generic_rubric():
    args = _parse(
        "--prompt-key",
        "custom",
        "--prompt",
        "Explain one interesting fact.",
        "--prompt-class",
        "custom",
    )

    spec = _resolve_prompt_spec(args)

    assert spec.key == "custom"
    assert spec.prompt == "Explain one interesting fact."
    assert spec.prompt_class == "custom"
    assert spec.to_record() == build_generic_prompt_spec(
        "Explain one interesting fact."
    ).to_record()


def test_cli_runs_basin_protocol_with_prebuilt_backend(tmp_path):
    args = _parse(
        "--prompt-key",
        "water_cycle",
        "--max-tokens",
        "4",
        "--continuation-tokens",
        "2",
        "--max-episodes",
        "3",
        "--magnitude",
        "0",
        "--runs-dir",
        str(tmp_path),
    )

    summary = _run_one(args, seed=0, backend_result=make_fake_backend())

    assert summary["mode"] == "basins"
    assert summary["protocol_validity"]["clean_reference_replay_exact"] is True
    assert summary["metrics"]["identity_control_exact"] is True
