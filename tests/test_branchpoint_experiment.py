import json
from pathlib import Path

from runtime_lab.branchpoints.experiment import run_branchpoint_experiment
from runtime_lab.config.schemas import BranchpointConfig
from runtime_lab.core.io.hashing import hash_config
from tests.fakes import make_fake_backend


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def _branchpoint_config(
    *,
    intervention_type="scaling",
    magnitude=0.1,
    max_new_tokens=4,
):
    return BranchpointConfig(
        prompt="P",
        model_key="audit-fake",
        max_new_tokens=max_new_tokens,
        seed=0,
        intervention_layer=-1,
        intervention_type=intervention_type,
        intervention_magnitude=magnitude,
        with_diagnostics=True,
    )


def test_each_branchpoint_uses_the_same_clean_context(tmp_path):
    summary = run_branchpoint_experiment(
        _branchpoint_config(),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    rows = _read_jsonl(summary["artifacts"]["events_path"])

    assert all(row["contexts_matched"] for row in rows)
    assert [row["decision_index"] for row in rows] == [1, 2, 3]
    assert all("clean_predicted_token_id" in row for row in rows)
    assert all("perturbed_predicted_token_id" in row for row in rows)
    assert all("clean_diagnostics" in row for row in rows)
    assert [
        row["clean_diagnostics"]["svd"]["window_len"]
        for row in rows
    ] == [1, 2, 3]
    assert all("clean_top1_margin" in row for row in rows)
    assert all("logit_kl" in row and "logit_js" in row for row in rows)
    assert all("intervention_delta_norm" in row for row in rows)


def test_one_flip_does_not_label_later_clean_steps_positive(tmp_path):
    summary = run_branchpoint_experiment(
        _branchpoint_config(),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    rows = _read_jsonl(summary["artifacts"]["events_path"])

    assert [row["argmax_flip"] for row in rows] == [True, False, False]
    assert [row["sampled_flip"] for row in rows] == [True, False, False]


def test_perturbed_forks_never_change_the_clean_continuation(tmp_path):
    perturbed = run_branchpoint_experiment(
        _branchpoint_config(),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    no_effect = run_branchpoint_experiment(
        _branchpoint_config(intervention_type="scaling", magnitude=1.0),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )

    assert perturbed["clean_token_ids"] == no_effect["clean_token_ids"]


def test_zero_additive_intervention_is_exactly_equal(tmp_path):
    summary = run_branchpoint_experiment(
        _branchpoint_config(intervention_type="additive", magnitude=0.0),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    rows = _read_jsonl(summary["artifacts"]["events_path"])

    assert all(row["argmax_flip"] is False for row in rows)
    assert all(row["sampled_flip"] is False for row in rows)
    assert all(row["logit_kl"] == 0.0 for row in rows)
    assert all(row["logit_js"] == 0.0 for row in rows)
    assert all(row["intervention_delta_norm"] == 0.0 for row in rows)


def test_paired_sampling_uses_the_same_rng_draw(tmp_path):
    config = _branchpoint_config(
        intervention_type="additive",
        magnitude=0.0,
    )
    config.temperature = 1.0
    summary = run_branchpoint_experiment(
        config,
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    rows = _read_jsonl(summary["artifacts"]["events_path"])

    assert all(row["sampling_rng_matched"] is True for row in rows)
    assert all(row["sampled_flip"] is False for row in rows)


def test_branchpoint_artifacts_hash_the_final_resolved_config(tmp_path):
    summary = run_branchpoint_experiment(
        _branchpoint_config(),
        runs_dir=tmp_path,
        prebuilt_backend=make_fake_backend(),
    )
    saved = json.loads(Path(summary["artifacts"]["config_path"]).read_text())

    assert saved["config"]["intervention_layer_resolved"] == 2
    assert saved["config_hash"] == hash_config(saved["config"])
    assert summary["config_hash"] == saved["config_hash"]
