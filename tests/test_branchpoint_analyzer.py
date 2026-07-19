import json
import sys

import pytest
from scripts import analyze_branchpoints, analyze_branchpoints_legacy
from runtime_lab.core.io.hashing import hash_config


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_branchpoint_run(root, name, labels):
    run_dir = root / name
    run_dir.mkdir()
    config = {
        "mode": "branchpoints",
        "model": "audit-fake",
        "intervention_type": "scaling",
    }
    _write_json(
        run_dir / "config.json",
        {"config_hash": hash_config(config), "config": config},
    )
    _write_json(run_dir / "summary.json", {"mode": "branchpoints", "run_id": name})
    rows = []
    for decision_index, label in enumerate(labels, start=1):
        margin = 0.1 if label else 2.0
        rows.append(
            {
                "mode": "branchpoints",
                "decision_index": decision_index,
                "contexts_matched": True,
                "argmax_flip": bool(label),
                "sampled_flip": bool(label),
                "clean_top1_margin": margin,
                "clean_hidden_norm": float(decision_index),
                "clean_diagnostics": {
                    "divergence": 1.0 / margin,
                    "svd": {"effective_rank": float(decision_index)},
                    "health": {
                        "valid": True,
                        "degraded": False,
                        "issues": [],
                    },
                },
                "consumed_token_text": " A",
                "perturbed_hidden_norm": 99.0,
                "intervention_delta_norm": 88.0,
                "logit_kl": 77.0,
            }
        )
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return run_dir


def _write_control_pair_fixture(root):
    run_dir = root / "control_run_20260719_000000"
    run_dir.mkdir()
    _write_json(run_dir / "config.json", {"config": {"mode": "control"}})
    _write_json(run_dir / "summary.json", {"mode": "control"})
    (run_dir / "events.jsonl").write_text(
        json.dumps({"token_text": "wrong-source"}) + "\n",
        encoding="utf-8",
    )


def test_analyzer_reads_only_local_counterfactual_rows(tmp_path):
    _write_control_pair_fixture(tmp_path)
    _write_branchpoint_run(
        tmp_path,
        "branchpoint_run_fixture",
        [False, True, False],
    )

    rows = analyze_branchpoints.load_branchpoint_rows(
        tmp_path,
        label="argmax_flip",
    )

    assert len(rows) == 3
    assert {row["run_id"] for row in rows} == {"branchpoint_run_fixture"}


def test_analyzer_features_use_only_clean_pre_intervention_fields(tmp_path):
    _write_branchpoint_run(
        tmp_path,
        "branchpoint_run_fixture",
        [False, True, False],
    )

    rows = analyze_branchpoints.load_branchpoint_rows(tmp_path)
    feature_names = set(rows[0]["features"])

    assert "clean_top1_margin" in feature_names
    assert "clean_diagnostics.divergence" in feature_names
    assert not any("perturbed" in name for name in feature_names)
    assert not any("intervention" in name for name in feature_names)
    assert "logit_kl" not in feature_names


def test_evaluation_splits_by_whole_run(tmp_path):
    for index in range(6):
        _write_branchpoint_run(
            tmp_path,
            f"branchpoint_run_fixture_{index}",
            [False, True, False, True],
        )

    result = analyze_branchpoints.analyze_branchpoint_runs(
        tmp_path,
        repeats=5,
        train_frac=0.5,
        seed=7,
    )

    assert len(result["splits"]) == 5
    assert all(
        not set(split["train_runs"]) & set(split["test_runs"])
        for split in result["splits"]
    )
    assert "auroc" in result["aggregate"]
    assert "precision" in result["aggregate"]
    assert "recall" in result["aggregate"]
    assert result["aggregate"]["auroc"]["min"] <= result["aggregate"]["auroc"]["max"]


def test_legacy_analyzer_requires_explicit_invalid_label_acknowledgement():
    args = analyze_branchpoints_legacy.build_arg_parser().parse_args([])

    assert args.acknowledge_invalid_labels is False


def test_legacy_analyzer_refuses_to_run_without_acknowledgement(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["analyze_branchpoints_legacy.py"])

    with pytest.raises(SystemExit, match="shifted and cascading"):
        analyze_branchpoints_legacy.main()
