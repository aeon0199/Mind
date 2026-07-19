import json
from pathlib import Path

import pytest

from runtime_lab.basins.evaluators import PROMPT_SPECS
from runtime_lab.basins.replication import (
    M3R2_ANALYSIS_PROTOCOL,
    M3R2_AUROC_THRESHOLD,
    M3R2_CALIBRATION_PROTOCOL,
    M3R2_MINIMUM_VALID_PROMPT_SPLITS,
    M3R2_PREDICTOR_FEATURES,
    M3R2_PREDICTOR_L2_REGULARIZATION,
)
from runtime_lab.core.io.hashing import hash_config
from scripts.analyze_basins import (
    analyze_basin_root,
    render_markdown,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_artifact(root, *, prompt_count=5):
    cells = {}
    selected_specs = PROMPT_SPECS[:prompt_count]
    for spec_index, spec in enumerate(selected_specs):
        for magnitude in (0.0, 0.3):
            run_id = f"basins_run_{spec.key}_{magnitude}"
            run_dir = root / "runs" / run_id
            config = {
                "mode": "basins",
                "protocol": "causal-branch-continuation-v1",
                "selection_protocol": "outcome-blind-early-middle-late-v1",
                "continuation_protocol": "paired-position-rng-no-intervention-v1",
                "prompt": spec.prompt,
                "prompt_key": spec.key,
                "prompt_class": spec.prompt_class,
                "prompt_spec": spec.to_record(),
                "model": "qwen3-1.7b",
                "seed": spec_index,
                "intervention_type": "additive",
                "intervention_magnitude": magnitude,
            }
            config_hash = hash_config(config)
            _write_json(
                run_dir / "config.json",
                {"config_hash": config_hash, "config": config},
            )
            if magnitude == 0.0:
                episodes = [
                    {
                        "episode_id": "control",
                        "decision_index": 1,
                        "position_band": "early",
                        "selection_outcome_blind": True,
                        "context_replay_verified": True,
                        "clean_decision_replay_verified": True,
                        "pre_flip_features": {
                            "clean_top1_margin": 0.5,
                            "clean_diagnostics": {"hidden_velocity": 0.2},
                        },
                        "local_counterfactual": {
                            "sampled_flip": False,
                            "argmax_flip": False,
                        },
                        "exact_token_match": True,
                        "causal_classification": "exact_no_effect",
                        "score_comparison": {
                            "outcome": "tie",
                            "score_delta": 0.0,
                        },
                    }
                ]
                metrics = {
                    "branchpoint_rows": 4,
                    "local_effect_rows": 0,
                    "argmax_flips": 0,
                    "sampled_flips": 0,
                    "selected_episodes": 1,
                    "selected_sampled_flips": 0,
                    "identity_control": True,
                    "identity_control_exact": True,
                    "outcomes": {"improve": 0, "degrade": 0, "tie": 1},
                }
            else:
                episodes = []
                for row_index, margin in enumerate((0.1, 0.2, 0.8, 0.9)):
                    outcome = "improve" if margin > 0.5 else "degrade"
                    episodes.append(
                        {
                            "episode_id": f"active_{row_index}",
                            "decision_index": row_index + 1,
                            "position_band": (
                                "early" if row_index < 2 else "late"
                            ),
                            "selection_outcome_blind": True,
                            "context_replay_verified": True,
                            "clean_decision_replay_verified": True,
                            "pre_flip_features": {
                                "clean_top1_margin": margin,
                                "normalized_position": row_index / 4,
                                "unregistered_signal": (
                                    999999.0
                                    if outcome == "improve"
                                    else -999999.0
                                ),
                                "clean_diagnostics": {
                                    "hidden_velocity": margin * 2,
                                },
                            },
                            "local_counterfactual": {
                                "sampled_flip": True,
                                "argmax_flip": True,
                                "logit_kl": 9999 - margin,
                            },
                            "exact_token_match": False,
                            "causal_classification": "local_token_flip",
                            "score_comparison": {
                                "outcome": outcome,
                                "score_delta": (
                                    1000 + margin
                                    if outcome == "improve"
                                    else -1000 - margin
                                ),
                                "perturbed": {"total_score": 9999},
                            },
                        }
                    )
                episodes.append(
                    {
                        "episode_id": "preserved_no_flip",
                        "decision_index": 7,
                        "position_band": "late",
                        "selection_outcome_blind": True,
                        "context_replay_verified": True,
                        "clean_decision_replay_verified": True,
                        "pre_flip_features": {"clean_top1_margin": 0.4},
                        "local_counterfactual": {
                            "sampled_flip": False,
                            "argmax_flip": False,
                        },
                        "exact_token_match": True,
                        "causal_classification": "latent_state_effect_only",
                        "score_comparison": {
                            "outcome": "tie",
                            "score_delta": 0.0,
                        },
                    }
                )
                metrics = {
                    "branchpoint_rows": 8,
                    "local_effect_rows": 8,
                    "argmax_flips": 4,
                    "sampled_flips": 4,
                    "selected_episodes": 5,
                    "selected_sampled_flips": 4,
                    "identity_control": False,
                    "identity_control_exact": None,
                    "outcomes": {"improve": 2, "degrade": 2, "tie": 1},
                }
            _write_jsonl(run_dir / "episodes.jsonl", episodes)
            _write_jsonl(run_dir / "branchpoints.jsonl", [])
            _write_jsonl(run_dir / "trajectories.jsonl", [])
            summary = {
                "config_hash": config_hash,
                "mode": "basins",
                "run_id": run_id,
                "run_dir": str(run_dir),
                "prompt_key": spec.key,
                "prompt_class": spec.prompt_class,
                "protocol_validity": {
                    "protocol": "causal-branch-continuation-v1",
                    "pass1_all_contexts_matched": True,
                    "selection_outcome_blind": True,
                    "replay_all_contexts_verified": True,
                    "clean_reference_replay_exact": True,
                    "continuation_intervention_disabled": True,
                    "continuation_sampling_rng_paired": True,
                    "branchpoint_rows": metrics["branchpoint_rows"],
                    "selected_episodes": metrics["selected_episodes"],
                },
                "metrics": metrics,
                "artifacts": {
                    "run_dir": str(run_dir),
                    "config_path": str(run_dir / "config.json"),
                    "episodes_path": str(run_dir / "episodes.jsonl"),
                    "summary_path": str(run_dir / "summary.json"),
                },
            }
            _write_json(run_dir / "summary.json", summary)
            cell_id = f"{spec.key}:{magnitude}"
            cells[cell_id] = {
                "cell_id": cell_id,
                "status": "ok",
                "magnitude": magnitude,
                "prompt_key": spec.key,
                "prompt_class": spec.prompt_class,
                "prompt": spec.prompt,
                "seed": spec_index,
                "run_dir": str(run_dir),
                "summary_path": str(run_dir / "summary.json"),
                "config_hash": config_hash,
                **metrics,
            }
    manifest = {
        "protocol": "observer-m3r-basin-calibration-v1",
        "status": "complete",
        "root": str(root),
        "cell_count": len(cells),
        "completed_cells": len(cells),
        "design": {
            "model": "qwen3-1.7b",
            "prompt_keys": [spec.key for spec in selected_specs],
            "prompt_classes": sorted(
                {spec.prompt_class for spec in selected_specs}
            ),
            "seeds": list(range(prompt_count)),
            "magnitudes": [0.0, 0.3],
        },
        "cells": cells,
    }
    _write_json(root / "basin_manifest.json", manifest)
    return manifest


def test_analyzer_uses_only_pre_flip_features_and_holds_out_whole_prompts(tmp_path):
    _make_artifact(tmp_path, prompt_count=5)

    report = analyze_basin_root(tmp_path)

    assert report["artifact_validation"]["validated_runs"] == 10
    assert report["coverage"]["active_selected_episodes"] == 25
    assert report["coverage"]["selected_sampled_flip_episodes"] == 20
    assert report["coverage"]["active_no_flip_episodes"] == 5
    assert report["coverage"]["identity_controls_exact"] is True
    assert report["predictor"]["valid_split_count"] == 5
    assert report["predictor"]["aggregate"]["auroc"]["mean"] == 1.0
    assert report["stop_condition"]["met"] is True
    assert all(
        len(split["test_prompts"]) == 1
        and split["test_prompts"][0] not in split["train_prompts"]
        for split in report["predictor"]["splits"]
    )
    assert "clean_top1_margin" in report["predictor"]["feature_names"]
    assert not any(
        "score" in name
        or "perturbed" in name
        or "logit_kl" in name
        for name in report["predictor"]["feature_names"]
    )


def test_insufficient_prompt_splits_are_reported_not_promoted(tmp_path):
    _make_artifact(tmp_path, prompt_count=1)

    report = analyze_basin_root(tmp_path)

    assert report["predictor"]["valid_split_count"] == 0
    assert report["stop_condition"]["met"] is False
    assert report["stop_condition"]["status"] == "insufficient_valid_splits"


def test_m3r2_analyzer_enforces_manifest_preregistered_predictor(tmp_path):
    manifest = _make_artifact(tmp_path, prompt_count=5)
    manifest["protocol"] = M3R2_CALIBRATION_PROTOCOL
    manifest["design"].update(
        {
            "protocol": M3R2_CALIBRATION_PROTOCOL,
            "prompt_set": "m3r2",
            "predictor_feature_set": list(M3R2_PREDICTOR_FEATURES),
            "predictor_l2_regularization": (
                M3R2_PREDICTOR_L2_REGULARIZATION
            ),
            "minimum_valid_prompt_splits": (
                M3R2_MINIMUM_VALID_PROMPT_SPLITS
            ),
            "auroc_threshold": M3R2_AUROC_THRESHOLD,
        }
    )
    _write_json(tmp_path / "basin_manifest.json", manifest)

    report = analyze_basin_root(tmp_path)

    assert report["analysis_protocol"] == M3R2_ANALYSIS_PROTOCOL
    assert report["predictor"]["feature_set_source"] == "manifest_preregistered"
    assert report["predictor"]["l2_regularization"] == (
        M3R2_PREDICTOR_L2_REGULARIZATION
    )
    assert set(report["predictor"]["feature_names"]) <= set(
        M3R2_PREDICTOR_FEATURES
    )
    assert "unregistered_signal" not in report["predictor"]["feature_names"]
    assert report["stop_condition"]["minimum_valid_splits"] == (
        M3R2_MINIMUM_VALID_PROMPT_SPLITS
    )
    assert report["predictor"]["valid_split_count"] == 5
    assert report["stop_condition"]["met"] is True


def test_analyzer_rejects_tampered_config_hash(tmp_path):
    manifest = _make_artifact(tmp_path, prompt_count=2)
    first = next(iter(manifest["cells"].values()))
    config_path = Path(first["run_dir"]) / "config.json"
    record = json.loads(config_path.read_text())
    record["config"]["seed"] = 999
    config_path.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="config hash"):
        analyze_basin_root(tmp_path)


def test_markdown_preserves_coverage_and_stop_verdict(tmp_path):
    _make_artifact(tmp_path, prompt_count=5)
    report = analyze_basin_root(tmp_path)

    markdown = render_markdown(report)

    assert "# Observer M3R Basin Analysis" in markdown
    assert "Prompt-held-out AUROC" in markdown
    assert "No-flip selected episodes" in markdown
    assert "Stop condition: **MET**" in markdown
