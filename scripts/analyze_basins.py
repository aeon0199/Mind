#!/usr/bin/env python3
"""Analyze M3R causal basin episodes with whole-prompt holdout."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from runtime_lab.basins.replication import (
    M3R2_ANALYSIS_PROTOCOL,
    M3R2_AUROC_THRESHOLD,
    M3R2_CALIBRATION_PROTOCOL,
    M3R2_MINIMUM_VALID_PROMPT_SPLITS,
    M3R2_PREDICTOR_FEATURES,
    M3R2_PREDICTOR_L2_REGULARIZATION,
)
from runtime_lab.core.io.hashing import hash_config


ANALYSIS_PROTOCOL = "observer-m3r-prompt-held-out-analysis-v1"
CALIBRATION_PROTOCOL = "observer-m3r-basin-calibration-v1"
DEFAULT_AUROC_THRESHOLD = 0.70
DEFAULT_MIN_VALID_SPLITS = 3
DEFAULT_L2_REGULARIZATION = 1e-3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _flatten_numeric(
    prefix: str,
    payload: Any,
    output: dict[str, float],
) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric(next_prefix, value, output)
        return
    numeric = _safe_float(payload)
    if numeric is not None:
        output[prefix] = numeric


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.zeros(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index
        while (
            end + 1 < len(values)
            and values[order[end + 1]] == values[order[index]]
        ):
            end += 1
        ranks[order[index : end + 1]] = (
            0.5 * (index + end) + 1.0
        )
        index = end + 1
    return ranks


def _roc_auc_score(
    labels: Sequence[int],
    scores: Sequence[float],
) -> float:
    y = np.asarray(labels, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both positive and negative labels")
    ranks = _average_ranks(score_array)
    positive_rank_sum = float(np.sum(ranks[y == 1]))
    u_statistic = (
        positive_rank_sum - positives * (positives + 1) / 2.0
    )
    return float(u_statistic / (positives * negatives))


def _fit_preprocessing(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    train = np.where(np.isnan(train), medians, train)
    test = np.where(np.isnan(test), medians, test)
    means = np.mean(train, axis=0)
    deviations = np.std(train, axis=0)
    deviations = np.where(deviations < 1e-8, 1.0, deviations)
    return (
        np.clip(np.nan_to_num((train - means) / deviations), -20.0, 20.0),
        np.clip(np.nan_to_num((test - means) / deviations), -20.0, 20.0),
    )


def _fit_logistic_regression(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    *,
    learning_rate: float = 0.02,
    l2_regularization: float = 1e-3,
    iterations: int = 2500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.zeros(train_features.shape[1], dtype=float)
    bias = 0.0
    for _ in range(iterations):
        logits = np.clip(train_features @ weights + bias, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = probabilities - train_labels
        gradient = (
            train_features.T @ error
        ) / len(train_labels) + l2_regularization * weights
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm > 10.0:
            gradient *= 10.0 / gradient_norm
        weights -= learning_rate * gradient
        bias -= learning_rate * float(np.clip(np.mean(error), -1.0, 1.0))
        weights = np.clip(weights, -25.0, 25.0)
        bias = float(np.clip(bias, -25.0, 25.0))
    train_scores = 1.0 / (
        1.0
        + np.exp(
            -np.clip(train_features @ weights + bias, -30.0, 30.0)
        )
    )
    test_scores = 1.0 / (
        1.0
        + np.exp(
            -np.clip(test_features @ weights + bias, -30.0, 30.0)
        )
    )
    return train_scores, test_scores, weights


def _describe(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def _precision_recall(
    labels: Sequence[int],
    scores: Sequence[float],
) -> tuple[float, float]:
    y = np.asarray(labels, dtype=int)
    predicted = np.asarray(scores, dtype=float) >= 0.5
    true_positive = int(np.sum((predicted == 1) & (y == 1)))
    false_positive = int(np.sum((predicted == 1) & (y == 0)))
    false_negative = int(np.sum((predicted == 0) & (y == 1)))
    precision = (
        float(true_positive / (true_positive + false_positive))
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        float(true_positive / (true_positive + false_negative))
        if true_positive + false_negative
        else 0.0
    )
    return precision, recall


def _validate_run(
    cell_id: str,
    cell: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    summary_path = Path(str(cell["summary_path"]))
    run_dir = Path(str(cell.get("run_dir") or summary_path.parent))
    config_path = run_dir / "config.json"
    episodes_path = run_dir / "episodes.jsonl"
    if not config_path.exists() or not summary_path.exists():
        raise ValueError(f"{cell_id}: missing config or summary artifact")
    if not episodes_path.exists():
        raise ValueError(f"{cell_id}: missing episodes artifact")

    record = _load_json(config_path)
    config = record.get("config") or {}
    saved_hash = record.get("config_hash")
    if saved_hash != hash_config(config):
        raise ValueError(f"{cell_id}: config hash does not match saved config")
    if cell.get("config_hash") != saved_hash:
        raise ValueError(f"{cell_id}: manifest config hash does not match")
    summary = _load_json(summary_path)
    if summary.get("config_hash") != saved_hash:
        raise ValueError(f"{cell_id}: summary config hash does not match")
    if config.get("mode") != "basins" or summary.get("mode") != "basins":
        raise ValueError(f"{cell_id}: non-basin artifact in M3R manifest")
    if config.get("protocol") != "causal-branch-continuation-v1":
        raise ValueError(f"{cell_id}: unexpected basin protocol")
    if config.get("prompt_key") != cell.get("prompt_key"):
        raise ValueError(f"{cell_id}: prompt key mismatch")
    if config.get("prompt_class") != cell.get("prompt_class"):
        raise ValueError(f"{cell_id}: prompt class mismatch")
    episodes = _load_jsonl(episodes_path)
    expected_episodes = int(
        (summary.get("metrics") or {}).get("selected_episodes", -1)
    )
    if len(episodes) != expected_episodes:
        raise ValueError(f"{cell_id}: episode count does not match summary")
    for episode in episodes:
        if episode.get("selection_outcome_blind") is not True:
            raise ValueError(f"{cell_id}: outcome-aware selection detected")
        if episode.get("context_replay_verified") is not True:
            raise ValueError(f"{cell_id}: unverified replay context")
        if episode.get("clean_decision_replay_verified") is not True:
            raise ValueError(f"{cell_id}: unverified clean replay decision")
        if not isinstance(episode.get("pre_flip_features"), Mapping):
            raise ValueError(f"{cell_id}: missing pre_flip_features")
    return config, summary, episodes


def _slice_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    score_deltas = [
        float((row["score_comparison"] or {}).get("score_delta", 0.0))
        for row in rows
    ]
    return {
        "episodes": len(rows),
        "sampled_flip_episodes": sum(
            bool((row["local_counterfactual"] or {}).get("sampled_flip"))
            for row in rows
        ),
        "outcomes": {
            outcome: sum(
                (row["score_comparison"] or {}).get("outcome") == outcome
                for row in rows
            )
            for outcome in ("improve", "degrade", "tie")
        },
        "mean_score_delta": (
            float(np.mean(score_deltas)) if score_deltas else None
        ),
    }


def _build_predictor(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    minimum_valid_splits: int,
    feature_allowlist: Sequence[str] | None = None,
    l2_regularization: float = DEFAULT_L2_REGULARIZATION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    predictor_rows: list[dict[str, Any]] = []
    for row in rows:
        if not bool(
            (row["local_counterfactual"] or {}).get("sampled_flip")
        ):
            continue
        outcome = (row["score_comparison"] or {}).get("outcome")
        if outcome not in {"improve", "degrade"}:
            continue
        features: dict[str, float] = {}
        _flatten_numeric("", row["pre_flip_features"], features)
        predictor_rows.append(
            {
                "prompt_key": row["prompt_key"],
                "prompt_class": row["prompt_class"],
                "label": int(outcome == "improve"),
                "features": features,
            }
        )

    prompt_classes = sorted(
        {str(row["prompt_class"]) for row in predictor_rows}
    )
    for row in predictor_rows:
        for prompt_class in prompt_classes:
            row["features"][f"prompt_class.{prompt_class}"] = float(
                row["prompt_class"] == prompt_class
            )
    available_features = {
        feature
        for row in predictor_rows
        for feature in row["features"]
    }
    if feature_allowlist is None:
        feature_names = sorted(
            feature
            for feature in available_features
            if (
                sum(
                    feature in candidate["features"]
                    for candidate in predictor_rows
                )
                / max(1, len(predictor_rows))
            )
            >= 0.5
        )
    else:
        feature_names = [
            feature
            for feature in feature_allowlist
            if feature in available_features
        ]
    if predictor_rows and feature_names:
        matrix = np.asarray(
            [
                [
                    row["features"].get(feature, np.nan)
                    for feature in feature_names
                ]
                for row in predictor_rows
            ],
            dtype=float,
        )
        labels = np.asarray(
            [int(row["label"]) for row in predictor_rows],
            dtype=int,
        )
        row_prompts = np.asarray(
            [str(row["prompt_key"]) for row in predictor_rows],
            dtype=object,
        )
    else:
        matrix = np.empty((0, len(feature_names)), dtype=float)
        labels = np.empty((0,), dtype=int)
        row_prompts = np.empty((0,), dtype=object)

    splits: list[dict[str, Any]] = []
    invalid_splits: list[dict[str, Any]] = []
    weights: list[np.ndarray] = []
    prompts = sorted(set(row_prompts.tolist()))
    for held_prompt in prompts:
        test_mask = row_prompts == held_prompt
        train_mask = ~test_mask
        train_labels = labels[train_mask]
        test_labels = labels[test_mask]
        reason = None
        if len(train_labels) == 0 or len(test_labels) == 0:
            reason = "empty_train_or_test"
        elif len(set(train_labels.tolist())) < 2:
            reason = "train_missing_label_class"
        elif len(set(test_labels.tolist())) < 2:
            reason = "test_missing_label_class"
        if reason is not None:
            invalid_splits.append(
                {
                    "held_out_prompt": held_prompt,
                    "reason": reason,
                    "train_rows": int(np.sum(train_mask)),
                    "test_rows": int(np.sum(test_mask)),
                    "test_improve": int(np.sum(test_labels == 1)),
                    "test_degrade": int(np.sum(test_labels == 0)),
                }
            )
            continue
        train_features, test_features = _fit_preprocessing(
            matrix[train_mask],
            matrix[test_mask],
        )
        train_scores, test_scores, split_weights = (
            _fit_logistic_regression(
                train_features,
                train_labels,
                test_features,
                l2_regularization=float(l2_regularization),
            )
        )
        precision, recall = _precision_recall(test_labels, test_scores)
        splits.append(
            {
                "held_out_prompt": held_prompt,
                "train_prompts": sorted(
                    set(row_prompts[train_mask].tolist())
                ),
                "test_prompts": [held_prompt],
                "train_rows": int(np.sum(train_mask)),
                "test_rows": int(np.sum(test_mask)),
                "train_auroc": _roc_auc_score(
                    train_labels,
                    train_scores,
                ),
                "auroc": _roc_auc_score(test_labels, test_scores),
                "precision": precision,
                "recall": recall,
                "test_improve": int(np.sum(test_labels == 1)),
                "test_degrade": int(np.sum(test_labels == 0)),
            }
        )
        weights.append(split_weights)

    aggregate = {
        "auroc": _describe(split["auroc"] for split in splits),
        "precision": _describe(
            split["precision"] for split in splits
        ),
        "recall": _describe(split["recall"] for split in splits),
    }
    if weights:
        mean_abs_weights = np.mean(np.abs(np.stack(weights)), axis=0)
        feature_importance = [
            {"feature": feature, "mean_abs_weight": float(weight)}
            for feature, weight in sorted(
                zip(feature_names, mean_abs_weights.tolist()),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
    else:
        feature_importance = []

    valid_split_count = len(splits)
    mean_auroc = aggregate["auroc"]["mean"]
    if valid_split_count < int(minimum_valid_splits):
        status = "insufficient_valid_splits"
        met = False
    elif mean_auroc is not None and float(mean_auroc) >= float(threshold):
        status = "threshold_met"
        met = True
    else:
        status = "below_threshold"
        met = False
    predictor = {
        "label": "perturbed rubric score improves vs degrades",
        "eligible_episode_rule": (
            "active selected episode with a paired sampled-token flip and a "
            "non-tie declared rubric outcome"
        ),
        "input_boundary": (
            "pre_flip_features plus declared prompt_class only"
        ),
        "row_count": len(predictor_rows),
        "positive_rows": sum(row["label"] == 1 for row in predictor_rows),
        "negative_rows": sum(row["label"] == 0 for row in predictor_rows),
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "split_unit": "whole prompt",
        "valid_split_count": valid_split_count,
        "invalid_split_count": len(invalid_splits),
        "splits": splits,
        "invalid_splits": invalid_splits,
        "aggregate": aggregate,
    }
    if feature_allowlist is not None:
        predictor["feature_set_source"] = "manifest_preregistered"
        predictor["l2_regularization"] = float(l2_regularization)
    stop_condition = {
        "metric": "mean prompt-held-out AUROC",
        "threshold": float(threshold),
        "minimum_valid_splits": int(minimum_valid_splits),
        "valid_splits": valid_split_count,
        "observed_mean_auroc": mean_auroc,
        "status": status,
        "met": met,
    }
    return predictor, stop_condition


def analyze_basin_root(
    root: str | Path,
    *,
    auroc_threshold: float | None = None,
    minimum_valid_splits: int | None = None,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    manifest_path = root / "basin_manifest.json"
    manifest = _load_json(manifest_path)
    manifest_protocol = manifest.get("protocol")
    if manifest_protocol not in {
        CALIBRATION_PROTOCOL,
        M3R2_CALIBRATION_PROTOCOL,
    }:
        raise ValueError("Unexpected or missing M3R basin manifest protocol")
    design = manifest.get("design") or {}
    is_m3r2 = manifest_protocol == M3R2_CALIBRATION_PROTOCOL
    if is_m3r2:
        expected_design = {
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
        mismatched = [
            key
            for key, expected in expected_design.items()
            if design.get(key) != expected
        ]
        if mismatched:
            raise ValueError(
                "M3R-2 manifest does not match the frozen design: "
                f"{mismatched}"
            )
        feature_allowlist: Sequence[str] | None = tuple(
            design["predictor_feature_set"]
        )
        l2_regularization = float(
            design["predictor_l2_regularization"]
        )
        resolved_threshold = float(
            design["auroc_threshold"]
            if auroc_threshold is None
            else auroc_threshold
        )
        resolved_minimum_splits = int(
            design["minimum_valid_prompt_splits"]
            if minimum_valid_splits is None
            else minimum_valid_splits
        )
        analysis_protocol = M3R2_ANALYSIS_PROTOCOL
    else:
        feature_allowlist = None
        l2_regularization = DEFAULT_L2_REGULARIZATION
        resolved_threshold = float(
            DEFAULT_AUROC_THRESHOLD
            if auroc_threshold is None
            else auroc_threshold
        )
        resolved_minimum_splits = int(
            DEFAULT_MIN_VALID_SPLITS
            if minimum_valid_splits is None
            else minimum_valid_splits
        )
        analysis_protocol = ANALYSIS_PROTOCOL
    cells = manifest.get("cells") or {}
    if not isinstance(cells, Mapping):
        raise ValueError("Basin manifest cells must be an object")
    if manifest.get("status") != "complete":
        raise ValueError("Basin manifest is not complete")
    if int(manifest.get("completed_cells", -1)) != int(
        manifest.get("cell_count", -2)
    ):
        raise ValueError("Basin manifest completed cell count is inconsistent")
    if len(cells) != int(manifest.get("cell_count", -1)):
        raise ValueError("Basin manifest cell map is incomplete")

    all_episodes: list[dict[str, Any]] = []
    active_episodes: list[dict[str, Any]] = []
    control_episodes: list[dict[str, Any]] = []
    validated_run_dirs: set[str] = set()
    config_hashes: set[str] = set()
    total_branchpoints = 0
    total_local_sampled_flips = 0
    control_exact_flags: list[bool] = []
    for cell_id, cell in sorted(cells.items()):
        if cell.get("status") != "ok":
            raise ValueError(f"{cell_id}: cell is not complete")
        config, summary, episodes = _validate_run(cell_id, cell)
        run_dir = str(cell.get("run_dir") or summary.get("run_dir"))
        config_hash = str(summary["config_hash"])
        if run_dir in validated_run_dirs:
            raise ValueError(f"{cell_id}: duplicate run directory")
        if config_hash in config_hashes:
            raise ValueError(f"{cell_id}: duplicate config hash")
        validated_run_dirs.add(run_dir)
        config_hashes.add(config_hash)
        metrics = summary.get("metrics") or {}
        total_branchpoints += int(metrics.get("branchpoint_rows", 0))
        total_local_sampled_flips += int(metrics.get("sampled_flips", 0))
        magnitude = float(config["intervention_magnitude"])
        for episode in episodes:
            row = {
                **episode,
                "cell_id": cell_id,
                "run_id": summary.get("run_id"),
                "seed": config.get("seed"),
                "magnitude": magnitude,
                "prompt_key": config.get("prompt_key"),
                "prompt_class": config.get("prompt_class"),
            }
            all_episodes.append(row)
            if magnitude == 0.0:
                control_episodes.append(row)
            else:
                active_episodes.append(row)
        if magnitude == 0.0:
            exact = metrics.get("identity_control_exact") is True
            control_exact_flags.append(exact)
            if not exact:
                raise ValueError(f"{cell_id}: identity control is not exact")

    selected_sampled_flips = sum(
        bool((row["local_counterfactual"] or {}).get("sampled_flip"))
        for row in active_episodes
    )
    active_no_flips = len(active_episodes) - selected_sampled_flips
    predictor, stop_condition = _build_predictor(
        active_episodes,
        threshold=resolved_threshold,
        minimum_valid_splits=resolved_minimum_splits,
        feature_allowlist=feature_allowlist,
        l2_regularization=l2_regularization,
    )
    prompt_keys = sorted(
        {str(row["prompt_key"]) for row in active_episodes}
    )
    prompt_classes = sorted(
        {str(row["prompt_class"]) for row in active_episodes}
    )
    prompt_slices = {
        key: _slice_summary(
            [row for row in active_episodes if row["prompt_key"] == key]
        )
        for key in prompt_keys
    }
    class_slices = {
        key: _slice_summary(
            [
                row
                for row in active_episodes
                if row["prompt_class"] == key
            ]
        )
        for key in prompt_classes
    }
    outcome_counts = {
        outcome: sum(
            (row["score_comparison"] or {}).get("outcome") == outcome
            for row in active_episodes
        )
        for outcome in ("improve", "degrade", "tie")
    }
    flip_outcome_counts = {
        outcome: sum(
            bool(
                (row["local_counterfactual"] or {}).get("sampled_flip")
            )
            and (row["score_comparison"] or {}).get("outcome") == outcome
            for row in active_episodes
        )
        for outcome in ("improve", "degrade", "tie")
    }
    return {
        "analysis_protocol": analysis_protocol,
        "generated_at": _utc_now(),
        "artifact_root": str(root),
        "manifest_path": str(manifest_path),
        "design": design,
        "artifact_validation": {
            "manifest_status": manifest.get("status"),
            "expected_runs": int(manifest.get("cell_count", 0)),
            "validated_runs": len(validated_run_dirs),
            "unique_run_dirs": len(validated_run_dirs),
            "unique_config_hashes": len(config_hashes),
        },
        "coverage": {
            "total_branchpoint_rows": total_branchpoints,
            "total_local_sampled_flips": total_local_sampled_flips,
            "all_selected_episodes": len(all_episodes),
            "control_selected_episodes": len(control_episodes),
            "active_selected_episodes": len(active_episodes),
            "selected_sampled_flip_episodes": selected_sampled_flips,
            "active_no_flip_episodes": active_no_flips,
            "selection_flip_coverage": (
                float(selected_sampled_flips / total_local_sampled_flips)
                if total_local_sampled_flips
                else None
            ),
            "active_outcomes": outcome_counts,
            "sampled_flip_outcomes": flip_outcome_counts,
            "identity_control_runs": len(control_exact_flags),
            "identity_controls_exact": bool(
                control_exact_flags and all(control_exact_flags)
            ),
        },
        "prompt_slices": prompt_slices,
        "class_slices": class_slices,
        "predictor": predictor,
        "stop_condition": stop_condition,
        "interpretation_boundary": (
            "Outcome means higher or lower score under the predeclared "
            "deterministic prompt rubric. It is not a universal semantic "
            "quality judgment."
        ),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    coverage = report["coverage"]
    predictor = report["predictor"]
    stop = report["stop_condition"]
    verdict = "MET" if stop["met"] else "NOT MET"
    lines = [
        "# Observer M3R Basin Analysis",
        "",
        f"- Artifact root: `{report['artifact_root']}`",
        (
            f"- Validated runs: **{report['artifact_validation']['validated_runs']}**"
        ),
        (
            "- Outcome boundary: deterministic task-specific rubric; not a "
            "universal semantic-quality judgment."
        ),
        "",
        "## Coverage",
        "",
        "| Measure | Value |",
        "|---|---:|",
        (
            f"| Local branchpoint rows | "
            f"{coverage['total_branchpoint_rows']} |"
        ),
        (
            f"| Active selected episodes | "
            f"{coverage['active_selected_episodes']} |"
        ),
        (
            f"| Selected sampled-token flips | "
            f"{coverage['selected_sampled_flip_episodes']} |"
        ),
        (
            f"| No-flip selected episodes | "
            f"{coverage['active_no_flip_episodes']} |"
        ),
        (
            f"| Identity controls exact | "
            f"{coverage['identity_controls_exact']} |"
        ),
        (
            f"| Improve / degrade / tie | "
            f"{coverage['active_outcomes']['improve']} / "
            f"{coverage['active_outcomes']['degrade']} / "
            f"{coverage['active_outcomes']['tie']} |"
        ),
        "",
        "## Prompt-held-out predictor",
        "",
        (
            f"Stop condition: **{verdict}**. Prompt-held-out AUROC threshold "
            f"is {stop['threshold']:.2f}; observed mean is "
            f"**{_fmt(stop['observed_mean_auroc'])}** across "
            f"{stop['valid_splits']} valid splits (minimum "
            f"{stop['minimum_valid_splits']})."
        ),
        "",
        "| Held-out prompt | Rows | AUROC | Precision | Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in predictor["splits"]:
        lines.append(
            f"| {split['held_out_prompt']} | {split['test_rows']} | "
            f"{_fmt(split['auroc'])} | {_fmt(split['precision'])} | "
            f"{_fmt(split['recall'])} |"
        )
    if not predictor["splits"]:
        lines.append("| _(none valid)_ | 0 | n/a | n/a | n/a |")
    lines.extend(
        [
            "",
            "### Prompt slices",
            "",
            "| Prompt | Episodes | Flips | Improve | Degrade | Tie | Mean Δ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, row in report["prompt_slices"].items():
        lines.append(
            f"| {key} | {row['episodes']} | "
            f"{row['sampled_flip_episodes']} | "
            f"{row['outcomes']['improve']} | "
            f"{row['outcomes']['degrade']} | "
            f"{row['outcomes']['tie']} | "
            f"{_fmt(row['mean_score_delta'], 4)} |"
        )
    lines.extend(
        [
            "",
            "### Leading clean pre-flip features",
            "",
        ]
    )
    for row in predictor["feature_importance"][:15]:
        lines.append(
            f"- `{row['feature']}`: mean absolute weight "
            f"{row['mean_abs_weight']:.4f}"
        )
    if not predictor["feature_importance"]:
        lines.append("- No valid feature model could be fit.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(report["interpretation_boundary"]),
            (
                "Ties, no-flips, latent effects, and exact no-effects remain "
                "in coverage; they are not silently converted into binary "
                "predictor labels."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze M3R causal basin calibration artifacts."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--auroc-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--minimum-valid-splits",
        type=int,
        default=None,
    )
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.root).expanduser().resolve()
    report = analyze_basin_root(
        root,
        auroc_threshold=args.auroc_threshold,
        minimum_valid_splits=args.minimum_valid_splits,
    )
    json_path = (
        Path(args.json_out).expanduser().resolve()
        if args.json_out
        else root / "basin_analysis.json"
    )
    markdown_path = (
        Path(args.markdown_out).expanduser().resolve()
        if args.markdown_out
        else root / "BASIN_ANALYSIS.md"
    )
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(
        f"[m3r-analysis] validated="
        f"{report['artifact_validation']['validated_runs']}/"
        f"{report['artifact_validation']['expected_runs']}"
    )
    print(
        f"[m3r-analysis] valid_prompt_splits="
        f"{report['predictor']['valid_split_count']} "
        f"mean_auroc={_fmt(report['stop_condition']['observed_mean_auroc'])} "
        f"stop={report['stop_condition']['status']}"
    )
    print(f"[m3r-analysis] json={json_path}")
    print(f"[m3r-analysis] markdown={markdown_path}")


if __name__ == "__main__":
    main()
