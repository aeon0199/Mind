#!/usr/bin/env python3
"""Analyze only Observer's independent local counterfactual branchpoint rows."""

from __future__ import annotations

import argparse
import json
import math
import random
import string
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from runtime_lab.core.io.hashing import hash_config


DEFAULT_REPEATS = 20
DEFAULT_MIN_RUNS = 3


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def _flatten_clean_numeric(
    prefix: str,
    payload: Any,
    output: Dict[str, float],
) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_clean_numeric(next_prefix, value, output)
        return
    numeric = _safe_float(payload)
    if numeric is not None:
        output[prefix] = numeric


def _token_features(token_text: str) -> Dict[str, float]:
    stripped = token_text.strip()
    return {
        "clean_token.length": float(len(token_text)),
        "clean_token.blank": float(not stripped),
        "clean_token.leading_space": float(token_text.startswith(" ")),
        "clean_token.newline": float("\n" in token_text),
        "clean_token.numeric": float(bool(stripped) and stripped.isdigit()),
        "clean_token.punctuation": float(
            bool(stripped)
            and all(character in string.punctuation for character in stripped)
        ),
    }


def _clean_features(event: Dict[str, Any]) -> Dict[str, float]:
    """Return fields observable before the local perturbation is applied."""
    features: Dict[str, float] = {}
    for key in (
        "decision_index",
        "context_sequence_length",
        "clean_top1_margin",
        "clean_hidden_norm",
    ):
        value = _safe_float(event.get(key))
        if value is not None:
            features[key] = value

    _flatten_clean_numeric(
        "clean_diagnostics",
        event.get("clean_diagnostics") or {},
        features,
    )
    features.update(_token_features(str(event.get("consumed_token_text", ""))))
    return features


def load_branchpoint_rows(
    runs_dir: str | Path,
    *,
    label: str = "argmax_flip",
    model: str | None = None,
    intervention_type: str | None = None,
) -> list[Dict[str, Any]]:
    """Load valid local-fork rows; controller-pair artifacts are never considered."""
    if label not in {"argmax_flip", "sampled_flip"}:
        raise ValueError("label must be 'argmax_flip' or 'sampled_flip'")

    rows: list[Dict[str, Any]] = []
    for run_dir in sorted(Path(runs_dir).glob("branchpoint_run_*")):
        config_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"
        events_path = run_dir / "events.jsonl"
        if not all(path.exists() for path in (config_path, summary_path, events_path)):
            continue

        config_record = _load_json(config_path)
        config = config_record.get("config") or {}
        summary = _load_json(summary_path)
        if config.get("mode") != "branchpoints" or summary.get("mode") != "branchpoints":
            continue
        expected_hash = config_record.get("config_hash")
        if expected_hash != hash_config(config):
            raise ValueError(f"{run_dir.name}: config hash does not match saved config")
        if model is not None and config.get("model") != model:
            continue
        if (
            intervention_type is not None
            and config.get("intervention_type") != intervention_type
        ):
            continue

        for event in _load_jsonl(events_path):
            if event.get("mode") != "branchpoints":
                raise ValueError(f"{run_dir.name}: non-branchpoint event in events.jsonl")
            if event.get("contexts_matched") is not True:
                raise ValueError(f"{run_dir.name}: unmatched counterfactual context")
            if label not in event:
                raise ValueError(f"{run_dir.name}: missing label field {label!r}")
            rows.append(
                {
                    "run_id": run_dir.name,
                    "decision_index": int(event.get("decision_index", 0)),
                    "label": int(bool(event[label])),
                    "features": _clean_features(event),
                    "model": config.get("model"),
                    "intervention_type": config.get("intervention_type"),
                }
            )
    return rows


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
        ranks[order[index : end + 1]] = 0.5 * (index + end) + 1.0
        index = end + 1
    return ranks


def _roc_auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = np.asarray(labels, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both positive and negative labels")
    ranks = _average_ranks(score_array)
    positive_rank_sum = float(np.sum(ranks[y == 1]))
    u_statistic = positive_rank_sum - positives * (positives + 1) / 2.0
    return float(u_statistic / (positives * negatives))


def _precision_recall(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float = 0.5,
) -> tuple[float, float]:
    y = np.asarray(labels, dtype=int)
    predicted = np.asarray(scores, dtype=float) >= float(threshold)
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
        1.0 + np.exp(-np.clip(train_features @ weights + bias, -30.0, 30.0))
    )
    test_scores = 1.0 / (
        1.0 + np.exp(-np.clip(test_features @ weights + bias, -30.0, 30.0))
    )
    return train_scores, test_scores, weights


def _fit_preprocessing(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    train = np.where(np.isnan(train), medians, train)
    test = np.where(np.isnan(test), medians, test)

    means = np.mean(train, axis=0)
    standard_deviations = np.std(train, axis=0)
    standard_deviations = np.where(standard_deviations < 1e-8, 1.0, standard_deviations)
    train = (train - means) / standard_deviations
    test = (test - means) / standard_deviations
    return (
        np.clip(np.nan_to_num(train), -20.0, 20.0),
        np.clip(np.nan_to_num(test), -20.0, 20.0),
    )


def _split_run_ids(
    run_ids: Sequence[str],
    *,
    train_frac: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    shuffled = list(sorted(run_ids))
    random.Random(seed).shuffle(shuffled)
    split_at = int(round(len(shuffled) * float(train_frac)))
    split_at = max(1, min(len(shuffled) - 1, split_at))
    return sorted(shuffled[:split_at]), sorted(shuffled[split_at:])


def _describe(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def analyze_branchpoint_runs(
    runs_dir: str | Path,
    *,
    label: str = "argmax_flip",
    model: str | None = None,
    intervention_type: str | None = None,
    repeats: int = DEFAULT_REPEATS,
    train_frac: float = 0.8,
    seed: int = 42,
    min_runs: int = DEFAULT_MIN_RUNS,
    min_feature_coverage: float = 0.5,
) -> Dict[str, Any]:
    rows = load_branchpoint_rows(
        runs_dir,
        label=label,
        model=model,
        intervention_type=intervention_type,
    )
    run_ids = sorted({str(row["run_id"]) for row in rows})
    if len(run_ids) < int(min_runs):
        raise ValueError(
            f"Need at least {min_runs} branchpoint runs; found {len(run_ids)}"
        )
    if not 0.0 < float(train_frac) < 1.0:
        raise ValueError("train_frac must be between zero and one")

    feature_names = sorted(
        {
            feature_name
            for row in rows
            for feature_name in row["features"]
            if (
                sum(
                    feature_name in candidate["features"]
                    for candidate in rows
                )
                / len(rows)
            )
            >= float(min_feature_coverage)
        }
    )
    if not feature_names:
        raise ValueError("No clean features meet the requested coverage")

    feature_matrix = np.asarray(
        [
            [
                row["features"].get(feature_name, np.nan)
                for feature_name in feature_names
            ]
            for row in rows
        ],
        dtype=float,
    )
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    row_runs = np.asarray([str(row["run_id"]) for row in rows], dtype=object)

    splits: list[Dict[str, Any]] = []
    weight_vectors: list[np.ndarray] = []
    attempt = 0
    max_attempts = max(int(repeats) * 20, 20)
    while len(splits) < int(repeats) and attempt < max_attempts:
        train_runs, test_runs = _split_run_ids(
            run_ids,
            train_frac=train_frac,
            seed=int(seed) + attempt,
        )
        attempt += 1
        train_mask = np.isin(row_runs, train_runs)
        test_mask = np.isin(row_runs, test_runs)
        train_labels = labels[train_mask]
        test_labels = labels[test_mask]
        if (
            len(set(train_labels.tolist())) < 2
            or len(set(test_labels.tolist())) < 2
        ):
            continue

        train_features, test_features = _fit_preprocessing(
            feature_matrix[train_mask],
            feature_matrix[test_mask],
        )
        train_scores, test_scores, weights = _fit_logistic_regression(
            train_features,
            train_labels,
            test_features,
        )
        precision, recall = _precision_recall(test_labels, test_scores)
        splits.append(
            {
                "split_index": len(splits),
                "train_runs": train_runs,
                "test_runs": test_runs,
                "n_train_rows": int(np.sum(train_mask)),
                "n_test_rows": int(np.sum(test_mask)),
                "train_auroc": _roc_auc_score(train_labels, train_scores),
                "auroc": _roc_auc_score(test_labels, test_scores),
                "precision": precision,
                "recall": recall,
            }
        )
        weight_vectors.append(weights)

    if len(splits) < int(repeats):
        raise ValueError(
            f"Could create only {len(splits)} valid grouped splits out of "
            f"{repeats} requested"
        )

    mean_absolute_weights = np.mean(np.abs(np.stack(weight_vectors)), axis=0)
    feature_importance = [
        {
            "feature": feature_name,
            "mean_abs_weight": float(weight),
        }
        for feature_name, weight in sorted(
            zip(feature_names, mean_absolute_weights.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    return {
        "label": label,
        "run_count": len(run_ids),
        "row_count": len(rows),
        "positive_rows": int(np.sum(labels == 1)),
        "negative_rows": int(np.sum(labels == 0)),
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "split_counts": {
            "requested": int(repeats),
            "valid": len(splits),
            "attempted": attempt,
        },
        "splits": splits,
        "aggregate": {
            "auroc": _describe(split["auroc"] for split in splits),
            "precision": _describe(split["precision"] for split in splits),
            "recall": _describe(split["recall"] for split in splits),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repeated whole-run evaluation of independent local counterfactual "
            "branchpoint rows."
        )
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--label",
        default="argmax_flip",
        choices=["argmax_flip", "sampled_flip"],
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--intervention-type", default=None)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-runs", type=int, default=DEFAULT_MIN_RUNS)
    parser.add_argument("--min-feature-coverage", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--json-out", default=None)
    return parser


def _print_result(result: Dict[str, Any], top_n: int) -> None:
    print("Observer local branchpoint analysis")
    print(
        f"runs={result['run_count']} rows={result['row_count']} "
        f"positive={result['positive_rows']} negative={result['negative_rows']}"
    )
    print(
        f"grouped splits={result['split_counts']['valid']}/"
        f"{result['split_counts']['requested']}"
    )
    for metric in ("auroc", "precision", "recall"):
        values = result["aggregate"][metric]
        print(
            f"{metric}: mean={values['mean']:.4f} median={values['median']:.4f} "
            f"range=[{values['min']:.4f}, {values['max']:.4f}]"
        )
    print("top clean features:")
    for row in result["feature_importance"][: int(top_n)]:
        print(f"  {row['mean_abs_weight']:.4f}  {row['feature']}")


def main() -> None:
    args = build_arg_parser().parse_args()
    result = analyze_branchpoint_runs(
        args.runs_dir,
        label=args.label,
        model=args.model,
        intervention_type=args.intervention_type,
        repeats=args.repeats,
        train_frac=args.train_frac,
        seed=args.seed,
        min_runs=args.min_runs,
        min_feature_coverage=args.min_feature_coverage,
    )
    _print_result(result, args.top_n)
    if args.json_out:
        output_path = Path(args.json_out)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"JSON summary written to {output_path}")


if __name__ == "__main__":
    main()
