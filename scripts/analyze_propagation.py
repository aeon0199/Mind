#!/usr/bin/env python3
"""Aggregate Observer Q2 propagation artifacts at the run level."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from runtime_lab.core.io.hashing import hash_config
try:
    from scripts.run_propagation_calibration import (
        control_magnitude,
        validate_cell_summary,
    )
except ModuleNotFoundError:
    from run_propagation_calibration import (
        control_magnitude,
        validate_cell_summary,
    )


METRICS = (
    "delta_l2",
    "delta_relative_clean",
    "cosine_distance",
    "amplification_vs_injection",
)
NUMERIC_FLOOR_RELATIVE = 1e-6
NUMERIC_FLOOR_LOGIT_KL = 1e-6


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _describe(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)) if values else 0.0,
        "min": float(min(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
    }


def _mean_metric(
    summary: Mapping[str, Any],
    *,
    layer: int | None,
    metric: str,
) -> float:
    metrics = summary.get("metrics") or {}
    if layer is None:
        block = (
            (metrics.get("terminal_summary") or {}).get("final_norm") or {}
        )
    else:
        block = (
            (metrics.get("layer_summary") or {}).get(str(layer)) or {}
        )
    return float((block.get(metric) or {}).get("mean", 0.0))


def _curve_point(
    summaries: Sequence[Mapping[str, Any]],
    *,
    layer: int | None,
) -> dict[str, Any]:
    point: dict[str, Any] = {
        "target": "final_norm" if layer is None else f"layer_{layer}",
        "kind": "terminal" if layer is None else "transformer",
        "layer": layer,
    }
    for metric in METRICS:
        point[metric] = _describe(
            [
                _mean_metric(summary, layer=layer, metric=metric)
                for summary in summaries
            ]
        )
    return point


def _group_key(record: Mapping[str, Any]) -> tuple[str, float, int]:
    return (
        str(record["intervention_type"]),
        float(record["magnitude"]),
        int(record["injection_layer"]),
    )


def _prompt_slice(
    members: Sequence[Mapping[str, Any]],
    *,
    capture_layers: Sequence[int],
) -> dict[str, Any]:
    summaries = [member["summary"] for member in members]
    decision_count = sum(
        int((summary.get("protocol_validity") or {}).get("rows", 0))
        for summary in summaries
    )
    argmax_flips = sum(
        int((summary.get("metrics") or {}).get("argmax_flips", 0))
        for summary in summaries
    )
    layer_points = [
        _curve_point(summaries, layer=layer)
        for layer in capture_layers
    ]
    peak = max(
        layer_points,
        key=lambda point: point["amplification_vs_injection"]["mean"],
    )
    terminal = _curve_point(summaries, layer=None)
    return {
        "run_count": len(members),
        "decision_count": decision_count,
        "seeds": sorted({int(member["seed"]) for member in members}),
        "mean_logit_kl": _describe(
            [
                float(
                    (summary.get("metrics") or {}).get(
                        "mean_logit_kl",
                        0.0,
                    )
                )
                for summary in summaries
            ]
        ),
        "argmax_flips": argmax_flips,
        "argmax_flip_rate": (
            float(argmax_flips / decision_count) if decision_count else 0.0
        ),
        "peak_transformer_layer": int(peak["layer"]),
        "peak_transformer_amplification": float(
            peak["amplification_vs_injection"]["mean"]
        ),
        "final_norm_delta_relative_clean": terminal[
            "delta_relative_clean"
        ],
        "final_norm_amplification_vs_injection": terminal[
            "amplification_vs_injection"
        ],
    }


def aggregate_runs(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate equal-weight run means without treating decisions as runs."""
    grouped: dict[
        tuple[str, float, int],
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    all_records = list(records)
    for record in all_records:
        grouped[_group_key(record)].append(record)

    groups: list[dict[str, Any]] = []
    total_decisions = 0
    for key in sorted(grouped):
        intervention_type, magnitude, injection_layer = key
        members = grouped[key]
        summaries = [member["summary"] for member in members]
        capture_layers = [
            int(layer) for layer in summaries[0].get("capture_layers") or []
        ]
        if not capture_layers or capture_layers[0] != injection_layer:
            raise ValueError(
                f"Invalid capture curve for group {key}: {capture_layers}"
            )
        if any(
            [int(layer) for layer in summary.get("capture_layers") or []]
            != capture_layers
            for summary in summaries
        ):
            raise ValueError(f"Mismatched capture layers within group {key}")

        decision_count = sum(
            int((summary.get("protocol_validity") or {}).get("rows", 0))
            for summary in summaries
        )
        total_decisions += decision_count
        argmax_flips = sum(
            int((summary.get("metrics") or {}).get("argmax_flips", 0))
            for summary in summaries
        )
        sampled_flips = sum(
            int((summary.get("metrics") or {}).get("sampled_flips", 0))
            for summary in summaries
        )
        curve = [
            _curve_point(summaries, layer=layer)
            for layer in capture_layers
        ]
        curve.append(_curve_point(summaries, layer=None))
        transformer_points = [
            point for point in curve if point["kind"] == "transformer"
        ]
        peak_point = max(
            transformer_points,
            key=lambda point: point["amplification_vs_injection"]["mean"],
        )
        terminal = curve[-1]
        direct = curve[0]
        control = magnitude == control_magnitude(intervention_type)
        output = {
            "argmax_flips": argmax_flips,
            "argmax_flip_rate": (
                float(argmax_flips / decision_count)
                if decision_count
                else 0.0
            ),
            "sampled_flips": sampled_flips,
            "sampled_flip_rate": (
                float(sampled_flips / decision_count)
                if decision_count
                else 0.0
            ),
            "mean_logit_kl": _describe(
                [
                    float(
                        (summary.get("metrics") or {}).get(
                            "mean_logit_kl",
                            0.0,
                        )
                    )
                    for summary in summaries
                ]
            ),
            "mean_logit_js": _describe(
                [
                    float(
                        (summary.get("metrics") or {}).get(
                            "mean_logit_js",
                            0.0,
                        )
                    )
                    for summary in summaries
                ]
            ),
        }
        by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for member in members:
            by_prompt[str(member["prompt_key"])].append(member)
        prompt_slices = {
            prompt_key: _prompt_slice(
                prompt_members,
                capture_layers=capture_layers,
            )
            for prompt_key, prompt_members in sorted(by_prompt.items())
        }
        final_norm_near_numeric_floor = bool(
            not control
            and direct["delta_l2"]["mean"] > 1e-12
            and terminal["delta_relative_clean"]["max"]
            <= NUMERIC_FLOOR_RELATIVE
            and output["mean_logit_kl"]["max"] <= NUMERIC_FLOOR_LOGIT_KL
        )
        groups.append(
            {
                "intervention_type": intervention_type,
                "magnitude": magnitude,
                "injection_layer": injection_layer,
                "control": control,
                "run_count": len(members),
                "decision_count": decision_count,
                "prompt_keys": sorted(
                    {str(member["prompt_key"]) for member in members}
                ),
                "prompt_slices": prompt_slices,
                "seeds": sorted({int(member["seed"]) for member in members}),
                "curve": curve,
                "peak_transformer_layer": int(peak_point["layer"]),
                "peak_transformer_amplification": float(
                    peak_point["amplification_vs_injection"]["mean"]
                ),
                "final_norm_near_numeric_floor": (
                    final_norm_near_numeric_floor
                ),
                "output": output,
            }
        )

    controls = [group for group in groups if group["control"]]
    identity_controls_exact = bool(
        controls
        and all(
            all(
                point["delta_l2"]["max"] <= 1e-12
                for point in group["curve"]
            )
            and group["output"]["argmax_flips"] == 0
            and group["output"]["sampled_flips"] == 0
            and group["output"]["mean_logit_kl"]["max"] <= 1e-12
            and group["output"]["mean_logit_js"]["max"] <= 1e-12
            for group in controls
        )
    )
    return {
        "analysis_protocol": "observer-q2-run-level-analysis-v1",
        "measurement_protocol": "same-context-layer-propagation-v1",
        "aggregation_unit": "equal-weight run means",
        "run_count": len(all_records),
        "decision_count": total_decisions,
        "group_count": len(groups),
        "identity_control_group_count": len(controls),
        "identity_controls_exact": identity_controls_exact,
        "numeric_floor_criteria": {
            "final_norm_delta_relative_clean_max": NUMERIC_FLOOR_RELATIVE,
            "mean_logit_kl_run_max": NUMERIC_FLOOR_LOGIT_KL,
        },
        "groups": groups,
    }


def load_propagation_runs(
    root: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Load, rehash, and revalidate every completed manifest cell."""
    root = Path(root).expanduser().resolve()
    manifest_path = root / "propagation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(
            f"Propagation manifest is not complete: {manifest.get('status')}"
        )
    cells = manifest.get("cells") or {}
    expected_count = int(manifest.get("cell_count", 0))
    if len(cells) != expected_count:
        raise ValueError(
            f"Manifest has {len(cells)} cells; expected {expected_count}"
        )

    records: list[dict[str, Any]] = []
    run_dirs: set[str] = set()
    config_hashes: set[str] = set()
    for cell_id, cell in sorted(cells.items()):
        if cell.get("status") != "ok":
            raise ValueError(f"Cell is not valid: {cell_id}")
        summary_path = Path(cell["summary_path"]).expanduser().resolve()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config_path = summary_path.parent / "config.json"
        config_record = json.loads(config_path.read_text(encoding="utf-8"))
        config = config_record["config"]
        config_hash = str(config_record["config_hash"])
        if config_hash != hash_config(config):
            raise ValueError(f"Config hash mismatch: {cell_id}")
        if summary.get("config_hash") != config_hash:
            raise ValueError(f"Summary hash mismatch: {cell_id}")
        if cell.get("config_hash") != config_hash:
            raise ValueError(f"Manifest hash mismatch: {cell_id}")
        validate_cell_summary(cell, summary)
        run_dir = str(summary_path.parent)
        if run_dir in run_dirs:
            raise ValueError(f"Duplicate run directory: {run_dir}")
        if config_hash in config_hashes:
            raise ValueError(f"Duplicate config hash: {config_hash}")
        run_dirs.add(run_dir)
        config_hashes.add(config_hash)
        records.append(
            {
                "cell_id": cell_id,
                "intervention_type": str(cell["intervention_type"]),
                "magnitude": float(cell["magnitude"]),
                "injection_layer": int(cell["injection_layer"]),
                "prompt_key": str(cell["prompt_key"]),
                "seed": int(cell["seed"]),
                "summary_path": str(summary_path),
                "summary": summary,
            }
        )

    return (
        dict(manifest.get("design") or {}),
        records,
        {
            "manifest_path": str(manifest_path),
            "manifest_status": str(manifest.get("status")),
            "expected_cells": expected_count,
            "validated_cells": len(records),
            "unique_run_dirs": len(run_dirs),
            "unique_config_hashes": len(config_hashes),
        },
    )


def _number(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1e-4 or abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def render_markdown(
    report: Mapping[str, Any],
    *,
    artifact_root: str,
) -> str:
    """Render a neutral reader-facing report; conclusions remain explicit."""
    lines = [
        "# Q2 Per-Layer Propagation Analysis",
        "",
        f"- Artifact root: `{artifact_root}`",
        (
            f"- Measurement: `{report['measurement_protocol']}`; aggregation: "
            f"**{report['aggregation_unit']}**."
        ),
        (
            f"- Validated runs: **{report['run_count']}**; paired decisions: "
            f"**{report['decision_count']}**; groups: "
            f"**{report['group_count']}**."
        ),
        (
            "- Identity controls exact: "
            f"**{str(report['identity_controls_exact']).lower()}**."
        ),
        "",
        "Each cell below aggregates run means equally. A run with more paired "
        "decisions does not receive extra statistical weight. `Peak amp` is "
        "the largest mean absolute-delta amplification across transformer "
        "blocks. `Final norm` is measured after the model's terminal "
        "normalization and before logits.",
        "",
        "## Group overview",
        "",
        "| Type | Magnitude | Inject L | Runs | Direct rel. | Peak amp @ L | "
        "Final norm rel. | Final amp | Mean KL | Argmax flips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in report["groups"]:
        direct = group["curve"][0]
        terminal = group["curve"][-1]
        output = group["output"]
        magnitude = _number(group["magnitude"])
        if group["control"]:
            magnitude += " (control)"
        lines.append(
            "| "
            f"{group['intervention_type']} | {magnitude} | "
            f"{group['injection_layer']} | {group['run_count']} | "
            f"{_number(direct['delta_relative_clean']['mean'])} | "
            f"{_number(group['peak_transformer_amplification'])} @ "
            f"{group['peak_transformer_layer']} | "
            f"{_number(terminal['delta_relative_clean']['mean'])} | "
            f"{_number(terminal['amplification_vs_injection']['mean'])} | "
            f"{_number(output['mean_logit_kl']['mean'])} | "
            f"{output['argmax_flips']}/{group['decision_count']} |"
        )

    lines.extend(
        [
            "",
            "## Per-layer curves",
            "",
        ]
    )
    for group in report["groups"]:
        if group["control"]:
            continue
        lines.extend(
            [
                (
                    f"### {group['intervention_type']} "
                    f"{_number(group['magnitude'])} at layer "
                    f"{group['injection_layer']}"
                ),
                "",
                "| Target | Delta L2 | Relative to clean | Cosine distance | "
                "Amplification |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for point in group["curve"]:
            target = (
                "Final norm"
                if point["kind"] == "terminal"
                else f"Layer {point['layer']}"
            )
            lines.append(
                f"| {target} | {_number(point['delta_l2']['mean'])} | "
                f"{_number(point['delta_relative_clean']['mean'])} | "
                f"{_number(point['cosine_distance']['mean'])} | "
                f"{_number(point['amplification_vs_injection']['mean'])} |"
            )
        if group["final_norm_near_numeric_floor"]:
            lines.extend(
                [
                    "",
                    (
                        "Final normalization reduced this measured "
                        "perturbation to the configured numerical-floor "
                        "criterion."
                    ),
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate a complete Observer Q2 propagation manifest."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.root).expanduser().resolve()
    design, records, validation = load_propagation_runs(root)
    report = aggregate_runs(records)
    report.update(
        {
            "generated_at": _utc_now(),
            "artifact_root": str(root),
            "design": design,
            "artifact_validation": validation,
        }
    )
    json_path = (
        Path(args.json_out).expanduser().resolve()
        if args.json_out
        else root / "propagation_analysis.json"
    )
    markdown_path = (
        Path(args.markdown_out).expanduser().resolve()
        if args.markdown_out
        else root / "PROPAGATION_ANALYSIS.md"
    )
    _write_json(json_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        render_markdown(report, artifact_root=str(root)),
        encoding="utf-8",
    )
    print(
        f"[q2-analysis] validated={validation['validated_cells']}/"
        f"{validation['expected_cells']} groups={report['group_count']}"
    )
    print(
        f"[q2-analysis] controls_exact={report['identity_controls_exact']}"
    )
    print(f"[q2-analysis] json={json_path}")
    print(f"[q2-analysis] markdown={markdown_path}")


if __name__ == "__main__":
    main()
