#!/usr/bin/env python3
"""Run Observer's resumable M3R causal basin calibration."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from runtime_lab.basins.evaluators import (
    PROMPT_SPECS,
    BasinPromptSpec,
    get_prompt_spec,
    validate_prompt_specs,
)
from runtime_lab.basins.experiment import (
    CONTINUATION_PROTOCOL,
    PROTOCOL,
    SELECTION_PROTOCOL,
    run_basin_experiment,
)
from runtime_lab.cli._common import (
    backend_num_layers,
    resolve_probe_layers,
    resolve_semantic_layer,
)
from runtime_lab.config.schemas import BasinConfig, DiagnosticsConfig
from runtime_lab.core.backend.loader import load_model_with_backend
from runtime_lab.core.io.hashing import hash_config


CALIBRATION_PROTOCOL = "observer-m3r-basin-calibration-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def magnitude_slug(value: float) -> str:
    rendered = f"{float(value):.3f}".replace("-", "neg").replace(".", "p")
    return f"magnitude_{rendered}"


def build_basin_cells(
    *,
    root: str | Path,
    prompts: Sequence[BasinPromptSpec],
    seeds: Iterable[int],
    magnitudes: Iterable[float],
) -> list[dict[str, Any]]:
    root = Path(root).expanduser().resolve()
    cells: list[dict[str, Any]] = []
    for magnitude in magnitudes:
        value = float(magnitude)
        for spec in prompts:
            runs_dir = root / magnitude_slug(value) / spec.key
            for seed in seeds:
                seed = int(seed)
                cells.append(
                    {
                        "cell_id": (
                            f"{magnitude_slug(value)}:{spec.key}:seed_{seed}"
                        ),
                        "magnitude": value,
                        "prompt_key": spec.key,
                        "prompt_class": spec.prompt_class,
                        "prompt": spec.prompt,
                        "rubric": spec.to_record(),
                        "seed": seed,
                        "runs_dir": str(runs_dir),
                    }
                )
    return cells


def validate_cell_summary(
    cell: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    protocol = summary.get("protocol_validity") or {}
    metrics = summary.get("metrics") or {}
    assert summary.get("prompt_key") == cell["prompt_key"], (
        "summary prompt key does not match the calibration cell"
    )
    assert summary.get("prompt_class") == cell["prompt_class"], (
        "summary prompt class does not match the calibration cell"
    )
    assert protocol.get("protocol") == PROTOCOL, (
        "unexpected basin continuation protocol"
    )
    assert protocol.get("pass1_all_contexts_matched") is True, (
        "local basin contexts were not all matched"
    )
    assert protocol.get("selection_outcome_blind") is True, (
        "branchpoint selection observed continuation outcomes"
    )
    assert protocol.get("replay_all_contexts_verified") is True, (
        "selected contexts were not verified during replay"
    )
    assert protocol.get("clean_reference_replay_exact") is True, (
        "clean replay did not reproduce the mapped reference"
    )
    assert protocol.get("continuation_intervention_disabled") is True, (
        "intervention remained active during branch continuation"
    )
    assert protocol.get("continuation_sampling_rng_paired") is True, (
        "continuation sampling randomness was not paired"
    )
    assert int(protocol.get("branchpoint_rows", 0)) > 0, (
        "basin run emitted no local branchpoints"
    )
    assert int(protocol.get("selected_episodes", 0)) > 0, (
        "basin run emitted no continuation episodes"
    )

    if float(cell["magnitude"]) == 0.0:
        assert metrics.get("identity_control") is True, (
            "zero cell was not recognized as an identity control"
        )
        assert metrics.get("identity_control_exact") is True, (
            "identity control was not exact"
        )
        assert int(metrics.get("argmax_flips", -1)) == 0, (
            "identity control produced an argmax flip"
        )
        assert int(metrics.get("sampled_flips", -1)) == 0, (
            "identity control produced a sampled flip"
        )
        assert int(metrics.get("local_effect_rows", -1)) == 0, (
            "identity control produced a measurable local effect"
        )
    else:
        assert int(metrics.get("local_effect_rows", 0)) > 0, (
            "active basin intervention had no measurable direct effect"
        )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _build_design(
    args: argparse.Namespace,
    *,
    backend_result: Any,
) -> dict[str, Any]:
    num_layers = backend_num_layers(backend_result)
    resolved_layer = resolve_semantic_layer(args.layer, num_layers)
    probe_layers = resolve_probe_layers("auto", num_layers)
    if resolved_layer not in probe_layers:
        probe_layers = [resolved_layer, *probe_layers]
    return {
        "protocol": CALIBRATION_PROTOCOL,
        "measurement_protocol": PROTOCOL,
        "selection_protocol": SELECTION_PROTOCOL,
        "continuation_protocol": CONTINUATION_PROTOCOL,
        "model": args.model,
        "registry_path": str(Path(args.registry_path).resolve()),
        "loaded_model_key": backend_result.config.key,
        "loaded_model_hf_id": backend_result.config.hf_id,
        "backend": backend_result.backend,
        "backend_meta": dict(backend_result.backend_meta),
        "num_layers": num_layers,
        "prompts": [spec.to_record() for spec in args.prompts],
        "prompt_keys": [spec.key for spec in args.prompts],
        "prompt_classes": sorted(
            {spec.prompt_class for spec in args.prompts}
        ),
        "seeds": list(args.seeds),
        "magnitudes": list(args.magnitudes),
        "identity_control": 0.0,
        "max_tokens": int(args.max_tokens),
        "continuation_tokens": int(args.continuation_tokens),
        "max_episodes": int(args.max_episodes),
        "layer_requested": str(args.layer),
        "layer_resolved": resolved_layer,
        "probe_layers_resolved": probe_layers,
        "intervention_type": "additive",
        "intervention_seed": int(args.intervention_seed),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "tie_tolerance": float(args.tie_tolerance),
    }


def _expected_config(
    cell: Mapping[str, Any],
    *,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "mode": "basins",
        "protocol": PROTOCOL,
        "selection_protocol": SELECTION_PROTOCOL,
        "continuation_protocol": CONTINUATION_PROTOCOL,
        "prompt": cell["prompt"],
        "prompt_key": cell["prompt_key"],
        "prompt_class": cell["prompt_class"],
        "prompt_spec": cell["rubric"],
        "model": design["model"],
        "max_new_tokens": int(design["max_tokens"]),
        "continuation_tokens": int(design["continuation_tokens"]),
        "max_episodes": int(design["max_episodes"]),
        "seed": int(cell["seed"]),
        "temperature": float(design["temperature"]),
        "top_p": float(design["top_p"]),
        "top_k": int(design["top_k"]),
        "intervention_layer_requested": int(design["layer_resolved"]),
        "intervention_layer_resolved": int(design["layer_resolved"]),
        "intervention_type": "additive",
        "intervention_magnitude": float(cell["magnitude"]),
        "intervention_magnitude_relative": True,
        "intervention_seed": int(design["intervention_seed"]),
        "tie_tolerance": float(design["tie_tolerance"]),
    }


def _matches_expected_config(
    config: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    return all(config.get(key) == value for key, value in expected.items())


def _find_completed_summary(
    cell: Mapping[str, Any],
    *,
    design: Mapping[str, Any],
) -> tuple[dict[str, Any], Path] | None:
    expected = _expected_config(cell, design=design)
    for run_dir in sorted(
        Path(cell["runs_dir"]).glob("basins_run_*"),
        reverse=True,
    ):
        config_path = run_dir / "config.json"
        summary_path = run_dir / "summary.json"
        if not config_path.exists() or not summary_path.exists():
            continue
        try:
            record = json.loads(config_path.read_text(encoding="utf-8"))
            config = record["config"]
            if record.get("config_hash") != hash_config(config):
                continue
            if not _matches_expected_config(config, expected):
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("config_hash") != record["config_hash"]:
                continue
            validate_cell_summary(cell, summary)
        except (
            AssertionError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            continue
        return summary, summary_path
    return None


def _compact_result(
    cell: Mapping[str, Any],
    summary: Mapping[str, Any],
    summary_path: Path,
    *,
    disposition: str,
) -> dict[str, Any]:
    metrics = summary.get("metrics") or {}
    return {
        "status": "ok",
        "disposition": disposition,
        "completed_at": _utc_now(),
        "summary_path": str(summary_path),
        "run_dir": str(summary.get("run_dir") or summary_path.parent),
        "config_hash": summary.get("config_hash"),
        "branchpoint_rows": int(metrics.get("branchpoint_rows", 0)),
        "local_effect_rows": int(metrics.get("local_effect_rows", 0)),
        "argmax_flips": int(metrics.get("argmax_flips", 0)),
        "sampled_flips": int(metrics.get("sampled_flips", 0)),
        "selected_episodes": int(metrics.get("selected_episodes", 0)),
        "selected_sampled_flips": int(
            metrics.get("selected_sampled_flips", 0)
        ),
        "outcomes": dict(metrics.get("outcomes") or {}),
        "causal_classifications": dict(
            metrics.get("causal_classifications") or {}
        ),
        "identity_control_exact": metrics.get("identity_control_exact"),
    }


def _run_cell(
    cell: Mapping[str, Any],
    *,
    design: Mapping[str, Any],
    backend_result: Any,
) -> dict[str, Any]:
    spec = get_prompt_spec(str(cell["prompt_key"]))
    config = BasinConfig(
        prompt=spec.prompt,
        prompt_key=spec.key,
        prompt_class=spec.prompt_class,
        evaluator_id=spec.evaluator_id,
        model_key=str(design["model"]),
        max_new_tokens=int(design["max_tokens"]),
        continuation_tokens=int(design["continuation_tokens"]),
        max_episodes=int(design["max_episodes"]),
        backend=str(design["backend"]),
        seed=int(cell["seed"]),
        temperature=float(design["temperature"]),
        top_p=float(design["top_p"]),
        top_k=int(design["top_k"]),
        intervention_layer=int(design["layer_resolved"]),
        intervention_type="additive",
        intervention_magnitude=float(cell["magnitude"]),
        intervention_magnitude_relative=True,
        intervention_seed=int(design["intervention_seed"]),
        tie_tolerance=float(design["tie_tolerance"]),
        with_diagnostics=True,
    )
    diagnostics = DiagnosticsConfig(
        enabled=True,
        probe_layers=list(design["probe_layers_resolved"]),
    )
    return run_basin_experiment(
        config,
        registry_path=str(design["registry_path"]),
        runs_dir=str(cell["runs_dir"]),
        diagnostics_config=diagnostics,
        prebuilt_backend=backend_result,
        prompt_spec=spec,
    )


def _select_prompts(keys: Iterable[str]) -> list[BasinPromptSpec]:
    requested = list(keys)
    if requested == ["all"]:
        return list(PROMPT_SPECS)
    table = {spec.key: spec for spec in PROMPT_SPECS}
    unknown = sorted(set(requested) - set(table))
    if unknown:
        raise ValueError(f"Unknown basin prompt keys: {unknown}")
    return [table[key] for key in requested]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run M3R causal branch continuations over a resumable warm-model "
            "prompt × seed × control grid."
        )
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", default="qwen3-1.7b")
    parser.add_argument("--registry-path", default="models.json")
    parser.add_argument("--prompt-keys", default="all")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--magnitudes", default="0,0.3")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--continuation-tokens", type=int, default=48)
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--layer", default="late")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--intervention-seed", type=int, default=42)
    parser.add_argument("--tie-tolerance", type=float, default=0.025)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.seeds = _parse_ints(args.seeds)
    args.magnitudes = _parse_floats(args.magnitudes)
    args.prompts = _select_prompts(
        key.strip()
        for key in args.prompt_keys.split(",")
        if key.strip()
    )
    if not args.seeds or not args.prompts:
        raise ValueError("At least one seed and prompt are required")
    if 0.0 not in args.magnitudes:
        raise ValueError("M3R calibration requires magnitude 0 control")
    evaluator_validation = validate_prompt_specs()
    if not evaluator_validation["all_passed"]:
        raise RuntimeError("Basin evaluator fixture validation failed")

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "basin_manifest.json"
    print(f"[m3r] root={root}", flush=True)
    print(f"[m3r] loading model={args.model}", flush=True)
    backend_result = load_model_with_backend(
        model_key=args.model,
        registry_path=args.registry_path,
        backend="hf",
    )
    design = _build_design(args, backend_result=backend_result)
    cells = build_basin_cells(
        root=root,
        prompts=args.prompts,
        seeds=args.seeds,
        magnitudes=args.magnitudes,
    )

    existing: dict[str, Any] = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("design") != design:
            raise ValueError(
                "Existing basin manifest has a different design; choose a "
                "new --root"
            )
    manifest: dict[str, Any] = {
        "protocol": CALIBRATION_PROTOCOL,
        "status": "running",
        "created_at": existing.get("created_at", _utc_now()),
        "updated_at": _utc_now(),
        "root": str(root),
        "design": design,
        "evaluator_validation": evaluator_validation,
        "cell_count": len(cells),
        "cells": dict(existing.get("cells") or {}),
    }
    _write_json_atomic(manifest_path, manifest)

    for index, cell in enumerate(cells, start=1):
        print(f"[m3r] {index}/{len(cells)} {cell['cell_id']}", flush=True)
        completed = None
        if not args.no_resume:
            completed = _find_completed_summary(cell, design=design)
        try:
            if completed is None:
                summary = _run_cell(
                    cell,
                    design=design,
                    backend_result=backend_result,
                )
                summary_path = Path(summary["artifacts"]["summary_path"])
                disposition = "ran"
            else:
                summary, summary_path = completed
                disposition = "resumed"
            validate_cell_summary(cell, summary)
            compact = _compact_result(
                cell,
                summary,
                summary_path,
                disposition=disposition,
            )
            manifest["cells"][cell["cell_id"]] = {
                **cell,
                **compact,
            }
            print(
                f"[m3r]   ok ({disposition}) "
                f"flips={compact['sampled_flips']} "
                f"selected={compact['selected_episodes']} "
                f"outcomes={compact['outcomes']}",
                flush=True,
            )
        except Exception as error:
            manifest["cells"][cell["cell_id"]] = {
                **cell,
                "status": "failed",
                "failed_at": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
            manifest["status"] = "failed"
            manifest["updated_at"] = _utc_now()
            _write_json_atomic(manifest_path, manifest)
            raise
        manifest["updated_at"] = _utc_now()
        _write_json_atomic(manifest_path, manifest)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    manifest["status"] = "complete"
    manifest["completed_at"] = _utc_now()
    manifest["updated_at"] = _utc_now()
    manifest["completed_cells"] = sum(
        entry.get("status") == "ok"
        for entry in manifest["cells"].values()
    )
    _write_json_atomic(manifest_path, manifest)
    print(
        f"[m3r] complete cells={manifest['completed_cells']}/"
        f"{manifest['cell_count']}",
        flush=True,
    )
    print(f"[m3r] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
