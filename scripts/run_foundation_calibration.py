#!/usr/bin/env python3
"""Run Observer's M1R + MHR foundation calibration with one warm model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from runtime_lab.branchpoints.experiment import run_branchpoint_experiment
from runtime_lab.config.schemas import (
    BranchpointConfig,
    DiagnosticsConfig,
    HysteresisConfig,
)
from runtime_lab.core.backend.loader import load_model_with_backend
from runtime_lab.core.io.hashing import hash_config
from runtime_lab.hysteresis.runner import run_hysteresis_experiment
from runtime_lab.cli._common import (
    backend_num_layers,
    resolve_probe_layers,
    resolve_semantic_layer,
)


CALIBRATION_PROTOCOL = "observer-foundation-calibration-v1"
DEFAULT_PROMPTS = (
    {
        "key": "sourdough",
        "text": "Write step-by-step instructions for baking sourdough bread.",
    },
    {
        "key": "water_cycle",
        "text": "Describe the water cycle in a few sentences.",
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def magnitude_slug(value: float) -> str:
    rendered = f"{float(value):.3f}".replace("-", "neg").replace(".", "p")
    return f"magnitude_{rendered}"


def build_calibration_cells(
    *,
    root: str | Path,
    prompts: Sequence[dict[str, str]],
    seeds: Iterable[int],
    branchpoint_magnitudes: Iterable[float],
    hysteresis_magnitudes: Iterable[float],
) -> list[dict[str, Any]]:
    """Build a deterministic grid with one artifact root per mode/magnitude."""
    root = Path(root).expanduser().resolve()
    cells: list[dict[str, Any]] = []
    mode_magnitudes = (
        ("branchpoints", branchpoint_magnitudes),
        ("hysteresis", hysteresis_magnitudes),
    )
    for mode, magnitudes in mode_magnitudes:
        for magnitude in magnitudes:
            magnitude = float(magnitude)
            runs_dir = root / mode / magnitude_slug(magnitude)
            for prompt in prompts:
                for seed in seeds:
                    prompt_key = str(prompt["key"])
                    seed = int(seed)
                    cells.append(
                        {
                            "cell_id": (
                                f"{mode}:{magnitude_slug(magnitude)}:"
                                f"{prompt_key}:seed_{seed}"
                            ),
                            "mode": mode,
                            "magnitude": magnitude,
                            "prompt_key": prompt_key,
                            "prompt": str(prompt["text"]),
                            "seed": seed,
                            "runs_dir": str(runs_dir),
                        }
                    )
    return cells


def validate_cell_summary(
    cell: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    """Fail closed when a calibration cell violates its causal controls."""
    mode = str(cell["mode"])
    magnitude = float(cell["magnitude"])
    metrics = summary.get("metrics") or {}
    protocol = summary.get("protocol_validity") or {}
    epsilon = 1e-12

    if mode == "branchpoints":
        assert protocol.get("all_contexts_matched") is True, (
            "branchpoint contexts were not all matched"
        )
        assert int(protocol.get("rows", 0)) > 0, "branchpoint run emitted no rows"
        if magnitude == 0.0:
            assert int(metrics.get("argmax_flips", -1)) == 0, (
                "zero branchpoint control produced an argmax flip"
            )
            assert int(metrics.get("sampled_flips", -1)) == 0, (
                "zero branchpoint control produced a sampled flip"
            )
            assert abs(float(metrics.get("mean_logit_kl", 1.0))) <= epsilon, (
                "zero branchpoint control produced nonzero logit KL"
            )
            assert abs(float(metrics.get("mean_logit_js", 1.0))) <= epsilon, (
                "zero branchpoint control produced nonzero logit JS"
            )
        else:
            assert (
                float(metrics.get("mean_logit_kl", 0.0)) > 0.0
                or float(metrics.get("mean_logit_js", 0.0)) > 0.0
            ), "nonzero branchpoint intervention had no measurable direct effect"
        return

    if mode == "hysteresis":
        assert protocol.get("matched_exposure") is True, (
            "hysteresis exposure traces were not matched"
        )
        assert protocol.get("matched_recovery") is True, (
            "hysteresis recovery traces were not matched"
        )
        if magnitude == 0.0:
            assert metrics.get("perturbation_did_not_propagate") is True, (
                "zero hysteresis control reported propagation"
            )
            for field in (
                "direct_effect_peak",
                "propagated_distance",
                "residual_distance",
            ):
                assert abs(float(metrics.get(field, 1.0))) <= epsilon, (
                    f"zero hysteresis control produced nonzero {field}"
                )
        else:
            assert float(metrics.get("direct_effect_peak", 0.0)) > 0.0, (
                "nonzero hysteresis intervention had no direct effect"
            )
        return

    raise ValueError(f"Unsupported calibration mode: {mode}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _expected_config(
    cell: dict[str, Any],
    *,
    design: dict[str, Any],
) -> dict[str, Any]:
    common = {
        "mode": cell["mode"],
        "prompt": cell["prompt"],
        "model": design["model"],
        "seed": int(cell["seed"]),
        "temperature": float(design["temperature"]),
        "top_p": float(design["top_p"]),
        "top_k": int(design["top_k"]),
    }
    if cell["mode"] == "branchpoints":
        return {
            **common,
            "max_new_tokens": int(design["branchpoint_max_tokens"]),
            "intervention_layer_resolved": int(
                design["branchpoint_layer_resolved"]
            ),
            "intervention_type": "additive",
            "intervention_magnitude": float(cell["magnitude"]),
            "intervention_magnitude_relative": True,
            "intervention_seed": int(design["intervention_seed"]),
            "with_diagnostics": True,
        }
    return {
        **common,
        "protocol": "matched-exposure-recovery-v2",
        "noise_layer_resolved": int(design["hysteresis_layer_resolved"]),
        "noise_magnitude": float(cell["magnitude"]),
        "noise_start": int(design["noise_start"]),
        "noise_duration": int(design["noise_duration"]),
        "noise_seed": int(design["noise_seed"]),
        "exposure_steps": int(design["hysteresis_exposure_tokens"]),
        "recovery_tokens": int(design["recovery_tokens"]),
        "propagation_floor": float(design["propagation_floor"]),
    }


def _matches_expected_config(
    config: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    return all(config.get(key) == value for key, value in expected.items())


def _find_completed_summary(
    cell: dict[str, Any],
    *,
    design: dict[str, Any],
) -> tuple[dict[str, Any], Path] | None:
    run_prefix = (
        "branchpoint_run_*"
        if cell["mode"] == "branchpoints"
        else "hysteresis_run_*"
    )
    expected = _expected_config(cell, design=design)
    for run_dir in sorted(Path(cell["runs_dir"]).glob(run_prefix), reverse=True):
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
        except (AssertionError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        return summary, summary_path
    return None


def _compact_result(
    cell: dict[str, Any],
    summary: dict[str, Any],
    summary_path: Path,
    *,
    disposition: str,
) -> dict[str, Any]:
    compact = {
        "status": "ok",
        "disposition": disposition,
        "completed_at": _utc_now(),
        "summary_path": str(summary_path),
        "run_dir": str(summary.get("run_dir") or summary_path.parent),
        "config_hash": summary.get("config_hash"),
        "metrics": summary.get("metrics") or {},
        "protocol_validity": summary.get("protocol_validity") or {},
    }
    if cell["mode"] == "branchpoints":
        compact["clean_token_count"] = len(summary.get("clean_token_ids") or [])
    else:
        compact["clean_token_count"] = len(summary.get("clean_token_ids") or [])
        compact["perturbed_token_count"] = len(
            summary.get("perturbed_token_ids") or []
        )
    return compact


def _build_design(args: argparse.Namespace, *, backend_result: Any) -> dict[str, Any]:
    num_layers = backend_num_layers(backend_result)
    branchpoint_layer = resolve_semantic_layer(args.branchpoint_layer, num_layers)
    hysteresis_layer = resolve_semantic_layer(args.hysteresis_layer, num_layers)
    probe_layers = resolve_probe_layers("auto", num_layers)
    if branchpoint_layer not in probe_layers:
        probe_layers = [branchpoint_layer, *probe_layers]
    return {
        "protocol": CALIBRATION_PROTOCOL,
        "model": args.model,
        "registry_path": str(Path(args.registry_path).resolve()),
        "loaded_model_key": backend_result.config.key,
        "loaded_model_hf_id": backend_result.config.hf_id,
        "backend": backend_result.backend,
        "backend_meta": dict(backend_result.backend_meta),
        "num_layers": num_layers,
        "prompts": list(args.prompts),
        "seeds": list(args.seeds),
        "branchpoint_magnitudes": list(args.branchpoint_magnitudes),
        "hysteresis_magnitudes": list(args.hysteresis_magnitudes),
        "branchpoint_max_tokens": int(args.branchpoint_max_tokens),
        "hysteresis_exposure_tokens": int(args.hysteresis_exposure_tokens),
        "recovery_tokens": int(args.recovery_tokens),
        "branchpoint_layer_requested": str(args.branchpoint_layer),
        "branchpoint_layer_resolved": branchpoint_layer,
        "hysteresis_layer_requested": str(args.hysteresis_layer),
        "hysteresis_layer_resolved": hysteresis_layer,
        "probe_layers_resolved": probe_layers,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "intervention_seed": int(args.intervention_seed),
        "noise_seed": int(args.noise_seed),
        "noise_start": int(args.noise_start),
        "noise_duration": int(args.noise_duration),
        "propagation_floor": float(args.propagation_floor),
    }


def _run_cell(
    cell: dict[str, Any],
    *,
    design: dict[str, Any],
    backend_result: Any,
) -> dict[str, Any]:
    if cell["mode"] == "branchpoints":
        diagnostics = DiagnosticsConfig(
            enabled=True,
            probe_layers=list(design["probe_layers_resolved"]),
        )
        config = BranchpointConfig(
            prompt=cell["prompt"],
            model_key=design["model"],
            max_new_tokens=int(design["branchpoint_max_tokens"]),
            backend=design["backend"],
            seed=int(cell["seed"]),
            temperature=float(design["temperature"]),
            top_p=float(design["top_p"]),
            top_k=int(design["top_k"]),
            intervention_layer=int(design["branchpoint_layer_resolved"]),
            intervention_type="additive",
            intervention_magnitude=float(cell["magnitude"]),
            intervention_magnitude_relative=True,
            intervention_seed=int(design["intervention_seed"]),
            with_diagnostics=True,
        )
        return run_branchpoint_experiment(
            config,
            registry_path=design["registry_path"],
            runs_dir=cell["runs_dir"],
            diagnostics_config=diagnostics,
            prebuilt_backend=backend_result,
        )

    config = HysteresisConfig(
        prompt=cell["prompt"],
        model_key=design["model"],
        max_new_tokens=int(design["hysteresis_exposure_tokens"]),
        backend=design["backend"],
        seed=int(cell["seed"]),
        temperature=float(design["temperature"]),
        top_p=float(design["top_p"]),
        top_k=int(design["top_k"]),
        perturbation_mode="noise",
        noise_layer=int(design["hysteresis_layer_resolved"]),
        noise_magnitude=float(cell["magnitude"]),
        noise_start=int(design["noise_start"]),
        noise_duration=int(design["noise_duration"]),
        noise_seed=int(design["noise_seed"]),
        recovery_tokens=int(design["recovery_tokens"]),
        propagation_floor=float(design["propagation_floor"]),
    )
    return run_hysteresis_experiment(
        config,
        registry_path=design["registry_path"],
        runs_dir=cell["runs_dir"],
        prebuilt_backend=backend_result,
    )


def _select_prompts(keys: Iterable[str]) -> list[dict[str, str]]:
    requested = list(keys)
    table = {prompt["key"]: prompt for prompt in DEFAULT_PROMPTS}
    unknown = sorted(set(requested) - set(table))
    if unknown:
        raise ValueError(f"Unknown prompt keys: {unknown}")
    return [dict(table[key]) for key in requested]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run M1R local branchpoints and MHR matched hysteresis over one "
            "resumable, warm-model calibration grid."
        )
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", default="qwen3-1.7b")
    parser.add_argument("--registry-path", default="models.json")
    parser.add_argument("--prompt-keys", default="sourdough,water_cycle")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--branchpoint-magnitudes", default="0,0.15,0.3")
    parser.add_argument("--hysteresis-magnitudes", default="0,0.1,0.2,0.3")
    parser.add_argument("--branchpoint-max-tokens", type=int, default=48)
    parser.add_argument("--hysteresis-exposure-tokens", type=int, default=20)
    parser.add_argument("--recovery-tokens", type=int, default=16)
    parser.add_argument("--branchpoint-layer", default="late")
    parser.add_argument("--hysteresis-layer", default="mid")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--intervention-seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--noise-start", type=int, default=3)
    parser.add_argument("--noise-duration", type=int, default=8)
    parser.add_argument("--propagation-floor", type=float, default=1e-6)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not scan for already completed matching cells.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.seeds = _parse_ints(args.seeds)
    args.branchpoint_magnitudes = _parse_floats(
        args.branchpoint_magnitudes
    )
    args.hysteresis_magnitudes = _parse_floats(
        args.hysteresis_magnitudes
    )
    args.prompts = _select_prompts(
        key.strip() for key in args.prompt_keys.split(",") if key.strip()
    )
    if len(args.seeds) < 1 or len(args.prompts) < 1:
        raise ValueError("At least one seed and prompt are required")

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "calibration_manifest.json"
    print(f"[calibration] root={root}", flush=True)
    print(f"[calibration] loading model={args.model}", flush=True)
    backend_result = load_model_with_backend(
        model_key=args.model,
        registry_path=args.registry_path,
        backend="hf",
    )
    design = _build_design(args, backend_result=backend_result)
    cells = build_calibration_cells(
        root=root,
        prompts=args.prompts,
        seeds=args.seeds,
        branchpoint_magnitudes=args.branchpoint_magnitudes,
        hysteresis_magnitudes=args.hysteresis_magnitudes,
    )

    existing: dict[str, Any] = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("design") != design:
            raise ValueError(
                "Existing calibration manifest has a different design; "
                "choose a new --root"
            )
    manifest = {
        "protocol": CALIBRATION_PROTOCOL,
        "status": "running",
        "created_at": existing.get("created_at", _utc_now()),
        "updated_at": _utc_now(),
        "root": str(root),
        "design": design,
        "cell_count": len(cells),
        "cells": dict(existing.get("cells") or {}),
    }
    _write_json_atomic(manifest_path, manifest)

    for index, cell in enumerate(cells, start=1):
        print(
            f"[calibration] {index}/{len(cells)} {cell['cell_id']}",
            flush=True,
        )
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
            manifest["cells"][cell["cell_id"]] = {
                **cell,
                **_compact_result(
                    cell,
                    summary,
                    summary_path,
                    disposition=disposition,
                ),
            }
            print(
                f"[calibration]   ok ({disposition}) "
                f"run={summary_path.parent.name}",
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
        f"[calibration] complete cells={manifest['completed_cells']}/"
        f"{manifest['cell_count']}",
        flush=True,
    )
    print(f"[calibration] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
