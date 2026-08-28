#!/usr/bin/env python3
"""Run Observer's resumable Q2 per-layer propagation calibration."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from runtime_lab.cli._common import (
    backend_num_layers,
    resolve_semantic_layer,
)
from runtime_lab.config.schemas import PropagationConfig
from runtime_lab.core.backend.loader import load_model_with_backend
from runtime_lab.core.io.hashing import hash_config
from runtime_lab.propagation.experiment import run_propagation_experiment


CALIBRATION_PROTOCOL = "observer-q2-propagation-calibration-v1"
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
SUPPORTED_INTERVENTIONS = ("additive", "scaling")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def magnitude_slug(value: float) -> str:
    rendered = f"{float(value):.3f}".replace("-", "neg").replace(".", "p")
    return f"magnitude_{rendered}"


def control_magnitude(intervention_type: str) -> float:
    """Return the identity setting for an intervention family."""
    kind = str(intervention_type).lower().strip()
    if kind == "additive":
        return 0.0
    if kind == "scaling":
        return 1.0
    raise ValueError(f"Unsupported Q2 intervention type: {intervention_type}")


def is_control_cell(cell: Mapping[str, Any]) -> bool:
    return float(cell["magnitude"]) == control_magnitude(
        str(cell["intervention_type"])
    )


def build_propagation_cells(
    *,
    root: str | Path,
    prompts: Sequence[Mapping[str, str]],
    seeds: Iterable[int],
    injection_layers: Iterable[int],
    magnitudes_by_type: Mapping[str, Iterable[float]],
) -> list[dict[str, Any]]:
    """Build the deterministic Q2 grid with isolated artifact roots."""
    root = Path(root).expanduser().resolve()
    cells: list[dict[str, Any]] = []
    seed_values = [int(seed) for seed in seeds]
    layer_values = [int(layer) for layer in injection_layers]
    for intervention_type, magnitudes in magnitudes_by_type.items():
        kind = str(intervention_type).lower().strip()
        if kind not in SUPPORTED_INTERVENTIONS:
            raise ValueError(f"Unsupported Q2 intervention type: {kind}")
        for magnitude in magnitudes:
            value = float(magnitude)
            for injection_layer in layer_values:
                runs_dir = (
                    root
                    / kind
                    / magnitude_slug(value)
                    / f"layer_{injection_layer:02d}"
                )
                for prompt in prompts:
                    for seed in seed_values:
                        prompt_key = str(prompt["key"])
                        cells.append(
                            {
                                "cell_id": (
                                    f"{kind}:{magnitude_slug(value)}:"
                                    f"layer_{injection_layer:02d}:"
                                    f"{prompt_key}:seed_{seed}"
                                ),
                                "intervention_type": kind,
                                "magnitude": value,
                                "injection_layer": injection_layer,
                                "prompt_key": prompt_key,
                                "prompt": str(prompt["text"]),
                                "seed": seed,
                                "runs_dir": str(runs_dir),
                            }
                        )
    return cells


def validate_cell_summary(
    cell: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    """Fail closed when a Q2 cell violates paired-context controls."""
    layer = int(cell["injection_layer"])
    protocol = summary.get("protocol_validity") or {}
    metrics = summary.get("metrics") or {}
    terminal_capture = summary.get("terminal_capture") or {}
    capture_layers = [int(value) for value in summary.get("capture_layers") or []]
    epsilon = 1e-12

    assert protocol.get("protocol") == "same-context-layer-propagation-v1", (
        "unexpected propagation protocol"
    )
    assert protocol.get("independent_one_step_forks") is True, (
        "propagation forks were not independent"
    )
    assert protocol.get("continued_clean_only") is True, (
        "perturbed forks contaminated the clean trajectory"
    )
    assert protocol.get("all_contexts_matched") is True, (
        "propagation contexts were not all matched"
    )
    assert protocol.get("all_layers_captured") is True, (
        "not every downstream layer was captured"
    )
    assert int(protocol.get("rows", 0)) > 0, (
        "propagation run emitted no paired decisions"
    )
    assert capture_layers and capture_layers[0] == layer, (
        "capture curve does not begin at the injection layer"
    )
    assert capture_layers == list(
        range(capture_layers[0], capture_layers[-1] + 1)
    ), "capture curve is missing a downstream transformer layer"
    assert terminal_capture.get("available") is True, (
        "final normalization capture is required for Q2"
    )
    assert protocol.get("terminal_capture_complete") is True, (
        "final normalization capture was incomplete"
    )

    injection = (
        (metrics.get("layer_summary") or {})
        .get(str(layer), {})
        .get("delta_l2", {})
    )
    injection_mean = float(injection.get("mean", 0.0))
    if is_control_cell(cell):
        assert abs(float(metrics.get("max_delta_l2", 1.0))) <= epsilon, (
            "identity control produced a hidden-state delta"
        )
        assert int(metrics.get("argmax_flips", -1)) == 0, (
            "identity control produced an argmax flip"
        )
        assert int(metrics.get("sampled_flips", -1)) == 0, (
            "identity control produced a sampled flip"
        )
        assert abs(float(metrics.get("mean_logit_kl", 1.0))) <= epsilon, (
            "identity control produced nonzero logit KL"
        )
        assert abs(float(metrics.get("mean_logit_js", 1.0))) <= epsilon, (
            "identity control produced nonzero logit JS"
        )
    else:
        assert injection_mean > epsilon, (
            "noncontrol intervention had no measurable direct effect"
        )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _expected_config(
    cell: Mapping[str, Any],
    *,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "mode": "propagation",
        "protocol": "same-context-layer-propagation-v1",
        "prompt": cell["prompt"],
        "model": design["model"],
        "max_new_tokens": int(design["max_tokens"]),
        "seed": int(cell["seed"]),
        "temperature": float(design["temperature"]),
        "top_p": float(design["top_p"]),
        "top_k": int(design["top_k"]),
        "intervention_layer_requested": int(cell["injection_layer"]),
        "intervention_layer_resolved": int(cell["injection_layer"]),
        "intervention_type": str(cell["intervention_type"]),
        "intervention_magnitude": float(cell["magnitude"]),
        "intervention_magnitude_relative": True,
        "intervention_seed": int(design["intervention_seed"]),
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
        Path(cell["runs_dir"]).glob("propagation_run_*"),
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
    layer = str(cell["injection_layer"])
    return {
        "status": "ok",
        "disposition": disposition,
        "completed_at": _utc_now(),
        "summary_path": str(summary_path),
        "run_dir": str(summary.get("run_dir") or summary_path.parent),
        "config_hash": summary.get("config_hash"),
        "decision_count": int(
            (summary.get("protocol_validity") or {}).get("rows", 0)
        ),
        "metrics": {
            "argmax_flips": int(metrics.get("argmax_flips", 0)),
            "sampled_flips": int(metrics.get("sampled_flips", 0)),
            "mean_logit_kl": float(metrics.get("mean_logit_kl", 0.0)),
            "mean_logit_js": float(metrics.get("mean_logit_js", 0.0)),
            "max_delta_l2": float(metrics.get("max_delta_l2", 0.0)),
            "injection_layer": (
                (metrics.get("layer_summary") or {}).get(layer) or {}
            ),
            "final_norm": (
                (metrics.get("terminal_summary") or {}).get("final_norm")
                or {}
            ),
        },
    }


def _select_prompts(keys: Iterable[str]) -> list[dict[str, str]]:
    requested = list(keys)
    table = {prompt["key"]: prompt for prompt in DEFAULT_PROMPTS}
    unknown = sorted(set(requested) - set(table))
    if unknown:
        raise ValueError(f"Unknown prompt keys: {unknown}")
    return [dict(table[key]) for key in requested]


def _resolve_layers(
    requested_layers: Iterable[str],
    *,
    num_layers: int,
) -> list[int]:
    resolved: list[int] = []
    for requested in requested_layers:
        layer = resolve_semantic_layer(str(requested), num_layers)
        if layer not in resolved:
            resolved.append(layer)
    if not resolved:
        raise ValueError("At least one injection layer is required")
    return resolved


def _build_design(
    args: argparse.Namespace,
    *,
    backend_result: Any,
) -> dict[str, Any]:
    num_layers = backend_num_layers(backend_result)
    injection_layers = _resolve_layers(args.layers, num_layers=num_layers)
    return {
        "protocol": CALIBRATION_PROTOCOL,
        "measurement_protocol": "same-context-layer-propagation-v1",
        "model": args.model,
        "registry_path": str(Path(args.registry_path).resolve()),
        "loaded_model_key": backend_result.config.key,
        "loaded_model_hf_id": backend_result.config.hf_id,
        "backend": backend_result.backend,
        "backend_meta": dict(backend_result.backend_meta),
        "num_layers": num_layers,
        "prompts": list(args.prompts),
        "seeds": list(args.seeds),
        "layers_requested": list(args.layers),
        "injection_layers": injection_layers,
        "magnitudes_by_type": {
            "additive": list(args.additive_magnitudes),
            "scaling": list(args.scaling_magnitudes),
        },
        "identity_controls": {
            kind: control_magnitude(kind)
            for kind in SUPPORTED_INTERVENTIONS
        },
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "intervention_seed": int(args.intervention_seed),
    }


def _run_cell(
    cell: Mapping[str, Any],
    *,
    design: Mapping[str, Any],
    backend_result: Any,
) -> dict[str, Any]:
    config = PropagationConfig(
        prompt=str(cell["prompt"]),
        model_key=str(design["model"]),
        max_new_tokens=int(design["max_tokens"]),
        backend=str(design["backend"]),
        seed=int(cell["seed"]),
        temperature=float(design["temperature"]),
        top_p=float(design["top_p"]),
        top_k=int(design["top_k"]),
        intervention_layer=int(cell["injection_layer"]),
        intervention_type=str(cell["intervention_type"]),
        intervention_magnitude=float(cell["magnitude"]),
        intervention_magnitude_relative=True,
        intervention_seed=int(design["intervention_seed"]),
    )
    return run_propagation_experiment(
        config,
        registry_path=str(design["registry_path"]),
        runs_dir=str(cell["runs_dir"]),
        prebuilt_backend=backend_result,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Q2 same-context per-layer propagation over one resumable, "
            "warm-model calibration grid."
        )
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", default="qwen3-1.7b")
    parser.add_argument("--registry-path", default="models.json")
    parser.add_argument("--prompt-keys", default="sourdough,water_cycle")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--layers", default="early,mid,mid+,late")
    parser.add_argument("--additive-magnitudes", default="0,0.1,0.2,0.3")
    parser.add_argument("--scaling-magnitudes", default="1,0.5,1.5")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--intervention-seed", type=int, default=42)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not scan for already completed matching cells.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.seeds = _parse_ints(args.seeds)
    args.layers = [
        item.strip() for item in args.layers.split(",") if item.strip()
    ]
    args.additive_magnitudes = _parse_floats(args.additive_magnitudes)
    args.scaling_magnitudes = _parse_floats(args.scaling_magnitudes)
    args.prompts = _select_prompts(
        key.strip() for key in args.prompt_keys.split(",") if key.strip()
    )
    if not args.seeds or not args.prompts:
        raise ValueError("At least one seed and prompt are required")
    if control_magnitude("additive") not in args.additive_magnitudes:
        raise ValueError("Additive Q2 calibration requires magnitude 0 control")
    if control_magnitude("scaling") not in args.scaling_magnitudes:
        raise ValueError("Scaling Q2 calibration requires scale 1 control")

    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "propagation_manifest.json"
    print(f"[q2] root={root}", flush=True)
    print(f"[q2] loading model={args.model}", flush=True)
    backend_result = load_model_with_backend(
        model_key=args.model,
        registry_path=args.registry_path,
        backend="hf",
    )
    design = _build_design(args, backend_result=backend_result)
    cells = build_propagation_cells(
        root=root,
        prompts=args.prompts,
        seeds=args.seeds,
        injection_layers=design["injection_layers"],
        magnitudes_by_type=design["magnitudes_by_type"],
    )

    existing: dict[str, Any] = {}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("design") != design:
            raise ValueError(
                "Existing propagation manifest has a different design; "
                "choose a new --root"
            )
    manifest: dict[str, Any] = {
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
        print(f"[q2] {index}/{len(cells)} {cell['cell_id']}", flush=True)
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
            injection_mean = (
                manifest["cells"][cell["cell_id"]]["metrics"][
                    "injection_layer"
                ]
                .get("delta_l2", {})
                .get("mean", 0.0)
            )
            final_norm_mean = (
                manifest["cells"][cell["cell_id"]]["metrics"]["final_norm"]
                .get("delta_l2", {})
                .get("mean", 0.0)
            )
            print(
                f"[q2]   ok ({disposition}) "
                f"injection={float(injection_mean):.6g} "
                f"final_norm={float(final_norm_mean):.6g}",
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
        f"[q2] complete cells={manifest['completed_cells']}/"
        f"{manifest['cell_count']}",
        flush=True,
    )
    print(f"[q2] manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
