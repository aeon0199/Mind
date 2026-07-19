from __future__ import annotations

import argparse

from runtime_lab.basins.evaluators import (
    BasinPromptSpec,
    build_generic_prompt_spec,
    get_prompt_spec,
)
from runtime_lab.basins.experiment import run_basin_experiment
from runtime_lab.config.schemas import BasinConfig, DiagnosticsConfig
from ._common import (
    add_probe_layers_arg,
    add_sampling_args,
    add_seed_sweep_arg,
    backend_num_layers,
    load_backend_from_args,
    parse_seeds,
    resolve_probe_layers,
    resolve_semantic_layer,
)
from ._sweep import run_sweep


def add_basin_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--prompt-key",
        default="water_cycle",
        help=(
            "Registered M3R prompt key, or 'custom' for the generic "
            "exploratory rubric."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Must match a registered prompt exactly; required when "
            "--prompt-key custom."
        ),
    )
    parser.add_argument("--prompt-class", default="custom")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--continuation-tokens", type=int, default=48)
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument(
        "--layer",
        default="late",
        help="Intervention layer: integer or mid/mid-/mid+/late/early.",
    )
    parser.add_argument(
        "--type",
        default="additive",
        choices=["additive", "scaling", "projection"],
    )
    parser.add_argument("--magnitude", type=float, default=0.30)
    parser.add_argument("--absolute-magnitude", action="store_true")
    parser.add_argument("--intervention-seed", type=int, default=42)
    parser.add_argument("--projection-subspace-dim", type=int, default=1)
    parser.add_argument("--tie-tolerance", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="hf", choices=["hf", "nnsight"])
    parser.add_argument("--nnsight-remote", action="store_true")
    parser.add_argument("--nnsight-device", default=None)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--registry-path", default="models.json")
    parser.add_argument("--runs-dir", default=None)
    add_probe_layers_arg(parser)
    add_sampling_args(parser)
    add_seed_sweep_arg(parser)


def _resolve_prompt_spec(args: argparse.Namespace) -> BasinPromptSpec:
    key = str(args.prompt_key)
    if key == "custom":
        if not args.prompt:
            raise ValueError("--prompt is required when --prompt-key custom")
        return build_generic_prompt_spec(
            str(args.prompt),
            key="custom",
            prompt_class=str(args.prompt_class),
        )
    spec = get_prompt_spec(key)
    if args.prompt is not None and str(args.prompt) != spec.prompt:
        raise ValueError(
            f"--prompt does not match registered basin prompt {key!r}"
        )
    return spec


def _run_one(args, seed: int, backend_result=None):
    spec = _resolve_prompt_spec(args)
    backend_result = backend_result or load_backend_from_args(args)
    num_layers = backend_num_layers(backend_result)
    resolved_layer = resolve_semantic_layer(args.layer, num_layers)
    probe_layers = resolve_probe_layers(args.probe_layers, num_layers)
    if resolved_layer not in probe_layers:
        probe_layers = [resolved_layer, *probe_layers]
    config = BasinConfig(
        prompt=spec.prompt,
        prompt_key=spec.key,
        prompt_class=spec.prompt_class,
        evaluator_id=spec.evaluator_id,
        model_key=args.model,
        max_new_tokens=int(args.max_tokens),
        continuation_tokens=int(args.continuation_tokens),
        max_episodes=int(args.max_episodes),
        backend=args.backend,
        nnsight_remote=bool(args.nnsight_remote),
        nnsight_device=args.nnsight_device,
        seed=int(seed),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        intervention_layer=resolved_layer,
        intervention_type=str(args.type),
        intervention_magnitude=float(args.magnitude),
        intervention_magnitude_relative=not bool(
            args.absolute_magnitude
        ),
        intervention_seed=int(args.intervention_seed),
        projection_subspace_dim=max(
            1,
            int(args.projection_subspace_dim),
        ),
        tie_tolerance=float(args.tie_tolerance),
        with_diagnostics=not bool(args.no_diagnostics),
    )
    diagnostics = DiagnosticsConfig(
        enabled=not bool(args.no_diagnostics),
        probe_layers=probe_layers if not args.no_diagnostics else [],
    )
    return run_basin_experiment(
        config,
        registry_path=args.registry_path,
        runs_dir=args.runs_dir,
        diagnostics_config=diagnostics,
        prebuilt_backend=backend_result,
        prompt_spec=spec,
    )


def run_from_args(args) -> None:
    spec = _resolve_prompt_spec(args)
    backend_result = load_backend_from_args(args)
    num_layers = backend_num_layers(backend_result)
    seeds = parse_seeds(getattr(args, "seeds", None))
    if not seeds:
        _run_one(args, int(args.seed), backend_result)
        return
    run_sweep(
        mode="basins",
        seeds=seeds,
        runs_dir=args.runs_dir,
        run_once=lambda seed: _run_one(args, seed, backend_result),
        describe={
            "prompt": spec.prompt,
            "prompt_key": spec.key,
            "prompt_class": spec.prompt_class,
            "model": args.model,
            "max_tokens": int(args.max_tokens),
            "continuation_tokens": int(args.continuation_tokens),
            "max_episodes": int(args.max_episodes),
            "layer": {
                "raw": args.layer,
                "resolved": resolve_semantic_layer(
                    args.layer,
                    num_layers,
                ),
            },
            "type": args.type,
            "magnitude": float(args.magnitude),
            "relative": not bool(args.absolute_magnitude),
        },
    )
