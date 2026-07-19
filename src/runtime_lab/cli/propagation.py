from __future__ import annotations

import argparse

from runtime_lab.config.schemas import PropagationConfig
from runtime_lab.propagation.experiment import run_propagation_experiment
from ._common import (
    add_sampling_args,
    add_seed_sweep_arg,
    backend_num_layers,
    load_backend_from_args,
    parse_seeds,
    resolve_semantic_layer,
)
from ._sweep import run_sweep


def add_propagation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--prompt",
        default="Explain how airplanes fly in a clear, accurate way.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--layer",
        default="mid",
        help="Injection layer: integer or mid/mid-/mid+/late/early.",
    )
    parser.add_argument(
        "--type",
        default="additive",
        choices=["additive", "scaling", "projection"],
    )
    parser.add_argument("--magnitude", type=float, default=0.15)
    parser.add_argument("--absolute-magnitude", action="store_true")
    parser.add_argument("--intervention-seed", type=int, default=42)
    parser.add_argument("--projection-subspace-dim", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", default="hf", choices=["hf", "nnsight"])
    parser.add_argument("--nnsight-remote", action="store_true")
    parser.add_argument("--nnsight-device", default=None)
    parser.add_argument("--registry-path", default="models.json")
    parser.add_argument("--runs-dir", default=None)
    add_sampling_args(parser)
    add_seed_sweep_arg(parser)


def _run_one(args, seed: int, backend_result=None):
    backend_result = backend_result or load_backend_from_args(args)
    num_layers = backend_num_layers(backend_result)
    config = PropagationConfig(
        prompt=args.prompt,
        model_key=args.model,
        max_new_tokens=int(args.max_tokens),
        backend=args.backend,
        nnsight_remote=bool(args.nnsight_remote),
        nnsight_device=args.nnsight_device,
        seed=int(seed),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        intervention_layer=resolve_semantic_layer(
            args.layer,
            num_layers,
        ),
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
    )
    return run_propagation_experiment(
        config,
        registry_path=args.registry_path,
        runs_dir=args.runs_dir,
        prebuilt_backend=backend_result,
    )


def run_from_args(args) -> None:
    backend_result = load_backend_from_args(args)
    num_layers = backend_num_layers(backend_result)
    seeds = parse_seeds(getattr(args, "seeds", None))
    if not seeds:
        _run_one(args, int(args.seed), backend_result)
        return
    run_sweep(
        mode="propagation",
        seeds=seeds,
        runs_dir=args.runs_dir,
        run_once=lambda seed: _run_one(args, seed, backend_result),
        describe={
            "prompt": args.prompt,
            "model": args.model,
            "max_tokens": int(args.max_tokens),
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
