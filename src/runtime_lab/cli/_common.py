"""Shared CLI helpers.

Probe-layer parsing, seed-sweep iteration, semantic layer defaults, and
temperature-sampling args all live here so each mode's parser stays tight.

Semantic layer spec strings supported anywhere an int layer is accepted:

    "mid"   -> n // 2
    "mid-"  -> max(0, n//2 - 2)
    "mid+"  -> min(n-1, n//2 + 2)
    "late"  -> n - 1
    "early" -> max(0, n // 4 - 1)

These resolve at runtime (when the model is known) and the resolved value is
written back into the run summary so the LLM driving the next experiment can
see what "mid" actually was for this model.
"""
from __future__ import annotations

import argparse
import re
from typing import Any, List, Optional, Union


SEMANTIC_LAYER_ALIASES = {"mid", "mid-", "mid+", "late", "early", "auto"}


def add_probe_layers_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--probe-layers",
        default="auto",
        help=(
            "Layers to instrument for diagnostics. 'auto' picks mid-stack + "
            "final: [n//4, n//2, 3n//4, -1]. Or pass comma-separated ints "
            "(e.g. '2,6,10,14'). Negative indices count from the end."
        ),
    )


def add_seed_sweep_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "Run N seeds for aggregation. Either comma-separated "
            "('1,2,3,4,5') or a range ('0-9'). When omitted, a single run "
            "uses --seed."
        ),
    )


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature for token selection. 0 = greedy (default, "
            "legacy behavior). Use 0.7–1.0 to let logit-level perturbations "
            "actually flip tokens — essential for measuring stress/noise effects."
        ),
    )
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling cutoff.")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling cap (0 = off).")


def parse_seeds(spec: Optional[str]) -> List[int]:
    if not spec:
        return []
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        m = re.match(r"^(-?\d+)\s*-\s*(-?\d+)$", spec)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            return list(range(a, b + 1))
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return [int(p) for p in parts]


def resolve_semantic_layer(spec: Union[str, int], num_layers: Optional[int]) -> int:
    """Resolve one layer to a validated, non-negative model index."""
    if num_layers is None:
        raise ValueError("num_layers is required to resolve a layer")
    n = int(num_layers)
    if n <= 0:
        raise ValueError(f"num_layers must be positive, got {n}")

    if isinstance(spec, int):
        idx = int(spec)
        if idx < 0:
            idx += n
        if not 0 <= idx < n:
            raise ValueError(f"layer {spec!r} is out of range for num_layers={n}")
        return idx

    s = str(spec).strip().lower()
    if s == "" or s.lstrip("-").isdigit():
        return resolve_semantic_layer(int(s), n)
    mapping = {
        "mid":   max(0, n // 2 - 1),
        "mid-":  max(0, n // 2 - 3),
        "mid+":  min(n - 1, n // 2 + 1),
        "late":  n - 1,
        "early": max(0, n // 4 - 1),
        "auto":  max(0, n // 2 - 1),
    }
    if s not in mapping:
        raise ValueError(
            f"unknown layer specification {spec!r}; expected an integer or "
            f"one of {sorted(SEMANTIC_LAYER_ALIASES)}"
        )
    return mapping[s]


def resolve_probe_layers(spec: str, num_layers: Optional[int]) -> List[int]:
    """Resolve the --probe-layers argument against a concrete layer count.

    Every result is a validated, non-negative model index."""
    if num_layers is None:
        raise ValueError("num_layers is required to resolve probe layers")
    n = int(num_layers)
    if n <= 0:
        raise ValueError(f"num_layers must be positive, got {n}")

    spec = (spec or "").strip()
    if spec == "" or spec.lower() == "auto":
        return sorted({max(0, n // 4 - 1), max(0, n // 2 - 1), max(0, 3 * n // 4 - 1), n - 1})
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    resolved = [resolve_semantic_layer(int(part), n) for part in parts]
    return list(dict.fromkeys(resolved))


def backend_num_layers(backend_result: Any) -> int:
    from runtime_lab.core.model.layers import resolve_transformer_layers

    layers = resolve_transformer_layers(backend_result.model)
    num_layers = int(len(layers))
    if num_layers <= 0:
        raise ValueError("loaded backend exposes no transformer layers")
    return num_layers


def load_backend_from_args(args: Any):
    from runtime_lab.core.backend.loader import load_model_with_backend

    return load_model_with_backend(
        model_key=getattr(args, "model", None),
        registry_path=getattr(args, "registry_path", "models.json"),
        backend=getattr(args, "backend", "hf"),
        nnsight_remote=bool(getattr(args, "nnsight_remote", False)),
        nnsight_device=getattr(args, "nnsight_device", None),
    )


def describe_layer_resolution(raw: Union[str, int], resolved: int, num_layers: Optional[int]) -> str:
    if isinstance(raw, str) and raw.lower() in SEMANTIC_LAYER_ALIASES:
        if num_layers is not None:
            return f"{raw} -> layer {resolved} (of {num_layers})"
        return f"{raw} (deferred)"
    return str(resolved)
