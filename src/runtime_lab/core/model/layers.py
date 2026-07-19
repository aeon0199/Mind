from __future__ import annotations

from typing import Any, Dict, List


def resolve_transformer_layers(model) -> Any:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("transformer", "blocks"),
        ("model", "decoder", "layers"),
    ]

    for path in candidates:
        cur = model
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok:
            return cur

    raise AttributeError("Could not locate transformer layers on model")


def resolve_final_norm_module(model) -> tuple[str | None, Any | None]:
    candidates = [
        ("model", "norm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
        ("model", "decoder", "final_layer_norm"),
        ("transformer", "norm_f"),
    ]
    for path in candidates:
        current = model
        for attribute in path:
            if not hasattr(current, attribute):
                break
            current = getattr(current, attribute)
        else:
            return ".".join(path), current
    return None, None


def resolve_layer_index(requested_idx: int, num_layers: int) -> int:
    idx = int(requested_idx)
    if idx < 0:
        idx = num_layers + idx
    if not (0 <= idx < num_layers):
        raise IndexError(
            f"Layer index out of range: requested={requested_idx}, resolved={idx}, num_layers={num_layers}"
        )
    return idx


def resolve_probe_layers(requested: List[int], num_layers: int) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for raw in requested:
        out[int(raw)] = resolve_layer_index(int(raw), num_layers)
    return out
