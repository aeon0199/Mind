from __future__ import annotations

from typing import Any


def resolved_model_identity(
    requested: str | None,
    backend_result: Any,
) -> dict[str, str | None]:
    """Return the loaded model identity and reject prebuilt-backend mislabeling."""
    config = getattr(backend_result, "config", None)
    loaded_key = getattr(config, "key", None)
    hf_id = getattr(config, "hf_id", None)
    if not loaded_key:
        raise ValueError("loaded backend does not expose an authoritative model key")

    loaded_key = str(loaded_key)
    requested_key = None if requested is None else str(requested)
    if requested_key is not None and requested_key != loaded_key:
        raise ValueError(
            f"requested model {requested_key!r} does not match loaded model {loaded_key!r}"
        )

    return {
        "model": loaded_key,
        "model_requested": requested_key,
        "model_hf_id": None if hf_id is None else str(hf_id),
    }
