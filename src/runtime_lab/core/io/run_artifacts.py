from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from runtime_lab.core.io.hashing import hash_config
from runtime_lab.core.io.json import json_safe, save_json


def create_run_dir(runs_root: str | Path, mode: str) -> tuple[str, Path]:
    """Allocate a collision-resistant experiment directory without reusing data."""
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    normalized_mode = str(mode).strip().lower()
    if not normalized_mode:
        raise ValueError("mode must not be empty")

    for _ in range(8):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_id = f"{normalized_mode}_run_{stamp}_{uuid4().hex[:8]}"
        run_dir = root / run_id
        try:
            run_dir.mkdir(exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            continue

    raise FileExistsError(f"Could not allocate a unique {normalized_mode!r} run directory")


def write_config_record(run_dir: str | Path, config: dict[str, Any]) -> tuple[str, Path]:
    """Write and hash the exact JSON-safe resolved configuration."""
    path = Path(run_dir) / "config.json"
    resolved_config = json_safe(config)
    config_hash = hash_config(resolved_config)
    save_json(str(path), {"config_hash": config_hash, "config": resolved_config})
    return config_hash, path
