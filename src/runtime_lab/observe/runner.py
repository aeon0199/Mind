from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from runtime_lab.config.schemas import CommonRunConfig, DiagnosticsConfig
from runtime_lab.core.backend.identity import resolved_model_identity
from runtime_lab.core.backend.loader import load_model_with_backend
from runtime_lab.core.diagnostics.manager import DiagnosticsManager, summarize_diagnostics_health
from runtime_lab.core.io.json import save_json
from runtime_lab.core.io.run_artifacts import create_run_dir, write_config_record
from runtime_lab.core.runtime.engine import RuntimeEngine
from runtime_lab.core.runtime.events import runtime_event_to_record


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def run_observe_experiment(
    config: CommonRunConfig,
    registry_path: str = "models.json",
    runs_dir: Optional[str] = None,
    diagnostics_config: Optional[DiagnosticsConfig] = None,
    prebuilt_backend: Optional[Any] = None,
) -> Dict[str, Any]:
    _set_seed(config.seed)

    if prebuilt_backend is not None:
        backend_result = prebuilt_backend
    else:
        backend_result = load_model_with_backend(
            model_key=config.model_key,
            registry_path=registry_path,
            backend=config.backend,
            nnsight_remote=config.nnsight_remote,
            nnsight_device=config.nnsight_device,
        )

    tokenizer = backend_result.tokenizer
    model = backend_result.model
    device = backend_result.device
    model_identity = resolved_model_identity(config.model_key, backend_result)
    model_id = str(model_identity["model"])

    diag_cfg = diagnostics_config or DiagnosticsConfig(
        enabled=True,
        probe_layers=[-1],
    )
    diagnostics = DiagnosticsManager(diag_cfg)

    engine = RuntimeEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_idx=-1,
        diagnostics_manager=diagnostics,
        intervention=None,
        probe_layers=diag_cfg.probe_layers,
        mode="observe",
        temperature=float(getattr(config, "temperature", 0.0)),
        top_p=float(getattr(config, "top_p", 1.0)),
        top_k=int(getattr(config, "top_k", 0)),
    )

    run_config = {
        "mode": "observe",
        "prompt": config.prompt,
        **model_identity,
        "backend": backend_result.backend,
        "backend_meta": dict(backend_result.backend_meta),
        "max_new_tokens": int(config.max_new_tokens),
        "seed": int(config.seed) if config.seed is not None else None,
        "temperature": float(getattr(config, "temperature", 0.0)),
        "top_p": float(getattr(config, "top_p", 1.0)),
        "top_k": int(getattr(config, "top_k", 0)),
        "probe_layers_resolved": list(dict.fromkeys(engine.probe_resolved.values())),
        "measure_resolved_layer_idx": int(engine.measure_resolved_layer_idx),
        "act_resolved_layer_idx": int(engine.act_resolved_layer_idx),
        "num_layers": int(engine.num_layers),
    }

    run_id, run_dir = create_run_dir(
        runs_dir or os.environ.get("RUNS_DIR", "runs"),
        "observe",
    )

    events_path = Path(run_dir) / "events.jsonl"
    summary_path = Path(run_dir) / "summary.json"
    output_path = Path(run_dir) / "output.txt"
    cfg_hash, config_path = write_config_record(run_dir, run_config)

    generated_text = ""
    events: list[Dict[str, Any]] = []

    n_tokens = 0
    sum_div = 0.0

    try:
        prefill = engine.prefill(config.prompt, intervention_active=False)
        pending_token_id = int(prefill.next_token_id)
        prompt_len = int(prefill.prompt_len)
        past_key_values = prefill.past_key_values
        eos = getattr(tokenizer, "eos_token_id", None)

        with open(events_path, "w", encoding="utf-8") as events_f:
            for t in range(int(config.max_new_tokens)):
                consumed_token_id = int(pending_token_id)

                step = engine.step(
                    t=t,
                    consumed_token_id=consumed_token_id,
                    prompt_len=prompt_len,
                    past_key_values=past_key_values,
                    intervention_active=False,
                )

                past_key_values = step.past_key_values
                pending_token_id = int(step.predicted_next_token_id)

                event = runtime_event_to_record(step.event)

                events_f.write(json.dumps(event, ensure_ascii=False) + "\n")
                events.append(event)

                generated_text += step.consumed_token_text
                n_tokens += 1
                sum_div += float(step.diagnostics.get("divergence", 0.0))

                if eos is not None and int(consumed_token_id) == int(eos):
                    break

    finally:
        engine.close()

    full_text = config.prompt + generated_text
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    summary = {
        "config_hash": cfg_hash,
        "mode": "observe",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_id": model_id,
        "backend": backend_result.backend,
        "runtime": {
            "device": str(device),
            "resolved_dtype": backend_result.backend_meta.get("resolved_dtype"),
            "policy_notes": backend_result.backend_meta.get("policy_notes", []),
            "backend_meta": dict(backend_result.backend_meta),
        },
        "tokens": int(n_tokens),
        "avg_divergence": float(sum_div / max(1, n_tokens)),
        "diagnostics_health": summarize_diagnostics_health([event.get("diagnostics", {}) for event in events]),
        "artifacts": {
            "run_dir": str(run_dir),
            "events_path": str(events_path),
            "output_path": str(output_path),
            "summary_path": str(summary_path),
            "config_path": str(config_path),
        },
    }

    try:
        from runtime_lab.core.advisory import analyze as _analyze_advisory
        summary["advisory"] = _analyze_advisory("observe", summary)
    except Exception as e:
        summary["advisory"] = {"error": f"advisory failed: {e}"}

    save_json(str(summary_path), summary)
    return summary
