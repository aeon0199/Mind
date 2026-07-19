from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from runtime_lab.config.schemas import BranchpointConfig, DiagnosticsConfig
from runtime_lab.core.backend.identity import resolved_model_identity
from runtime_lab.core.backend.loader import load_model_with_backend
from runtime_lab.core.diagnostics.manager import (
    DiagnosticsManager,
    summarize_diagnostics_health,
)
from runtime_lab.core.interventions.factory import build_intervention
from runtime_lab.core.io.json import json_safe, save_json
from runtime_lab.core.io.run_artifacts import create_run_dir, write_config_record
from runtime_lab.core.model.cache_utils import (
    cache_sequence_length,
    compute_cache_fingerprint,
)
from runtime_lab.core.runtime.engine import RuntimeEngine
from runtime_lab.core.sampling import logit_jensen_shannon, logit_kl
from runtime_lab.core.trajectory.generation import generation_state_from_seed_cache
from runtime_lab.stress.seed_cache import build_seed_cache


def _set_deterministic_state(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        import numpy as np
        import random

        np.random.seed(int(seed))
        random.seed(int(seed))
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _top1_margin(logits: torch.Tensor) -> float:
    values = torch.topk(logits.detach().float().reshape(-1), k=min(2, logits.numel())).values
    if values.numel() < 2:
        return 0.0
    return float((values[0] - values[1]).item())


def _argmax_token(logits: torch.Tensor) -> int:
    return int(logits.detach().float().reshape(-1).argmax().item())


def _hidden_distances(
    clean_hidden: Optional[torch.Tensor],
    perturbed_hidden: Optional[torch.Tensor],
) -> tuple[float | None, float | None]:
    if not isinstance(clean_hidden, torch.Tensor) or not isinstance(perturbed_hidden, torch.Tensor):
        return None, None
    clean = clean_hidden.detach().float().reshape(-1)
    perturbed = perturbed_hidden.detach().float().reshape(-1)
    if clean.numel() != perturbed.numel() or clean.numel() == 0:
        return None, None
    clean_norm = float(clean.norm().item())
    perturbed_norm = float(perturbed.norm().item())
    if clean_norm <= 1e-12 or perturbed_norm <= 1e-12:
        cosine_distance = None
    else:
        cosine = float(torch.dot(clean, perturbed).item()) / (
            clean_norm * perturbed_norm
        )
        cosine_distance = float(1.0 - max(-1.0, min(1.0, cosine)))
    relative_l2 = float((clean - perturbed).norm().item()) / max(clean_norm, 1e-12)
    return cosine_distance, relative_l2


def _build_intervention(config: BranchpointConfig, kwargs: Dict[str, Any]):
    kind = str(config.intervention_type).lower().strip()
    common = dict(kwargs)
    common.setdefault("seed", int(config.intervention_seed))
    if kind == "scaling":
        common.setdefault("scale", float(config.intervention_magnitude))
    elif kind == "projection":
        common.setdefault(
            "subspace_dim",
            max(1, int(config.projection_subspace_dim)),
        )
    else:
        common.setdefault("magnitude", float(config.intervention_magnitude))
        common.setdefault(
            "relative",
            bool(config.intervention_magnitude_relative),
        )
    return build_intervention(kind, **common)


def run_branchpoint_experiment(
    config: BranchpointConfig,
    registry_path: str = "models.json",
    runs_dir: Optional[str] = None,
    diagnostics_config: Optional[DiagnosticsConfig] = None,
    intervention_kwargs: Optional[Dict[str, Any]] = None,
    prebuilt_backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """Measure independent one-step counterfactuals along one clean trajectory."""
    _set_deterministic_state(config.seed)

    backend_result = prebuilt_backend or load_model_with_backend(
        model_key=config.model_key,
        registry_path=registry_path,
        backend=config.backend,
        nnsight_remote=config.nnsight_remote,
        nnsight_device=config.nnsight_device,
    )
    identity = resolved_model_identity(config.model_key, backend_result)
    model = backend_result.model
    tokenizer = backend_result.tokenizer
    device = backend_result.device

    seed_cache = build_seed_cache(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=config.prompt,
        intervention_layer=int(config.intervention_layer),
    )
    resolved_layer = int(seed_cache.resolved_layer_idx)
    diag_cfg = diagnostics_config or DiagnosticsConfig(
        enabled=bool(config.with_diagnostics),
        probe_layers=[resolved_layer] if config.with_diagnostics else [],
    )
    diagnostics = DiagnosticsManager(diag_cfg)
    intervention = _build_intervention(config, intervention_kwargs or {})

    engine = RuntimeEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_idx=resolved_layer,
        diagnostics_manager=diagnostics,
        intervention=intervention,
        probe_layers=diag_cfg.probe_layers,
        mode="branchpoints",
        temperature=float(config.temperature),
        top_p=float(config.top_p),
        top_k=int(config.top_k),
    )

    run_config = {
        "mode": "branchpoints",
        "prompt": config.prompt,
        **identity,
        "backend": backend_result.backend,
        "backend_meta": dict(backend_result.backend_meta),
        "max_new_tokens": int(config.max_new_tokens),
        "seed": int(config.seed) if config.seed is not None else None,
        "temperature": float(config.temperature),
        "top_p": float(config.top_p),
        "top_k": int(config.top_k),
        "intervention_layer_requested": int(config.intervention_layer),
        "intervention_layer_resolved": resolved_layer,
        "intervention_type": str(config.intervention_type),
        "intervention_magnitude": float(config.intervention_magnitude),
        "intervention_magnitude_relative": bool(config.intervention_magnitude_relative),
        "intervention_seed": int(config.intervention_seed),
        "projection_subspace_dim": int(config.projection_subspace_dim),
        "with_diagnostics": bool(config.with_diagnostics),
        "probe_layers_resolved": list(dict.fromkeys(engine.probe_resolved.values())),
        "num_layers": int(engine.num_layers),
        "seed_cache_fingerprint": seed_cache.fingerprint,
        "prompt_tokens": int(seed_cache.seq_len),
    }
    run_id, run_dir = create_run_dir(
        runs_dir or os.environ.get("RUNS_DIR", "runs"),
        "branchpoint",
    )
    config_hash, config_path = write_config_record(run_dir, run_config)
    events_path = Path(run_dir) / "events.jsonl"
    output_path = Path(run_dir) / "output.txt"
    summary_path = Path(run_dir) / "summary.json"

    rows: list[Dict[str, Any]] = []
    state = None
    try:
        if int(config.max_new_tokens) > 0:
            state = generation_state_from_seed_cache(
                seed_cache,
                temperature=float(config.temperature),
                top_p=float(config.top_p),
                top_k=int(config.top_k),
            )

        eos = getattr(tokenizer, "eos_token_id", None)
        with events_path.open("w", encoding="utf-8") as events_file:
            while (
                state is not None
                and state.next_decision_index < int(config.max_new_tokens)
            ):
                consumed_token_id = int(state.pending_token_id)
                if eos is not None and consumed_token_id == int(eos):
                    break

                source_fingerprint = compute_cache_fingerprint(state.past_key_values)
                clean_cache = state.fork_cache()
                perturbed_cache = state.fork_cache()
                clean_fingerprint = compute_cache_fingerprint(clean_cache)
                perturbed_fingerprint = compute_cache_fingerprint(perturbed_cache)
                source_length = cache_sequence_length(state.past_key_values)
                contexts_matched = bool(
                    source_fingerprint not in {"unavailable", "empty"}
                    and source_fingerprint == clean_fingerprint == perturbed_fingerprint
                    and cache_sequence_length(clean_cache) == source_length
                    and cache_sequence_length(perturbed_cache) == source_length
                )
                if not contexts_matched:
                    raise RuntimeError(
                        "Could not prove that clean and perturbed branchpoint contexts match"
                    )

                decision_index = int(state.next_decision_index)
                rng_before = torch.random.get_rng_state()
                clean_step = engine.step(
                    t=decision_index,
                    consumed_token_id=consumed_token_id,
                    prompt_len=int(state.prompt_len),
                    past_key_values=clean_cache,
                    intervention_active=False,
                    collect_diagnostics=True,
                )
                clean_rng_after = torch.random.get_rng_state()
                torch.random.set_rng_state(rng_before)
                try:
                    perturbed_step = engine.step(
                        t=decision_index,
                        consumed_token_id=consumed_token_id,
                        prompt_len=int(state.prompt_len),
                        past_key_values=perturbed_cache,
                        intervention_active=True,
                        collect_diagnostics=False,
                    )
                finally:
                    torch.random.set_rng_state(clean_rng_after)

                clean_argmax = _argmax_token(clean_step.logits)
                perturbed_argmax = _argmax_token(perturbed_step.logits)
                clean_hidden = clean_step.hidden_post
                perturbed_hidden = perturbed_step.hidden_post
                cosine_distance, relative_l2 = _hidden_distances(
                    clean_hidden,
                    perturbed_hidden,
                )
                pre_norm = float(perturbed_step.event.hidden_pre_norm)
                delta_norm = float(perturbed_step.event.hidden_delta_norm)

                row = {
                    "mode": "branchpoints",
                    "decision_index": decision_index,
                    "t": decision_index,
                    "context_fingerprint": source_fingerprint,
                    "clean_context_fingerprint": clean_fingerprint,
                    "perturbed_context_fingerprint": perturbed_fingerprint,
                    "context_sequence_length": source_length,
                    "contexts_matched": contexts_matched,
                    "sampling_rng_matched": True,
                    "consumed_token_id": consumed_token_id,
                    "consumed_token_text": clean_step.consumed_token_text,
                    "clean_predicted_token_id": int(clean_step.predicted_next_token_id),
                    "clean_predicted_token_text": tokenizer.decode(
                        [int(clean_step.predicted_next_token_id)],
                        skip_special_tokens=False,
                    ),
                    "perturbed_predicted_token_id": int(
                        perturbed_step.predicted_next_token_id
                    ),
                    "perturbed_predicted_token_text": tokenizer.decode(
                        [int(perturbed_step.predicted_next_token_id)],
                        skip_special_tokens=False,
                    ),
                    "clean_argmax_token_id": clean_argmax,
                    "perturbed_argmax_token_id": perturbed_argmax,
                    "argmax_flip": bool(clean_argmax != perturbed_argmax),
                    "sampled_flip": bool(
                        clean_step.predicted_next_token_id
                        != perturbed_step.predicted_next_token_id
                    ),
                    "clean_top1_margin": _top1_margin(clean_step.logits),
                    "clean_diagnostics": clean_step.diagnostics,
                    "clean_hidden_norm": float(clean_step.event.hidden_post_norm),
                    "perturbed_hidden_norm": float(
                        perturbed_step.event.hidden_post_norm
                    ),
                    "hidden_cosine_distance": cosine_distance,
                    "hidden_l2_relative": relative_l2,
                    "logit_kl": float(logit_kl(clean_step.logits, perturbed_step.logits)),
                    "logit_js": float(
                        logit_jensen_shannon(
                            clean_step.logits,
                            perturbed_step.logits,
                        )
                    ),
                    "intervention_delta_norm": delta_norm,
                    "intervention_delta_relative": (
                        float(delta_norm / pre_norm) if pre_norm > 1e-12 else None
                    ),
                    "intervention_layer_resolved": resolved_layer,
                    "intervention_type": str(config.intervention_type),
                }
                rows.append(row)
                events_file.write(
                    json.dumps(json_safe(row), ensure_ascii=False) + "\n"
                )
                events_file.flush()

                state.advance_clean(clean_step)
                if (
                    eos is not None
                    and int(clean_step.predicted_next_token_id) == int(eos)
                ):
                    break
    finally:
        engine.close()

    clean_token_ids = [] if state is None else list(state.generated_token_ids)
    clean_text = tokenizer.decode(clean_token_ids, skip_special_tokens=True)
    output_path.write_text(config.prompt + clean_text, encoding="utf-8")
    row_count = len(rows)
    argmax_flips = sum(int(row["argmax_flip"]) for row in rows)
    sampled_flips = sum(int(row["sampled_flip"]) for row in rows)
    summary = {
        "config_hash": config_hash,
        "mode": "branchpoints",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_id": identity["model"],
        "backend": backend_result.backend,
        "runtime": {
            "device": str(device),
            "resolved_dtype": backend_result.backend_meta.get("resolved_dtype"),
            "policy_notes": backend_result.backend_meta.get("policy_notes", []),
            "backend_meta": dict(backend_result.backend_meta),
        },
        "clean_token_ids": clean_token_ids,
        "clean_output": clean_text,
        "protocol_validity": {
            "independent_one_step_forks": True,
            "continued_clean_only": True,
            "all_contexts_matched": all(
                row["contexts_matched"] for row in rows
            ),
            "rows": row_count,
        },
        "metrics": {
            "argmax_flips": int(argmax_flips),
            "argmax_flip_rate": float(argmax_flips / row_count) if row_count else 0.0,
            "sampled_flips": int(sampled_flips),
            "sampled_flip_rate": (
                float(sampled_flips / row_count) if row_count else 0.0
            ),
            "mean_logit_kl": (
                float(sum(row["logit_kl"] for row in rows) / row_count)
                if row_count
                else 0.0
            ),
            "mean_logit_js": (
                float(sum(row["logit_js"] for row in rows) / row_count)
                if row_count
                else 0.0
            ),
        },
        "diagnostics_health": summarize_diagnostics_health(
            [row["clean_diagnostics"] for row in rows]
        ),
        "config": run_config,
        "artifacts": {
            "run_dir": str(run_dir),
            "config_path": str(config_path),
            "events_path": str(events_path),
            "output_path": str(output_path),
            "summary_path": str(summary_path),
        },
    }
    save_json(str(summary_path), summary)
    return summary
