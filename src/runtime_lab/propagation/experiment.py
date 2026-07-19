from __future__ import annotations

import json
import os
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Optional

import numpy as np
import torch

from runtime_lab.config.schemas import PropagationConfig
from runtime_lab.core.backend.identity import resolved_model_identity
from runtime_lab.core.backend.loader import load_model_with_backend
from runtime_lab.core.interventions.factory import build_intervention
from runtime_lab.core.io.json import json_safe, save_json
from runtime_lab.core.io.run_artifacts import (
    create_run_dir,
    write_config_record,
)
from runtime_lab.core.model.cache_utils import (
    cache_sequence_length,
    compute_cache_fingerprint,
)
from runtime_lab.core.model.layers import (
    resolve_final_norm_module,
    resolve_transformer_layers,
)
from runtime_lab.core.runtime.engine import RuntimeEngine
from runtime_lab.core.runtime.hooks import HiddenCaptureHook
from runtime_lab.core.sampling import logit_jensen_shannon, logit_kl
from runtime_lab.core.trajectory.generation import (
    generation_state_from_seed_cache,
)
from runtime_lab.stress.seed_cache import build_seed_cache


def _set_deterministic_state(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_intervention(config: PropagationConfig):
    kind = str(config.intervention_type).lower().strip()
    kwargs: Dict[str, Any] = {"seed": int(config.intervention_seed)}
    if kind == "scaling":
        kwargs["scale"] = float(config.intervention_magnitude)
    elif kind == "projection":
        kwargs["subspace_dim"] = max(
            1,
            int(config.projection_subspace_dim),
        )
    else:
        kwargs["magnitude"] = float(config.intervention_magnitude)
        kwargs["relative"] = bool(
            config.intervention_magnitude_relative
        )
    return build_intervention(kind, **kwargs)


def _argmax_token(logits: torch.Tensor) -> int:
    return int(logits.detach().float().reshape(-1).argmax().item())


def _layer_delta(
    clean_hidden: torch.Tensor,
    perturbed_hidden: torch.Tensor,
) -> Dict[str, float]:
    clean = clean_hidden.detach().float().reshape(-1)
    perturbed = perturbed_hidden.detach().float().reshape(-1)
    if clean.numel() == 0 or clean.numel() != perturbed.numel():
        raise ValueError("paired layer states have incompatible shapes")

    clean_norm = float(clean.norm().item())
    perturbed_norm = float(perturbed.norm().item())
    if torch.equal(clean, perturbed):
        delta_l2 = 0.0
        cosine_distance = 0.0
    else:
        delta_l2 = float((perturbed - clean).norm().item())
        if clean_norm <= 1e-12 or perturbed_norm <= 1e-12:
            cosine_distance = 0.0
        else:
            cosine = float(torch.dot(clean, perturbed).item()) / (
                clean_norm * perturbed_norm
            )
            cosine_distance = float(
                1.0 - max(-1.0, min(1.0, cosine))
            )
    return {
        "clean_norm": clean_norm,
        "perturbed_norm": perturbed_norm,
        "delta_l2": delta_l2,
        "delta_relative_clean": (
            float(delta_l2 / clean_norm) if clean_norm > 1e-12 else 0.0
        ),
        "cosine_distance": cosine_distance,
    }


def _describe(values: list[float]) -> Dict[str, float]:
    return {
        "mean": float(mean(values)) if values else 0.0,
        "min": float(min(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
    }


def run_propagation_experiment(
    config: PropagationConfig,
    registry_path: str = "models.json",
    runs_dir: Optional[str] = None,
    prebuilt_backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """Measure one-step clean/perturbed deltas at every downstream layer."""
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
    injection_layer = int(seed_cache.resolved_layer_idx)
    num_layers = len(resolve_transformer_layers(model))
    capture_layers = list(range(injection_layer, num_layers))
    intervention = _build_intervention(config)
    engine = RuntimeEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_idx=injection_layer,
        diagnostics_manager=None,
        intervention=intervention,
        probe_layers=capture_layers,
        mode="propagation",
        temperature=float(config.temperature),
        top_p=float(config.top_p),
        top_k=int(config.top_k),
    )
    final_norm_name, final_norm_module = resolve_final_norm_module(model)
    final_norm_capture = (
        HiddenCaptureHook() if final_norm_module is not None else None
    )
    final_norm_handle = (
        final_norm_module.register_forward_hook(final_norm_capture)
        if final_norm_module is not None
        else None
    )

    run_config = {
        "mode": "propagation",
        "protocol": "same-context-layer-propagation-v1",
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
        "intervention_layer_resolved": injection_layer,
        "intervention_type": str(config.intervention_type),
        "intervention_magnitude": float(config.intervention_magnitude),
        "intervention_magnitude_relative": bool(
            config.intervention_magnitude_relative
        ),
        "intervention_seed": int(config.intervention_seed),
        "projection_subspace_dim": int(config.projection_subspace_dim),
        "capture_layers": capture_layers,
        "terminal_capture": {
            "name": final_norm_name,
            "available": final_norm_capture is not None,
        },
        "num_layers": num_layers,
        "seed_cache_fingerprint": seed_cache.fingerprint,
        "prompt_tokens": int(seed_cache.seq_len),
    }
    run_id, run_dir = create_run_dir(
        runs_dir or os.environ.get("RUNS_DIR", "runs"),
        "propagation",
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

                source_fingerprint = compute_cache_fingerprint(
                    state.past_key_values
                )
                clean_cache = state.fork_cache()
                perturbed_cache = state.fork_cache()
                source_length = cache_sequence_length(
                    state.past_key_values
                )
                contexts_matched = bool(
                    source_fingerprint not in {"unavailable", "empty"}
                    and source_fingerprint
                    == compute_cache_fingerprint(clean_cache)
                    == compute_cache_fingerprint(perturbed_cache)
                    and cache_sequence_length(clean_cache) == source_length
                    and cache_sequence_length(perturbed_cache)
                    == source_length
                )
                if not contexts_matched:
                    raise RuntimeError(
                        "Could not prove matched propagation contexts"
                    )

                decision_index = int(state.next_decision_index)
                rng_before = torch.random.get_rng_state()
                if final_norm_capture is not None:
                    final_norm_capture.reset()
                clean_step = engine.step(
                    t=decision_index,
                    consumed_token_id=consumed_token_id,
                    prompt_len=int(state.prompt_len),
                    past_key_values=clean_cache,
                    intervention_active=False,
                    collect_diagnostics=False,
                )
                clean_final_norm = (
                    final_norm_capture.last_hidden.clone()
                    if final_norm_capture is not None
                    and final_norm_capture.last_hidden is not None
                    else None
                )
                clean_rng_after = torch.random.get_rng_state()
                torch.random.set_rng_state(rng_before)
                if final_norm_capture is not None:
                    final_norm_capture.reset()
                try:
                    perturbed_step = engine.step(
                        t=decision_index,
                        consumed_token_id=consumed_token_id,
                        prompt_len=int(state.prompt_len),
                        past_key_values=perturbed_cache,
                        intervention_active=True,
                        collect_diagnostics=False,
                    )
                    perturbed_final_norm = (
                        final_norm_capture.last_hidden.clone()
                        if final_norm_capture is not None
                        and final_norm_capture.last_hidden is not None
                        else None
                    )
                finally:
                    torch.random.set_rng_state(clean_rng_after)

                if not all(
                    layer in clean_step.layer_states
                    and layer in perturbed_step.layer_states
                    for layer in capture_layers
                ):
                    raise RuntimeError(
                        "A requested downstream layer was not captured"
                    )

                layer_deltas = {
                    str(layer): _layer_delta(
                        clean_step.layer_states[layer],
                        perturbed_step.layer_states[layer],
                    )
                    for layer in capture_layers
                }
                injection_delta = float(
                    layer_deltas[str(injection_layer)]["delta_l2"]
                )
                for values in layer_deltas.values():
                    values["amplification_vs_injection"] = (
                        float(values["delta_l2"] / injection_delta)
                        if injection_delta > 1e-12
                        else 0.0
                    )
                terminal_deltas: Dict[str, Dict[str, float]] = {}
                if final_norm_capture is not None:
                    if (
                        clean_final_norm is None
                        or perturbed_final_norm is None
                    ):
                        raise RuntimeError(
                            "Final normalization module was not captured"
                        )
                    final_norm_delta = _layer_delta(
                        clean_final_norm,
                        perturbed_final_norm,
                    )
                    final_norm_delta[
                        "amplification_vs_injection"
                    ] = (
                        float(
                            final_norm_delta["delta_l2"]
                            / injection_delta
                        )
                        if injection_delta > 1e-12
                        else 0.0
                    )
                    terminal_deltas["final_norm"] = final_norm_delta

                clean_argmax = _argmax_token(clean_step.logits)
                perturbed_argmax = _argmax_token(
                    perturbed_step.logits
                )
                row = {
                    "mode": "propagation",
                    "protocol": "same-context-layer-propagation-v1",
                    "decision_index": decision_index,
                    "context_fingerprint": source_fingerprint,
                    "context_sequence_length": source_length,
                    "contexts_matched": True,
                    "sampling_rng_matched": True,
                    "consumed_token_id": consumed_token_id,
                    "consumed_token_text": clean_step.consumed_token_text,
                    "clean_predicted_token_id": int(
                        clean_step.predicted_next_token_id
                    ),
                    "perturbed_predicted_token_id": int(
                        perturbed_step.predicted_next_token_id
                    ),
                    "clean_argmax_token_id": clean_argmax,
                    "perturbed_argmax_token_id": perturbed_argmax,
                    "argmax_flip": bool(
                        clean_argmax != perturbed_argmax
                    ),
                    "sampled_flip": bool(
                        clean_step.predicted_next_token_id
                        != perturbed_step.predicted_next_token_id
                    ),
                    "logit_kl": float(
                        logit_kl(
                            clean_step.logits,
                            perturbed_step.logits,
                        )
                    ),
                    "logit_js": float(
                        logit_jensen_shannon(
                            clean_step.logits,
                            perturbed_step.logits,
                        )
                    ),
                    "intervention_layer_resolved": injection_layer,
                    "capture_layers": capture_layers,
                    "layer_deltas": layer_deltas,
                    "terminal_deltas": terminal_deltas,
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
        if final_norm_handle is not None:
            final_norm_handle.remove()

    clean_token_ids = [] if state is None else list(
        state.generated_token_ids
    )
    clean_text = tokenizer.decode(
        clean_token_ids,
        skip_special_tokens=True,
    )
    output_path.write_text(config.prompt + clean_text, encoding="utf-8")

    layer_summary: Dict[str, Any] = {}
    for layer in capture_layers:
        key = str(layer)
        layer_summary[key] = {
            "delta_l2": _describe(
                [float(row["layer_deltas"][key]["delta_l2"]) for row in rows]
            ),
            "delta_relative_clean": _describe(
                [
                    float(
                        row["layer_deltas"][key][
                            "delta_relative_clean"
                        ]
                    )
                    for row in rows
                ]
            ),
            "cosine_distance": _describe(
                [
                    float(
                        row["layer_deltas"][key]["cosine_distance"]
                    )
                    for row in rows
                ]
            ),
            "amplification_vs_injection": _describe(
                [
                    float(
                        row["layer_deltas"][key][
                            "amplification_vs_injection"
                        ]
                    )
                    for row in rows
                ]
            ),
        }

    argmax_flips = sum(int(row["argmax_flip"]) for row in rows)
    sampled_flips = sum(int(row["sampled_flip"]) for row in rows)
    terminal_summary: Dict[str, Any] = {}
    if final_norm_capture is not None:
        for metric in (
            "delta_l2",
            "delta_relative_clean",
            "cosine_distance",
            "amplification_vs_injection",
        ):
            terminal_summary[metric] = _describe(
                [
                    float(row["terminal_deltas"]["final_norm"][metric])
                    for row in rows
                ]
            )
    summary = {
        "config_hash": config_hash,
        "mode": "propagation",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_id": identity["model"],
        "backend": backend_result.backend,
        "runtime": {
            "device": str(device),
            "resolved_dtype": backend_result.backend_meta.get(
                "resolved_dtype"
            ),
            "policy_notes": backend_result.backend_meta.get(
                "policy_notes",
                [],
            ),
            "backend_meta": dict(backend_result.backend_meta),
        },
        "capture_layers": capture_layers,
        "terminal_capture": {
            "name": final_norm_name,
            "available": final_norm_capture is not None,
        },
        "clean_token_ids": clean_token_ids,
        "clean_output": clean_text,
        "protocol_validity": {
            "protocol": "same-context-layer-propagation-v1",
            "independent_one_step_forks": True,
            "continued_clean_only": True,
            "all_contexts_matched": all(
                row["contexts_matched"] for row in rows
            ),
            "all_layers_captured": all(
                set(row["layer_deltas"])
                == {str(layer) for layer in capture_layers}
                for row in rows
            ),
            "terminal_capture_complete": bool(
                final_norm_capture is None
                or all(
                    "final_norm" in row["terminal_deltas"]
                    for row in rows
                )
            ),
            "rows": len(rows),
        },
        "metrics": {
            "argmax_flips": argmax_flips,
            "argmax_flip_rate": (
                float(argmax_flips / len(rows)) if rows else 0.0
            ),
            "sampled_flips": sampled_flips,
            "sampled_flip_rate": (
                float(sampled_flips / len(rows)) if rows else 0.0
            ),
            "final_layer_argmax_flips": argmax_flips,
            "mean_logit_kl": (
                float(mean(row["logit_kl"] for row in rows))
                if rows
                else 0.0
            ),
            "mean_logit_js": (
                float(mean(row["logit_js"] for row in rows))
                if rows
                else 0.0
            ),
            "max_delta_l2": max(
                (
                    float(values["delta_l2"])
                    for row in rows
                    for values in (
                        list(row["layer_deltas"].values())
                        + list(row["terminal_deltas"].values())
                    )
                ),
                default=0.0,
            ),
            "layer_summary": layer_summary,
            "terminal_summary": {
                "final_norm": terminal_summary,
            },
        },
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
