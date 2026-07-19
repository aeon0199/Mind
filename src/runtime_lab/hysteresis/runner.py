from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from runtime_lab.config.schemas import HysteresisConfig
from runtime_lab.core.backend.identity import resolved_model_identity
from runtime_lab.core.backend.loader import load_model_with_backend
from runtime_lab.core.interventions.factory import build_intervention
from runtime_lab.core.io.json import json_safe, save_json
from runtime_lab.core.io.run_artifacts import create_run_dir, write_config_record
from runtime_lab.core.model.cache_utils import (
    cache_sequence_length,
    compute_cache_fingerprint,
)
from runtime_lab.core.runtime.engine import RuntimeEngine, StepResult
from runtime_lab.core.sampling import logit_jensen_shannon, logit_kl
from runtime_lab.core.trajectory.generation import (
    GenerationState,
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


def _argmax_token(logits: torch.Tensor) -> int:
    return int(logits.detach().float().reshape(-1).argmax().item())


def _hidden_distances(
    clean_hidden: Optional[torch.Tensor],
    perturbed_hidden: Optional[torch.Tensor],
) -> tuple[float | None, float | None]:
    if not isinstance(clean_hidden, torch.Tensor) or not isinstance(
        perturbed_hidden,
        torch.Tensor,
    ):
        return None, None
    clean = clean_hidden.detach().float().reshape(-1)
    perturbed = perturbed_hidden.detach().float().reshape(-1)
    if clean.numel() == 0 or clean.numel() != perturbed.numel():
        return None, None
    if torch.equal(clean, perturbed):
        return 0.0, 0.0
    clean_norm = float(clean.norm().item())
    perturbed_norm = float(perturbed.norm().item())
    relative_l2 = float((clean - perturbed).norm().item()) / max(
        clean_norm,
        1e-12,
    )
    if clean_norm <= 1e-12 or perturbed_norm <= 1e-12:
        return None, relative_l2
    cosine = float(torch.dot(clean, perturbed).item()) / (
        clean_norm * perturbed_norm
    )
    cosine = max(-1.0, min(1.0, cosine))
    return float(1.0 - cosine), relative_l2


def _distance_point(
    *,
    phase: str,
    phase_step: int,
    global_decision_index: int,
    clean_step: StepResult,
    perturbed_step: StepResult,
    intervention_active: bool,
) -> Dict[str, Any]:
    hidden_cosine, hidden_l2_relative = _hidden_distances(
        clean_step.hidden_post,
        perturbed_step.hidden_post,
    )
    js = float(logit_jensen_shannon(clean_step.logits, perturbed_step.logits))
    kl = float(logit_kl(clean_step.logits, perturbed_step.logits))
    distance_components = [
        value
        for value in (hidden_cosine, hidden_l2_relative, js)
        if value is not None
    ]
    distance = float(max(distance_components, default=0.0))
    clean_argmax = _argmax_token(clean_step.logits)
    perturbed_argmax = _argmax_token(perturbed_step.logits)
    return {
        "phase": phase,
        "phase_step": int(phase_step),
        "global_decision_index": int(global_decision_index),
        "intervention_active": bool(intervention_active),
        "distance": distance,
        "hidden_cosine_distance": hidden_cosine,
        "hidden_l2_relative": hidden_l2_relative,
        "logit_js": js,
        "logit_kl": kl,
        "clean_argmax_token_id": clean_argmax,
        "perturbed_argmax_token_id": perturbed_argmax,
        "argmax_flip": bool(clean_argmax != perturbed_argmax),
        "sampled_flip": bool(
            clean_step.predicted_next_token_id
            != perturbed_step.predicted_next_token_id
        ),
    }


def _trace_record(
    *,
    phase: str,
    branch: str,
    phase_step: int,
    global_decision_index: int,
    context_length: int | None,
    step: StepResult,
    intervention_active: bool,
    paired_distance: float,
) -> Dict[str, Any]:
    return {
        "mode": "hysteresis",
        "phase": phase,
        "branch": branch,
        "phase_step": int(phase_step),
        "global_decision_index": int(global_decision_index),
        "context_sequence_length": context_length,
        "consumed_token_id": int(step.consumed_token_id),
        "consumed_token_text": step.consumed_token_text,
        "predicted_next_token_id": int(step.predicted_next_token_id),
        "argmax_token_id": _argmax_token(step.logits),
        "hidden_pre_norm": float(step.event.hidden_pre_norm),
        "hidden_post_norm": float(step.event.hidden_post_norm),
        "hidden_delta_norm": float(step.event.hidden_delta_norm),
        "intervention_active": bool(intervention_active),
        "sampling_rng_paired": True,
        "paired_distance": float(paired_distance),
    }


def _paired_forward(
    *,
    engine: RuntimeEngine,
    clean_state: GenerationState,
    perturbed_state: GenerationState,
    phase: str,
    phase_step: int,
    intervention_active: bool,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if clean_state.next_decision_index != perturbed_state.next_decision_index:
        raise RuntimeError("paired trajectories lost decision-index alignment")
    global_decision_index = int(clean_state.next_decision_index)
    clean_context_length = cache_sequence_length(clean_state.past_key_values)
    perturbed_context_length = cache_sequence_length(
        perturbed_state.past_key_values
    )

    rng_before = torch.random.get_rng_state()
    clean_step = engine.step(
        t=global_decision_index,
        consumed_token_id=int(clean_state.pending_token_id),
        prompt_len=int(clean_state.prompt_len),
        past_key_values=clean_state.past_key_values,
        intervention_active=False,
        collect_diagnostics=False,
    )
    clean_rng_after = torch.random.get_rng_state()
    torch.random.set_rng_state(rng_before)
    try:
        perturbed_step = engine.step(
            t=global_decision_index,
            consumed_token_id=int(perturbed_state.pending_token_id),
            prompt_len=int(perturbed_state.prompt_len),
            past_key_values=perturbed_state.past_key_values,
            intervention_active=bool(intervention_active),
            collect_diagnostics=False,
        )
    finally:
        torch.random.set_rng_state(clean_rng_after)

    distance = _distance_point(
        phase=phase,
        phase_step=phase_step,
        global_decision_index=global_decision_index,
        clean_step=clean_step,
        perturbed_step=perturbed_step,
        intervention_active=intervention_active,
    )
    clean_record = _trace_record(
        phase=phase,
        branch="clean",
        phase_step=phase_step,
        global_decision_index=global_decision_index,
        context_length=clean_context_length,
        step=clean_step,
        intervention_active=False,
        paired_distance=distance["distance"],
    )
    perturbed_record = _trace_record(
        phase=phase,
        branch="perturbed",
        phase_step=phase_step,
        global_decision_index=global_decision_index,
        context_length=perturbed_context_length,
        step=perturbed_step,
        intervention_active=intervention_active,
        paired_distance=distance["distance"],
    )
    clean_state.advance(clean_step)
    perturbed_state.advance(perturbed_step)
    return clean_record, perturbed_record, distance


def _pending_is_eos(
    clean_state: GenerationState,
    perturbed_state: GenerationState,
    eos_token_id: int | None,
) -> bool:
    if eos_token_id is None:
        return False
    return bool(
        int(clean_state.pending_token_id) == int(eos_token_id)
        or int(perturbed_state.pending_token_id) == int(eos_token_id)
    )


def _classify_regime(recovery: float) -> str:
    if recovery > 0.8:
        return "elastic"
    if recovery > 0.4:
        return "partial"
    if recovery >= 0.0:
        return "plastic"
    return "runaway"


def _endpoint_frame(
    *,
    stage: str,
    trace: list[Dict[str, Any]],
    text: str,
    source: str,
) -> Dict[str, Any]:
    endpoint = trace[-1] if trace else {}
    return {
        "stage": stage,
        "compatibility_view": True,
        "metric_source": source,
        "token_text": text,
        "hidden_norm": endpoint.get("hidden_post_norm", 0.0),
        "intervention_active": endpoint.get("intervention_active", False),
        "extra": {
            "note": (
                "Endpoint compatibility view only. Recovery metrics come from "
                "the aligned paired distance curves in summary.json."
            )
        },
    }


def run_hysteresis_experiment(
    config: HysteresisConfig,
    registry_path: str = "models.json",
    runs_dir: Optional[str] = None,
    prebuilt_backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run matched clean/perturbed exposure and intervention-free recovery."""
    if str(config.perturbation_mode).lower().strip() != "noise":
        raise ValueError(
            "Matched internal hysteresis requires perturbation_mode='noise'. "
            "Historical prompt/reflection runs measure prompt contamination, "
            "not internal recovery."
        )
    exposure_steps_requested = int(config.max_new_tokens)
    recovery_steps_requested = int(config.recovery_tokens)
    if exposure_steps_requested < 1:
        raise ValueError("max_new_tokens must be at least 1 exposure step")
    if recovery_steps_requested < 1:
        raise ValueError("recovery_tokens must be at least 1")
    if int(config.noise_start) < 1:
        raise ValueError("noise_start must be at least 1")
    if int(config.noise_duration) < 0:
        raise ValueError("noise_duration must not be negative")

    _set_deterministic_state(config.seed)
    backend_result = prebuilt_backend or load_model_with_backend(
        model_key=config.model_key,
        registry_path=registry_path,
        backend=config.backend,
        nnsight_remote=config.nnsight_remote,
        nnsight_device=config.nnsight_device,
    )
    identity = resolved_model_identity(config.model_key, backend_result)
    tokenizer = backend_result.tokenizer
    model = backend_result.model
    device = backend_result.device

    seed_cache = build_seed_cache(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=config.prompt,
        intervention_layer=int(config.noise_layer),
    )
    resolved_layer = int(seed_cache.resolved_layer_idx)
    intervention = build_intervention(
        "additive",
        magnitude=float(config.noise_magnitude),
        relative=True,
        seed=int(config.noise_seed),
    )
    engine = RuntimeEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_idx=resolved_layer,
        diagnostics_manager=None,
        intervention=intervention,
        probe_layers=[],
        mode="hysteresis",
        temperature=float(config.temperature),
        top_p=float(config.top_p),
        top_k=int(config.top_k),
    )

    initial_state = generation_state_from_seed_cache(
        seed_cache,
        temperature=float(config.temperature),
        top_p=float(config.top_p),
        top_k=int(config.top_k),
    )
    clean_state = initial_state.clone()
    perturbed_state = initial_state.clone()
    initial_fingerprints = {
        "source": compute_cache_fingerprint(initial_state.past_key_values),
        "clean": compute_cache_fingerprint(clean_state.past_key_values),
        "perturbed": compute_cache_fingerprint(perturbed_state.past_key_values),
    }
    initial_contexts_matched = bool(
        initial_fingerprints["source"] not in {"unavailable", "empty"}
        and len(set(initial_fingerprints.values())) == 1
    )
    if not initial_contexts_matched:
        engine.close()
        raise RuntimeError("Could not prove matched initial hysteresis contexts")

    run_config = {
        "mode": "hysteresis",
        "protocol": "matched-exposure-recovery-v2",
        "prompt": config.prompt,
        **identity,
        "backend": backend_result.backend,
        "backend_meta": dict(backend_result.backend_meta),
        "seed": int(config.seed) if config.seed is not None else None,
        "temperature": float(config.temperature),
        "top_p": float(config.top_p),
        "top_k": int(config.top_k),
        "perturbation_mode": "noise",
        "noise_layer_requested": int(config.noise_layer),
        "noise_layer_resolved": resolved_layer,
        "noise_magnitude": float(config.noise_magnitude),
        "noise_start": int(config.noise_start),
        "noise_duration": int(config.noise_duration),
        "noise_seed": int(config.noise_seed),
        "exposure_steps": exposure_steps_requested,
        "recovery_tokens": recovery_steps_requested,
        "propagation_floor": float(config.propagation_floor),
        "num_layers": int(engine.num_layers),
        "seed_cache_fingerprint": seed_cache.fingerprint,
        "prompt_tokens": int(seed_cache.seq_len),
    }
    run_id, run_dir = create_run_dir(
        runs_dir or os.environ.get("RUNS_DIR", "runs"),
        "hysteresis",
    )
    config_hash, config_path = write_config_record(run_dir, run_config)

    traces: Dict[str, list[Dict[str, Any]]] = {
        "clean_exposure": [],
        "perturbed_exposure": [],
        "clean_recovery": [],
        "perturbed_recovery": [],
    }
    exposure_curve: list[Dict[str, Any]] = []
    recovery_curve: list[Dict[str, Any]] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_truncated_phase: str | None = None

    try:
        for phase_step in range(1, exposure_steps_requested + 1):
            if _pending_is_eos(clean_state, perturbed_state, eos_token_id):
                eos_truncated_phase = "exposure"
                break
            intervention_active = bool(
                int(config.noise_start)
                <= phase_step
                < int(config.noise_start) + int(config.noise_duration)
            )
            clean_record, perturbed_record, distance = _paired_forward(
                engine=engine,
                clean_state=clean_state,
                perturbed_state=perturbed_state,
                phase="exposure",
                phase_step=phase_step,
                intervention_active=intervention_active,
            )
            traces["clean_exposure"].append(clean_record)
            traces["perturbed_exposure"].append(perturbed_record)
            exposure_curve.append(distance)

        clean_exposure_token_ids = list(clean_state.generated_token_ids)
        perturbed_exposure_token_ids = list(perturbed_state.generated_token_ids)

        if eos_truncated_phase is None:
            for phase_step in range(1, recovery_steps_requested + 1):
                if _pending_is_eos(clean_state, perturbed_state, eos_token_id):
                    eos_truncated_phase = "recovery"
                    break
                clean_record, perturbed_record, distance = _paired_forward(
                    engine=engine,
                    clean_state=clean_state,
                    perturbed_state=perturbed_state,
                    phase="recovery",
                    phase_step=phase_step,
                    intervention_active=False,
                )
                traces["clean_recovery"].append(clean_record)
                traces["perturbed_recovery"].append(perturbed_record)
                recovery_curve.append(distance)
    finally:
        engine.close()

    exposure_window_complete = bool(
        int(config.noise_duration) == 0
        or (
            len(exposure_curve)
            >= int(config.noise_start) + int(config.noise_duration)
        )
    )
    matched_exposure = bool(
        initial_contexts_matched
        and len(traces["clean_exposure"]) == exposure_steps_requested
        and len(traces["perturbed_exposure"]) == exposure_steps_requested
    )
    matched_recovery = bool(
        matched_exposure
        and exposure_window_complete
        and len(traces["clean_recovery"]) == recovery_steps_requested
        and len(traces["perturbed_recovery"]) == recovery_steps_requested
    )

    endpoint_valid = bool(matched_exposure and exposure_window_complete)
    propagated_distance = (
        float(exposure_curve[-1]["distance"])
        if endpoint_valid and exposure_curve
        else None
    )
    direct_effect_peak = max(
        (
            float(point["distance"])
            for point in exposure_curve
            if point["intervention_active"]
        ),
        default=0.0,
    )
    exposure_peak = max(
        (float(point["distance"]) for point in exposure_curve),
        default=0.0,
    )
    residual_distance = (
        float(recovery_curve[-1]["distance"])
        if matched_recovery and recovery_curve
        else None
    )
    did_not_propagate = (
        None
        if propagated_distance is None
        else bool(propagated_distance <= float(config.propagation_floor))
    )
    if not endpoint_valid or not matched_recovery:
        recovery = None
        regime = "invalid"
    elif did_not_propagate:
        recovery = None
        regime = "no-propagation"
    else:
        recovery = float(
            1.0 - float(residual_distance) / float(propagated_distance)
        )
        regime = _classify_regime(recovery)

    clean_exposure_text = tokenizer.decode(
        clean_exposure_token_ids,
        skip_special_tokens=True,
    )
    perturbed_exposure_text = tokenizer.decode(
        perturbed_exposure_token_ids,
        skip_special_tokens=True,
    )
    clean_recovery_text = tokenizer.decode(
        clean_state.generated_token_ids,
        skip_special_tokens=True,
    )
    perturbed_recovery_text = tokenizer.decode(
        perturbed_state.generated_token_ids,
        skip_special_tokens=True,
    )

    events_path = Path(run_dir) / "events.jsonl"
    output_paths = {
        "output_clean_exposure": Path(run_dir) / "output_clean_exposure.txt",
        "output_perturbed_exposure": (
            Path(run_dir) / "output_perturbed_exposure.txt"
        ),
        "output_clean_recovery": Path(run_dir) / "output_clean_recovery.txt",
        "output_perturbed_recovery": (
            Path(run_dir) / "output_perturbed_recovery.txt"
        ),
    }
    with events_path.open("w", encoding="utf-8") as events_file:
        for phase in ("exposure", "recovery"):
            for branch in ("clean", "perturbed"):
                for row in traces[f"{branch}_{phase}"]:
                    events_file.write(
                        json.dumps(json_safe(row), ensure_ascii=False) + "\n"
                    )
    for key, path in output_paths.items():
        text = {
            "output_clean_exposure": clean_exposure_text,
            "output_perturbed_exposure": perturbed_exposure_text,
            "output_clean_recovery": clean_recovery_text,
            "output_perturbed_recovery": perturbed_recovery_text,
        }[key]
        path.write_text(config.prompt + text, encoding="utf-8")

    frame_base = _endpoint_frame(
        stage="base",
        trace=traces["clean_exposure"],
        text=clean_exposure_text,
        source="clean_exposure endpoint",
    )
    frame_perturb = _endpoint_frame(
        stage="perturb",
        trace=traces["perturbed_exposure"],
        text=perturbed_exposure_text,
        source="perturbed_exposure endpoint",
    )
    frame_reask = _endpoint_frame(
        stage="reask",
        trace=traces["perturbed_recovery"],
        text=perturbed_recovery_text,
        source="perturbed_recovery endpoint compatibility alias",
    )
    save_json(str(Path(run_dir) / "frame_base.json"), frame_base)
    save_json(str(Path(run_dir) / "frame_perturb.json"), frame_perturb)
    save_json(str(Path(run_dir) / "frame_reask.json"), frame_reask)
    (Path(run_dir) / "output_base.txt").write_text(
        config.prompt + clean_exposure_text,
        encoding="utf-8",
    )
    (Path(run_dir) / "output_perturb.txt").write_text(
        config.prompt + perturbed_exposure_text,
        encoding="utf-8",
    )
    (Path(run_dir) / "output_reask.txt").write_text(
        config.prompt + perturbed_recovery_text,
        encoding="utf-8",
    )
    (Path(run_dir) / "delta.txt").write_text(
        (
            "[matched hidden-state perturbation; no prompt delta]\n"
            f"layer={resolved_layer}\n"
            f"magnitude={float(config.noise_magnitude)} relative\n"
            f"start={int(config.noise_start)}\n"
            f"duration={int(config.noise_duration)}\n"
            f"seed={int(config.noise_seed)}\n"
        ),
        encoding="utf-8",
    )

    protocol_validity = {
        "protocol": "matched-exposure-recovery-v2",
        "initial_contexts_matched": initial_contexts_matched,
        "initial_context_fingerprints": initial_fingerprints,
        "matched_exposure": matched_exposure,
        "matched_recovery": matched_recovery,
        "equal_phase_instructions": True,
        "recovery_instruction": None,
        "intervention_disabled_during_recovery": all(
            not row["intervention_active"]
            for key in ("clean_recovery", "perturbed_recovery")
            for row in traces[key]
        ),
        "exposure_window_complete": exposure_window_complete,
        "clean_exposure_steps": len(traces["clean_exposure"]),
        "perturbed_exposure_steps": len(traces["perturbed_exposure"]),
        "clean_recovery_steps": len(traces["clean_recovery"]),
        "perturbed_recovery_steps": len(traces["perturbed_recovery"]),
        "requested_exposure_steps": exposure_steps_requested,
        "requested_recovery_steps": recovery_steps_requested,
        "eos_truncated_phase": eos_truncated_phase,
    }
    metrics = {
        "drift": propagated_distance,
        "hysteresis": residual_distance,
        "recovery": recovery,
        "regime": regime,
        "direct_effect_peak": direct_effect_peak,
        "exposure_peak_distance": exposure_peak,
        "propagated_distance": propagated_distance,
        "residual_distance": residual_distance,
        "perturbation_did_not_propagate": did_not_propagate,
        "propagation_floor": float(config.propagation_floor),
        "distance_curve_exposure": exposure_curve,
        "distance_curve_recovery": recovery_curve,
    }
    summary_path = Path(run_dir) / "summary.json"
    summary = {
        "config_hash": config_hash,
        "mode": "hysteresis",
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
        "config": run_config,
        "seed_cache": {
            "seq_len": int(seed_cache.seq_len),
            "fingerprint": seed_cache.fingerprint,
            "fingerprint_available": bool(
                seed_cache.fingerprint not in {"unavailable", "empty"}
            ),
        },
        "protocol_validity": protocol_validity,
        "metrics": metrics,
        "traces": traces,
        "clean_token_ids": list(clean_state.generated_token_ids),
        "perturbed_token_ids": list(perturbed_state.generated_token_ids),
        "telemetry": {
            "base": frame_base,
            "perturb": frame_perturb,
            "reask": frame_reask,
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "config_path": str(config_path),
            "events_path": str(events_path),
            "summary_path": str(summary_path),
            **{key: str(path) for key, path in output_paths.items()},
            "frame_base": str(Path(run_dir) / "frame_base.json"),
            "frame_perturb": str(Path(run_dir) / "frame_perturb.json"),
            "frame_reask": str(Path(run_dir) / "frame_reask.json"),
            "output_base": str(Path(run_dir) / "output_base.txt"),
            "output_perturb": str(Path(run_dir) / "output_perturb.txt"),
            "output_reask": str(Path(run_dir) / "output_reask.txt"),
            "delta": str(Path(run_dir) / "delta.txt"),
        },
    }
    try:
        from runtime_lab.core.advisory import analyze as analyze_advisory

        summary["advisory"] = analyze_advisory("hysteresis", summary)
    except Exception as error:
        summary["advisory"] = {"error": f"advisory failed: {error}"}
    save_json(str(summary_path), summary)
    return summary
