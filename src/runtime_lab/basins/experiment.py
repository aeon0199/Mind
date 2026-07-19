from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch

from runtime_lab.basins.evaluators import (
    BasinPromptSpec,
    build_generic_prompt_spec,
    compare_outputs,
    get_prompt_spec,
    score_output,
)
from runtime_lab.config.schemas import BasinConfig, DiagnosticsConfig
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
from runtime_lab.core.runtime.engine import RuntimeEngine, StepResult
from runtime_lab.core.sampling import logit_jensen_shannon, logit_kl
from runtime_lab.core.trajectory.generation import (
    GenerationState,
    generation_state_from_seed_cache,
)
from runtime_lab.stress.seed_cache import build_seed_cache


PROTOCOL = "causal-branch-continuation-v1"
SELECTION_PROTOCOL = "outcome-blind-early-middle-late-v1"
CONTINUATION_PROTOCOL = "paired-position-rng-no-intervention-v1"


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


def _argmax_token(logits: torch.Tensor) -> int:
    return int(logits.detach().float().reshape(-1).argmax().item())


def _top1_margin(logits: torch.Tensor) -> float:
    values = torch.topk(
        logits.detach().float().reshape(-1),
        k=min(2, logits.numel()),
    ).values
    if values.numel() < 2:
        return 0.0
    return float((values[0] - values[1]).item())


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
    if clean.numel() != perturbed.numel() or clean.numel() == 0:
        return None, None
    difference = clean - perturbed
    if torch.equal(clean, perturbed):
        return 0.0, 0.0
    clean_norm = float(clean.norm().item())
    perturbed_norm = float(perturbed.norm().item())
    cosine_distance = None
    if clean_norm > 1e-12 and perturbed_norm > 1e-12:
        cosine = float(torch.dot(clean, perturbed).item()) / (
            clean_norm * perturbed_norm
        )
        cosine_distance = float(1.0 - max(-1.0, min(1.0, cosine)))
    relative_l2 = float(difference.norm().item()) / max(clean_norm, 1e-12)
    return cosine_distance, relative_l2


def _build_intervention(config: BasinConfig):
    kind = str(config.intervention_type).lower().strip()
    common: dict[str, Any] = {"seed": int(config.intervention_seed)}
    if kind == "scaling":
        common["scale"] = float(config.intervention_magnitude)
    elif kind == "projection":
        common["subspace_dim"] = max(
            1,
            int(config.projection_subspace_dim),
        )
    else:
        common["magnitude"] = float(config.intervention_magnitude)
        common["relative"] = bool(config.intervention_magnitude_relative)
    return build_intervention(kind, **common)


def _is_identity_control(config: BasinConfig) -> bool:
    kind = str(config.intervention_type).lower().strip()
    magnitude = float(config.intervention_magnitude)
    if kind == "additive":
        return magnitude == 0.0
    if kind == "scaling":
        return magnitude == 1.0
    return False


def _position_band(decision_index: int, max_new_tokens: int) -> str:
    decision_count = max(1, int(max_new_tokens) - 1)
    position = max(0, min(decision_count - 1, int(decision_index) - 1))
    band_index = min(2, (position * 3) // decision_count)
    return ("early", "middle", "late")[band_index]


def _has_measurable_local_effect(row: Mapping[str, Any]) -> bool:
    return bool(
        abs(float(row.get("logit_kl", 0.0))) > 1e-12
        or abs(float(row.get("logit_js", 0.0))) > 1e-12
        or abs(float(row.get("intervention_delta_norm", 0.0))) > 1e-12
    )


def _selection_copy(
    row: Mapping[str, Any],
    *,
    band: str,
    basis: str,
) -> dict[str, Any]:
    excluded = {
        "outcome",
        "score_comparison",
        "clean_score",
        "perturbed_score",
    }
    return {
        **{
            str(key): value
            for key, value in row.items()
            if key not in excluded
        },
        "position_band": band,
        "selection_basis": basis,
        "selection_protocol": SELECTION_PROTOCOL,
        "selection_outcome_blind": True,
    }


def _select_branchpoints(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_new_tokens: int,
    max_episodes: int,
) -> list[dict[str, Any]]:
    """Select branchpoints without observing any continuation outcome."""
    ordered = sorted(rows, key=lambda row: int(row["decision_index"]))
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    for band in ("early", "middle", "late"):
        candidates = [
            row
            for row in ordered
            if _position_band(
                int(row["decision_index"]),
                max_new_tokens,
            )
            == band
        ]
        if not candidates:
            continue
        sampled_flips = [
            row for row in candidates if bool(row.get("sampled_flip"))
        ]
        measurable = [
            row for row in candidates if _has_measurable_local_effect(row)
        ]
        if sampled_flips:
            chosen = sampled_flips[0]
            basis = "earliest_sampled_flip"
        elif measurable:
            chosen = measurable[0]
            basis = "earliest_latent_effect"
        else:
            chosen = candidates[0]
            basis = "exact_no_effect_anchor"
        selected.append(_selection_copy(chosen, band=band, basis=basis))
        selected_indices.add(int(chosen["decision_index"]))
        if len(selected) >= int(max_episodes):
            return sorted(selected, key=lambda row: int(row["decision_index"]))

    if len(selected) < int(max_episodes):
        for row in ordered:
            decision_index = int(row["decision_index"])
            if decision_index in selected_indices or not bool(
                row.get("sampled_flip")
            ):
                continue
            band = _position_band(decision_index, max_new_tokens)
            selected.append(
                _selection_copy(
                    row,
                    band=band,
                    basis="additional_sampled_flip",
                )
            )
            selected_indices.add(decision_index)
            if len(selected) >= int(max_episodes):
                break
    return sorted(selected, key=lambda row: int(row["decision_index"]))


def _prefix_features(
    state: GenerationState,
    *,
    decision_index: int,
    max_new_tokens: int,
    context_sequence_length: int | None,
    consumed_token_text: str,
    clean_step: StepResult,
) -> dict[str, Any]:
    tokens = list(state.generated_token_ids)
    unique_ratio = (
        float(len(set(tokens)) / len(tokens))
        if tokens
        else 1.0
    )
    stripped = consumed_token_text.strip()
    return {
        "decision_index": int(decision_index),
        "normalized_position": float(
            int(decision_index) / max(1, int(max_new_tokens) - 1)
        ),
        "context_sequence_length": context_sequence_length,
        "prefix_token_count": len(tokens),
        "prefix_unique_token_ratio": unique_ratio,
        "prefix_repetition_score": float(1.0 - unique_ratio),
        "clean_top1_margin": _top1_margin(clean_step.logits),
        "clean_hidden_norm": float(clean_step.event.hidden_post_norm),
        "consumed_token": {
            "length": len(consumed_token_text),
            "blank": bool(not stripped),
            "leading_space": consumed_token_text.startswith(" "),
            "newline": "\n" in consumed_token_text,
            "numeric": bool(stripped and stripped.isdigit()),
        },
        "clean_diagnostics": clean_step.diagnostics,
    }


def _local_fork(
    *,
    engine: RuntimeEngine,
    state: GenerationState,
    decision_index: int,
    collect_diagnostics: bool,
) -> tuple[StepResult, StepResult, dict[str, Any]]:
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
        raise RuntimeError("Could not prove matched basin branchpoint contexts")

    rng_before = torch.random.get_rng_state()
    clean_step = engine.step(
        t=decision_index,
        consumed_token_id=int(state.pending_token_id),
        prompt_len=int(state.prompt_len),
        past_key_values=clean_cache,
        intervention_active=False,
        collect_diagnostics=collect_diagnostics,
    )
    clean_rng_after = torch.random.get_rng_state()
    torch.random.set_rng_state(rng_before)
    try:
        perturbed_step = engine.step(
            t=decision_index,
            consumed_token_id=int(state.pending_token_id),
            prompt_len=int(state.prompt_len),
            past_key_values=perturbed_cache,
            intervention_active=True,
            collect_diagnostics=False,
        )
    finally:
        torch.random.set_rng_state(clean_rng_after)
    return (
        clean_step,
        perturbed_step,
        {
            "context_fingerprint": source_fingerprint,
            "clean_context_fingerprint": clean_fingerprint,
            "perturbed_context_fingerprint": perturbed_fingerprint,
            "context_sequence_length": source_length,
            "contexts_matched": contexts_matched,
        },
    )


def _branchpoint_row(
    *,
    config: BasinConfig,
    state: GenerationState,
    clean_step: StepResult,
    perturbed_step: StepResult,
    context: Mapping[str, Any],
    tokenizer: Any,
    resolved_layer: int,
) -> dict[str, Any]:
    clean_argmax = _argmax_token(clean_step.logits)
    perturbed_argmax = _argmax_token(perturbed_step.logits)
    cosine_distance, relative_l2 = _hidden_distances(
        clean_step.hidden_post,
        perturbed_step.hidden_post,
    )
    pre_norm = float(perturbed_step.event.hidden_pre_norm)
    delta_norm = float(perturbed_step.event.hidden_delta_norm)
    decision_index = int(state.next_decision_index)
    consumed_token_id = int(state.pending_token_id)
    pre_flip_features = _prefix_features(
        state,
        decision_index=decision_index,
        max_new_tokens=int(config.max_new_tokens),
        context_sequence_length=context.get("context_sequence_length"),
        consumed_token_text=clean_step.consumed_token_text,
        clean_step=clean_step,
    )
    return {
        "mode": "basins",
        "protocol": PROTOCOL,
        "decision_index": decision_index,
        "t": decision_index,
        **dict(context),
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
        "clean_top1_margin": pre_flip_features["clean_top1_margin"],
        "clean_diagnostics": clean_step.diagnostics,
        "pre_flip_features": pre_flip_features,
        "clean_hidden_norm": float(clean_step.event.hidden_post_norm),
        "perturbed_hidden_norm": float(
            perturbed_step.event.hidden_post_norm
        ),
        "hidden_cosine_distance": cosine_distance,
        "hidden_l2_relative": relative_l2,
        "logit_kl": float(logit_kl(clean_step.logits, perturbed_step.logits)),
        "logit_js": float(
            logit_jensen_shannon(clean_step.logits, perturbed_step.logits)
        ),
        "intervention_delta_norm": delta_norm,
        "intervention_delta_relative": (
            float(delta_norm / pre_norm) if pre_norm > 1e-12 else None
        ),
        "intervention_layer_resolved": int(resolved_layer),
        "intervention_type": str(config.intervention_type),
    }


def _continuation_seed(
    run_seed: int | None,
    decision_index: int,
    continuation_index: int,
) -> int:
    payload = (
        f"{0 if run_seed is None else int(run_seed)}:"
        f"{int(decision_index)}:{int(continuation_index)}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def _pending_is_eos(
    state: GenerationState,
    eos_token_id: int | None,
) -> bool:
    return bool(
        eos_token_id is not None
        and int(state.pending_token_id) == int(eos_token_id)
    )


def _continue_episode(
    *,
    engine: RuntimeEngine,
    clean_state: GenerationState,
    perturbed_state: GenerationState,
    decision_index: int,
    continuation_tokens: int,
    run_seed: int | None,
    eos_token_id: int | None,
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    outer_rng = torch.random.get_rng_state()
    try:
        for continuation_index in range(1, int(continuation_tokens) + 1):
            clean_active = not _pending_is_eos(clean_state, eos_token_id)
            perturbed_active = not _pending_is_eos(
                perturbed_state,
                eos_token_id,
            )
            if not clean_active and not perturbed_active:
                break
            paired_seed = _continuation_seed(
                run_seed,
                decision_index,
                continuation_index,
            )
            clean_step: StepResult | None = None
            perturbed_step: StepResult | None = None
            if clean_active:
                torch.manual_seed(paired_seed)
                clean_step = engine.step(
                    t=int(clean_state.next_decision_index),
                    consumed_token_id=int(clean_state.pending_token_id),
                    prompt_len=int(clean_state.prompt_len),
                    past_key_values=clean_state.past_key_values,
                    intervention_active=False,
                    collect_diagnostics=False,
                )
            if perturbed_active:
                torch.manual_seed(paired_seed)
                perturbed_step = engine.step(
                    t=int(perturbed_state.next_decision_index),
                    consumed_token_id=int(perturbed_state.pending_token_id),
                    prompt_len=int(perturbed_state.prompt_len),
                    past_key_values=perturbed_state.past_key_values,
                    intervention_active=False,
                    collect_diagnostics=False,
                )

            cosine_distance = None
            relative_l2 = None
            cross_context_kl = None
            cross_context_js = None
            if clean_step is not None and perturbed_step is not None:
                cosine_distance, relative_l2 = _hidden_distances(
                    clean_step.hidden_post,
                    perturbed_step.hidden_post,
                )
                cross_context_kl = float(
                    logit_kl(clean_step.logits, perturbed_step.logits)
                )
                cross_context_js = float(
                    logit_jensen_shannon(
                        clean_step.logits,
                        perturbed_step.logits,
                    )
                )
            row = {
                "protocol": CONTINUATION_PROTOCOL,
                "branchpoint_decision_index": int(decision_index),
                "continuation_index": continuation_index,
                "paired_sampling_seed": paired_seed,
                "sampling_seed_paired": True,
                "clean_active": clean_active,
                "perturbed_active": perturbed_active,
                "intervention_active_clean": False,
                "intervention_active_perturbed": False,
                "clean_consumed_token_id": (
                    int(clean_state.pending_token_id)
                    if clean_active
                    else None
                ),
                "perturbed_consumed_token_id": (
                    int(perturbed_state.pending_token_id)
                    if perturbed_active
                    else None
                ),
                "clean_predicted_token_id": (
                    int(clean_step.predicted_next_token_id)
                    if clean_step is not None
                    else None
                ),
                "perturbed_predicted_token_id": (
                    int(perturbed_step.predicted_next_token_id)
                    if perturbed_step is not None
                    else None
                ),
                "cross_context_hidden_cosine_distance": cosine_distance,
                "cross_context_hidden_l2_relative": relative_l2,
                "cross_context_logit_kl": cross_context_kl,
                "cross_context_logit_js": cross_context_js,
            }
            trace.append(row)
            if clean_step is not None:
                clean_state.advance(clean_step)
            if perturbed_step is not None:
                perturbed_state.advance(perturbed_step)
    finally:
        torch.random.set_rng_state(outer_rng)
    return trace


def _first_divergence(
    clean: list[int],
    perturbed: list[int],
) -> int | None:
    length = max(len(clean), len(perturbed))
    for index in range(length):
        clean_value = clean[index] if index < len(clean) else None
        perturbed_value = (
            perturbed[index] if index < len(perturbed) else None
        )
        if clean_value != perturbed_value:
            return index
    return None


def _token_match_rate(clean: list[int], perturbed: list[int]) -> float:
    length = max(len(clean), len(perturbed))
    if length == 0:
        return 1.0
    matches = sum(
        index < len(clean)
        and index < len(perturbed)
        and clean[index] == perturbed[index]
        for index in range(length)
    )
    return float(matches / length)


def _causal_classification(
    local_row: Mapping[str, Any],
    *,
    exact_token_match: bool,
) -> str:
    if bool(local_row.get("sampled_flip")):
        return "local_token_flip"
    if not exact_token_match:
        return "latent_token_divergence"
    if _has_measurable_local_effect(local_row):
        return "latent_state_effect_only"
    return "exact_no_effect"


def _write_jsonl_row(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(json_safe(dict(row)), ensure_ascii=False) + "\n")
    handle.flush()


def _validate_prompt_config(
    config: BasinConfig,
    prompt_spec: BasinPromptSpec | None,
) -> BasinPromptSpec:
    if prompt_spec is not None:
        spec = prompt_spec
    elif str(config.prompt_key) == "custom":
        spec = build_generic_prompt_spec(
            config.prompt,
            key=config.prompt_key,
            prompt_class=config.prompt_class,
        )
    else:
        spec = get_prompt_spec(config.prompt_key)
    if str(config.prompt) != spec.prompt:
        raise ValueError(
            "Basin config prompt does not match registered prompt "
            f"{spec.key!r}"
        )
    if str(config.prompt_class) != spec.prompt_class:
        raise ValueError(
            "Basin config prompt_class does not match registered prompt"
        )
    if str(config.evaluator_id) != spec.evaluator_id:
        raise ValueError(
            "Basin config evaluator_id does not match registered prompt"
        )
    return spec


def run_basin_experiment(
    config: BasinConfig,
    registry_path: str = "models.json",
    runs_dir: Optional[str] = None,
    diagnostics_config: Optional[DiagnosticsConfig] = None,
    prebuilt_backend: Optional[Any] = None,
    prompt_spec: BasinPromptSpec | None = None,
) -> Dict[str, Any]:
    """Map local forks, then causally continue an outcome-blind selection."""
    if int(config.max_new_tokens) < 2:
        raise ValueError("max_new_tokens must be at least 2")
    if int(config.continuation_tokens) < 1:
        raise ValueError("continuation_tokens must be at least 1")
    if int(config.max_episodes) < 1:
        raise ValueError("max_episodes must be at least 1")
    if float(config.tie_tolerance) < 0.0:
        raise ValueError("tie_tolerance must not be negative")
    spec = _validate_prompt_config(config, prompt_spec)

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
    engine = RuntimeEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_idx=resolved_layer,
        diagnostics_manager=diagnostics,
        intervention=_build_intervention(config),
        probe_layers=diag_cfg.probe_layers,
        mode="basins",
        temperature=float(config.temperature),
        top_p=float(config.top_p),
        top_k=int(config.top_k),
    )

    rubric_hash = score_output(spec, "")["rubric_hash"]
    run_config = {
        "mode": "basins",
        "protocol": PROTOCOL,
        "selection_protocol": SELECTION_PROTOCOL,
        "continuation_protocol": CONTINUATION_PROTOCOL,
        "prompt": config.prompt,
        "prompt_key": spec.key,
        "prompt_class": spec.prompt_class,
        "prompt_spec": spec.to_record(),
        "evaluator_id": spec.evaluator_id,
        "rubric_hash": rubric_hash,
        **identity,
        "backend": backend_result.backend,
        "backend_meta": dict(backend_result.backend_meta),
        "max_new_tokens": int(config.max_new_tokens),
        "continuation_tokens": int(config.continuation_tokens),
        "max_episodes": int(config.max_episodes),
        "seed": int(config.seed) if config.seed is not None else None,
        "temperature": float(config.temperature),
        "top_p": float(config.top_p),
        "top_k": int(config.top_k),
        "intervention_layer_requested": int(config.intervention_layer),
        "intervention_layer_resolved": resolved_layer,
        "intervention_type": str(config.intervention_type),
        "intervention_magnitude": float(config.intervention_magnitude),
        "intervention_magnitude_relative": bool(
            config.intervention_magnitude_relative
        ),
        "intervention_seed": int(config.intervention_seed),
        "projection_subspace_dim": int(config.projection_subspace_dim),
        "tie_tolerance": float(config.tie_tolerance),
        "with_diagnostics": bool(config.with_diagnostics),
        "probe_layers_resolved": list(
            dict.fromkeys(engine.probe_resolved.values())
        ),
        "num_layers": int(engine.num_layers),
        "seed_cache_fingerprint": seed_cache.fingerprint,
        "prompt_tokens": int(seed_cache.seq_len),
    }
    run_id, run_dir = create_run_dir(
        runs_dir or os.environ.get("RUNS_DIR", "runs"),
        "basins",
    )
    config_hash, config_path = write_config_record(run_dir, run_config)
    run_dir = Path(run_dir)
    branchpoints_path = run_dir / "branchpoints.jsonl"
    episodes_path = run_dir / "episodes.jsonl"
    trajectory_path = run_dir / "trajectories.jsonl"
    summary_path = run_dir / "summary.json"

    branchpoint_rows: list[dict[str, Any]] = []
    clean_reference_token_ids: list[int] = []
    replay_reference_token_ids: list[int] = []
    episodes: list[dict[str, Any]] = []
    replayed_indices: set[int] = set()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    try:
        state = generation_state_from_seed_cache(
            seed_cache,
            temperature=float(config.temperature),
            top_p=float(config.top_p),
            top_k=int(config.top_k),
        )
        with branchpoints_path.open("w", encoding="utf-8") as branchpoint_file:
            while state.next_decision_index < int(config.max_new_tokens):
                if _pending_is_eos(state, eos_token_id):
                    break
                decision_index = int(state.next_decision_index)
                clean_step, perturbed_step, context = _local_fork(
                    engine=engine,
                    state=state,
                    decision_index=decision_index,
                    collect_diagnostics=True,
                )
                row = _branchpoint_row(
                    config=config,
                    state=state,
                    clean_step=clean_step,
                    perturbed_step=perturbed_step,
                    context=context,
                    tokenizer=tokenizer,
                    resolved_layer=resolved_layer,
                )
                branchpoint_rows.append(row)
                _write_jsonl_row(branchpoint_file, row)
                state.advance_clean(clean_step)
        clean_reference_token_ids = list(state.generated_token_ids)
        if not branchpoint_rows:
            raise RuntimeError(
                "Basin run emitted no branchpoint decisions; the prompt ended "
                "before a causal fork could be measured"
            )

        selected = _select_branchpoints(
            branchpoint_rows,
            max_new_tokens=int(config.max_new_tokens),
            max_episodes=int(config.max_episodes),
        )
        selected_by_index = {
            int(row["decision_index"]): row for row in selected
        }

        _set_deterministic_state(config.seed)
        engine.reset()
        replay_state = generation_state_from_seed_cache(
            seed_cache,
            temperature=float(config.temperature),
            top_p=float(config.top_p),
            top_k=int(config.top_k),
        )
        with (
            episodes_path.open("w", encoding="utf-8") as episodes_file,
            trajectory_path.open("w", encoding="utf-8") as trajectory_file,
        ):
            while replay_state.next_decision_index < int(
                config.max_new_tokens
            ):
                if _pending_is_eos(replay_state, eos_token_id):
                    break
                decision_index = int(replay_state.next_decision_index)
                selected_row = selected_by_index.get(decision_index)
                if selected_row is None:
                    clean_step = engine.step(
                        t=decision_index,
                        consumed_token_id=int(replay_state.pending_token_id),
                        prompt_len=int(replay_state.prompt_len),
                        past_key_values=replay_state.past_key_values,
                        intervention_active=False,
                        collect_diagnostics=False,
                    )
                    replay_state.advance_clean(clean_step)
                    continue

                perturbed_source = replay_state.clone()
                shared_prefix_count = len(replay_state.generated_token_ids)
                clean_step, perturbed_step, context = _local_fork(
                    engine=engine,
                    state=replay_state,
                    decision_index=decision_index,
                    collect_diagnostics=False,
                )
                context_replay_verified = bool(
                    context["context_fingerprint"]
                    == selected_row["context_fingerprint"]
                )
                clean_decision_replay_verified = bool(
                    int(clean_step.predicted_next_token_id)
                    == int(selected_row["clean_predicted_token_id"])
                    and _argmax_token(clean_step.logits)
                    == int(selected_row["clean_argmax_token_id"])
                )
                if not context_replay_verified:
                    raise RuntimeError(
                        f"Decision {decision_index}: replay context mismatch"
                    )
                if not clean_decision_replay_verified:
                    raise RuntimeError(
                        f"Decision {decision_index}: clean replay mismatch"
                    )

                replay_state.advance_clean(clean_step)
                clean_branch = replay_state.clone()
                perturbed_source.advance(perturbed_step)
                perturbed_branch = perturbed_source
                trace = _continue_episode(
                    engine=engine,
                    clean_state=clean_branch,
                    perturbed_state=perturbed_branch,
                    decision_index=decision_index,
                    continuation_tokens=int(config.continuation_tokens),
                    run_seed=config.seed,
                    eos_token_id=eos_token_id,
                )

                clean_full = list(clean_branch.generated_token_ids)
                perturbed_full = list(
                    perturbed_branch.generated_token_ids
                )
                clean_branch_tokens = clean_full[shared_prefix_count:]
                perturbed_branch_tokens = perturbed_full[
                    shared_prefix_count:
                ]
                clean_text = tokenizer.decode(
                    clean_full,
                    skip_special_tokens=True,
                )
                perturbed_text = tokenizer.decode(
                    perturbed_full,
                    skip_special_tokens=True,
                )
                score_comparison = compare_outputs(
                    spec,
                    clean_text=clean_text,
                    perturbed_text=perturbed_text,
                    tie_tolerance=float(config.tie_tolerance),
                )
                exact_token_match = clean_full == perturbed_full
                episode_id = (
                    f"episode_{len(episodes) + 1:03d}_"
                    f"decision_{decision_index:03d}"
                )
                local_counterfactual = {
                    key: selected_row.get(key)
                    for key in (
                        "argmax_flip",
                        "sampled_flip",
                        "logit_kl",
                        "logit_js",
                        "hidden_cosine_distance",
                        "hidden_l2_relative",
                        "intervention_delta_norm",
                        "intervention_delta_relative",
                    )
                }
                episode = {
                    "mode": "basins",
                    "protocol": PROTOCOL,
                    "episode_id": episode_id,
                    "decision_index": decision_index,
                    "position_band": selected_row["position_band"],
                    "selection_basis": selected_row["selection_basis"],
                    "selection_protocol": SELECTION_PROTOCOL,
                    "selection_outcome_blind": True,
                    "context_fingerprint": context[
                        "context_fingerprint"
                    ],
                    "context_replay_verified": context_replay_verified,
                    "clean_decision_replay_verified": (
                        clean_decision_replay_verified
                    ),
                    "pre_flip_features": selected_row[
                        "pre_flip_features"
                    ],
                    "local_counterfactual": local_counterfactual,
                    "shared_prefix_token_count": shared_prefix_count,
                    "clean_branch_token_ids": clean_branch_tokens,
                    "perturbed_branch_token_ids": perturbed_branch_tokens,
                    "clean_full_token_ids": clean_full,
                    "perturbed_full_token_ids": perturbed_full,
                    "clean_output": clean_text,
                    "perturbed_output": perturbed_text,
                    "continuation_tokens_requested": int(
                        config.continuation_tokens
                    ),
                    "continuation_steps_recorded": len(trace),
                    "continuation_protocol": CONTINUATION_PROTOCOL,
                    "continuation_intervention_disabled": all(
                        not row["intervention_active_clean"]
                        and not row["intervention_active_perturbed"]
                        for row in trace
                    ),
                    "sampling_seed_paired": all(
                        row["sampling_seed_paired"] for row in trace
                    ),
                    "exact_token_match": exact_token_match,
                    "first_divergence_offset": _first_divergence(
                        clean_branch_tokens,
                        perturbed_branch_tokens,
                    ),
                    "token_match_rate": _token_match_rate(
                        clean_branch_tokens,
                        perturbed_branch_tokens,
                    ),
                    "causal_classification": _causal_classification(
                        selected_row,
                        exact_token_match=exact_token_match,
                    ),
                    "score_comparison": score_comparison,
                }
                episodes.append(episode)
                replayed_indices.add(decision_index)
                _write_jsonl_row(episodes_file, episode)
                for trace_row in trace:
                    _write_jsonl_row(
                        trajectory_file,
                        {
                            "mode": "basins",
                            "episode_id": episode_id,
                            **trace_row,
                        },
                    )
                (run_dir / f"{episode_id}_clean.txt").write_text(
                    config.prompt + "\n\n" + clean_text,
                    encoding="utf-8",
                )
                (run_dir / f"{episode_id}_perturbed.txt").write_text(
                    config.prompt + "\n\n" + perturbed_text,
                    encoding="utf-8",
                )
        replay_reference_token_ids = list(
            replay_state.generated_token_ids
        )
    finally:
        engine.close()

    selected_indices = {
        int(row["decision_index"])
        for row in _select_branchpoints(
            branchpoint_rows,
            max_new_tokens=int(config.max_new_tokens),
            max_episodes=int(config.max_episodes),
        )
    }
    replay_all_contexts_verified = bool(
        replayed_indices == selected_indices
        and all(
            row["context_replay_verified"]
            and row["clean_decision_replay_verified"]
            for row in episodes
        )
    )
    clean_reference_replay_exact = bool(
        clean_reference_token_ids == replay_reference_token_ids
    )
    if not replay_all_contexts_verified:
        raise RuntimeError(
            "Basin replay did not verify every selected branchpoint context"
        )
    if not clean_reference_replay_exact:
        raise RuntimeError(
            "Basin episodes altered or failed to reproduce the clean reference"
        )
    identity_control = _is_identity_control(config)
    identity_control_exact = (
        bool(
            bool(branchpoint_rows)
            and bool(episodes)
            and all(
                not row["argmax_flip"]
                and not row["sampled_flip"]
                and abs(float(row["logit_kl"])) <= 1e-12
                and abs(float(row["logit_js"])) <= 1e-12
                for row in branchpoint_rows
            )
            and all(
                row["exact_token_match"]
                and row["score_comparison"]["outcome"] == "tie"
                for row in episodes
            )
        )
        if identity_control
        else None
    )
    outcomes = {
        outcome: sum(
            row["score_comparison"]["outcome"] == outcome
            for row in episodes
        )
        for outcome in ("improve", "degrade", "tie")
    }
    classifications = {
        label: sum(
            row["causal_classification"] == label
            for row in episodes
        )
        for label in (
            "local_token_flip",
            "latent_token_divergence",
            "latent_state_effect_only",
            "exact_no_effect",
        )
    }
    protocol_validity = {
        "protocol": PROTOCOL,
        "pass1_independent_one_step_forks": True,
        "pass1_continued_clean_only": True,
        "pass1_all_contexts_matched": all(
            row["contexts_matched"] for row in branchpoint_rows
        ),
        "selection_protocol": SELECTION_PROTOCOL,
        "selection_outcome_blind": True,
        "selected_decision_indices": sorted(selected_indices),
        "replayed_decision_indices": sorted(replayed_indices),
        "replay_all_contexts_verified": replay_all_contexts_verified,
        "clean_reference_replay_exact": clean_reference_replay_exact,
        "continuation_protocol": CONTINUATION_PROTOCOL,
        "continuation_intervention_disabled": all(
            row["continuation_intervention_disabled"]
            for row in episodes
        ),
        "continuation_sampling_rng_paired": all(
            row["sampling_seed_paired"] for row in episodes
        ),
        "branchpoint_rows": len(branchpoint_rows),
        "selected_episodes": len(episodes),
    }
    metrics = {
        "branchpoint_rows": len(branchpoint_rows),
        "local_effect_rows": sum(
            int(_has_measurable_local_effect(row))
            for row in branchpoint_rows
        ),
        "argmax_flips": sum(
            int(row["argmax_flip"]) for row in branchpoint_rows
        ),
        "sampled_flips": sum(
            int(row["sampled_flip"]) for row in branchpoint_rows
        ),
        "selected_episodes": len(episodes),
        "mean_logit_kl": (
            float(
                sum(float(row["logit_kl"]) for row in branchpoint_rows)
                / len(branchpoint_rows)
            )
            if branchpoint_rows
            else 0.0
        ),
        "mean_logit_js": (
            float(
                sum(float(row["logit_js"]) for row in branchpoint_rows)
                / len(branchpoint_rows)
            )
            if branchpoint_rows
            else 0.0
        ),
        "selected_sampled_flips": sum(
            int(row["local_counterfactual"]["sampled_flip"])
            for row in episodes
        ),
        "outcomes": outcomes,
        "causal_classifications": classifications,
        "identity_control": identity_control,
        "identity_control_exact": identity_control_exact,
    }
    summary = {
        "config_hash": config_hash,
        "mode": "basins",
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
        "prompt_key": spec.key,
        "prompt_class": spec.prompt_class,
        "rubric_hash": rubric_hash,
        "clean_reference_token_ids": clean_reference_token_ids,
        "clean_reference_output": tokenizer.decode(
            clean_reference_token_ids,
            skip_special_tokens=True,
        ),
        "replay_reference_token_ids": replay_reference_token_ids,
        "protocol_validity": protocol_validity,
        "metrics": metrics,
        "diagnostics_health": summarize_diagnostics_health(
            [row["clean_diagnostics"] for row in branchpoint_rows]
        ),
        "config": run_config,
        "artifacts": {
            "run_dir": str(run_dir),
            "config_path": str(config_path),
            "branchpoints_path": str(branchpoints_path),
            "episodes_path": str(episodes_path),
            "trajectory_path": str(trajectory_path),
            "summary_path": str(summary_path),
        },
    }
    save_json(str(summary_path), summary)
    return summary
