from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Optional

import torch


def _vector(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    vector = value.detach().float().cpu().reshape(-1)
    if not torch.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values")
    return vector


def _logit_metrics(logits: Optional[torch.Tensor]) -> tuple[float | None, float | None]:
    if logits is None:
        return None, None
    vector = _vector(logits, name="logits")
    if vector.numel() == 0:
        return None, None

    log_probs = torch.log_softmax(vector, dim=-1)
    probabilities = log_probs.exp()
    entropy = float(-(probabilities * log_probs).sum().item())
    if vector.numel() < 2:
        margin = None
    else:
        top_two = torch.topk(vector, k=2).values
        margin = float((top_two[0] - top_two[1]).item())
    return entropy, margin


class BaselineProbe:
    """Simple, directly interpretable token-to-token diagnostic baselines."""

    def __init__(self) -> None:
        self._previous_hidden: Optional[torch.Tensor] = None
        self._previous_velocity: Optional[torch.Tensor] = None
        self._steps = 0

    def reset(self) -> None:
        self._previous_hidden = None
        self._previous_velocity = None
        self._steps = 0

    def step(
        self,
        hidden: torch.Tensor,
        *,
        logits: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        current = _vector(hidden, name="hidden")
        if (
            self._previous_hidden is not None
            and current.numel() != self._previous_hidden.numel()
        ):
            raise ValueError("hidden dimension changed within one baseline trajectory")

        if self._previous_hidden is None:
            velocity_vector = torch.zeros_like(current)
            velocity = 0.0
            acceleration = 0.0
        else:
            velocity_vector = current - self._previous_hidden
            velocity = float(torch.linalg.vector_norm(velocity_vector).item())
            if self._previous_velocity is None:
                acceleration = 0.0
            else:
                acceleration = float(
                    torch.linalg.vector_norm(
                        velocity_vector - self._previous_velocity
                    ).item()
                )

        entropy, margin = _logit_metrics(logits)
        self._previous_hidden = current.clone()
        self._previous_velocity = velocity_vector.clone()
        self._steps += 1
        return {
            "hidden_velocity": velocity,
            "hidden_acceleration": acceleration,
            "logit_entropy": entropy,
            "top1_margin": margin,
            "baseline_step": int(self._steps),
        }


def run_baseline_probe(
    hidden_states: Iterable[torch.Tensor],
    *,
    logits: Optional[Sequence[Optional[torch.Tensor]]] = None,
) -> list[dict[str, Any]]:
    states = list(hidden_states)
    if logits is not None and len(logits) != len(states):
        raise ValueError("logits and hidden_states must have equal lengths")

    probe = BaselineProbe()
    return [
        probe.step(
            hidden,
            logits=None if logits is None else logits[index],
        )
        for index, hidden in enumerate(states)
    ]
