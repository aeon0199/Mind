from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from runtime_lab.core.model.cache_utils import clone_past_key_values
from runtime_lab.core.runtime.engine import StepResult
from runtime_lab.core.sampling import sample_token_id
from runtime_lab.stress.seed_cache import SeedCache


@dataclass
class GenerationState:
    """The clean continuation state at one local causal decision boundary."""

    prompt_len: int
    past_key_values: Any
    pending_token_id: int
    generated_token_ids: list[int] = field(default_factory=list)
    next_decision_index: int = 1
    last_hidden: Optional[torch.Tensor] = None
    last_logits: Optional[torch.Tensor] = None

    def fork_cache(self) -> Any:
        return clone_past_key_values(self.past_key_values)

    def advance_clean(self, step: StepResult) -> None:
        self.past_key_values = step.past_key_values
        self.pending_token_id = int(step.predicted_next_token_id)
        self.generated_token_ids.append(int(step.predicted_next_token_id))
        self.next_decision_index += 1
        self.last_hidden = (
            step.measure_hidden.detach().cpu()
            if isinstance(step.measure_hidden, torch.Tensor)
            else None
        )
        self.last_logits = step.logits.detach().cpu()


def generation_state_from_seed_cache(
    seed_cache: SeedCache,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> GenerationState:
    first_token_id = sample_token_id(
        seed_cache.next_token_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return GenerationState(
        prompt_len=int(seed_cache.seq_len),
        past_key_values=clone_past_key_values(seed_cache.past_key_values),
        pending_token_id=int(first_token_id),
        generated_token_ids=[int(first_token_id)],
        next_decision_index=1,
        last_hidden=(
            seed_cache.seed_hidden.detach().cpu()
            if seed_cache.seed_hidden_available
            else None
        ),
        last_logits=seed_cache.next_token_logits.detach().cpu(),
    )
