from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RuntimeEvent:
    t: int
    consumed_token_id: int
    consumed_token_text: str
    predicted_next_token_id: Optional[int]
    resolved_layer_idx: int
    hidden_pre_norm: float
    hidden_post_norm: float
    hidden_delta_norm: float
    measure_resolved_layer_idx: Optional[int] = None
    act_resolved_layer_idx: Optional[int] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    intervention_active: bool = False
    mode: str = "runtime"


@dataclass
class ControlEvent(RuntimeEvent):
    scale_used: float = 1.0
    next_scale: float = 1.0
    status: str = "STABLE"
    shadow: bool = False


def runtime_event_to_record(event: RuntimeEvent) -> Dict[str, Any]:
    """Serialize one causal token decision with explicit and legacy names."""
    record = asdict(event)
    record["token_id"] = int(event.consumed_token_id)
    record["token_text"] = str(event.consumed_token_text)
    return record
