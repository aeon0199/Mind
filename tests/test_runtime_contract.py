from pathlib import Path
from types import SimpleNamespace

import torch

from runtime_lab.core.runtime.engine import RuntimeEngine
from runtime_lab.core.runtime.events import RuntimeEvent, runtime_event_to_record
from runtime_lab.core.model.cache_utils import (
    clone_past_key_values,
    compute_cache_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]


class _FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return f"<{int(token_ids[0])}>"


class _FakeLayer(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = SimpleNamespace(h=torch.nn.ModuleList([_FakeLayer()]))
        self.last_attention_mask = None

    def forward(
        self,
        input_ids,
        attention_mask,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    ):
        del past_key_values, use_cache, return_dict
        self.last_attention_mask = attention_mask.detach().cpu()
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        for layer in self.transformer.h:
            hidden = layer(hidden)
        logits = torch.zeros((*input_ids.shape, 16), device=input_ids.device)
        logits[..., 11] = 1.0
        cache_len = int(attention_mask.shape[-1])
        key = torch.zeros((1, 1, cache_len, 2), device=input_ids.device)
        return SimpleNamespace(logits=logits, past_key_values=((key, key.clone()),))


def _sample_runtime_event():
    return RuntimeEvent(
        t=3,
        consumed_token_id=7,
        consumed_token_text="<7>",
        predicted_next_token_id=11,
        resolved_layer_idx=4,
        measure_resolved_layer_idx=2,
        act_resolved_layer_idx=4,
        hidden_pre_norm=1.0,
        hidden_post_norm=1.2,
        hidden_delta_norm=0.2,
        diagnostics={"divergence": 0.1},
        intervention_active=True,
        mode="stress",
    )


def test_runtime_event_record_preserves_both_sides_of_the_decision():
    record = runtime_event_to_record(_sample_runtime_event())

    assert record["consumed_token_id"] == 7
    assert record["consumed_token_text"] == "<7>"
    assert record["predicted_next_token_id"] == 11
    assert record["token_id"] == 7
    assert record["token_text"] == "<7>"
    assert record["measure_resolved_layer_idx"] == 2
    assert record["act_resolved_layer_idx"] == 4


def test_step_attention_mask_uses_cache_length_plus_input_token():
    model = _FakeModel()
    engine = RuntimeEngine(
        model=model,
        tokenizer=_FakeTokenizer(),
        device=torch.device("cpu"),
        layer_idx=0,
    )
    key = torch.zeros((1, 1, 5, 2))
    cache_with_length_five = ((key, key.clone()),)

    try:
        engine.step(
            t=99,
            consumed_token_id=3,
            prompt_len=2,
            past_key_values=cache_with_length_five,
        )
    finally:
        engine.close()

    assert model.last_attention_mask.shape[-1] == 6


def test_observe_and_control_use_the_canonical_event_serializer():
    observe = (ROOT / "src" / "runtime_lab" / "observe" / "runner.py").read_text()
    control = (ROOT / "src" / "runtime_lab" / "control" / "adaptive_runner.py").read_text()

    assert "runtime_event_to_record(step.event)" in observe
    assert "runtime_event_to_record(ctrl_evt)" in control


def test_transformers_v5_layer_cache_can_be_cloned_and_fingerprinted():
    key = torch.arange(24, dtype=torch.float32).reshape(1, 1, 6, 4)

    class V5Cache:
        def __init__(self):
            self.layers = [
                SimpleNamespace(keys=key.clone(), values=(key + 1).clone())
            ]

        def get_seq_length(self):
            return int(self.layers[0].keys.shape[-2])

    source = V5Cache()
    clone = clone_past_key_values(source)

    assert compute_cache_fingerprint(source) != "unavailable"
    assert compute_cache_fingerprint(source) == compute_cache_fingerprint(clone)
    clone.layers[0].keys.add_(100)
    assert not torch.equal(source.layers[0].keys, clone.layers[0].keys)
