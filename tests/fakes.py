from __future__ import annotations

from types import SimpleNamespace

import torch

from runtime_lab.core.backend.loader import BackendLoadResult, ModelConfig
from runtime_lab.core.model.cache_utils import cache_sequence_length


class FakeTokenizer:
    eos_token_id = 3
    eos_token = "<eos>"
    pad_token_id = 3
    pad_token = "<eos>"

    _tokens = {
        0: "A",
        1: "B",
        2: "P",
        3: "<eos>",
    }

    def __call__(self, text, return_tensors="pt"):
        del text, return_tensors
        input_ids = torch.tensor([[2]], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().reshape(-1).tolist()
        elif isinstance(token_ids, int):
            token_ids = [token_ids]
        pieces = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id == self.eos_token_id:
                continue
            pieces.append(self._tokens.get(token_id, f"<{token_id}>"))
        return "".join(pieces)


class _FakeLayer(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class _FakeTransformer(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.h = torch.nn.ModuleList([_FakeLayer() for _ in range(num_layers)])


class FakeCausalModel(torch.nn.Module):
    """Tiny hookable model with one deliberately low-margin local decision."""

    def __init__(self, num_layers=3):
        super().__init__()
        self.transformer = _FakeTransformer(num_layers)
        self.last_attention_mask = None

    def forward(
        self,
        input_ids,
        attention_mask,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    ):
        del use_cache, return_dict
        self.last_attention_mask = attention_mask.detach().cpu()
        previous_length = cache_sequence_length(past_key_values) or 0
        input_length = int(input_ids.shape[1])
        total_length = previous_length + input_length

        positions = torch.arange(
            previous_length + 1,
            total_length + 1,
            device=input_ids.device,
            dtype=torch.long,
        )
        strengths = torch.where(
            positions == 2,
            torch.ones_like(positions, dtype=torch.float32),
            torch.full_like(positions, 10, dtype=torch.float32),
        )
        hidden = torch.zeros(
            (input_ids.shape[0], input_length, 4),
            device=input_ids.device,
            dtype=torch.float32,
        )
        hidden[..., 0] = strengths
        hidden[..., 1] = input_ids.float()

        for layer in self.transformer.h:
            hidden = layer(hidden)

        logits = torch.full(
            (input_ids.shape[0], input_length, 4),
            -10.0,
            device=input_ids.device,
        )
        logits[..., 0] = hidden[..., 0]
        logits[..., 1] = 0.5

        key = torch.zeros((1, 1, total_length, 2), device=input_ids.device)
        if past_key_values is not None and previous_length:
            old_key = past_key_values[0][0]
            key[:, :, :previous_length, :] = old_key
        key[:, :, previous_length:, 0] = input_ids.float().reshape(1, 1, input_length)
        key[:, :, previous_length:, 1] = positions.float().reshape(1, 1, input_length)
        value = key.clone()

        return SimpleNamespace(
            logits=logits,
            past_key_values=((key, value),),
        )


def make_fake_backend(num_layers=3):
    model = FakeCausalModel(num_layers=num_layers)
    model.eval()
    return BackendLoadResult(
        tokenizer=FakeTokenizer(),
        model=model,
        device=torch.device("cpu"),
        config=ModelConfig(
            key="audit-fake",
            hf_id="local/audit-fake",
            device="cpu",
            dtype="float32",
        ),
        backend="hf",
        backend_meta={
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "requested_dtype": "float32",
            "resolved_dtype": "float32",
            "policy_notes": [],
            "device_map": "cpu",
        },
    )
