# v1/hooks.py

"""
Telemetry Hooks
---------------
Forward hooks for capturing final-token hidden states and logits.

Design goals:
- Keep tensors on the same device as the model (no .cpu()) to avoid device mismatch
  with runner-side computations (e.g., random projection).
- Normalize captured representations to 1D vectors:
  - hidden: [H]
  - logits: [V]
- Be robust across common HF decoder-only model layouts.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def _as_tensor(x: Any) -> Optional[torch.Tensor]:
    """Best-effort: extract a tensor from possible tuple/list outputs."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)) and len(x) > 0:
        # Many modules return tuples like (hidden_states, ...)
        if isinstance(x[0], torch.Tensor):
            return x[0]
    return None


def _last_token_vector(x: torch.Tensor) -> torch.Tensor:
    """
    Convert common shapes to a 1D vector for batch=0, last token.

    Supported shapes:
    - [H] -> [H]
    - [1, H] -> [H]
    - [B, H] -> [H] (takes batch 0)
    - [B, T, H] -> [H] (takes batch 0, last token)
    - [T, H] -> [H] (takes last token)
    """
    if x.dim() == 1:
        return x
    if x.dim() == 2:
        # Could be [B, H] or [T, H] or [1, H]
        # Prefer batch=0 if B present; if it's [T, H], batch=0 is also fine
        return x[0]
    if x.dim() == 3:
        return x[0, -1, :]
    # Fallback: flatten (should not happen in normal decoder-only hidden/logits)
    return x.reshape(-1)


class HiddenStateCapture:
    """Hook that captures the final-token hidden state as a 1D vector [H]."""

    def __init__(self):
        self.captured: Optional[torch.Tensor] = None

    def __call__(self, module, inputs, output):
        hs = _as_tensor(output)
        if hs is None:
            self.captured = None
            return
        self.captured = _last_token_vector(hs).detach()

    def reset(self):
        self.captured = None


class LogitCapture:
    """Hook that captures final-token logits as a 1D vector [V]."""

    def __init__(self):
        self.captured: Optional[torch.Tensor] = None

    def __call__(self, module, inputs, output):
        out = _as_tensor(output)
        if out is None:
            self.captured = None
            return
        self.captured = _last_token_vector(out).detach()

    def reset(self):
        self.captured = None


class HookManager:
    """Registers and manages hooks on common decoder-only HF models."""

    def __init__(self, model):
        self.model = model
        self.hidden_hook = HiddenStateCapture()
        self.logit_hook = LogitCapture()
        self.handles = []

    def _find_final_block(self):
        """
        Try common locations for the final transformer block across HF architectures.
        Returns a module or raises AttributeError.
        """
        candidates = []

        # Common patterns:
        # - LLaMA/Qwen-style: model.model.layers
        # - GPT-NeoX style: model.gpt_neox.layers
        # - GPT-2 style: model.transformer.h
        # - Falcon style: model.transformer.h
        # - MPT style: model.transformer.blocks
        m = self.model

        if hasattr(m, "model") and hasattr(m.model, "layers"):
            candidates.append(m.model.layers)
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            candidates.append(m.gpt_neox.layers)
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            candidates.append(m.transformer.h)
        if hasattr(m, "transformer") and hasattr(m.transformer, "blocks"):
            candidates.append(m.transformer.blocks)

        for layer_list in candidates:
            try:
                if len(layer_list) > 0:
                    return layer_list[-1]
            except Exception:
                continue

        raise AttributeError("Could not locate final transformer block for hidden hook.")

    def register(self):
        """Attach hooks to final transformer block and lm_head (or equivalent)."""
        # Hidden-state hook
        try:
            final_block = self._find_final_block()
            h1 = final_block.register_forward_hook(self.hidden_hook)
            self.handles.append(h1)
        except Exception as e:
            print(f"[V1] ERROR attaching hidden-state hook: {e}")

        # Logit hook: typically lm_head
        try:
            if hasattr(self.model, "lm_head"):
                lm_head = self.model.lm_head
            elif hasattr(self.model, "embed_out"):
                # Some models use embed_out
                lm_head = self.model.embed_out
            else:
                raise AttributeError("Could not locate lm_head/embed_out for logit hook.")

            h2 = lm_head.register_forward_hook(self.logit_hook)
            self.handles.append(h2)
        except Exception as e:
            print(f"[V1] ERROR attaching logit hook: {e}")

        print("[V1] Hooks registered.")

    def reset(self):
        self.hidden_hook.reset()
        self.logit_hook.reset()

    def cleanup(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
