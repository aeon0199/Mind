"""utils.py

Utilities for V1.5 observability.

This keeps the original helpers (basic_stats, svd_compress, save_json) and adds
compact, *usable* state signatures for single-pass runners:

- simhash64(hidden): stable 64-bit sign-hash of the hidden vector
- project_hidden(hidden, out_dim): low-dim deterministic random projection
- WindowedSVDProbe: meaningful SVD signature over a sliding window of hidden states

Why windowed SVD?
- SVD of a single vector (treated as 1×D) yields exactly one singular value,
  which is just the vector norm. A windowed SVD over the last W states captures
  local trajectory rank/energy distribution and is actually informative.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


_EPS = 1e-8

# ---------------------------------------------------------------------------
# Basic tensor stats
# ---------------------------------------------------------------------------

def tensor_norm(x: torch.Tensor) -> float:
    if not isinstance(x, torch.Tensor):
        return 0.0
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def tensor_entropy(logits: torch.Tensor) -> float:
    """Entropy of a logit vector (1D or [1,V]) in nats."""
    if not isinstance(logits, torch.Tensor):
        return 0.0
    x = logits.detach().float()
    if x.dim() > 1:
        x = x.view(-1)
    probs = torch.softmax(x, dim=-1)
    ent = -(probs * (probs + _EPS).log()).sum()
    return float(ent.item())


def topk_values(logits: torch.Tensor, k: int = 5) -> Dict[str, Any]:
    """Return top-k token ids, logits, and probabilities from a logit vector."""
    if not isinstance(logits, torch.Tensor):
        return {"k": k, "ids": [], "logits": [], "probs": []}

    x = logits.detach().float()

    # 🔒 Guardrail: scalar or empty logits are invalid for top-k
    if x.dim() == 0 or x.numel() <= 1:
        return {"k": k, "ids": [], "logits": [], "probs": []}

    if x.dim() > 1:
        x = x.view(-1)

    k = int(min(k, x.numel()))
    vals, idx = torch.topk(x, k=k)
    probs = torch.softmax(x, dim=-1).gather(0, idx)

    return {
        "k": k,
        "ids": [int(i) for i in idx.tolist()],
        "logits": [float(v) for v in vals.tolist()],
        "probs": [float(p) for p in probs.tolist()],
    }



def basic_stats(hidden: torch.Tensor, logits: torch.Tensor) -> Dict[str, Any]:
    """Standard per-token stats used across runners."""
    return {
        "hidden_norm": tensor_norm(hidden),
        "logit_norm": tensor_norm(logits),
        "entropy": tensor_entropy(logits),
        "topk": topk_values(logits, k=5),
    }

# ---------------------------------------------------------------------------
# SVD signatures
# ---------------------------------------------------------------------------

def svd_compress(x: torch.Tensor, k: int = 8) -> List[float]:
    """Return up to k singular values of x.

    Notes:
    - If x is 1D (a single hidden vector), this returns a single value equal to ||x||.
      That is mathematically correct but not a rich state representation.
    - For a meaningful SVD signature in V1.5, use WindowedSVDProbe below.
    """
    if not isinstance(x, torch.Tensor):
        return []
    t = x.detach().float()
    if t.dim() == 1:
        t = t.unsqueeze(0)  # 1×D
    try:
        s = torch.linalg.svdvals(t)
        s = s[: int(k)]
        return [float(v) for v in s.tolist()]
    except Exception:
        return []

# ---------------------------------------------------------------------------
# Deterministic projections and compact state hashes
# ---------------------------------------------------------------------------

_PROJ_CACHE: Dict[Tuple[int, int, int], torch.Tensor] = {}


def _get_rademacher_proj(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    """Deterministic ±1/sqrt(out_dim) projection matrix on CPU."""
    key = (int(in_dim), int(out_dim), int(seed))
    if key in _PROJ_CACHE:
        return _PROJ_CACHE[key]

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    m = torch.randint(0, 2, (in_dim, out_dim), generator=g, dtype=torch.int8)
    m = (m * 2 - 1).to(torch.float32)  # {-1, +1}
    m = m / math.sqrt(float(out_dim))
    _PROJ_CACHE[key] = m
    return m


def project_hidden(hidden: torch.Tensor, out_dim: int = 32, seed: int = 0) -> List[float]:
    """Deterministic low-dim projection of a hidden vector, returned as a list."""
    if not isinstance(hidden, torch.Tensor):
        return []
    h = hidden.detach().float().cpu().view(-1)
    proj = _get_rademacher_proj(h.numel(), int(out_dim), int(seed))
    z = h @ proj  # [out_dim]
    return [float(v) for v in z.tolist()]


def simhash64(hidden: torch.Tensor, seed: int = 0) -> str:
    """64-bit sign-hash of hidden state, returned as 16-hex characters."""
    if not isinstance(hidden, torch.Tensor):
        return "0" * 16
    h = hidden.detach().float().cpu().view(-1)
    proj = _get_rademacher_proj(h.numel(), 64, int(seed))
    scores = h @ proj  # [64]
    bits = (scores > 0).to(torch.int64).tolist()

    value = 0
    for i, b in enumerate(bits):
        value |= (int(b) & 1) << i
    return f"{value:016x}"


# ---------------------------------------------------------------------------
# Windowed SVD probe (meaningful state signature)
# ---------------------------------------------------------------------------

@dataclass
class WindowedSVDProbeConfig:
    window_size: int = 8
    top_k: int = 8
    center: bool = True


class WindowedSVDProbe:
    """SVD signature over a rolling window of hidden vectors.

    Uses the Gram trick: for X (T×D), eigenvalues of (X X^T) give singular values^2,
    so the computation is O(T^2 D + T^3) with small T (window_size).
    """

    def __init__(self, config: Optional[WindowedSVDProbeConfig] = None):
        self.config = config or WindowedSVDProbeConfig()
        if self.config.window_size < 2:
            raise ValueError("window_size must be >= 2")
        self._buf: List[torch.Tensor] = []

    def reset(self) -> None:
        self._buf.clear()

    def step(self, hidden: torch.Tensor) -> Dict[str, Any]:
        if not isinstance(hidden, torch.Tensor):
            return {"window_len": len(self._buf), "singular_values": [], "effective_rank": 0.0}

        h = hidden.detach().float().cpu().view(-1)
        self._buf.append(h)
        if len(self._buf) > int(self.config.window_size):
            self._buf.pop(0)

        t = len(self._buf)
        if t < 2:
            return {"window_len": t, "singular_values": [], "effective_rank": 0.0}

        X = torch.stack(self._buf, dim=0)  # (T, D)
        if self.config.center:
            X = X - X.mean(dim=0, keepdim=True)

        G = X @ X.T  # (T, T)
        # Eigenvalues are singular_values^2
        eig = torch.linalg.eigvalsh(G).clamp(min=0.0)
        eig = torch.flip(eig, dims=[0])  # descending

        sv = torch.sqrt(eig + _EPS)
        sv_k = sv[: int(self.config.top_k)]
        s2 = eig
        total = float(s2.sum().item()) + _EPS
        p = s2 / total

        # Effective rank = exp(entropy(p))
        eff_rank = float(torch.exp(-(p * (p + _EPS).log()).sum()).item())

        top1_frac = float((s2[0] / total).item()) if s2.numel() else 0.0

        return {
            "window_len": t,
            "singular_values": [float(v) for v in sv_k.tolist()],
            "effective_rank": eff_rank,
            "top1_energy_frac": top1_frac,
        }

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)