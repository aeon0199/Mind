"""predictor.py

Sliding-window divergence predictor for V1.5.

Goal:
- Provide a cheap, streaming scalar that increases when the current hidden state
  is poorly explained by recent local dynamics.

Design constraints:
- Must run per-token without allocating O(D^2) objects (D = model hidden size).
- Must be stable for short windows (window_size ~ 8).

Approach:
1) Project the hidden vector h (D) into a low-dimensional space z (k) using a
   fixed, deterministic Rademacher projection (±1 / sqrt(k)).
2) Fit a VAR(1) model in z-space on the recent window via ridge regression:
      z_{t} ≈ z_{t-1} A
3) Divergence = combined normalized L2 error and cosine distance between
   the predicted z_t and the observed z_t.

The projection makes the predictor tractable and the scalar comparable across
models with different hidden sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import torch


_EPS = 1e-8


def _to_1d_np(x: torch.Tensor) -> np.ndarray:
    """Detach -> float32 -> CPU -> flatten."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    return x.detach().float().cpu().view(-1).numpy().astype(np.float32, copy=False)


def _rademacher_projection(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    """Deterministic ±1/sqrt(out_dim) projection matrix (in_dim, out_dim)."""
    rng = np.random.default_rng(seed)
    # 0/1 -> -1/+1
    m = rng.integers(0, 2, size=(in_dim, out_dim), dtype=np.int8)
    m = (m * 2 - 1).astype(np.float32)
    m /= np.sqrt(float(out_dim))
    return m


@dataclass
class DivergenceDetails:
    n: int
    proj_dim: int
    l2: float
    cosine: float
    combined: float


class StateWindow:
    """Fixed-length FIFO for low-dim state vectors."""

    def __init__(self, maxlen: int):
        if maxlen < 2:
            raise ValueError("StateWindow maxlen must be >= 2")
        self.maxlen = int(maxlen)
        self._buf: List[np.ndarray] = []

    def reset(self) -> None:
        self._buf.clear()

    def add(self, v: np.ndarray) -> None:
        if v.ndim != 1:
            v = v.reshape(-1)
        self._buf.append(v.astype(np.float32, copy=False))
        if len(self._buf) > self.maxlen:
            self._buf.pop(0)

    def __len__(self) -> int:
        return len(self._buf)

    def matrix(self) -> np.ndarray:
        if not self._buf:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(self._buf, axis=0)  # (T, k)


def _fit_var1_ridge(states: np.ndarray, ridge: float) -> np.ndarray:
    """Fit A in z_t ≈ z_{t-1} A using ridge regression.

    states: (T, k) with T >= 2
    returns: A (k, k)
    """
    x = states[:-1, :]  # (T-1, k)
    y = states[1:, :]   # (T-1, k)

    # Solve (X^T X + λI) A = X^T Y
    xtx = x.T @ x
    k = xtx.shape[0]
    rhs = x.T @ y
    a = np.linalg.solve(xtx + ridge * np.eye(k, dtype=np.float32), rhs).astype(np.float32, copy=False)
    return a


def _divergence(pred: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """Return normalized L2 and cosine distance metrics.

    Robustness note:
    - The original normalization divided by ||actual||, which can produce rare,
      extremely large spikes when ||actual|| is small (a common edge case after
      low-dimensional projections).
    - We use a symmetric denominator to keep the scale stable across tokens.
    """
    pred = pred.astype(np.float32, copy=False)
    actual = actual.astype(np.float32, copy=False)

    diff = pred - actual

    pred_n = float(np.linalg.norm(pred))
    act_n = float(np.linalg.norm(actual))

    # Symmetric scale (reduces blow-ups when actual is near-zero).
    denom_l2 = 0.5 * (pred_n + act_n) + _EPS

    # Optional floor to prevent pathological spikes in degenerate windows.
    denom_l2 = float(max(denom_l2, 1e-3))

    l2 = float(np.linalg.norm(diff) / denom_l2)

    denom = (pred_n * act_n) + _EPS
    cosine = float(1.0 - float(np.dot(pred, actual) / denom))
    cosine = float(np.clip(cosine, 0.0, 2.0))

    # Combined score: primarily magnitude error, with directional penalty.
    combined = float(0.7 * l2 + 0.3 * cosine)
    return {"l2": l2, "cosine": cosine, "combined": combined}


class DivergencePredictor:
    """Streaming divergence predictor (scalar per token)."""

    def __init__(
        self,
        window_size: int = 8,
        proj_dim: int = 64,
        proj_seed: int = 0,
        ridge: float = 1e-2,
    ):
        if window_size < 3:
            # Need at least 3 to fit and evaluate one-step prediction robustly.
            raise ValueError("window_size must be >= 3")
        if proj_dim < 4:
            raise ValueError("proj_dim must be >= 4")
        self.window_size = int(window_size)
        self.proj_dim = int(proj_dim)
        self.proj_seed = int(proj_seed)
        self.ridge = float(ridge)

        self._window = StateWindow(self.window_size)
        self._proj: Optional[np.ndarray] = None  # (D, k), created lazily

        self.last_details: Optional[DivergenceDetails] = None

    def reset(self) -> None:
        self._window.reset()
        self.last_details = None
        # Keep projection cached; it is deterministic per hidden size.

    def _project(self, hidden: torch.Tensor) -> np.ndarray:
        h = _to_1d_np(hidden)  # (D,)
        if self._proj is None or self._proj.shape[0] != h.shape[0]:
            self._proj = _rademacher_projection(h.shape[0], self.proj_dim, self.proj_seed)
        z = h @ self._proj  # (k,)
        return z.astype(np.float32, copy=False)

    def step(self, hidden: torch.Tensor) -> float:
        """Consume current hidden state and return scalar divergence.

        Research-grade note:
        This is a *held-out* one-step prediction error.

        We fit the VAR(1) dynamics on the recent window excluding the newest
        state, then predict the newest state from the previous one.

        This avoids the "in-sample" effect where fitting on the same transition
        being evaluated can drive divergence toward ~0.
        """
        z = self._project(hidden)
        self._window.add(z)

        # Need at least 3 states so we can train on >=2 states (>=1 transition)
        # and then evaluate on the held-out last transition.
        if len(self._window) < 3:
            self.last_details = DivergenceDetails(
                n=len(self._window),
                proj_dim=self.proj_dim,
                l2=0.0,
                cosine=0.0,
                combined=0.0,
            )
            return 0.0

        states = self._window.matrix()  # (T, k)

        # Train on states[0..T-2] only (exclude the newest state).
        train_states = states[:-1, :]
        a = _fit_var1_ridge(train_states, ridge=self.ridge)

        # Held-out prediction: z_t_hat = z_{t-1} A, compare to z_t.
        pred = states[-2, :] @ a
        actual = states[-1, :]

        d = _divergence(pred, actual)
        self.last_details = DivergenceDetails(
            n=int(states.shape[0]),
            proj_dim=self.proj_dim,
            l2=float(d["l2"]),
            cosine=float(d["cosine"]),
            combined=float(d["combined"]),
        )
        return float(d["combined"])
