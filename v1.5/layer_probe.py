"""layer_probe.py

Layer-wise stiffness / elasticity probe.

What it measures
----------------
For each probed layer L, we maintain a sliding window of last-token vectors
h_t^L \in R^H. From that window we compute:

- velocity norms: v_t = ||h_t - h_{t-1}||_2
- stiffness: mean(v_t) over the window (higher => more volatile / less stable)
- stiffness_trend: linear slope of v_t across the window (positive => accelerating)
- elasticity: 1 / (1 + stiffness) (bounded, higher => more stable)

This is a *diagnostic* proxy. It is not a physical elasticity; it is a
bounded stability score derived from local trajectory smoothness.

Design notes
------------
- All outputs are JSON-serializable.
- Accepts hidden states as torch tensors with shapes like [H], [B,H], [B,T,H].
- Uses CPU-side numpy for simple regressions; overhead is small at window sizes <= 10.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


@dataclass
class LayerProbeConfig:
    layers: List[int]
    window_size: int = 5


def _last_token_1d(x: torch.Tensor) -> np.ndarray:
    """Convert a hidden state tensor to a 1D numpy vector for the last token."""
    if x is None:
        raise ValueError("layer state is None")

    # Common shapes:
    # [H]
    # [B, H]
    # [B, T, H]
    if x.dim() == 3:
        x = x[:, -1, :]
    if x.dim() == 2:
        x = x[0]
    if x.dim() != 1:
        x = x.reshape(-1)

    return x.detach().float().cpu().numpy()


class LayerProbe:
    def __init__(self, config: LayerProbeConfig):
        self.config = config
        self._buffers: Dict[int, List[np.ndarray]] = {int(i): [] for i in config.layers}
        self._prev_stiffness: Dict[int, float] = {int(i): 0.0 for i in config.layers}

    def reset(self) -> None:
        for k in list(self._buffers.keys()):
            self._buffers[k].clear()
            self._prev_stiffness[k] = 0.0

    def step(self, layer_states: Dict[int, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """Update probe with the current token's layer states."""
        out: Dict[str, Dict[str, float]] = {}

        for layer_idx in self.config.layers:
            if layer_idx not in layer_states:
                # If a layer wasn't provided this step, return a minimal record.
                out[str(layer_idx)] = {
                    "window_len": float(len(self._buffers[int(layer_idx)])),
                    "stiffness": float(self._prev_stiffness[int(layer_idx)]),
                    "stiffness_trend": 0.0,
                    "elasticity": float(1.0 / (1.0 + self._prev_stiffness[int(layer_idx)])),
                }
                continue

            vec = _last_token_1d(layer_states[layer_idx])
            buf = self._buffers[int(layer_idx)]
            buf.append(vec)
            if len(buf) > int(self.config.window_size):
                buf.pop(0)

            metrics = self._compute_metrics(buf, prev=self._prev_stiffness[int(layer_idx)])
            self._prev_stiffness[int(layer_idx)] = float(metrics["stiffness"])
            out[str(layer_idx)] = metrics

        return out

    @staticmethod
    def _compute_metrics(window: List[np.ndarray], prev: float) -> Dict[str, float]:
        n = len(window)
        if n < 2:
            stiffness = float(prev) if prev > 0 else 0.0
            return {
                "window_len": float(n),
                "stiffness": float(stiffness),
                "stiffness_trend": 0.0,
                "elasticity": float(1.0 / (1.0 + stiffness)),
            }

        # velocities between successive states
        v = []
        for i in range(1, n):
            d = window[i] - window[i - 1]
            v.append(float(np.linalg.norm(d)))

        v_arr = np.asarray(v, dtype=np.float32)
        stiffness = float(v_arr.mean())

        # trend via simple least-squares slope
        # x = 0..len(v)-1
        if len(v_arr) >= 2:
            x = np.arange(len(v_arr), dtype=np.float32)
            x = x - x.mean()
            denom = float((x * x).sum()) + 1e-8
            slope = float((x * (v_arr - v_arr.mean())).sum() / denom)
        else:
            slope = 0.0

        elasticity = float(1.0 / (1.0 + stiffness))

        return {
            "window_len": float(n),
            "stiffness": float(stiffness),
            "stiffness_trend": float(slope),
            "elasticity": float(elasticity),
        }
