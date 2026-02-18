"""spectral.py

Spectral / frequency-style diagnostics for a *single* hidden state vector.

Why FFT over SVD here?
- In V1.5 runner, we typically have only the last-token hidden vector (shape [D]).
- SVD on a 1×D matrix returns a single singular value (equivalent to the vector norm),
  which is not informative for 'spectral leakage' style diagnostics.

This module therefore treats the hidden vector as a 1D signal over its feature index
and computes an FFT-based power spectrum. While the feature index is not a physical
axis, these metrics still provide a stable, cheap measure of how 'rough' vs 'smooth'
the activations are across dimensions, and whether energy is concentrated in a narrow
band or spread broadly.

Returned metrics are normalized and robust to scale.
"""

from __future__ import annotations

from typing import Dict, Any

import math
import torch


_EPS = 1e-8


def spectral_energy_metrics(hidden_state: torch.Tensor, n_bands: int = 6) -> Dict[str, Any]:
    """Compute FFT-based spectral diagnostics on a 1D hidden state vector.

    Args:
        hidden_state: torch.Tensor with shape [D] or [1, D] (or any tensor flattenable to 1D)
        n_bands: number of equal-width frequency bands for band energy fractions.

    Returns:
        Dict with:
          - total_power
          - spectral_entropy
          - spectral_flatness
          - centroid (0..1, normalized frequency)
          - rolloff_85 (0..1, normalized frequency where cumulative power reaches 85%)
          - high_frac (top 20% frequencies)
          - low_frac (bottom 20% frequencies, excluding DC)
          - band_fracs (list length n_bands)
    """
    if not isinstance(hidden_state, torch.Tensor):
        raise TypeError(f"hidden_state must be torch.Tensor, got {type(hidden_state)}")

    def _compute(x: torch.Tensor) -> Dict[str, Any]:
        x = x.detach().float()
        x = x.view(-1)  # [D]
        d = int(x.numel())

        # Remove mean to avoid DC dominance.
        x = x - x.mean()

        # rFFT -> power spectrum
        spec = torch.fft.rfft(x, dim=0)
        power = (spec.real ** 2 + spec.imag ** 2).float()  # [F]
        # Drop DC component for most ratios
        if power.numel() > 1:
            power_no_dc = power[1:]
        else:
            power_no_dc = power

        total_power = float(power_no_dc.sum().item())
        if total_power <= 0.0:
            # Degenerate (all zeros)
            return {
                "total_power": 0.0,
                "spectral_entropy": 0.0,
                "spectral_flatness": 0.0,
                "centroid": 0.0,
                "rolloff_85": 0.0,
                "high_frac": 0.0,
                "low_frac": 0.0,
                "band_fracs": [0.0 for _ in range(max(1, int(n_bands)))],
                "freq_bins": int(power_no_dc.numel()),
                "signal_dim": d,
            }

        p = power_no_dc / (power_no_dc.sum() + _EPS)  # normalized distribution

        # Spectral entropy (normalized to [0,1])
        ent = -(p * (p + _EPS).log()).sum()
        ent_norm = float((ent / math.log(float(p.numel()) + _EPS)).item())

        # Spectral flatness: geometric mean / arithmetic mean
        geom = torch.exp(torch.mean(torch.log(power_no_dc + _EPS)))
        arith = torch.mean(power_no_dc)
        flat = float((geom / (arith + _EPS)).item())

        # Spectral centroid (normalized frequency index in [0,1])
        freqs = torch.linspace(0.0, 1.0, steps=int(p.numel()), device=p.device, dtype=p.dtype)
        centroid = float((freqs * p).sum().item())

        # Rolloff 85%
        cdf = torch.cumsum(p, dim=0)
        roll_idx = int((cdf >= 0.85).nonzero(as_tuple=False)[0].item()) if p.numel() > 0 else 0
        rolloff = float(roll_idx / max(1, (p.numel() - 1)))

        # Low/high fractions
        k = max(1, int(0.2 * p.numel()))
        low_frac = float(p[:k].sum().item())
        high_frac = float(p[-k:].sum().item())

        # Equal-width band fractions
        nb = max(1, int(n_bands))
        band_fracs = []
        for b in range(nb):
            a = int((b * p.numel()) / nb)
            z = int(((b + 1) * p.numel()) / nb)
            band_fracs.append(float(p[a:z].sum().item()))

        return {
            "total_power": total_power,
            "spectral_entropy": ent_norm,
            "spectral_flatness": flat,
            "centroid": centroid,
            "rolloff_85": rolloff,
            "high_frac": high_frac,
            "low_frac": low_frac,
            "band_fracs": band_fracs,
            "freq_bins": int(p.numel()),
            "signal_dim": d,
        }

    # Try on the native device first; fallback to CPU if FFT is unsupported.
    try:
        return _compute(hidden_state)
    except Exception as e:
        try:
            x_cpu = hidden_state.detach().float().cpu()
            out = _compute(x_cpu)
            out["fft_fallback"] = "cpu"
            out["fft_error"] = str(e)
            return out
        except Exception as e2:
            return {
                "error": f"FFT failed on device and cpu fallback: {e}; fallback: {e2}",
                "signal_dim": int(hidden_state.numel()) if isinstance(hidden_state, torch.Tensor) else 0,
            }
