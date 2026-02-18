"""
plotter.py — V1.5 Summary Plot Generator
----------------------------------------
Generates 3 lab / investor-friendly plots from a V1.5-style summary dict
or summary.json.

Design note:
V1.5 always records full telemetry (no basic/full modes). This plotter
assumes dense per-token signals and intentionally avoids conditional logic
for missing diagnostics.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _safe_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _get_nested(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _minmax_scale(arr: np.ndarray) -> np.ndarray:
    """
    Scales to [0, 1] ignoring NaNs.
    If constant (or all NaN), returns a flat 0.5 (or NaNs).
    """
    out = arr.astype(float).copy()
    mask = ~np.isnan(out)
    if mask.sum() == 0:
        return out
    vmin = np.nanmin(out)
    vmax = np.nanmax(out)
    if np.isclose(vmin, vmax):
        out[mask] = 0.5
        return out
    out[mask] = (out[mask] - vmin) / (vmax - vmin)
    return out


# ─────────────────────────────────────────────
# Data extraction
# ─────────────────────────────────────────────

def _extract_series(summary: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    tel = summary.get("telemetry", []) or []
    n = len(tel)
    x = np.arange(n, dtype=float)

    entropy = np.array(
        [_safe_float(_get_nested(t.get("stats", {}), ["entropy"])) for t in tel],
        dtype=float,
    )
    dyn_div = np.array(
        [_safe_float(_get_nested(t.get("stats", {}), ["dynamic_divergence"])) for t in tel],
        dtype=float,
    )
    hidden_norm = np.array(
        [_safe_float(_get_nested(t.get("stats", {}), ["hidden_norm"])) for t in tel],
        dtype=float,
    )
    emitted_logprob = np.array(
        [_safe_float(_get_nested(t.get("stats", {}), ["emitted_logprob"])) for t in tel],
        dtype=float,
    )

    # Prefer spectral_entropy from spectral_energy_metrics(); fall back to older keys if present.
    spectral_spread = np.array(
        [_safe_float(_get_nested(t.get("stats", {}), ["spectral", "spectral_entropy"])) for t in tel],
        dtype=float,
    )

    # Back-compat: some older summaries used spectral_dispersion
    if np.isnan(spectral_spread).all():
        spectral_spread = np.array(
            [_safe_float(_get_nested(t.get("stats", {}), ["spectral", "spectral_dispersion"])) for t in tel],
            dtype=float,
        )

    # Back-compat: some older summaries used total/tail energy.
    if np.isnan(spectral_spread).all():
        total = np.array(
            [_safe_float(_get_nested(t.get("stats", {}), ["spectral", "spectral_total_energy"])) for t in tel],
            dtype=float,
        )
        tail = np.array(
            [_safe_float(_get_nested(t.get("stats", {}), ["spectral", "spectral_tail_energy"])) for t in tel],
            dtype=float,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            spectral_spread = tail / total

    series = {
        "entropy": entropy,
        "dynamic_divergence": dyn_div,
        "hidden_norm": hidden_norm,
        "spectral_spread": spectral_spread,
        "emitted_logprob": emitted_logprob,
    }
    return x, series


def _extract_svd_matrix(
    summary: Dict[str, Any],
    k_max: int = 5,
) -> Tuple[np.ndarray, Optional[int]]:
    tel = summary.get("telemetry", []) or []
    n = len(tel)

    sv_lists: List[List[float]] = []
    windows: List[int] = []

    for t in tel:
        sv = _get_nested(t.get("stats", {}), ["svd", "singular_values"])
        w = _get_nested(t.get("stats", {}), ["svd", "window"])
        if isinstance(w, int):
            windows.append(w)

        if isinstance(sv, list) and len(sv) > 0:
            sv_lists.append([_safe_float(v) for v in sv])
        else:
            sv_lists.append([])

    inferred_k = min(k_max, max((len(v) for v in sv_lists), default=0))
    if inferred_k <= 0:
        return np.empty((n, 0), dtype=float), None

    mat = np.full((n, inferred_k), np.nan, dtype=float)
    for i, vals in enumerate(sv_lists):
        for j in range(min(inferred_k, len(vals))):
            mat[i, j] = vals[j]

    # Infer a representative window size (median, ignoring early zeros)
    window_size = None
    valid_w = [w for w in windows if isinstance(w, int) and w > 1]
    if valid_w:
        window_size = int(np.median(valid_w))

    return mat, window_size


# ─────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────

def plot_timeline_vitals(summary: Dict[str, Any], out_dir: str) -> str:
    model = str(_get_nested(summary, ["model"]) or _get_nested(summary, ["config", "model"]) or "")
    ts = str(_get_nested(summary, ["timestamp"]) or "")
    x, series = _extract_series(summary)

    y_entropy = _minmax_scale(series["entropy"])
    y_div = _minmax_scale(series["dynamic_divergence"])
    y_hn = _minmax_scale(series["hidden_norm"])
    y_spec = _minmax_scale(series["spectral_spread"])

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.plot(x, y_entropy, label="entropy (scaled)")
    ax.plot(x, y_div, label="dynamic_divergence (scaled)")
    ax.plot(x, y_hn, label="hidden_norm (scaled)")
    ax.plot(x, y_spec, label="spectral_entropy / tail_ratio (scaled)")

    title = "V1.5 Token Timeline — Core Vitals (Min–Max Scaled)"
    if model or ts:
        title += f"  [{model}  {ts}]"
    ax.set_title(title)
    ax.set_xlabel("token index")
    ax.set_ylabel("scaled value [0,1]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "timeline_vitals.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_svd_signature(
    summary: Dict[str, Any],
    out_dir: str,
    k_max: int = 5,
) -> Optional[str]:
    model = str(_get_nested(summary, ["model"]) or _get_nested(summary, ["config", "model"]) or "")
    ts = str(_get_nested(summary, ["timestamp"]) or "")
    svd_mat, window_size = _extract_svd_matrix(summary, k_max=k_max)
    if svd_mat.shape[1] == 0:
        return None

    x = np.arange(svd_mat.shape[0], dtype=float)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    for j in range(svd_mat.shape[1]):
        ax.plot(x, svd_mat[:, j], label=f"σ{j+1}")

    title = f"V1.5 SVD Signature Over Tokens (top-{svd_mat.shape[1]})"
    if window_size is not None:
        title += f" — window={window_size}"
    if model or ts:
        title += f"  [{model}  {ts}]"

    ax.set_title(title)
    ax.set_xlabel("token index")
    ax.set_ylabel("singular value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "svd_signature.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_scatter_entropy_vs_divergence(summary: Dict[str, Any], out_dir: str) -> str:
    model = str(_get_nested(summary, ["model"]) or _get_nested(summary, ["config", "model"]) or "")
    ts = str(_get_nested(summary, ["timestamp"]) or "")
    _, series = _extract_series(summary)

    entropy = series["entropy"]
    div = series["dynamic_divergence"]

    mask = (~np.isnan(entropy)) & (~np.isnan(div))

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)

    ax.scatter(entropy[mask], div[mask], s=18)
    title = "V1.5 Phase Space — Entropy vs Dynamic Divergence"
    if model or ts:
        title += f"  [{model}  {ts}]"
    ax.set_title(title)
    ax.set_xlabel("entropy")
    ax.set_ylabel("dynamic_divergence")
    ax.grid(True, alpha=0.3)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "scatter_entropy_vs_divergence.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_headline_summary(summary: Dict[str, Any], out_dir: str) -> str:
    """A compact 'scorecard' with aggregated stats (for quick screenshots)."""
    _, series = _extract_series(summary)
    model = str(_get_nested(summary, ["model"]) or _get_nested(summary, ["config", "model"]) or "")
    ts = str(_get_nested(summary, ["timestamp"]) or "")
    cfg_hash = str(_get_nested(summary, ["config_hash"]) or "")

    def _q(arr: np.ndarray, q: float) -> float:
        if arr.size == 0 or np.isnan(arr).all():
            return float("nan")
        return float(np.nanpercentile(arr, q))

    def _avg(arr: np.ndarray) -> float:
        if arr.size == 0 or np.isnan(arr).all():
            return float("nan")
        return float(np.nanmean(arr))

    entropy = series["entropy"]
    div = series["dynamic_divergence"]
    logprob = series["emitted_logprob"]

    stats_lines = [
        "V1.5 — Run Summary (Aggregates)",
        f"Model: {model}",
        f"Timestamp: {ts}",
        f"Config hash: {cfg_hash}",
        "",
        f"Tokens: {len(entropy)}",
        "",
        f"Entropy:   mean={_avg(entropy):.4g}  p5={_q(entropy,5):.4g}  p95={_q(entropy,95):.4g}",
        f"Divergence mean={_avg(div):.4g}  p5={_q(div,5):.4g}  p95={_q(div,95):.4g}",
        f"Logprob:   mean={_avg(logprob):.4g}  p5={_q(logprob,5):.4g}  p95={_q(logprob,95):.4g}",
    ]

    fig = plt.figure(figsize=(8.5, 4))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.02, 0.98, "\n".join(stats_lines), va="top", ha="left", family="monospace", fontsize=11)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "headline_summary.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path

# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def generate_plots(summary: Dict[str, Any], run_dir: str) -> Dict[str, Optional[str]]:
    """
    Generates all plots into {run_dir}/plots and returns their paths.
    """
    plots_dir = os.path.join(run_dir, "plots")
    out: Dict[str, Optional[str]] = {}

    out["timeline_vitals"] = plot_timeline_vitals(summary, plots_dir)
    out["svd_signature"] = plot_svd_signature(summary, plots_dir, k_max=5)
    out["scatter_entropy_vs_divergence"] = plot_scatter_entropy_vs_divergence(summary, plots_dir)
    out["headline_summary"] = plot_headline_summary(summary, plots_dir)

    return out


def load_summary_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate V1.5 plots from summary.json")
    parser.add_argument("summary_json", help="Path to runs/.../summary.json")
    args = parser.parse_args()

    summary = load_summary_json(args.summary_json)
    run_dir = os.path.dirname(os.path.abspath(args.summary_json))

    paths = generate_plots(summary, run_dir)
    print("Generated plots:")
    for k, v in paths.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
