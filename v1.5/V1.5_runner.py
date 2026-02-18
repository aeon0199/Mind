"""
V1.5 Runner — Real-Time Inference Observability
----------------------------------------------
Single-pass diagnostic runner with:
- Sliding-window divergence prediction
- Spectral leakage diagnostics
- Layer-wise stiffness / elasticity gradient

NO branching
NO correction
NO control
"""

import os
import time
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional

import torch

from model_loader import load_model
from hooks import HookManager
from utils import basic_stats, save_json
from predictor import DivergencePredictor
from spectral import spectral_energy_metrics
from layer_probe import LayerProbe, LayerProbeConfig

# NEW: plotting
try:
    from plotter import generate_plots
except Exception:
    generate_plots = None


RUNS_DIR = os.environ.get("RUNS_DIR", "runs")
VERSION = "1.5"


# ─────────────────────────────────────────────
# Windowed SVD (V1.4 semantics) + guardrail
# ─────────────────────────────────────────────

def _windowed_svd_signature(
    hidden_window: List[torch.Tensor],
    topk: int = 8,
) -> Dict[str, Any]:
    """
    Compute a meaningful SVD signature by stacking a window of hidden vectors
    into a matrix (W x D) and returning the top singular values.

    Guardrail: refuses to compute SVD on a single vector.
    """
    assert len(hidden_window) >= 2, (
        "SVD guardrail triggered: attempted SVD with window < 2. "
        "Single-vector SVD is invalid and indicates a regression."
    )

    mat = torch.stack([h.detach().float() for h in hidden_window], dim=0)

    assert mat.shape[0] > 1, (
        "SVD guardrail triggered: matrix has only one row. "
        "Temporal SVD requires W >= 2."
    )

    s = torch.linalg.svdvals(mat)
    k = max(1, min(int(topk), int(s.numel())))

    return {
        "window": int(mat.shape[0]),
        "topk": int(k),
        "singular_values": s[:k].tolist(),
    }


# ─────────────────────────────────────────────
# Telemetry container
# ─────────────────────────────────────────────

@dataclass
class TokenTelemetry:
    token_id: int
    token_text: str
    timestamp: float
    stats: Dict[str, Any]


# ─────────────────────────────────────────────
# Streaming generation with telemetry
# ─────────────────────────────────────────────

def generate_with_streaming_telemetry(
    model,
    tokenizer,
    device: torch.device,
    hook_mgr: HookManager,
    predictor: DivergencePredictor,
    layer_probe: LayerProbe,
    prompt: str,
    max_new_tokens: int,
) -> List[TokenTelemetry]:
    """
    Generate tokens one by one, capturing telemetry,
    predictive divergence, spectral leakage, and layer-wise stiffness.
    """
    hook_mgr.reset()
    predictor.reset()
    layer_probe.reset()

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    telemetry: List[TokenTelemetry] = []

    # Windowed SVD buffer (V1.4 semantics)
    hidden_buf: List[torch.Tensor] = []
    svd_window = 8
    svd_topk = 8

    with torch.no_grad():
        # First pass: process full prompt once, initialize KV cache.
        hook_mgr.reset()
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        # Needed for hidden_states indexing alignment below.
        # Some HF models return hidden_states with embeddings prepended (len = num_layers + 1).
        num_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
        if not isinstance(num_layers, int) or num_layers <= 0:
            try:
                num_layers = max(0, len(outputs.hidden_states) - 1)
            except Exception:
                num_layers = 0

        for _ in range(max_new_tokens):
            # Next token decision from current context.
            # Next-token decision and its logprob
            probs = torch.softmax(logits, dim=-1)
            logprobs = torch.log(probs + 1e-12)
            next_token = logits.argmax(dim=-1, keepdim=True)
            emitted_logprob = float(logprobs[0, next_token.item()].item())

            # Final-layer hidden state (prefer hook, fallback to outputs.hidden_states)
            hidden = hook_mgr.hidden_hook.captured  # expected shape [H]
            hidden_source = "hook"
            if hidden is None:
                # Robust fallback if hooks fail to attach on a given model.
                # outputs.hidden_states[-1] is last layer, shape [B, T, H].
                hidden = outputs.hidden_states[-1][0, -1, :].detach()
                hidden_source = "outputs_hidden_states"

            # ✅ FIX: capture logits directly from model output
            logit_vec = logits.squeeze(0)  # shape [V]

            stats = basic_stats(hidden, logit_vec)
            stats["emitted_logprob"] = emitted_logprob
            stats["hidden_source"] = hidden_source
            stats["kv_cache"] = True

            # ── Windowed SVD (guarded)
            h_vec = hidden.detach().squeeze(0)
            hidden_buf.append(h_vec)
            if len(hidden_buf) > svd_window:
                hidden_buf = hidden_buf[-svd_window:]

            if len(hidden_buf) >= 2:
                stats["svd"] = _windowed_svd_signature(hidden_buf, topk=svd_topk)
            else:
                stats["svd"] = {"window": len(hidden_buf), "singular_values": []}

            # Sliding-window divergence
            stats["dynamic_divergence"] = predictor.step(hidden)

            # Spectral leakage
            stats["spectral"] = spectral_energy_metrics(hidden)

            # Layer-wise stiffness probe
            # Align HF hidden_states indexing: hidden_states[0] is embeddings, then layer 0..L-1 at 1..L
            layer_states: Dict[int, torch.Tensor] = {}
            hs_list = outputs.hidden_states
            for layer_idx in layer_probe.config.layers:
                target_idx = layer_idx + 1 if len(hs_list) == (num_layers + 1) else layer_idx
                if 0 <= target_idx < len(hs_list):
                    layer_states[layer_idx] = hs_list[target_idx][:, -1, :]

            stats["layer_stiffness"] = layer_probe.step(layer_states)

            token_id = int(next_token.item())
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)

            telemetry.append(
                TokenTelemetry(
                    token_id=token_id,
                    token_text=token_text,
                    timestamp=time.time(),
                    stats=stats,
                )
            )

            if token_id == tokenizer.eos_token_id:
                break

            # Advance one token using KV cache (O(1) context length per step).
            hook_mgr.reset()
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

    return telemetry


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

def run_v1_5_observer(
    prompt: str,
    model_key: Optional[str] = None,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """
    Run V1.5 real-time observability pass.
    """
    print("\n" + "═" * 60)
    print("  V1.5 — Real-Time Inference Observability")
    print("═" * 60 + "\n")

    tokenizer, model, device, model_config = load_model(model_key=model_key)

    hook_mgr = HookManager(model)
    hook_mgr.register()

    predictor = DivergencePredictor(window_size=8)

    # Choose representative early / mid / late layers
    num_layers = model.config.num_hidden_layers
    probe_layers = [
        num_layers // 4,
        num_layers // 2,
        num_layers - 1,
    ]

    layer_probe = LayerProbe(
        LayerProbeConfig(
            layers=probe_layers,
            window_size=5,
        )
    )

    os.makedirs(RUNS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, f"v1_5_run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    telemetry = generate_with_streaming_telemetry(
        model=model,
        tokenizer=tokenizer,
        device=device,
        hook_mgr=hook_mgr,
        predictor=predictor,
        layer_probe=layer_probe,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

    decoded_text = prompt + "".join(t.token_text for t in telemetry)

    summary = {
        "version": VERSION,
        "timestamp": stamp,
        "model": model_key or model_config.key,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "num_generated_tokens": len(telemetry),
        "layer_probe_layers": probe_layers,
        "telemetry": [
            {
                "token_id": t.token_id,
                "token_text": t.token_text,
                "timestamp": t.timestamp,
                "stats": t.stats,
            }
            for t in telemetry
        ],
    }

    save_json(os.path.join(run_dir, "summary.json"), summary)

    with open(os.path.join(run_dir, "output.txt"), "w", encoding="utf-8") as f:
        f.write(decoded_text)

    if generate_plots is not None:
        try:
            paths = generate_plots(summary, run_dir)
            print("[V1.5] Plots generated:")
            for k, v in paths.items():
                print(f"  - {k}: {v}")
        except Exception as e:
            print(f"[V1.5] Plot generation failed (non-fatal): {e}")
    else:
        print("[V1.5] plotter.py not available; skipping plot generation.")

    hook_mgr.cleanup()

    print(f"\n[V1.5] Run complete. Output saved to: {run_dir}\n")

    return summary


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="V1.5 — Real-Time Inference Observability"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt", default="Explain how airplanes fly.")
    parser.add_argument("--max_new_tokens", type=int, default=128)

    args = parser.parse_args()

    run_v1_5_observer(
        prompt=args.prompt,
        model_key=args.model,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
