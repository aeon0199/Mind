# LLM Stability Runtime Stack

Inference-time control research for LLMs: measure when generation destabilizes, apply targeted interventions, and quantify recovery.

## Why This Exists

Most intervention demos show that outputs change. This stack is built to answer harder research questions:

- When does a model diverge from its expected trajectory?
- How long does perturbation memory persist?
- Which interventions recover behavior fastest?
- Can closed-loop control damp instability in real time?

## What Makes This Different

- Deterministic branchpoint experiments via `SeedCache` (baseline vs intervention from identical prompt-pass state)
- Explicit hysteresis protocol (`BASE -> PERTURB -> REASK`) to test persistence, not just immediate drift
- Unified adaptive controller with token-level diagnostics and intervention decisions
- Optional SAE steering and NNsight backend support for advanced intervention workflows

## Runtime Components

- `baseline_hysteresis_v1/`
  - Protocol layer for persistence/hysteresis experiments.
- `v1.5/`
  - Standalone observability runner with streaming diagnostics and plotting.
- `intervention_engine_v1.5_v2/`
  - Deterministic baseline-vs-intervention runner with recovery metrics.
- `adaptive_controller_system4/`
  - Closed-loop controller (`observe`, `stress`, `control`) with per-token event logs.

## 2-Minute Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run one command from each runtime:

```bash
# 1) Hysteresis baseline protocol
python baseline_hysteresis_v1/runner.py observer \
  --prompt "Explain how airplanes fly." \
  --max-new-tokens 128

# 2) Standalone V1.5 observability run
python v1.5/V1.5_runner.py \
  --prompt "Explain how airplanes fly." \
  --max_new_tokens 128

# 3) Deterministic intervention stress run
python intervention_engine_v1.5_v2/intervention.py run \
  --prompt "Explain how airplanes fly." \
  --max-tokens 64 \
  --layer -1 \
  --type additive \
  --magnitude 2.0 \
  --start 5 \
  --duration 10

# 4) Adaptive closed-loop control (shadow mode)
python adaptive_controller_system4/adaptive_runner.py control \
  --prompt "Explain how airplanes fly." \
  --shadow
```

## Advanced Modes (Optional)

Install optional packages:

```bash
pip install -r requirements-optional.txt
```

```bash
# Intervention engine with NNsight backend
python intervention_engine_v1.5_v2/intervention.py run \
  --backend nnsight \
  --nnsight-remote \
  --prompt "Explain how airplanes fly." \
  --layer -1 \
  --type scaling \
  --magnitude 0.9

# Intervention engine with SAE steering
python intervention_engine_v1.5_v2/intervention.py run \
  --prompt "Explain how airplanes fly." \
  --type sae \
  --layer -1 \
  --sae-repo "apollo-research/llama-3.1-70b-sae" \
  --sae-feature-idx 42 \
  --sae-strength 5.0

# Adaptive controller with SAE + dashboard
python adaptive_controller_system4/adaptive_loop.py \
  --prompt "Explain how airplanes fly." \
  --type sae \
  --sae-repo "apollo-research/llama-3.1-70b-sae" \
  --sae-feature-idx 42 \
  --sae-strength 5.0
```

## What Data You Get

This stack is designed to produce reusable research artifacts, not just text outputs.

- `intervention_engine_v1.5_v2` runs:
  - deterministic config hash + seed cache fingerprint
  - baseline/intervention trajectories
  - recovery and divergence metrics
- `v1.5` runs:
  - token-by-token telemetry with divergence/spectral/stiffness diagnostics
  - optional generated plot artifacts via `plotter.py`
- `adaptive_controller_system4` runs:
  - token-level `events.jsonl` with diagnostics + control actions
  - `summary.json` with regime counts and aggregate control stats
  - optional `dashboard.html`
- `baseline_hysteresis_v1` runs:
  - staged frames (`base`, `perturb`, `reask`)
  - hysteresis/recovery summary metrics

## Research Use-Cases

- Compare intervention families (`additive`, `projection`, `scaling`, `sae`) under matched branchpoints
- Evaluate intervention persistence and recovery timing
- Test controller policies in shadow mode before active deployment
- Generate publishable per-token traces for stability/control analysis

## Reproducibility

- Deterministic branchpointing before baseline/intervention split
- Config hashing and run metadata in artifacts
- Release/reporting checklist in `REPRODUCIBILITY.md`

## Project Layout

- `intervention_engine_v1.5_v2/intervention.py`: main intervention runner
- `v1.5/V1.5_runner.py`: standalone observability runner
- `adaptive_controller_system4/adaptive_runner.py`: unified entrypoint (`observe`, `stress`, `control`)
- `adaptive_controller_system4/adaptive_loop.py`: adaptive control runtime
- `.github/workflows/ci.yml`: compile/smoke checks

## Citation

If this project helps your research, cite via `CITATION.cff`.
