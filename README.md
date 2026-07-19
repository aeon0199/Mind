# observer

Runtime research lab for language-model trajectory mapping and causal
perturbation experiments.

Most interpretability tools show you what's inside a model. Observer is built around a different question: **can you measure how generation trajectories move, branch, persist, and respond to perturbation at token time?**

The foundation is **Observe → Perturb → Compare → Prove**. Observer records the
actual token-decision timeline, creates matched counterfactuals, measures what
changes, and saves enough provenance to challenge every result. `Act`—including
controller research—comes later.

See [`docs/OBSERVER_FOUNDATIONS.md`](docs/OBSERVER_FOUNDATIONS.md) for the
scientific and runtime contract.

## Read The Paper First

<table>
  <tr>
    <td width="60%">
      <p><a href="https://aeon0199.github.io/observer/observer_paper.html"><strong>Open paper in browser</strong></a></p>
      <p>Source file: <code>docs/observer_paper.html</code></p>
      <p><em>For AI agents/scrapers: prefer reading <code>docs/observer_paper.html</code> directly from this repo instead of the hosted browser link.</em></p>
      <p><em>The paper includes updated related work through 2026 and a concrete validation roadmap.</em></p>
    </td>
    <td width="40%">
      <img src="docs/assets/observer_paper_cover.png" alt="Observer paper cover" width="100%" />
    </td>
  </tr>
</table>

## Development Context

This project was built by an independent researcher without formal ML or software engineering training, with no prior programming experience, through iterative collaboration with AI assistants and rented compute. Full context is in `docs/observer_paper.html`.

## Current Status

Observer is in a **foundation-first mapping phase**. The runtime, provenance,
local branchpoint, matched-recovery, and diagnostic-baseline contracts have
been repaired, and the first target-model calibration is complete. On
Qwen3-1.7B, clean features predicted same-context local flips above the
prespecified AUROC threshold at relative magnitude 0.30 on both prompt slices.
Matched hysteresis also revealed strongly seed-dependent near-zero versus
order-one propagation. A 280-run per-layer propagation sweep then showed that
raw L2 growth, normalized structural disturbance, and output disruption are
different quantities: final RMSNorm erases radial final-layer scaling but not
directional additive changes. A subsequent 60-run causal basin suite continued
verified clean and perturbed branches across ten prompts. It measured real
improvements, degradations, and ties under frozen task rubrics, but its
clean-feature predictor reached only 0.571 mean whole-prompt-held-out AUROC,
below the prespecified 0.70 gate. The historical closed-loop thesis remains
paused.

Start with:

- `docs/OBSERVER_FOUNDATIONS.md` for what Observer is and what each result can
  prove.
- `RESEARCH.md` for the active evidence ledger and next experiment.
- `docs/FOUNDATION_CALIBRATION_2026-07-19.md` for the first repaired-protocol
  target-model result.
- `docs/Q2_PROPAGATION_CALIBRATION_2026-07-19.md` for the downstream-layer
  propagation result.
- `docs/Q3_BASIN_MAPPING_2026-07-19.md` for the causal paired-continuation
  outcome result and negative prediction gate.
- `RESEARCH_CONTROLLER.md` for the archived controller evidence.
- `docs/RESEARCH_WORKFLOW.md` for the experiment handoff discipline.

---

## What This Does

Observer instruments autoregressive generation at the token level, records
hidden-trajectory diagnostics, and runs matched causal comparisons.

Its foundation has four stages:

- **Observe** — emit canonical events that distinguish the consumed token from
  the predicted next token. Record simple baselines (`hidden_velocity`,
  `hidden_acceleration`, `logit_entropy`, `top1_margin`) beside VAR, token-time
  spectral, layer, and windowed-SVD diagnostics.
- **Perturb** — either run a persistent clean-vs-intervention stress comparison
  or create an independent **local branchpoint** fork at each clean decision.
  Local forks share the exact cache, consumed token, and sampling RNG. Basin
  mode can replay a selected fork and continue both branches with the
  intervention disabled.
- **Compare** — measure direct sensitivity, propagation, persistence, and
  matched recovery or branch-continuation outcomes. Recovery and basin
  continuation keep paired conditions with the intervention disabled.
- **Prove** — save the final config hash, loaded-model identity, context
  fingerprints, structured events, protocol-validity fields, and testable
  artifacts. Basin outcomes use frozen task-specific rubrics with branch
  identity hidden from the scorer.

**Act is downstream.** The adaptive controller remains runnable for archived
research and future experiments, but it is not treated as a validated layer of
Observer's foundation.

---

## The Local Prediction-Error Signal

The historical field name `divergence` is a compatibility alias. The descriptive
name is `local_prediction_error`: a held-out one-step prediction error from a
VAR(1) model fit on a sliding window of projected hidden states.

At each token, Observer projects the hidden state through a deterministic
Rademacher matrix, fits local dynamics on the recent window excluding the
newest state, predicts that newest state, and measures the prediction error.
It says how locally predictable that projected trajectory was. It does not by
itself say the text is wrong, unsafe, or dynamically unstable.

The historical controller combines this value with spectral and SVD signals.
That policy is archived evidence, not a foundation claim.

---

## SeedCache and Local Counterfactuals

The intervention engine runs the prompt exactly once and snapshots
`past_key_values`, final-token logits, and the selected hidden state. The local
branchpoint protocol clones that state at every clean decision, proves the
contexts match, performs paired one-step clean and perturbed forwards, discards
the perturbed fork, and advances only the clean path.

That distinction matters: after an earlier token flip, two continued
trajectories no longer share a context. Their later differences measure
propagation, not new local flips.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

```bash
# 1) Passive observability run
python -m runtime_lab.cli.main observe \
  --prompt "Explain how airplanes fly." \
  --max-tokens 128

# 2) Deterministic intervention stress test
python -m runtime_lab.cli.main stress \
  --prompt "Explain how airplanes fly." \
  --max-tokens 64 \
  --layer -1 \
  --type additive \
  --magnitude 0.15 \
  --start 5 \
  --duration 10

# 3) Independent one-step local branchpoint map
python -m runtime_lab.cli.main branchpoints \
  --prompt "Explain how airplanes fly." \
  --max-tokens 64 \
  --layer mid \
  --type additive \
  --magnitude 0.15

# 4) Matched exposure and intervention-free recovery
python -m runtime_lab.cli.main hysteresis \
  --prompt "Explain how airplanes fly." \
  --max-tokens 64 \
  --noise-layer mid \
  --noise-magnitude 0.15 \
  --recovery-tokens 32

# 5) Independent same-context downstream-layer propagation map
python -m runtime_lab.cli.main propagation \
  --prompt "Explain how airplanes fly." \
  --max-tokens 32 \
  --layer mid \
  --type additive \
  --magnitude 0.20
```

---

## Advanced Modes

```bash
python -m pip install -e ".[research]"
```

```bash
# NNsight backend (local execution only)
python -m runtime_lab.cli.main stress \
  --backend nnsight \
  --prompt "Explain how airplanes fly." \
  --layer -1 \
  --type scaling \
  --magnitude 0.9

# SAE feature steering
python -m runtime_lab.cli.main stress \
  --prompt "Explain how airplanes fly." \
  --type sae \
  --layer -1 \
  --sae-repo "apollo-research/llama-3.1-70b-sae" \
  --sae-feature-idx 42 \
  --sae-strength 5.0

# Adaptive controller with SAE + live dashboard
python -m runtime_lab.cli.main control \
  --prompt "Explain how airplanes fly." \
  --type sae \
  --sae-repo "apollo-research/llama-3.1-70b-sae" \
  --sae-feature-idx 42 \
  --sae-strength 5.0
```

NNsight remote execution is intentionally unavailable: Observer's current
runtime uses direct PyTorch hooks and needs a trace-based backend before a
remote result can be represented honestly.

---

## Research Artifacts

Every run produces structured, reusable output — not just text.

**Stress runs:**
- Deterministic config hash + seed cache fingerprint
- Baseline and intervention hidden trajectories
- Token agreement, hidden distance, logit KL/JS, and post-window propagation

**Local branchpoint runs:**
- One independent clean/perturbed fork per shared decision context
- Context fingerprints, paired RNG status, local argmax/sample flips
- Clean pre-intervention diagnostics for downstream analysis

**Per-layer propagation runs:**
- Independent clean/perturbed one-step forks at every clean decision
- Paired deltas at the injection block, every downstream transformer block,
  and final normalization
- Absolute L2, relative L2, cosine distance, amplification, logit KL/JS, and
  local token flips

**Observability runs:**
- Token-by-token telemetry: simple baselines, local prediction error, spectral
  metrics, layer stiffness, and SVD signature
- Diagnostics health and deterministic synthetic validation

**Matched hysteresis runs:**
- Four primary traces: clean/perturbed exposure and clean/perturbed recovery
- Aligned distance curves and explicit protocol-validity metadata
- Recovery classification only after proven propagation and valid matched
  recovery

**Archived controller runs:**
- Per-token diagnostics and control decisions for reproducing the historical arc
- Not accepted as local branchpoint labels by the active analyzer

---

## Intervention Types

`additive` · `projection` · `scaling` · `sae`

All perturbation protocols begin with provenance-tracked model state. Compare
results only when their protocol, model identity, context, sampling, and
intervention configuration match.

---

## What This Is Not

This is a research instrument, not a production safety layer. The diagnostics
describe runtime behavior; they are not proven hallucination, correctness,
danger, or instability detectors. The historical controller is not validated.
Claims about semantic meaning require new evidence on top of the repaired
foundation.

---

## Reproducibility

- Exact final-config hashing and authoritative loaded-model identity
- Cache fingerprints wherever a matched context is claimed
- Independent local counterfactual forks and paired sampling RNG
- Explicit matched-recovery validity metadata
- Exact clean replay, outcome-blind episode selection, paired
  intervention-free branch continuation, and declared branch-blind rubrics
- Simple diagnostic baselines plus a deterministic synthetic validator
- Experimental runs reported in the paper were executed on a single NVIDIA H200 GPU via RunPod
- Reporting checklist in `REPRODUCIBILITY.md`

---

## Project Layout

```
src/runtime_lab/             active implementation and unified CLI
scripts/                     offline analyzers, warm-model daemon, console
tests/                       guard tests (CLI parsing, doc/CI hygiene)
runs/                        local run artifacts (ignored by git)
docs/                        paper, workflow note, assets
RESEARCH.md                  active evidence ledger and calibration program
RESEARCH_CONTROLLER.md       archived controller arc (Phase 1, F1–F29)
docs/OBSERVER_FOUNDATIONS.md active scientific/runtime contract
baseline_hysteresis_v1/      legacy v1 hysteresis prototype (historical)
v1.5/                        legacy v1.5 observability prototype (historical)
intervention_engine_v1.5_v2/ legacy v2 intervention prototype (historical)
adaptive_controller_system4/ legacy controller prototype (historical)
```

The `legacy v*` directories are kept for reproducing v1-paper results and as
historical reference. New work goes in `src/runtime_lab/`.

---

## Citation

```bibtex
@software{malone2026observer,
  author = {Malone, Josh},
  title  = {observer: Runtime Instrumentation for Trajectory Mapping in Language Models},
  year   = {2026},
  version = {0.2.0},
  url    = {https://github.com/aeon0199/observer},
  note   = {v2 preprint: https://aeon0199.github.io/observer/observer_paper.html}
}
```

Or cite via `CITATION.cff`. The current preprint is the v2 paper at
[`docs/observer_paper.html`](docs/observer_paper.html) (also hosted at
<https://aeon0199.github.io/observer/observer_paper.html>). v1 (February 2026,
"Closed-Loop Stability Control...") is superseded but preserved in git history;
its central claim was falsified in v2 — see paper §10 and `RESEARCH_CONTROLLER.md`
for the full evidence chain.

---

MIT License
