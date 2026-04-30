# Reproducibility Checklist

## Required for Public Claims

1. Pin the commit hash (or a specific run_id) in every figure / table caption.
2. Report model key (from `models.json`), backend, seed, intervention settings, and `measure_layer` / `act_layer` separately when they differ.
3. Run at least 3 seeds for each claimed comparison; ≥5 if claiming reproducibility (DSR analysis).
4. Report mean + stdev (or a confidence interval), not the best run.
5. Publish raw `summary.json` and `events.jsonl` artifacts used for any plot or table.
6. For shadow/active comparisons: record `intervention_applied` counts per run — a "passing" cell where the controller never fired is a measurement-layer confound (see `RESEARCH_CONTROLLER.md` F26).

## Standard Run Metadata

The unified `runtime_lab.cli.main` runners (and the warm `observer_daemon.py`) all
write:

- `runs/<mode>_run_<timestamp>/config.json` — full experiment configuration with
  `config_hash` (SHA-256 of sorted-key JSON).
- `runs/<mode>_run_<timestamp>/summary.json` — aggregate metrics, runtime info
  (device / dtype / policy notes), and an `advisory` block with structured
  `observations` / `likely_causes` / `next_actions` / `flags` for LLM-driven workflows.
- `runs/<mode>_run_<timestamp>/events.jsonl` — per-token diagnostics including
  `raw_div`, spectral metrics (token-time axis), windowed SVD, layer stiffness
  for all probed layers, and (for control mode) `intervention_applied`,
  `scale_used`, `controller_drift_norm`, `controller_reference_hidden_norm`,
  resolved `measure_resolved_layer_idx` and `act_resolved_layer_idx` for
  provenance.
- `runs/<mode>_run_<timestamp>/output.txt` — the generated text.

Mode-specific extras:

- **observe**: pure passive run, no intervention fields.
- **stress**: A/B branchpoint comparison via SeedCache. Writes `results.json`
  with `metrics.logit_kl_series`, `logit_kl_mean_during`, `logit_kl_mean_after`,
  `token_match_rate`, `recovery_ratio`, `regime`. Writes both `baseline_output.txt`
  and `intervention_output.txt`.
- **hysteresis**: BASE → PERTURB → REASK protocol. Writes `frame_base.json`,
  `frame_perturb.json`, `frame_reask.json` with per-stage telemetry; `delta.txt`
  with the perturbation prefix used; `output_base.txt` / `output_perturb.txt` /
  `output_reask.txt`. `summary.json.metrics` includes `drift`, `hysteresis`,
  `recovery`, `regime` (one of `elastic` / `partial` / `plastic` / `runaway` /
  `no-propagation`), and a `perturbation_did_not_propagate` flag when `drift`
  falls below the propagation threshold.
- **control**: closed-loop run. Writes `events.jsonl` per token, `summary.json`
  with `status_counts` (WARNING / CRITICAL / COOLDOWN), `avg_raw_div_mean`,
  `avg_score_mean`, and the `advisory` block.
- **sweep_<mode>_<timestamp>/**: when invoked with `--seeds`, an aggregated
  `sweep.json` is written alongside the individual run directories with mean /
  stdev / range / per-seed run_ids for every numeric metric in the per-run
  summaries.

## Suggested Reporting

1. For every comparison cell, report the count of valid pairs after the
   shadow/active pairing rules in `scripts/analyze_branchpoints.py` apply.
   "Valid" requires shared (model, prompt, seed, max_tokens, temperature,
   measure_layer, act_layer) and `intervention_applied > 0` for active.
2. When a claim depends on cross-prompt or cross-model generalization, run the
   analysis on each slice independently and report the per-slice number alongside
   any combined number — combined AUROCs can be inflated by lucky pair-level
   splits across heterogeneous variants (see `RESEARCH_CONTROLLER.md` F30/F31).
3. For controller-mode A/B claims, always report `intervention_applied` totals
   per cell. If the active cell barely fired (or fired more than expected) the
   measurement is suspect.

## Public Release Packaging

1. Tag a release on `main` (e.g., `v0.2.0` for the v2 paper era).
2. If publishing run artifacts, upload them as a release asset rather than
   committing to the repo (`runs/` is gitignored — keep it that way).
3. Keep `CITATION.cff` in sync with the most recent paper revision.
4. (Optional) connect the repo to Zenodo for DOI minting on tagged releases.
