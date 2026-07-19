# Reproducibility Checklist

## Required for Public Claims

1. Pin the commit hash (or a specific run_id) in every figure / table caption.
2. Report the authoritative loaded-model identity, backend, resolved device/dtype,
   seed, sampling settings, intervention settings, and requested/resolved layers.
3. Run at least 3 seeds for each claimed comparison; ≥5 if claiming reproducibility (DSR analysis).
4. Report mean + stdev (or a confidence interval), not the best run.
5. Publish raw `summary.json` and `events.jsonl` artifacts used for any plot or table.
6. Name the protocol. A persistent trajectory comparison, an independent local
   branchpoint, and matched recovery answer different causal questions.
7. Include a zero-perturbation control. It must preserve matched tokens and
   report zero causal distance within the protocol's numerical contract.
8. For local flippability claims, use only `branchpoint_run_*` artifacts created
   by independent same-context one-step forks. Controller shadow/active rows are
   not local branchpoint labels.
9. For recovery claims, require `protocol_validity.matched_recovery=true` and
   proven propagation above the configured floor.
10. For archived shadow/active controller comparisons, record
    `intervention_applied` counts per run—a passing cell where the controller
    never fired is a measurement-layer confound.

## Standard Run Metadata

The unified `runtime_lab.cli.main` runners (and the warm
`observer_daemon.py`) write provenance before interpretation:

- `runs/<mode>_run_<timestamp>_<unique-suffix>/config.json` — the exact final
  experiment configuration with `config_hash` (SHA-256 of sorted-key JSON).
- `summary.json` — aggregate metrics, runtime info
  (device / dtype / policy notes), and an `advisory` block with structured
  `observations` / `likely_causes` / `next_actions` / `flags` for LLM-driven workflows.
- Canonical events distinguish the `consumed_token_*` input to a decision from
  its `predicted_next_token_id`. Legacy `token_*` fields alias the consumed
  token only.
- Diagnostic records expose simple baselines plus
  `local_prediction_error`; `divergence` remains an equivalent compatibility
  field.

Mode-specific extras:

- **observe**: pure passive run. Writes `events.jsonl`, `summary.json`, and
  `output.txt`.
- **stress**: A/B branchpoint comparison via SeedCache. Writes `results.json`
  with `metrics.logit_kl_series`, `logit_kl_mean_during`, `logit_kl_mean_after`,
  `token_match_rate`, `recovery_ratio`, `regime`. Writes both `baseline_output.txt`
  and `intervention_output.txt`. These continued branches measure direct effect
  plus propagation; rows after a token flip are not independent local labels.
- **branchpoints**: independent same-context one-step counterfactuals. Writes
  `events.jsonl` with context fingerprints, clean/perturbed predictions,
  argmax/sample flips, hidden distances, KL/JS, and clean diagnostics.
  `protocol_validity.independent_one_step_forks` and
  `all_contexts_matched` must be true.
- **hysteresis**: `matched-exposure-recovery-v2`. Writes four primary traces:
  `clean_exposure`, `perturbed_exposure`, `clean_recovery`, and
  `perturbed_recovery`, plus aligned distance curves in `summary.json` and
  `events.jsonl`. `initial_intervention_distance` is the first active paired
  distance; `active_window_peak_distance` may also include accumulated
  consequences after a prior flip. Both recovery branches receive no new
  instruction and no intervention. Legacy `frame_base.json` /
  `frame_perturb.json` / `frame_reask.json` files are labeled endpoint
  compatibility views, not metric sources.
- **control**: closed-loop run. Writes `events.jsonl` per token, `summary.json`
  with `status_counts` (WARNING / CRITICAL / COOLDOWN), `avg_raw_div_mean`,
  `avg_score_mean`, and the `advisory` block. This mode is historical/downstream
  and is excluded by the active local-branchpoint analyzer.
- **sweep_<mode>_<timestamp>/**: when invoked with `--seeds`, an aggregated
  `sweep.json` is written alongside the individual run directories with mean /
  stdev / range / per-seed run_ids for every numeric metric in the per-run
  summaries.

## Suggested Reporting

1. For every branchpoint analysis, report total runs, whole-run train/test
   membership, contexts matched, rows, positive labels, AUROC, precision,
   recall, and min/max across unique run-grouped splits. When the requested
   split count exceeds the finite number of unique group assignments, report
   the smaller honest count rather than weighting duplicate splits.
2. Fit imputation, scaling, feature filtering, and classifiers using training
   runs only.
3. When a claim depends on cross-prompt or cross-model generalization, run the
   analysis on each slice independently and report the per-slice number alongside
   any combined number.
4. For matched hysteresis, report direct-effect peak, propagated exposure
   endpoint, residual recovery endpoint, propagation floor, EOS truncation, and
   all protocol-validity fields.
5. For controller-mode A/B claims, always report `intervention_applied` totals
   per cell. If the active cell barely fired (or fired more than expected) the
   measurement is suspect.

## Diagnostic Validation

Before assigning meaning to advanced signals, run:

```bash
python scripts/validate_diagnostics.py \
  --json-out /tmp/observer-diagnostic-validation.json
```

The command exits nonzero unless constant, linear-drift, oscillatory, and shock
trajectories pass explicit checks through the actual diagnostic manager. This
validates mechanics and basic ordering, not downstream semantic meaning.

## Public Release Packaging

1. Tag a release on `main` (e.g., `v0.2.0` for the v2 paper era).
2. If publishing run artifacts, upload them as a release asset rather than
   committing to the repo (`runs/` is gitignored — keep it that way).
3. Keep `CITATION.cff` in sync with the most recent paper revision.
4. (Optional) connect the repo to Zenodo for DOI minting on tagged releases.
