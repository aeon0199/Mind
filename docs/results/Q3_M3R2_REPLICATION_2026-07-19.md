# Q3 M3R-2 Prompt-Breadth Replication — 2026-07-19

## Verdict

M3R-2 successfully replicated Observer's causal basin measurement across 15
entirely new prompts, but it did **not** clear the preregistered outcome
prediction gate:

- 90/90 planned Qwen3-1.7B runs and all artifact hashes validated;
- all 45/45 identity-control runs were exact;
- 5,670 independent same-context local branchpoints were mapped;
- the 45 active runs retained 217 selected episodes: 195 sampled-token flips
  and 22 no-flips;
- sampled-token flips produced **67 / 49 / 79** improve / degrade / tie
  outcomes under the frozen task rubrics;
- 13 valid whole-prompt holdouts produced mean AUROC **0.596**, below the
  preregistered 0.70 threshold; 2 invalid prompt holdouts were reported
  separately because their binary rows lacked degradation examples.

Stop condition: **BELOW THRESHOLD**. The minimum evidence requirement was
satisfied, but the predictive requirement was not. Observer can measure
intervention-caused branch outcomes in this scope; the frozen clean pre-flip
feature set does not reliably predict better versus worse outcomes on unseen
prompts.

The original M3R procedural slice had 12 improvements, no degradations, and 6
ties. Across three new procedural prompts, M3R-2 produced **16 improve, 19
degrade, and 9 ties**. The scoped procedural pattern did not replicate.

**Controller remains paused.**

## Frozen-before-data design

The complete protocol, prompt registry, evaluator fixtures, predictor feature
set, regularization, and stop condition were committed before the smoke or
full-model results were generated.

| Dimension | Frozen value |
|---|---|
| Calibration protocol | `observer-m3r2-basin-replication-v1` |
| Measurement protocol | `causal-branch-continuation-v1` |
| Model | Qwen3-1.7B, Hugging Face backend, Apple MPS float32 |
| Prompts | 15 fresh prompts; 3 each factual, procedural, creative, reasoning, code |
| Seeds | 0, 1, 2 |
| Intervention | layer-27 relative additive direction, seed 42 |
| Magnitudes | 0 exact identity control; 0.30 active |
| Sampling | temperature 0.8, top-p 1.0, top-k 0 |
| Clean mapping | 64 generated tokens requested |
| Branch continuation | 96 intervention-free tokens requested per branch |
| Episode selection | at most 5 outcome-blind episodes per run |
| Tie tolerance | 0.025 rubric-score delta |
| Predictor | deterministic logistic regression, L2 0.05 |
| Evaluation split | hold out one complete prompt |
| Stop condition | mean held-out AUROC at least 0.70 across at least 5 valid prompts |

The 15 keys were disjoint from the original M3R registry:

- factual: `seasons_tilt`, `rock_cycle`, `electric_circuit`;
- procedural: `rice_steps`, `flat_tire_steps`, `tomato_transplant_steps`;
- creative: `glass_violin`, `mars_orchard`, `undersea_library`;
- reasoning: `museum_tickets`, `rectangle_garden`, `discount_tax`;
- code: `dedupe_order_code`, `clamp_code`, `word_frequencies_code`.

## What the outcome label means

Each prompt used a deterministic rubric and validated good/degraded fixtures
that were frozen before generation. The scorer received the prompt definition
and continuation text, but not clean versus perturbed branch identity.

“Improve” means that the perturbed continuation scored higher under that
prompt's declared rubric. “Degrade” means it scored lower.
It is not a universal semantic-quality judgment. It is also not a
human-preference score, correctness oracle, or safety label.

The v2 code evaluator extracts the requested function from surrounding prose
or fences and parses it with `ast.parse`. It never imports or executes model
output. Its source contract resolves more truncated code than the M3R v1
rubric, but it still checks static syntax and declared source criteria rather
than runtime semantics. Markdown fences and extra chatter are penalized by a
separate frozen rubric component.

## Artifact validation and causal coverage

- Manifest status was complete with 90/90 cells.
- All 90 run directories and all 90 config hashes were unique.
- Every episode passed outcome-blind selection, context replay, clean-decision
  replay, paired per-position sampling, and intervention-free continuation
  checks.
- All 45 identity controls were exact. Their 135 selected episodes had no
  local effects or flips and tied exactly.
- Pass one mapped 5,670 branchpoint rows and found 447 local sampled-token
  flips.
- The outcome-blind budget selected 195 of those flips, for 43.6% flip
  coverage.
- The 45 active runs retained 217 episodes: 195 sampled flips and 22 no-flips.
- All active selected episodes produced 67 improve, 49 degrade, and 101 tie
  outcomes.
- Sampled-token flips alone produced 67 improve, 49 degrade, and 79 ties.
- Ties and no-flips remained visible in coverage and were never converted into
  binary predictor labels.

The durable manifest SHA-256 at analysis time was:

```text
4d7c540700a2d2e8367c0b43fac116125cca149c668b27b891664dde907548b1
```

## Outcome slices

| Prompt class | Episodes | Sampled flips | Improve | Degrade | Tie |
|---|---:|---:|---:|---:|---:|
| Factual | 45 | 43 | 20 | 11 | 14 |
| Procedural | 44 | 42 | 16 | 19 | 9 |
| Creative | 44 | 41 | 14 | 13 | 17 |
| Reasoning | 39 | 26 | 10 | 4 | 25 |
| Code | 45 | 43 | 7 | 2 | 36 |
| **Total** | **217** | **195** | **67** | **49** | **101** |

This is structured but not one-directional. Factual runs leaned toward
improvement, creative and procedural runs contained nearly balanced
improvement/degradation counts, reasoning contained many ties and no-flips,
and code was dominated by rubric ties.

## Prompt-held-out predictor

The predictor used exactly the 23 preregistered fields:

```text
clean_top1_margin
clean_hidden_norm
normalized_position
context_sequence_length
prefix_repetition_score
prefix_unique_token_ratio
consumed_token.length
consumed_token.leading_space
clean_diagnostics.local_prediction_error
clean_diagnostics.logit_entropy
clean_diagnostics.hidden_velocity
clean_diagnostics.hidden_acceleration
clean_diagnostics.spectral.spectral_entropy
clean_diagnostics.spectral.permutation_change
clean_diagnostics.svd.effective_rank
clean_diagnostics.svd.top1_energy_frac
clean_diagnostics.layer_stiffness.27.stiffness
clean_diagnostics.layer_stiffness.27.stiffness_trend
prompt_class.code
prompt_class.creative
prompt_class.factual
prompt_class.procedural
prompt_class.reasoning
```

No perturbed-state, local-effect, selected-token, continuation, rubric score,
or post-outcome field entered the model. Imputation and scaling were fit on
training prompts only. L2 regularization remained fixed at 0.05.

There were 116 eligible sampled-flip, non-tie rows: 67 improve and 49 degrade.
Thirteen prompts contained both labels and generated valid holdouts. Two were
invalid:

- `rectangle_garden`: 2 improve / 0 degrade binary rows;
- `word_frequencies_code`: 1 improve / 0 degrade binary row.

| Held-out prompt | Rows | Improve | Degrade | AUROC | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| clamp code | 5 | 4 | 1 | 1.000 | 0.800 | 1.000 |
| dedupe order code | 3 | 2 | 1 | 1.000 | 0.667 | 1.000 |
| discount tax | 8 | 5 | 3 | 0.467 | 0.571 | 0.800 |
| electric circuit | 11 | 7 | 4 | 0.536 | 0.700 | 1.000 |
| flat tire steps | 11 | 5 | 6 | 0.900 | 1.000 | 0.200 |
| glass violin | 9 | 5 | 4 | 0.300 | 0.333 | 0.200 |
| mars orchard | 12 | 7 | 5 | 0.571 | 0.500 | 0.714 |
| museum tickets | 4 | 3 | 1 | 1.000 | 0.750 | 1.000 |
| rice steps | 13 | 7 | 6 | 0.476 | 0.400 | 0.286 |
| rock cycle | 12 | 6 | 6 | 0.611 | 0.545 | 1.000 |
| seasons tilt | 8 | 7 | 1 | 0.000 | 0.667 | 0.286 |
| tomato transplant steps | 11 | 4 | 7 | 0.393 | 0.167 | 0.250 |
| undersea library | 6 | 2 | 4 | 0.500 | 1.000 | 0.500 |
| **Mean** | — | — | — | **0.596** | **0.623** | **0.634** |

Held-out AUROC ranged from 0.000 to 1.000 with standard deviation 0.292. The
model can separate some prompts while reversing or approaching chance on
others. The preregistered minimum of five valid splits was comfortably met,
so this is a measured **below threshold** result rather than an
insufficient-data result.

## Comparison with M3R

| Measure | M3R | M3R-2 |
|---|---:|---:|
| Fresh prompts | 10 | 15 |
| Total runs | 60 | 90 |
| Active selected episodes | 90 | 217 |
| Eligible binary rows | 42 | 116 |
| Valid prompt holdouts | 5 | 13 |
| Mean held-out AUROC | 0.571 | 0.596 |
| Gate | 0.70 | 0.70 |
| Result | below threshold | below threshold |

M3R-2 increased prompt breadth, outcome balance, predictor rows, holdout count,
continuation length, episode coverage, code-rubric resolution, and
regularization. Mean AUROC moved from 0.571 to 0.596 but remained materially
below 0.70. The negative Q3 gate therefore replicated under a stronger design.

The M3R procedural slice was 12 improve / 0 degrade / 6 tie across two prompts.
M3R-2's three new procedural prompts were:

- `rice_steps`: 7 improve / 6 degrade / 2 tie;
- `flat_tire_steps`: 5 improve / 6 degrade / 4 tie;
- `tomato_transplant_steps`: 4 improve / 7 degrade / 3 tie.

Combined: **16 improve, 19 degrade, and 9 ties**. The procedural pattern did
not replicate and is retired as a directional hypothesis in this scope.

## Manual artifact audit

Representative saved episodes were inspected without rescoring:

1. **Improve plus v2 parse:** `clamp_code`, seed 0, decision 3 changed the
   sampled token and moved the rubric from 0.72 to 0.80. Both branches yielded
   extractable, parseable `clamp` functions; the perturbed branch satisfied
   one additional frozen source criterion. Both still used Markdown fences,
   so the separate no-fence component scored zero.
2. **Degrade:** `clamp_code`, seed 1, decision 4 changed the sampled token and
   moved the rubric from 0.20 to 0.00 because the perturbed continuation was
   truncated before a complete function.
3. **Sampled-flip tie:** `clamp_code`, seed 0, decision 9 changed the sampled
   token, but both repetitive, incomplete continuations scored 0.20.
4. **Exact no-flip continuation:** `clamp_code`, seed 0, decision 22 had a
   measurable latent-state effect but no sampled-token flip. The paired
   continuations were token-identical and both scored 0.92.
5. **Static evaluator boundary:** the v2 parser proved source extraction and
   syntax/contract criteria without executing generated code. It cannot by
   itself certify runtime correctness or general code quality.

These cases confirm that the artifact record preserves the distinctions among
local flip, latent-only effect, exact downstream match, operational rubric
change, and evaluator resolution.

## Stop-condition decision

The frozen rule required:

1. at least five valid whole-prompt holdouts; and
2. mean held-out AUROC at least 0.70.

M3R-2 produced 13 valid holdouts but mean AUROC 0.596.

**Q3 status: replicated measured negative on Qwen3-1.7B.** Observer's causal
measurement and artifact discipline are strong; the proposed compact clean
pre-flip predictor does not generalize well enough to gate an intervention.

The controller-return program still lacks two qualifying results:

1. Q1 cleared scoped AUROC but not a controller-ready precision/recall trigger.
2. Q2 produced mechanistic propagation maps but not a prompt-robust bounded
   controller layer.
3. Q3 twice missed its frozen prompt-held-out prediction gate, and the original
   procedural directional pattern failed on fresh prompts.

**Controller remains paused.**

## Durable artifacts

- Root:
  `/Users/jmalone/Documents/Artifacts/Observer/m3r2-basin-replication-2026-07-19`
- Manifest: `basin_manifest.json`
- Machine-readable analysis: `basin_analysis.json`
- Generated analysis: `BASIN_ANALYSIS.md`
- Independent reanalysis:
  `/tmp/observer-m3r2-analysis-independent.json`
- Frozen runner: `experiments/run_basin_calibration.py`
- Frozen analyzer: `analysis/analyze_basins.py`

The independent analyzer rerun matched the durable analysis exactly after
removing only `generated_at`.
