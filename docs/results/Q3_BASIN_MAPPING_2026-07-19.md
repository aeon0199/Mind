# Q3 Causal Basin Mapping — 2026-07-19

## Verdict

Observer can now causally continue a clean and a perturbed branch from the same
verified decision context, score both continuations without exposing branch
identity to the evaluator, and preserve flips, no-flips, improvements,
degradations, and ties in one auditable record.

The first Qwen3-1.7B calibration produced real outcome changes, but the
prespecified generalization gate did not pass:

- 60/60 planned runs and all artifact hashes validated;
- 2,820 same-context local branchpoint rows were mapped;
- 69 selected active episodes changed the paired sampled token;
- those sampled-token flips produced **31 / 11 / 27**
  improve / degrade / tie outcomes under the declared rubrics;
- a predictor restricted to clean pre-flip/context features reached mean
  whole-prompt-held-out AUROC **0.571**, below the required 0.70.

Stop condition: **NOT MET**. Observer has established causal basin-outcome
measurement in this scope, not a prompt-robust rule for deciding when a flip
will help. **Controller remains paused.**

## What “better” means here

Each of the ten prompts had a deterministic rubric declared before the
calibration ran. Factual prompts checked requested concepts and constraints;
procedural prompts checked ordered steps and required quantities; creative
prompts checked requested story elements and endings; reasoning prompts checked
the stated calculation and answer; code prompts parsed source with Python's AST
and checked a declared contract without executing generated code.

The scorer received continuation text, prompt key, and rubric—not clean versus
perturbed branch identity. A score difference within 0.025 was a tie.

Therefore, “improve” means a higher score under that prompt's predeclared
deterministic rubric. It is **not a universal semantic-quality judgment**,
human-preference label, or safety assessment.

## Causal protocol

The `causal-branch-continuation-v1` experiment uses two passes:

1. Map every clean decision with independent, same-context clean and perturbed
   one-step forks. Record clean diagnostics before the intervention, local
   hidden/logit effects, argmax flips, and paired sampled-token flips. Advance
   only the clean path.
2. Choose at most one early, middle, and late episode using the outcome-blind
   rule `outcome-blind-early-middle-late-v1`: take the earliest sampled flip in
   each band, otherwise the earliest latent-effect anchor, otherwise an exact
   anchor.
3. Reset and replay the exact clean trajectory. Reject the run if the context,
   decision, or fingerprint does not reproduce.
4. Continue clean and perturbed branches for 48 tokens with the intervention
   disabled. At every continuation position, both branches use deterministic
   paired sampling RNG.
5. Score the two saved continuations with the branch-blind declared rubric.

The protocol deliberately separates the local cause from its trajectory-level
consequence. It also keeps negative controls and selected no-flip episodes
instead of silently dropping them.

## Design

| Dimension | Value |
|---|---|
| Model | Qwen3-1.7B, Hugging Face backend, MPS float32 |
| Prompt classes | factual, procedural, creative, reasoning, code |
| Prompts | 10 total; 2 independently declared prompts per class |
| Seeds | 0, 1, 2 |
| Intervention | layer-27 relative additive direction |
| Magnitudes | 0 identity control; 0.30 active |
| Sampling | temperature 0.8, top-p 1.0 |
| Clean mapping | 48 generated tokens requested |
| Continued branch | 48 tokens requested per branch |
| Selected episodes | at most 3 per run: early, middle, late |
| Total | 60 runs; 2,820 local rows; 180 selected episodes |

The 30 active runs and 30 identity-control runs use the same prompt/seed grid.

## Artifact validation and coverage

- 60/60 manifest cells completed and revalidated.
- 60 unique run directories and 60 unique final-config hashes.
- Every selected episode passed exact clean replay and context-fingerprint
  checks before continuation.
- All 30 identity-control runs were exact: 90/90 selected control episodes had
  identical tokens and tied scores.
- Active runs contained 90 selected episodes: 69 sampled-token flips and 21
  selected no-flips.
- Across all active selected episodes, outcomes were 31 improve, 11 degrade,
  and 48 tie.
- Across the 69 sampled-token flips alone, outcomes were 31 improve, 11
  degrade, and 27 tie.
- Pass one found 228 local sampled-token flips. The outcome-blind three-episode
  budget selected 69 of them, or 30.3%. The result therefore describes the
  declared early/middle/late sample, not every local flip.

Manual inspection also found all four expected cases in the saved artifacts:
a flip with a higher rubric score, a flip with a lower score, a changed
continuation that tied under the fixed code rubric, and a selected no-flip
whose two continuations were exactly identical.

## Prompt-held-out predictor

The binary predictor used only:

- clean diagnostics recorded before the intervention;
- clean hidden norm and top-1 margin;
- consumed-token and prefix-shape features;
- decision position and context length;
- the declared prompt class.

It did not use the perturbed hidden state, local intervention effect, selected
token outcome, continuation, or rubric result. Preprocessing and logistic
regression fitting occurred inside each training fold, and each test fold held
out a complete prompt.

Ties and no-flips remained in coverage but were not relabeled as improve or
degrade. That left 42 eligible binary rows: 31 improve and 11 degrade.

| Held-out prompt | Test rows | AUROC | Precision | Recall |
|---|---:|---:|---:|---:|
| clockwork fox | 5 | 0.000 | 0.000 | 0.000 |
| lighthouse future | 6 | 0.889 | 0.667 | 0.667 |
| photosynthesis | 5 | 0.667 | 0.750 | 1.000 |
| train distance | 4 | 0.500 | 0.333 | 0.500 |
| water cycle | 7 | 0.800 | 1.000 | 0.800 |
| **Mean** | — | **0.571** | **0.550** | **0.593** |

Five held-out prompts contained both labels and produced valid AUROC folds.
Four other prompts had binary rows from only one outcome class and were
reported as invalid folds. The palindrome prompt produced only ties and
therefore no binary row.

Training AUROC was 0.991–1.000 while held-out results ranged from 0.000 to
0.889. With 42 rows and many diagnostic features, this is strong evidence of
overfit rather than a stable general predictor. Feature weights are retained
for hypothesis generation, not promoted as causal laws.

## Scoped secondary observation

The procedural class produced **12 improve, 0 degrade, and 6 ties** across two
prompts and three seeds. Sixteen of its 18 selected episodes contained a
sampled-token flip. This is the strongest descriptive slice in the run and a
reasonable replication target.

It does not replace the prespecified Q3 gate:

- it covers only two prompts in one model;
- the evaluator rewards explicit requested steps and quantities;
- two procedural held-out folds lacked degradation examples and could not
  produce AUROC;
- the prompt-held-out predictor as a whole remained below threshold.

Calling this a controller-ready “procedural prompts improve” rule would move
the goalposts after seeing the results. Observer records it as a scoped
secondary hypothesis only.

## Stop-condition decision

The declared requirement was mean prompt-held-out AUROC at least 0.70 across at
least three valid prompt splits. Five splits were valid, but observed mean
AUROC was 0.571.

**Q3 status: measured negative at this calibration.** The causal instrument and
outcome protocol work; the proposed pre-flip/context feature set does not yet
generalize well enough to predict better versus worse outcomes across unseen
prompts.

The controller-return program still has none of its three stricter gates:

1. Q1 cleared scoped AUROC but not its required precision/recall trigger.
2. Q2 found a candidate layer but not a prompt-robust bounded-cascade layer.
3. Q3 did not clear its prompt-held-out outcome-prediction threshold.

**Controller remains paused.**

## Honest next research step

If basin mapping continues, the next experiment should be a preregistered
replication rather than controller work:

- add independent procedural prompts and prompts designed to produce both
  improvement and degradation labels;
- increase unique prompt-level holdouts before expanding the feature set;
- improve outcome-rubric resolution, especially for truncated code
  continuations, while freezing the revised rubric before new runs;
- keep the same exact controls, replay checks, outcome-blind selection, and
  prompt-held-out analysis boundary.

## Durable artifacts

- Root:
  `/Users/jmalone/Documents/Artifacts/Observer/m3r-basin-mapping-2026-07-19`
- Manifest: `basin_manifest.json`
- Machine-readable analysis: `basin_analysis.json`
- Generated analysis table: `BASIN_ANALYSIS.md`
- Reproducible runner: `experiments/run_basin_calibration.py`
- Independent analyzer: `analysis/analyze_basins.py`
