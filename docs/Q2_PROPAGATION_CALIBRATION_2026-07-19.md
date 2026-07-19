# Q2 Propagation Calibration — 2026-07-19

## Verdict

Q2 now has a **scoped mechanistic answer** on Qwen3-1.7B:

- raw hidden-state L2 grows strongly toward the end of the transformer because
  the residual stream itself changes scale;
- relative and angular disturbance usually contracts instead of exploding;
- final RMSNorm removes a purely radial final-layer scaling intervention while
  preserving directional additive changes;
- output disruption depends on intervention family, scaling direction, depth,
  prompt, and trajectory. Raw L2 amplification alone does not identify a
  destructive cascade.

Layer 13 had the lowest aggregate mean logit KL at every tested non-control
setting, making it the best **candidate** in this narrow grid. It did not
uniformly minimize argmax flips within each prompt slice, so Q2's
controller-return criterion is not met. **Controller remains paused.**

## Protocol

The `same-context-layer-propagation-v1` experiment clones one verified source
cache at every clean decision:

1. run one clean next-token step;
2. restore paired sampling RNG and run one perturbed step from the same cache;
3. capture the output of the injection block, every downstream transformer
   block, and the model's final normalization;
4. compare hidden L2, relative L2, cosine distance, logits, and local token
   decisions;
5. discard the perturbed fork and advance only the clean trajectory.

This is a local propagation map. It does not mix later context divergence into
the label.

## Design

| Dimension | Value |
|---|---|
| Model | Qwen3-1.7B, Hugging Face backend, MPS float32 |
| Prompts | Sourdough procedural prompt; water-cycle descriptive prompt |
| Seeds | 0, 1, 2, 3, 4 |
| Sampling | temperature 0.8, top-p 1.0 |
| Injection layers | 6, 13, 20, 27 |
| Additive magnitudes | 0 control; 0.10, 0.20, 0.30 relative |
| Scaling factors | 1.0 control; 0.5, 1.5 |
| Requested generation | 16 tokens per run |
| Measured forks | 15 subsequent decisions per run |
| Total | 280 runs; 4,200 paired decisions; 28 aggregate cells |

The first token is sampled by the prompt seed-cache pass; the following 15
decisions are the independently forked propagation measurements.

## Artifact validation

- 280/280 cells completed.
- 280 unique run directories.
- 280 unique final-config hashes; every config was rehashed during analysis.
- Every context pair matched.
- Every requested transformer block and terminal normalization was captured.
- All 80 identity-control runs were exact no-ops across 1,200 paired decisions:
  zero hidden delta, zero logit KL/JS, and zero local token flips.
- Analysis uses equal-weight run means. A run does not receive extra
  statistical weight for containing more decisions.

## Finding 1 — Raw L2 growth is not the same as structural destruction

Representative additive magnitude 0.30 results:

| Inject layer | Direct relative delta | Peak raw-L2 amplification | Final-norm relative delta | Mean logit KL | Argmax flips |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.300 | 8.662 at L27 | 0.0699 | 0.0347 | 10/150 |
| 13 | 0.300 | 4.214 at L27 | 0.0975 | 0.0227 | 9/150 |
| 20 | 0.300 | 1.684 at L27 | 0.1929 | 0.0629 | 12/150 |
| 27 | 0.300 | 1.000 at L27 | 0.2800 | 0.0861 | 18/150 |

Earlier additive injections grow more in **absolute** L2, but their disturbance
shrinks more as a fraction of the clean hidden state. A layer-6 injection ends
at about 23% of its starting relative size; a layer-27 injection retains about
93%. The old phrase “earlier layer = worse” therefore does not hold as a
universal local law. Historical F27 used repeated closed-loop intervention and
can include accumulated context and exposure effects that this one-step
protocol deliberately removes.

## Finding 2 — Final RMSNorm is selective, not a general eraser

At layer 27, both non-identity scaling settings begin with relative delta 0.50:

| Intervention | Final-norm relative delta | Mean logit KL | Argmax flips |
|---|---:|---:|---:|
| scale 0.5 | 0 exactly | 0 exactly | 0/150 |
| scale 1.5 | 7.54e-8 | 1.33e-8 | 0/150 |
| additive 0.30 | 0.2800 | 0.0861 | 18/150 |

RMSNorm removes the radial scaling degree of freedom at numerical precision.
It does not erase a directional additive perturbation. This independently
replicates and mechanistically closes historical F17/F22.

## Finding 3 — Direction and prompt matter

Scaling down and scaling up have the same direct relative magnitude but
different consequences:

| Inject layer | KL, scale 0.5 | KL, scale 1.5 |
|---:|---:|---:|
| 6 | 0.1947 | 0.0447 |
| 13 | 0.1140 | 0.0299 |
| 20 | 0.1154 | 0.0448 |
| 27 | 0 | 1.33e-8 |

Prompt slices also differ substantially. For scale 0.5 at layer 6, mean KL was
0.0672 on sourdough and 0.3223 on water cycle. For additive 0.30 at layer 6,
it was 0.0114 and 0.0580. A layer/magnitude-only controller rule would discard
material causal structure.

## Finding 4 — Layer 13 is a candidate, not a controller answer

Aggregate mean logit KL by injection layer:

| Setting | L6 | L13 | L20 | L27 |
|---|---:|---:|---:|---:|
| additive 0.10 | 0.00341 | **0.00245** | 0.00604 | 0.00909 |
| additive 0.20 | 0.01581 | **0.01002** | 0.02603 | 0.03731 |
| additive 0.30 | 0.03468 | **0.02265** | 0.06285 | 0.08615 |
| scale 0.5 | 0.19472 | **0.11397** | 0.11543 | 0 |
| scale 1.5 | 0.04469 | **0.02988** | 0.04483 | 1.33e-8 |

The layer-27 scaling zeros are normalization no-ops, not useful control
actions. Among layers where both intervention families survive, layer 13 has
the lowest aggregate KL in every tested setting. However:

- it does not always have the fewest argmax flips;
- the prompt-specific ordering is not perfectly stable;
- this protocol applies one intervention at a time, not a repeated controller;
- semantic output quality was not measured.

It is therefore a principled location for later mapping, not permission to
resume controller work.

## Scope and next step

This result covers one model, two prompts, five seeds, four layers, one fixed
additive direction, two scaling directions, and local one-step propagation.
It answers how deltas move in this scope and shows why raw L2 alone was an
insufficient definition of cascade.

The next foundation task is to rebuild Q3 around causal branch continuations
and explicit outcome evaluation. The historical M3.1 proposal depended on the
old controller and cannot serve as the root protocol.

## Durable artifacts

- Root:
  `/Users/jmalone/Documents/Artifacts/Observer/q2-propagation-2026-07-19`
- Manifest: `propagation_manifest.json`
- Machine-readable aggregation: `propagation_analysis.json`
- Full generated curve tables: `PROPAGATION_ANALYSIS.md`
