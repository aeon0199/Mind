# Contributing

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional research extras:

```bash
pip install -r requirements-optional.txt
```

## Pull Request Guidelines

1. Keep changes scoped to one subsystem when possible.
2. Include a short reproducibility note in PR description:
   - command used
   - seed(s)
   - key output paths
3. Do not merge claims without multi-seed results.
4. Keep CLI behavior backward compatible unless discussed first.

## Research Workflow

For experiment-driven work, use the repo workflow note:

- [docs/RESEARCH_WORKFLOW.md](docs/RESEARCH_WORKFLOW.md)

It covers:
- how to orient from `RESEARCH.md`
- how to define success criteria before runs
- how to verify provenance before interpretation
- how to update the research log after each session

## Quick Checks

These match what CI runs:

```bash
# Compile-check the active runtime + scripts
python -m compileall -q src scripts

# Run the test suite
python -m unittest discover -s tests
```

The test suite is small but guards against drift — CLI semantic-layer parsing,
branchpoint-analyzer defaults, README quickstart commands pointing at the unified
runtime. Add tests when adding features that other agents will rely on.

The legacy modules (`baseline_hysteresis_v1/`, `v1.5/`, `intervention_engine_v1.5_v2/`,
`adaptive_controller_system4/`) are kept as historical reference for the v1
paper. Production research uses `src/runtime_lab/` — invoked via
`python -m runtime_lab.cli.main {observe|stress|hysteresis|control}` or the
warm-model daemon at `scripts/observer_daemon.py`. Don't add new code to the
legacy directories.
