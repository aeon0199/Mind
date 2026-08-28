# Contributing

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Optional research extras:

```bash
python -m pip install -e ".[research]"
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

- [docs/workflow/RESEARCH_WORKFLOW.md](docs/workflow/RESEARCH_WORKFLOW.md)

It covers:
- how to orient from `docs/research/RESEARCH.md`
- how to define success criteria before runs
- how to verify provenance before interpretation
- how to update the research log after each session

## Quick Checks

These match what CI runs:

```bash
# Compile-check the active runtime and support tools
python -m compileall -q src analysis experiments tools

# Run the test suite
python -m pytest -q
```

The test suite is small but guards against drift — CLI semantic-layer parsing,
branchpoint-analyzer defaults, README quickstart commands pointing at the unified
runtime. Add tests when adding features that other agents will rely on.

The legacy modules (`archive/prototypes/`, `archive/controller/`, and `archive/analysis/`) are kept as historical reference for the v1
paper. Production research uses `src/runtime_lab/` — invoked via
`python -m runtime_lab.cli.main {observe|stress|hysteresis|control}` or the
warm-model daemon at `tools/observer_daemon.py`. Don't add new code to the
legacy directories.
