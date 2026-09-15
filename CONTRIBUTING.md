# Contributing to Aster

Thanks for helping build a useful and reproducible research tool. Small changes with
clear motivation and evidence are easiest to review.

## Get started

Fork the repository, clone your fork, create a virtual environment, and install:

```bash
python -m pip install -e '.[dev]'
pytest
```

Use a descriptive branch such as `fix/seed-validation` or `feat/asha-scheduler`.
Open an issue before large API changes so the design can be discussed. Small fixes,
documentation corrections, and regression tests can go directly to a pull request.

## Before a pull request

- Explain the concrete problem and resulting behavior.
- Add meaningful tests for changes to scheduling, statistics, seeds, budgets, or failures.
- Run `pytest`, `ruff check .`, `ruff format --check .`, and `python -m build`.
- Update documentation when behavior or the public API changes.
- Keep unrelated changes separate and preserve backward compatibility where practical.

## Research standards

State assumptions, keep optimization and audit data separate, and label heuristic methods.
Do not introduce performance claims based on one seed or unbalanced hardware budgets.
Report losing cases as well as winning cases. Benchmark contributions should include the
command, code revision, hardware, thread settings, seeds, dependencies, and raw results.
Do not commit private financial data or credentials.

Good first contributions include input-validation tests, a new controlled-noise objective,
Windows process-pool verification, and improvements to the result schema documentation.
Larger priorities are in [ROADMAP.md](ROADMAP.md).

## Review and licensing

The maintainer reviews API and research-method changes. By submitting a contribution,
you agree that it may be distributed under this project's MIT license. Only submit code,
documentation, and data that you have the right to share. No contributor license agreement
is required.
