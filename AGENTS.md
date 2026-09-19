# Repository Guidelines

## Project Structure & Module Organization

This repository uses a Python `src` layout. Keep stable public imports in `src/llm2jev/__init__.py`. Protocol and domain objects live in `src/llm2jev/models/`; binary compilation, prompt rendering, normalization, and response assembly live in `src/llm2jev/inference/`. Shared internal JSON and probability helpers belong in `src/llm2jev/utils/`, not the public package API. Tests mirror these responsibilities under `tests/` using `test_<module>.py` names.

Design documents live in `docs/`. Read `docs/concepts.md` for the target wire behavior, `docs/binary-question-design.md` for the binary-reranker approach, and `docs/implementation-plan.md` for status and sequencing. Update the plan when completing a stage or changing an architectural decision.

## Build, Test, and Development Commands

- `uv sync --extra transformers` creates `.venv` and installs the package with its inference dependencies.
- `uv run python -m unittest discover -s tests -v` runs the complete test suite.
- `uv run python -m compileall -q src tests` recursively checks syntax for source and tests.
- `uv build` verifies source and wheel package construction.

Python 3.10 or newer is required.

## Coding Style & Naming Conventions

Use four-space indentation, type annotations, and standard-library features unless a dependency has a demonstrated need. Use `PascalCase` for public classes, `snake_case` for functions and modules, and leading underscores for internal helpers. Keep public APIs small and re-export them deliberately from `llm2jev.__init__`. Prefer frozen, keyword-only dataclasses for immutable protocol objects. No formatter or linter is currently configured; follow the existing style and avoid unrelated formatting changes.

## Testing Guidelines

Use `unittest.TestCase` and descriptive `test_<behavior>` methods. Every validation rule needs both a valid serialization case and an invalid-input case. Model-facing work should first use deterministic fake backends; tests must not require network access or model downloads. Run the full suite and recursive compile check before submitting changes.

## Commit & Pull Request Guidelines

History currently contains only `Initial commit`, so no formal convention exists. Use concise imperative subjects, for example `Add binary question compiler`. Keep commits focused. Pull requests should explain the behavior change, list verification commands, link relevant issues, and identify any protocol or probability assumptions. Include sample JSON for wire-format changes; screenshots are unnecessary unless a UI is added.

## Architecture Notes

The intended inference path is binary and prefill-only: compile each question candidate, read the next-token `yes`/`no` logits, and assemble Jev-compatible probabilities. Do not introduce text generation or freeze prompt field names without supporting evaluation evidence.
