# Repository Guidelines

## Project Structure & Module Organization
- `src/` – Core code: `llm_client.py`, `config.py`, `document_segmenter.py`, `test_sequential_reading.py`, `main.py`.
- `tests/` – Pytest suite: `test_config.py`, `test_llm_client.py`, `test_contract_reader.py`.
- `data/` – Sample inputs and contract corpus (large text files).
- `output/` – Generated QA results and reports; do not rely on committed outputs.
- Top-level helpers: `simple_agent.py`, `main.py`, `pyproject.toml`, `README.md`, `CLAUDE.md`.

## Build, Test, and Development Commands
- Install (recommended): `uv sync` and test extras with `uv sync --extra test`.
- Alt install (pip): `pip install -e .` and `pip install -e .[test]`.
- Run QA system: `uv run python -m src.test_sequential_reading`.
- Run agent demo: `uv run python simple_agent.py`.
- Run tests: `uv run pytest` (specific file: `uv run pytest tests/test_config.py`).
- Optional coverage: `pytest --cov=src` (requires `pytest-cov`).

## Coding Style & Naming Conventions
- Python ≥ 3.13; follow PEP 8 with 4-space indentation and type hints.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, module names are lowercase.
- Docstrings: concise, with Args/Returns (see `src/llm_client.py`).
- Imports: from outside use `from src.module import …`; within `src/`, sibling imports as in existing files.
- No enforced linter/formatter; keep imports ordered and lines ≤ 100 chars. If you use tools locally, prefer Black + Ruff.

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio`, `pytest-mock`.
- Location/pattern: `tests/test_*.py`; keep tests unit-scoped and deterministic.
- Async tests: use `pytest.mark.asyncio` where applicable.
- Integration tests may require a real `OPENROUTER_API_KEY` (use `-m integration`).

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise summary; optional type prefix (`feat:`, `fix:`, `docs:`). Examples in history: “Enhance document analysis…”, “feat: add evaluation cache…”.
- PRs: include purpose, scope, and key changes; link issues; add/updated tests; note config/env needs; attach relevant logs or QA metrics (e.g., paths under `output/legal/`).

## Security & Configuration Tips
- Secrets: store in `.env`; never commit. Required: `OPENROUTER_API_KEY`. Optional: `LITELLM_MODEL`, `LITELLM_TEMPERATURE`, `MAX_TOKENS`.
- Large artifacts: avoid committing generated outputs; prefer reproducing via commands above.
