# prompt-eval

FastAPI + Jinja2 + htmx web app for evaluating LLM prompts. SQLite via SQLModel. Python 3.12, managed with uv.

## Commands

```bash
uv sync                      # install
uv run pytest                # tests (no network; engines are faked)
uv run ruff check . && uv run ruff format .
uv run uvicorn prompt_eval.main:create_app --factory --reload   # the user runs this, not you
```

## Map (`src/prompt_eval/`)

- `models.py`: all tables. `db.py`: engine, `init_db`, additive column migration.
- `engines/`: providers (Anthropic SDK, OpenAI-compatible over httpx), `EngineClient` (rate limits + retries), `EngineRegistry` (one client per engine).
- `generation.py` test case generation · `runner.py` runs as asyncio tasks · `static_checks.py` · `judge.py` · `report.py` · `suggestions.py`.
- `routes/`: one module per area. `templates/`: pages set `section`, `active_project_id`, `active_prompt_id` for the sidebar.

## Rules

- Every model call goes through `EngineClient.complete()` so limits apply. Structured output: set `ChatRequest.json_schema`.
- Anthropic: don't send `temperature`; use `effort`. Default models `claude-sonnet-5-5` / `claude-opus-5-5`.
- Schema changes are additive only: new columns must be nullable or have a scalar default (see `db.add_missing_columns`).
- Prompt versions are immutable. Runs snapshot scoring settings; results snapshot test case and engine details.
- Endpoints that start or cancel runs must be `async def` (they need the event loop).
- Tests: swap `app.state.engines.provider_factory` for fakes; never call real APIs.
- Don't start the dev server; give the user the command. No CI. No `Co-Authored-By` trailers in commits.
- Repo text is English.
