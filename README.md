# prompt-eval

A small web app for evaluating LLM prompts: define projects and prompts, generate test cases,
run them against target models, score the outputs with static checks and an LLM judge,
and get a report on what to change in the prompt.

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
uv sync
cp .env.example .env   # add your ANTHROPIC_API_KEY and/or LM Studio URL
uv run uvicorn prompt_eval.main:create_app --factory --reload
```

Then open http://localhost:8000.

## Engines

An engine is a model endpoint plus its limits. Add them on the **Engines** page:

- **Anthropic**: model id (e.g. `claude-opus-5-5`) and optional effort. The key is read from
  `ANTHROPIC_API_KEY` in `.env`; keys are never stored in the database.
- **OpenAI-compatible** (LM Studio and similar): model name and base URL (defaults to
  `LMSTUDIO_BASE_URL`).

Each engine has its own parallel-request cap, optional requests/min and tokens/min limits,
and retries with exponential backoff on 429/5xx (honouring `retry-after`; a 429 pauses the
whole engine). All work on one engine shares these limits.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```
