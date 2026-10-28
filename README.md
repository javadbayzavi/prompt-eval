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

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```
