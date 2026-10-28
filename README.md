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

## How it works

1. **Projects and prompts.** A prompt has a task description, an expected output format
   (JSON, YAML, XML, Markdown or text), an optional JSON Schema, and versions. Versions are
   immutable; editing the text saves the next one. Templates use `{{variable}}` placeholders.
2. **Test cases.** Add them by hand or generate them with any engine: choose how many and the
   mix of normal, edge and adversarial cases, preview, then keep the ones you want.
3. **Runs.** Run a version on several engines at once. Each engine has its own limits, and
   results are saved with the exact prompt that was sent.
4. **Scoring.**
   - *Static checks* (parse, schema, no extra text, not cut off, length, regex), each with its
     own weight as a percent of the static score.
   - *Judge*: a model grades each output on weighted criteria, with reasoning per criterion.
   - *Final score* = static share × static + the rest × judge.
5. **Report.** Scores by engine and category, check pass rates, criterion averages, every test
   with the reasons for its score, and a Markdown download. A model can then suggest prompt
   changes and a revised prompt, which you can save as the next version and compare.

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
