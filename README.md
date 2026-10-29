# prompt-eval

A small self-hosted web app for **evaluating LLM prompts**.

You describe what a prompt should do and what its output should look like. The app generates
test cases, runs them on one or more models, scores every answer with rule-based checks and an
LLM judge, and produces a report that explains each score and suggests what to change in the
prompt. Save the suggestion as a new version, run it again, and compare.

```
define prompt → generate test cases → run on models → score (static + judge) → report → improve → compare
```

## Features

- **Projects, prompts and versions.** Each prompt has a task description, an expected output
  format (JSON, YAML, XML, Markdown or text), an optional JSON Schema, and immutable versions.
  Prompt text uses `{{variable}}` placeholders that test cases fill in.
- **Test cases.** Write them by hand or generate them with a model: choose how many and the mix
  of normal, edge and adversarial cases, review the preview, keep what you want.
- **Runs on several models at once.** Each model ("engine") has its own parallel-request cap,
  requests/min and tokens/min limits, and retries with backoff on rate limits and server errors.
- **Static checks**, each with its own weight: parses, matches the JSON Schema, no extra text
  around the payload, not cut off, max length, must / must not match a regex.
- **LLM judge** that grades each answer on weighted criteria (correctness, completeness,
  instruction following by default), with reasoning per criterion.
- **Final score** = a configurable share of the static score + the rest from the judge.
- **Report** per run: scores by engine and test category, check pass rates, criterion averages,
  every test with the reasons for its score, and a Markdown download.
- **Prompt suggestions**: a model reads the report and proposes concrete changes and a revised
  prompt, which you can save as the next version with one click.
- **Compare runs** of the same prompt, by engine and by test case.

Supported providers: **Anthropic** (Claude) and any **OpenAI-compatible** server, such as
[LM Studio](https://lmstudio.ai/) for local models.

## Requirements

- [uv](https://docs.astral.sh/uv/). It installs Python 3.12+ for you if needed.
- At least one model:
  - an Anthropic API key from the [Claude Console](https://console.anthropic.com/), and/or
  - LM Studio (or another OpenAI-compatible server) running locally.

Install uv if you don't have it:

```bash
brew install uv
```

or

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

```bash
git clone https://github.com/javadbayzavi/prompt-eval.git
cd prompt-eval
uv sync
cp .env.example .env
```

Then edit `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | (empty) | Needed for Claude engines. Never stored in the database. |
| `LMSTUDIO_BASE_URL` | `http://localhost:1234/v1` | Default base URL for OpenAI-compatible engines. |
| `DATABASE_URL` | `sqlite:///./data/prompt_eval.db` | Where data is kept. The file is created on first run. |

## Running

```bash
uv run uvicorn prompt_eval.main:create_app --factory --reload
```

Open <http://localhost:8000>. Use `--port 8080` for another port, or `--host 0.0.0.0` to reach
it from other machines on your network. Drop `--reload` when you're not editing the code.

To use a local model, start the server in LM Studio (Developer tab → Start server) and load a
model first.

## Quick start

1. **Engines → New engine.** Add the model you want to test, e.g. `claude-sonnet-5-5`, and a
   stronger one to judge, e.g. `claude-opus-5-5` with effort `medium`. Press **Test** to check
   the connection. For LM Studio, pick *OpenAI-compatible* and use the model id LM Studio shows.
2. **Projects → New project**, then **New prompt**: describe the task, pick the output format,
   optionally paste a JSON Schema, and write the prompt with `{{variables}}`.
3. **Test cases → Generate** (or **Add manually**).
4. **Scoring**: adjust the static check weights, and pick the judge engine and its criteria.
5. **New run**: choose the version and one or more engines. Results and scores fill in live.
6. **Report and suggestions**: read why each test scored as it did, then **Suggest
   improvements** → **Save as the next version** → run again → **Compare**.

### Choosing limits

Set an engine's limits a little below your account's limits (Claude Console → Settings →
Limits). Anthropic counts input and output tokens separately; the app counts them together,
so setting **Tokens / min** to about 80% of your *output* tokens/min limit is safe. If you
leave the limits blank, a 429 from the API still pauses that engine and retries after the
wait the API asks for.

Two engines that use the same Claude model share one Anthropic quota, but each has its own
limiter. Split the quota between them, or use one engine for both jobs.

For local models, set **Parallel requests** to what your LM Studio setup can serve (often 1)
and leave the per-minute limits blank.

## How scoring works

- **Static score (0–100)**: each enabled check has a weight; the weights add up to 100.
  Defaults depend on the output format. For JSON with a schema: parses 40, matches schema 40,
  no extra text 10, not cut off 10.
- **Judge score (0–100)**: the judge rates each criterion 0–10 with reasoning; criteria have
  weights that add up to 100. The model's output reaches the judge as data, so instructions
  hidden in an answer can't steer the grading.
- **Final score** = static share × static + (100 − static share) × judge. With no judge,
  the final score is the static score.

Runs keep a copy of the scoring settings they used. After changing settings, use **Rescore**
(static checks only, no model calls) or **Re-judge** on an existing run.

## Project layout

```
src/prompt_eval/
  main.py            app factory
  models.py          database tables
  db.py              SQLite setup and additive column migration
  engines/           Anthropic and OpenAI-compatible providers, rate limiting, retries
  generation.py      test case generation
  runner.py          background runs
  static_checks.py   rule-based checks
  judge.py           LLM judge and final score
  report.py          run reports and comparison
  suggestions.py     prompt improvement suggestions
  routes/            web routes
  templates/         Jinja2 pages (htmx for live updates)
tests/
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

Tests use fake engines and never call a real API.

The database schema only grows: new columns are added to an existing database at startup.
Removing or renaming a column needs a manual migration (or a fresh `data/prompt_eval.db`).
