from dataclasses import dataclass

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.datastructures import FormData

from prompt_eval import static_checks
from prompt_eval.models import Prompt, Run
from prompt_eval.runner import rescore
from prompt_eval.static_checks import CheckConfig, CheckSpec
from prompt_eval.web import SessionDep, get_or_404, redirect, templates

router = APIRouter()


@dataclass
class CheckRow:
    name: str
    spec: CheckSpec
    enabled: bool
    weight: str
    param: str


def rows_from_config(prompt: Prompt) -> list[CheckRow]:
    current = {c.name: c for c in static_checks.prompt_config(prompt)}
    return [
        CheckRow(
            name=name,
            spec=spec,
            enabled=name in current,
            weight=f"{current[name].weight:g}" if name in current else "",
            param=current[name].param if name in current else "",
        )
        for name, spec in static_checks.checks_for(prompt.output_format).items()
    ]


def rows_from_form(prompt: Prompt, form: FormData) -> list[CheckRow]:
    return [
        CheckRow(
            name=name,
            spec=spec,
            enabled=form.get(f"enabled__{name}") == "on",
            weight=str(form.get(f"weight__{name}", "")).strip(),
            param=str(form.get(f"param__{name}", "")).strip(),
        )
        for name, spec in static_checks.checks_for(prompt.output_format).items()
    ]


def scoring_context(prompt: Prompt, rows: list[CheckRow], **extra) -> dict:
    return {
        "prompt": prompt,
        "rows": rows,
        "is_default": prompt.static_checks is None,
        **extra,
    }


@router.post("/prompts/{prompt_id}/scoring", response_class=HTMLResponse)
async def save_scoring(request: Request, session: SessionDep, prompt_id: int) -> HTMLResponse:
    """Save the static checks (htmx fragment); invalid settings are shown, not saved."""
    prompt = get_or_404(session, Prompt, prompt_id)
    form = await request.form()

    if form.get("reset"):
        prompt.static_checks = None
        session.commit()
        context = scoring_context(prompt, rows_from_config(prompt), saved="Reset to defaults.")
        return templates.TemplateResponse(request, "prompts/_scoring.html", context)

    rows = rows_from_form(prompt, form)
    configs, problems = [], []
    for row in rows:
        if not row.enabled:
            continue
        try:
            weight = float(row.weight)
        except ValueError:
            problems.append(f"{row.spec.label}: weight must be a number")
            continue
        configs.append(CheckConfig(row.name, weight, row.param))
    problems += static_checks.validate_config(configs, prompt.output_format) if not problems else []
    if any(c.name == "schema" for c in configs) and not static_checks.has_json_schema(
        prompt.output_schema
    ):
        problems.append("Matches schema: set a JSON Schema in the prompt settings first")

    if problems:
        context = scoring_context(prompt, rows, problems=problems)
        return templates.TemplateResponse(request, "prompts/_scoring.html", context)

    prompt.static_checks = static_checks.configs_to_json(configs)
    session.commit()
    context = scoring_context(prompt, rows_from_config(prompt), saved="Saved.")
    return templates.TemplateResponse(request, "prompts/_scoring.html", context)


@router.post("/runs/{run_id}/rescore")
def rescore_run(request: Request, session: SessionDep, run_id: int) -> RedirectResponse:
    run = get_or_404(session, Run, run_id)
    if not request.app.state.runs.is_active(run_id):
        rescore(session, run)
    return redirect(f"/runs/{run_id}")
