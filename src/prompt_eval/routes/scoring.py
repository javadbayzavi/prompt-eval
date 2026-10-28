from dataclasses import asdict, dataclass

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import select
from starlette.datastructures import FormData

from prompt_eval import judge, static_checks
from prompt_eval.models import Engine, Prompt, Run
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


MAX_CRITERIA = 8


@dataclass
class CriterionRow:
    name: str = ""
    description: str = ""
    weight: str = ""


def judge_context(session, prompt: Prompt, rows: list[CriterionRow] | None = None, **extra) -> dict:
    if rows is None:
        rows = [
            CriterionRow(c.name, c.description, f"{c.weight:g}")
            for c in judge.criteria_from_json(prompt.judge_criteria)
        ]
    rows = rows + [CriterionRow() for _ in range(MAX_CRITERIA - len(rows))]
    return {
        "prompt": prompt,
        "criterion_rows": rows,
        "judge_engines": session.exec(select(Engine).order_by(Engine.name)).all(),
        "criteria_default": prompt.judge_criteria is None,
        **extra,
    }


@router.post("/prompts/{prompt_id}/judge", response_class=HTMLResponse)
async def save_judge(request: Request, session: SessionDep, prompt_id: int) -> HTMLResponse:
    """Save judge engine, criteria and the static/judge split (htmx fragment)."""
    prompt = get_or_404(session, Prompt, prompt_id)
    form = await request.form()

    def render(**extra) -> HTMLResponse:
        context = judge_context(session, prompt, **extra)
        return templates.TemplateResponse(request, "prompts/_judge.html", context)

    if form.get("reset"):
        prompt.judge_criteria = None
        session.commit()
        return render(saved="Criteria reset to defaults.")

    rows = [
        CriterionRow(
            name=str(form.get(f"crit_name_{i}", "")).strip(),
            description=str(form.get(f"crit_desc_{i}", "")).strip(),
            weight=str(form.get(f"crit_weight_{i}", "")).strip(),
        )
        for i in range(MAX_CRITERIA)
    ]
    rows = [r for r in rows if r.name or r.description or r.weight]
    problems = []
    criteria = []
    for row in rows:
        if not row.name:
            problems.append("Every criterion needs a name")
            continue
        try:
            criteria.append(judge.Criterion(row.name, row.description, float(row.weight)))
        except ValueError:
            problems.append(f"{row.name}: weight must be a number")
    if not problems:
        problems = judge.validate_criteria(criteria)

    engine_id = str(form.get("judge_engine_id", "")).strip()
    judge_engine = session.get(Engine, int(engine_id)) if engine_id.isdigit() else None
    if engine_id and judge_engine is None:
        problems.append("Pick an existing judge engine")
    try:
        static_weight = float(str(form.get("static_weight", "")))
        if not 0 <= static_weight <= 100:
            raise ValueError
    except ValueError:
        problems.append("Static share must be a number from 0 to 100")
        static_weight = prompt.static_weight

    if problems:
        return render(rows=rows, problems=problems)

    prompt.judge_engine_id = judge_engine.id if judge_engine else None
    prompt.static_weight = static_weight
    prompt.judge_criteria = [asdict(c) for c in criteria]
    prompt.judge_instructions = str(form.get("judge_instructions", "")).strip()
    session.commit()
    return render(saved="Saved.")


@router.post("/runs/{run_id}/rejudge")
async def rejudge_run(request: Request, session: SessionDep, run_id: int) -> RedirectResponse:
    run = get_or_404(session, Run, run_id)
    if run.prompt.judge_engine_id is None:
        raise HTTPException(status_code=400, detail="Set a judge engine on the prompt first")
    request.app.state.runs.start_rejudge(run_id)
    return redirect(f"/runs/{run_id}")
