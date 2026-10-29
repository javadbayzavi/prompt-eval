from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from sqlmodel import select

from prompt_eval import suggestions
from prompt_eval.engines import EngineError
from prompt_eval.models import Engine, PromptVersion, Run
from prompt_eval.report import build_report, compare
from prompt_eval.templating import find_variables
from prompt_eval.web import SessionDep, get_or_404, redirect, templates

router = APIRouter()


def report_context(session, run: Run, **extra) -> dict:
    others = [r for r in run.prompt.runs if r.id != run.id]
    return {
        "report": build_report(run),
        "run": run,
        "engines": session.exec(select(Engine).order_by(Engine.name)).all(),
        "other_runs": others,
        **extra,
    }


@router.get("/runs/{run_id}/report", response_class=HTMLResponse)
def show_report(request: Request, session: SessionDep, run_id: int) -> HTMLResponse:
    run = get_or_404(session, Run, run_id)
    return templates.TemplateResponse(request, "runs/report.html", report_context(session, run))


@router.get("/runs/{run_id}/report.md")
def download_report(request: Request, session: SessionDep, run_id: int) -> Response:
    run = get_or_404(session, Run, run_id)
    body = templates.get_template("runs/report.md").render(report=build_report(run), run=run)
    return Response(
        body,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="run-{run.id}-report.md"'},
    )


@router.post("/runs/{run_id}/suggestions", response_class=HTMLResponse)
async def generate_suggestions(
    request: Request, session: SessionDep, run_id: int, engine_id: Annotated[int, Form()]
) -> HTMLResponse:
    """Ask a model for prompt improvements (htmx fragment); the result is saved on the run."""
    run = get_or_404(session, Run, run_id)
    engine = session.get(Engine, engine_id)
    context: dict = {"run": run}
    if engine is None:
        context["error"] = "Pick an engine."
        return templates.TemplateResponse(request, "runs/_suggestions.html", context)

    report = build_report(run)
    variables = find_variables(run.version.system_prompt, run.version.template)
    try:
        client = await request.app.state.engines.get(engine)
        response = await client.complete(suggestions.build_request(report))
        if response.refused:
            raise EngineError("The model refused the request")
        if response.stop_reason in ("max_tokens", "length"):
            raise EngineError("The answer hit the engine's max tokens; raise it and try again")
        run.suggestions = suggestions.parse(response.text, variables)
        run.suggestions_engine = engine.name
        session.commit()
    except EngineError as e:
        context["error"] = str(e)
    return templates.TemplateResponse(request, "runs/_suggestions.html", context)


@router.post("/runs/{run_id}/suggestions/apply")
def apply_suggestions(session: SessionDep, run_id: int) -> RedirectResponse:
    """Save the suggested prompt as the prompt's next version."""
    run = get_or_404(session, Run, run_id)
    if not run.suggestions:
        raise HTTPException(status_code=400, detail="Generate suggestions first")
    prompt = run.prompt
    latest = prompt.latest_version
    prompt.versions.append(
        PromptVersion(
            version=latest.version + 1 if latest else 1,
            system_prompt=run.suggestions["revised_system_prompt"],
            template=run.suggestions["revised_template"],
            notes=f"Suggested from run #{run.id} (v{run.version.version})",
        )
    )
    session.commit()
    return redirect(f"/prompts/{prompt.id}")


@router.get("/runs/{run_id}/compare", response_class=HTMLResponse)
def compare_runs(request: Request, session: SessionDep, run_id: int, other: int) -> HTMLResponse:
    run = get_or_404(session, Run, run_id)
    other_run = get_or_404(session, Run, other)
    if other_run.prompt_id != run.prompt_id:
        raise HTTPException(status_code=400, detail="Both runs must be of the same prompt")
    # Show the older run as "before".
    before, after = sorted([run, other_run], key=lambda r: r.id)
    engine_rows, rows = compare(before, after)
    return templates.TemplateResponse(
        request,
        "runs/compare.html",
        {"before": before, "after": after, "engine_rows": engine_rows, "rows": rows},
    )
