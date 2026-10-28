from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import select

from prompt_eval.models import (
    Engine,
    Prompt,
    PromptVersion,
    ResultStatus,
    Run,
    TestCase,
    TestCaseCategory,
)
from prompt_eval.runner import create_run
from prompt_eval.web import SessionDep, get_or_404, redirect, templates

router = APIRouter()


@dataclass
class EngineSummary:
    name: str
    model: str
    total: int = 0
    done: int = 0
    errors: int = 0
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def finished(self) -> int:
        return self.done + self.errors

    @property
    def avg_latency_ms(self) -> int:
        return self.latency_ms // self.done if self.done else 0


def summarize(run: Run) -> list[EngineSummary]:
    summaries: dict[str, EngineSummary] = {}
    for r in run.results:
        s = summaries.setdefault(r.engine_name, EngineSummary(name=r.engine_name, model=r.model))
        s.total += 1
        if r.status == ResultStatus.DONE:
            s.done += 1
            s.latency_ms += r.latency_ms
        elif r.status in (ResultStatus.ERROR, ResultStatus.CANCELLED):
            s.errors += 1
        s.input_tokens += r.input_tokens
        s.output_tokens += r.output_tokens
    return list(summaries.values())


def run_context(request: Request, run: Run) -> dict:
    total = len(run.results)
    finished = sum(
        r.status not in (ResultStatus.PENDING, ResultStatus.RUNNING) for r in run.results
    )
    return {
        "run": run,
        "summaries": summarize(run),
        "total": total,
        "finished": finished,
        "failed": sum(
            r.status in (ResultStatus.ERROR, ResultStatus.CANCELLED) for r in run.results
        ),
        "active": request.app.state.runs.is_active(run.id) or run.status.active,
    }


@router.post("/prompts/{prompt_id}/runs")
async def start_run(
    request: Request,
    session: SessionDep,
    prompt_id: int,
    version_id: Annotated[int, Form()],
    engine_ids: Annotated[list[int], Form()] = [],  # noqa: B006 (FastAPI form default)
    categories: Annotated[list[TestCaseCategory], Form()] = [],  # noqa: B006
) -> RedirectResponse:
    prompt = get_or_404(session, Prompt, prompt_id)
    version = session.get(PromptVersion, version_id)
    if version is None or version.prompt_id != prompt.id:
        raise HTTPException(status_code=400, detail="Pick a version of this prompt")
    engines = session.exec(select(Engine).where(Engine.id.in_(engine_ids))).all()
    if not engines:
        raise HTTPException(status_code=400, detail="Pick at least one engine")
    chosen = set(categories) or set(TestCaseCategory)
    test_cases = session.exec(
        select(TestCase)
        .where(TestCase.prompt_id == prompt.id, TestCase.category.in_(chosen))
        .order_by(TestCase.id)
    ).all()
    if not test_cases:
        raise HTTPException(status_code=400, detail="No test cases match")

    run = create_run(session, version, list(engines), list(test_cases))
    request.app.state.runs.start(run.id)
    return redirect(f"/runs/{run.id}")


@router.get("/runs/{run_id}", response_class=HTMLResponse)
def show_run(request: Request, session: SessionDep, run_id: int) -> HTMLResponse:
    run = get_or_404(session, Run, run_id)
    return templates.TemplateResponse(request, "runs/detail.html", run_context(request, run))


@router.get("/runs/{run_id}/progress", response_class=HTMLResponse)
def run_progress(request: Request, session: SessionDep, run_id: int) -> HTMLResponse:
    run = get_or_404(session, Run, run_id)
    return templates.TemplateResponse(request, "runs/_progress.html", run_context(request, run))


@router.post("/runs/{run_id}/cancel")
async def cancel_run(request: Request, session: SessionDep, run_id: int) -> RedirectResponse:
    get_or_404(session, Run, run_id)
    request.app.state.runs.cancel(run_id)
    return redirect(f"/runs/{run_id}")


@router.post("/runs/{run_id}/retry")
async def retry_run(request: Request, session: SessionDep, run_id: int) -> RedirectResponse:
    get_or_404(session, Run, run_id)
    request.app.state.runs.retry_failed(run_id)
    return redirect(f"/runs/{run_id}")


@router.post("/runs/{run_id}/delete")
def delete_run(request: Request, session: SessionDep, run_id: int) -> RedirectResponse:
    run = get_or_404(session, Run, run_id)
    if request.app.state.runs.is_active(run_id):
        raise HTTPException(status_code=409, detail="Cancel the run before deleting it")
    prompt_id = run.prompt_id
    session.delete(run)
    session.commit()
    return redirect(f"/prompts/{prompt_id}#runs")
