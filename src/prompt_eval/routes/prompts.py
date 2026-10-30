"""Prompt pages. Each tab (prompt, test cases, scoring, runs, settings) is its own page."""

from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlmodel import Session, select

from prompt_eval import exporting
from prompt_eval.generation import DEFAULT_MIX, MAX_TEST_CASES
from prompt_eval.models import (
    Engine,
    OutputFormat,
    Project,
    Prompt,
    PromptVersion,
    TestCaseCategory,
)
from prompt_eval.routes.projects import download
from prompt_eval.routes.scoring import judge_context, rows_from_config, scoring_context
from prompt_eval.templating import find_variables
from prompt_eval.web import SessionDep, get_or_404, redirect, require_text, templates

router = APIRouter()


def all_engines(session: Session) -> list[Engine]:
    return list(session.exec(select(Engine).order_by(Engine.name)).all())


def latest_variables(prompt: Prompt) -> list[str]:
    latest = prompt.latest_version
    return find_variables(latest.system_prompt, latest.template) if latest else []


def render_tab(request: Request, prompt: Prompt, tab: str, template: str, **extra) -> HTMLResponse:
    context = {"prompt": prompt, "tab": tab, **extra}
    return templates.TemplateResponse(request, template, context)


@router.get("/projects/{project_id}/prompts/new", response_class=HTMLResponse)
def new_prompt(request: Request, session: SessionDep, project_id: int) -> HTMLResponse:
    project = get_or_404(session, Project, project_id)
    return templates.TemplateResponse(
        request, "prompts/new.html", {"project": project, "output_formats": list(OutputFormat)}
    )


@router.post("/projects/{project_id}/prompts")
def create_prompt(
    session: SessionDep,
    project_id: int,
    name: Annotated[str, Form()],
    template: Annotated[str, Form()],
    task_description: Annotated[str, Form()] = "",
    output_format: Annotated[OutputFormat, Form()] = OutputFormat.TEXT,
    output_schema: Annotated[str, Form()] = "",
    system_prompt: Annotated[str, Form()] = "",
) -> RedirectResponse:
    project = get_or_404(session, Project, project_id)
    prompt = Prompt(
        project=project,
        name=require_text(name, "Name"),
        task_description=task_description.strip(),
        output_format=output_format,
        output_schema=output_schema.strip(),
    )
    prompt.versions.append(
        PromptVersion(
            version=1,
            system_prompt=system_prompt.strip(),
            template=require_text(template, "Template"),
        )
    )
    session.add(prompt)
    session.commit()
    return redirect(f"/prompts/{prompt.id}")


@router.get("/prompts/{prompt_id}", response_class=HTMLResponse)
def show_prompt(request: Request, session: SessionDep, prompt_id: int) -> HTMLResponse:
    prompt = get_or_404(session, Prompt, prompt_id)
    return render_tab(
        request,
        prompt,
        "prompt",
        "prompts/prompt.html",
        latest=prompt.latest_version,
        variables=latest_variables(prompt),
    )


@router.get("/prompts/{prompt_id}/test-cases", response_class=HTMLResponse)
def prompt_test_cases(request: Request, session: SessionDep, prompt_id: int) -> HTMLResponse:
    prompt = get_or_404(session, Prompt, prompt_id)
    return render_tab(request, prompt, "test-cases", "prompts/test_cases.html")


@router.get("/prompts/{prompt_id}/test-cases/new", response_class=HTMLResponse)
def new_test_case(request: Request, session: SessionDep, prompt_id: int) -> HTMLResponse:
    prompt = get_or_404(session, Prompt, prompt_id)
    return render_tab(
        request,
        prompt,
        "test-cases",
        "test_cases/new.html",
        variables=latest_variables(prompt),
        categories=list(TestCaseCategory),
    )


@router.get("/prompts/{prompt_id}/test-cases/generate", response_class=HTMLResponse)
def generate_page(request: Request, session: SessionDep, prompt_id: int) -> HTMLResponse:
    prompt = get_or_404(session, Prompt, prompt_id)
    return render_tab(
        request,
        prompt,
        "test-cases",
        "prompts/generate.html",
        engines=all_engines(session),
        default_mix=DEFAULT_MIX,
        max_test_cases=MAX_TEST_CASES,
        variables=latest_variables(prompt),
    )


@router.get("/prompts/{prompt_id}/scoring", response_class=HTMLResponse)
def prompt_scoring(request: Request, session: SessionDep, prompt_id: int) -> HTMLResponse:
    prompt = get_or_404(session, Prompt, prompt_id)
    # Both helpers also return "prompt", which render_tab sets itself.
    context = scoring_context(prompt, rows_from_config(prompt)) | judge_context(session, prompt)
    context.pop("prompt")
    return render_tab(request, prompt, "scoring", "prompts/scoring.html", **context)


@router.get("/prompts/{prompt_id}/runs", response_class=HTMLResponse)
def prompt_runs(request: Request, session: SessionDep, prompt_id: int) -> HTMLResponse:
    prompt = get_or_404(session, Prompt, prompt_id)
    return render_tab(request, prompt, "runs", "prompts/runs.html")


@router.get("/prompts/{prompt_id}/runs/new", response_class=HTMLResponse)
def new_run(request: Request, session: SessionDep, prompt_id: int) -> HTMLResponse:
    prompt = get_or_404(session, Prompt, prompt_id)
    counts = {cat: sum(tc.category == cat for tc in prompt.test_cases) for cat in TestCaseCategory}
    return render_tab(
        request,
        prompt,
        "runs",
        "runs/new.html",
        engines=all_engines(session),
        category_counts=counts,
    )


@router.get("/prompts/{prompt_id}/settings", response_class=HTMLResponse)
def prompt_settings(
    request: Request, session: SessionDep, prompt_id: int, saved: bool = False
) -> HTMLResponse:
    prompt = get_or_404(session, Prompt, prompt_id)
    return render_tab(
        request,
        prompt,
        "settings",
        "prompts/settings.html",
        output_formats=list(OutputFormat),
        saved=saved,
    )


@router.post("/prompts/{prompt_id}")
def update_prompt(
    session: SessionDep,
    prompt_id: int,
    name: Annotated[str, Form()],
    task_description: Annotated[str, Form()] = "",
    output_format: Annotated[OutputFormat, Form()] = OutputFormat.TEXT,
    output_schema: Annotated[str, Form()] = "",
) -> RedirectResponse:
    prompt = get_or_404(session, Prompt, prompt_id)
    prompt.name = require_text(name, "Name")
    prompt.task_description = task_description.strip()
    prompt.output_format = output_format
    prompt.output_schema = output_schema.strip()
    session.commit()
    return redirect(f"/prompts/{prompt.id}/settings?saved=1")


@router.get("/prompts/{prompt_id}/export")
def export_prompt(session: SessionDep, prompt_id: int) -> JSONResponse:
    prompt = get_or_404(session, Prompt, prompt_id)
    return download(exporting.prompt_file(session, prompt), prompt.name)


@router.post("/prompts/{prompt_id}/delete")
def delete_prompt(session: SessionDep, prompt_id: int) -> RedirectResponse:
    prompt = get_or_404(session, Prompt, prompt_id)
    project_id = prompt.project_id
    session.delete(prompt)
    session.commit()
    return redirect(f"/projects/{project_id}")


@router.post("/prompts/{prompt_id}/versions")
def create_version(
    session: SessionDep,
    prompt_id: int,
    template: Annotated[str, Form()],
    system_prompt: Annotated[str, Form()] = "",
    notes: Annotated[str, Form()] = "",
) -> RedirectResponse:
    prompt = get_or_404(session, Prompt, prompt_id)
    latest = prompt.latest_version
    prompt.versions.append(
        PromptVersion(
            version=(latest.version + 1) if latest else 1,
            system_prompt=system_prompt.strip(),
            template=require_text(template, "Template"),
            notes=notes.strip(),
        )
    )
    session.commit()
    return redirect(f"/prompts/{prompt.id}")
