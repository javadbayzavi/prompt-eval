from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import select

from prompt_eval.generation import DEFAULT_MIX, MAX_TEST_CASES
from prompt_eval.models import (
    Engine,
    OutputFormat,
    Project,
    Prompt,
    PromptVersion,
    TestCaseCategory,
)
from prompt_eval.routes.scoring import rows_from_config, scoring_context
from prompt_eval.templating import find_variables
from prompt_eval.web import SessionDep, get_or_404, redirect, require_text, templates

router = APIRouter()


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
    latest = prompt.latest_version
    variables = find_variables(latest.system_prompt, latest.template) if latest else []
    return templates.TemplateResponse(
        request,
        "prompts/detail.html",
        {
            "prompt": prompt,
            "latest": latest,
            "variables": variables,
            "output_formats": list(OutputFormat),
            "categories": list(TestCaseCategory),
            "engines": session.exec(select(Engine).order_by(Engine.name)).all(),
            "default_mix": DEFAULT_MIX,
            "max_test_cases": MAX_TEST_CASES,
            **scoring_context(prompt, rows_from_config(prompt)),
        },
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
    return redirect(f"/prompts/{prompt.id}")


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
