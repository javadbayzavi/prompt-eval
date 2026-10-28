from typing import Annotated

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import select

from prompt_eval.models import OutputFormat, Project
from prompt_eval.web import SessionDep, get_or_404, redirect, require_text, templates

router = APIRouter()


@router.get("/projects", response_class=HTMLResponse)
def list_projects(request: Request, session: SessionDep) -> HTMLResponse:
    projects = session.exec(select(Project).order_by(Project.created_at.desc())).all()
    return templates.TemplateResponse(request, "projects/list.html", {"projects": projects})


@router.post("/projects")
def create_project(
    session: SessionDep,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
) -> RedirectResponse:
    project = Project(name=require_text(name, "Name"), description=description.strip())
    session.add(project)
    session.commit()
    return redirect(f"/projects/{project.id}")


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def show_project(request: Request, session: SessionDep, project_id: int) -> HTMLResponse:
    project = get_or_404(session, Project, project_id)
    return templates.TemplateResponse(
        request,
        "projects/detail.html",
        {"project": project, "output_formats": list(OutputFormat)},
    )


@router.post("/projects/{project_id}")
def update_project(
    session: SessionDep,
    project_id: int,
    name: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
) -> RedirectResponse:
    project = get_or_404(session, Project, project_id)
    project.name = require_text(name, "Name")
    project.description = description.strip()
    session.commit()
    return redirect(f"/projects/{project.id}")


@router.post("/projects/{project_id}/delete")
def delete_project(session: SessionDep, project_id: int) -> RedirectResponse:
    session.delete(get_or_404(session, Project, project_id))
    session.commit()
    return redirect("/projects")
