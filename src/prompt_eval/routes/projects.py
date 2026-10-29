from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import select

from prompt_eval import importing
from prompt_eval.models import Project
from prompt_eval.web import SessionDep, get_or_404, redirect, require_text, templates

router = APIRouter()


@router.get("/projects", response_class=HTMLResponse)
def list_projects(request: Request, session: SessionDep) -> HTMLResponse:
    projects = session.exec(select(Project).order_by(Project.name)).all()
    return templates.TemplateResponse(request, "projects/list.html", {"projects": projects})


@router.get("/projects/new", response_class=HTMLResponse)
def new_project(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "projects/new.html")


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


# Before /projects/{project_id}, so "import" is not read as a project id.
@router.get("/projects/import", response_class=HTMLResponse)
def import_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "projects/import.html", {"format": importing.FORMAT})


@router.post("/projects/import", response_model=None)
async def import_project(
    request: Request,
    session: SessionDep,
    file: Annotated[UploadFile | None, File()] = None,
    content: Annotated[str, Form()] = "",
) -> HTMLResponse | RedirectResponse:
    """Creates a project from a JSON file (or pasted JSON); problems are shown, nothing saved."""
    text = (await file.read()).decode("utf-8") if file and file.filename else content
    try:
        project = importing.import_project(session, importing.parse(text))
    except importing.InvalidImport as e:
        context = {"format": importing.FORMAT, "problems": e.problems, "content": text}
        return templates.TemplateResponse(request, "projects/import.html", context, status_code=400)
    return redirect(f"/projects/{project.id}")


@router.get("/projects/{project_id}", response_class=HTMLResponse)
def show_project(request: Request, session: SessionDep, project_id: int) -> HTMLResponse:
    project = get_or_404(session, Project, project_id)
    return templates.TemplateResponse(request, "projects/detail.html", {"project": project})


@router.get("/projects/{project_id}/settings", response_class=HTMLResponse)
def project_settings(request: Request, session: SessionDep, project_id: int) -> HTMLResponse:
    project = get_or_404(session, Project, project_id)
    return templates.TemplateResponse(request, "projects/settings.html", {"project": project})


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
