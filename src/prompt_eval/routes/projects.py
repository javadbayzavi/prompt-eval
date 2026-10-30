from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlmodel import select

from prompt_eval import exporting, importing
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
    return _import_page(request, None)


@router.post("/projects/import", response_model=None)
async def import_project(
    request: Request,
    session: SessionDep,
    file: Annotated[UploadFile | None, File()] = None,
    content: Annotated[str, Form()] = "",
) -> HTMLResponse | RedirectResponse:
    """Creates a project from a JSON file (or pasted JSON); problems are shown, nothing saved."""
    text = await _uploaded_text(file, content)
    try:
        project = importing.import_project(session, importing.parse(text))
    except importing.InvalidImport as e:
        return _import_page(request, None, problems=e.problems, content=text)
    return redirect(f"/projects/{project.id}")


@router.get("/projects/{project_id}/import", response_class=HTMLResponse)
def import_into_page(request: Request, session: SessionDep, project_id: int) -> HTMLResponse:
    return _import_page(request, get_or_404(session, Project, project_id))


@router.post("/projects/{project_id}/import", response_model=None)
async def import_into(
    request: Request,
    session: SessionDep,
    project_id: int,
    file: Annotated[UploadFile | None, File()] = None,
    content: Annotated[str, Form()] = "",
) -> HTMLResponse | RedirectResponse:
    """Adds the file's prompts to this project; problems are shown, nothing saved."""
    project = get_or_404(session, Project, project_id)
    text = await _uploaded_text(file, content)
    try:
        prompts = importing.import_into(session, project, importing.parse(text))
    except importing.InvalidImport as e:
        return _import_page(request, project, problems=e.problems, content=text)
    return redirect(f"/prompts/{prompts[0].id}" if len(prompts) == 1 else f"/projects/{project.id}")


@router.get("/projects/{project_id}/export")
def export_project(session: SessionDep, project_id: int) -> JSONResponse:
    project = get_or_404(session, Project, project_id)
    return download(exporting.project_file(session, project), project.name)


def download(data: dict, name: str) -> JSONResponse:
    headers = {"Content-Disposition": f'attachment; filename="{exporting.file_name(name)}"'}
    return JSONResponse(data, headers=headers)


async def _uploaded_text(file: UploadFile | None, content: str) -> str:
    return (await file.read()).decode("utf-8") if file and file.filename else content


def _import_page(request: Request, project: Project | None, **extra) -> HTMLResponse:
    context = {"format": importing.FORMAT, "project": project, **extra}
    status = 400 if extra.get("problems") else 200
    return templates.TemplateResponse(request, "projects/import.html", context, status_code=status)


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
