"""Shared web helpers: template rendering and request-scoped dependencies."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import selectinload
from sqlmodel import Session, SQLModel, select

from prompt_eval.db import session_scope
from prompt_eval.models import Project

BASE_DIR = Path(__file__).parent


@dataclass
class NavPrompt:
    id: int
    name: str


@dataclass
class NavProject:
    id: int
    name: str
    prompts: list[NavPrompt]


def sidebar_context(request: Request) -> dict:
    """Projects and their prompts for the sidebar on every page."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return {"nav_projects": []}
    with Session(engine) as session:
        projects = session.exec(
            select(Project).options(selectinload(Project.prompts)).order_by(Project.name)
        ).all()
        nav = [
            NavProject(p.id, p.name, [NavPrompt(pr.id, pr.name) for pr in p.prompts])
            for p in projects
        ]
    return {"nav_projects": nav}


templates = Jinja2Templates(directory=BASE_DIR / "templates", context_processors=[sidebar_context])


def get_session(request: Request) -> Iterator[Session]:
    yield from session_scope(request.app.state.engine)


SessionDep = Annotated[Session, Depends(get_session)]


def get_or_404[ModelT: SQLModel](session: Session, model: type[ModelT], id: int) -> ModelT:
    obj = session.get(model, id)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"{model.__name__} {id} not found")
    return obj


def redirect(url: str) -> RedirectResponse:
    # 303 so the browser follows a POST with a GET.
    return RedirectResponse(url, status_code=303)


def require_text(value: str, field: str) -> str:
    value = value.strip()
    if not value:
        raise HTTPException(status_code=400, detail=f"{field} is required")
    return value
