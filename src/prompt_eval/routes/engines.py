from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from prompt_eval.engines import ChatRequest, EngineError
from prompt_eval.models import EFFORT_LEVELS, Engine, EngineProvider, utcnow
from prompt_eval.web import SessionDep, get_or_404, redirect, require_text, templates

router = APIRouter()

MODEL_SUGGESTIONS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
PING_PROMPT = "Reply with exactly one word: pong"


class EngineForm(BaseModel):
    # Numbers arrive as strings so blank fields can mean "not set".
    name: str
    provider: EngineProvider
    model: str
    base_url: str = ""
    max_tokens: str = ""
    effort: str = ""
    temperature: str = ""
    max_concurrency: str = ""
    requests_per_minute: str = ""
    tokens_per_minute: str = ""
    max_retries: str = ""
    timeout_seconds: str = ""


def optional_int(value: str, field: str, *, minimum: int = 1) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        number = int(value)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{field} must be a whole number") from None
    if number < minimum:
        raise HTTPException(status_code=400, detail=f"{field} must be at least {minimum}")
    return number


def optional_float(value: str, field: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{field} must be a number") from None


def apply_form(session: Session, engine: Engine, form: EngineForm) -> None:
    name = require_text(form.name, "Name")
    clash = session.exec(select(Engine).where(Engine.name == name, Engine.id != engine.id)).first()
    if clash:
        raise HTTPException(status_code=400, detail=f"An engine named {name!r} already exists")
    if form.effort and form.effort not in EFFORT_LEVELS:
        raise HTTPException(status_code=400, detail=f"Effort must be one of {EFFORT_LEVELS}")

    engine.name = name
    engine.provider = form.provider
    engine.model = require_text(form.model, "Model")
    engine.base_url = form.base_url.strip()
    engine.max_tokens = optional_int(form.max_tokens, "Max tokens") or 8192
    engine.effort = form.effort
    engine.temperature = optional_float(form.temperature, "Temperature")
    engine.max_concurrency = optional_int(form.max_concurrency, "Max concurrency") or 1
    engine.requests_per_minute = optional_int(form.requests_per_minute, "Requests per minute")
    engine.tokens_per_minute = optional_int(form.tokens_per_minute, "Tokens per minute")
    retries = optional_int(form.max_retries, "Max retries", minimum=0)
    engine.max_retries = 5 if retries is None else retries
    engine.timeout_seconds = optional_float(form.timeout_seconds, "Timeout") or 600.0
    engine.updated_at = utcnow()


def form_context(**extra) -> dict:
    return {
        "providers": list(EngineProvider),
        "effort_levels": EFFORT_LEVELS,
        "model_suggestions": MODEL_SUGGESTIONS,
        **extra,
    }


@router.get("/engines", response_class=HTMLResponse)
def list_engines(request: Request, session: SessionDep) -> HTMLResponse:
    engines = session.exec(select(Engine).order_by(Engine.name)).all()
    return templates.TemplateResponse(
        request, "engines/list.html", form_context(engines=engines, engine=None)
    )


@router.post("/engines")
def create_engine(session: SessionDep, form: Annotated[EngineForm, Form()]) -> RedirectResponse:
    engine = Engine(name="", provider=form.provider, model="")
    apply_form(session, engine, form)
    session.add(engine)
    session.commit()
    return redirect("/engines")


@router.get("/engines/{engine_id}/edit", response_class=HTMLResponse)
def edit_engine(request: Request, session: SessionDep, engine_id: int) -> HTMLResponse:
    engine = get_or_404(session, Engine, engine_id)
    return templates.TemplateResponse(request, "engines/edit.html", form_context(engine=engine))


@router.post("/engines/{engine_id}")
def update_engine(
    session: SessionDep, engine_id: int, form: Annotated[EngineForm, Form()]
) -> RedirectResponse:
    engine = get_or_404(session, Engine, engine_id)
    apply_form(session, engine, form)
    session.commit()
    return redirect("/engines")


@router.post("/engines/{engine_id}/delete")
async def delete_engine(request: Request, session: SessionDep, engine_id: int) -> RedirectResponse:
    session.delete(get_or_404(session, Engine, engine_id))
    session.commit()
    await request.app.state.engines.forget(engine_id)
    return redirect("/engines")


@router.post("/engines/{engine_id}/test", response_class=HTMLResponse)
async def test_engine(request: Request, session: SessionDep, engine_id: int) -> HTMLResponse:
    """Send one tiny request and render the outcome (an htmx fragment)."""
    engine = get_or_404(session, Engine, engine_id)
    context: dict = {"engine": engine}
    try:
        client = await request.app.state.engines.get(engine)
        context["response"] = await client.complete(ChatRequest(user=PING_PROMPT, max_tokens=1024))
    except EngineError as e:
        context["error"] = str(e)
    return templates.TemplateResponse(request, "engines/_test_result.html", context)
