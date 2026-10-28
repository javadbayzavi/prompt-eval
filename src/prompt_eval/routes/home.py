from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from prompt_eval.web import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
