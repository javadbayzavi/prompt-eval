import json
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, ValidationError

from prompt_eval import generation
from prompt_eval.engines import EngineError
from prompt_eval.models import Engine, Prompt, TestCase, TestCaseCategory, TestCaseSource
from prompt_eval.web import SessionDep, get_or_404, redirect, templates

router = APIRouter()


class GenerateForm(BaseModel):
    engine_id: int
    count: int = 10
    mix_normal: float = 60
    mix_edge: float = 25
    mix_adversarial: float = 15
    guidance: str = ""


class SavedCase(BaseModel):
    """One previewed case, round-tripped through the preview form as JSON."""

    name: str
    category: TestCaseCategory
    inputs: dict[str, str]
    criteria: str = ""
    expected_output: str = ""


@router.post("/prompts/{prompt_id}/generate", response_class=HTMLResponse)
async def generate_test_cases(
    request: Request, session: SessionDep, prompt_id: int, form: Annotated[GenerateForm, Form()]
) -> HTMLResponse:
    """Generate a preview of test cases (htmx fragment). Nothing is saved yet."""
    prompt = get_or_404(session, Prompt, prompt_id)
    context: dict = {"prompt": prompt}

    def fail(message: str) -> HTMLResponse:
        context["error"] = message
        return templates.TemplateResponse(request, "prompts/_generated.html", context)

    engine = session.get(Engine, form.engine_id)
    if engine is None:
        return fail("Pick an engine to generate with.")
    if not 1 <= form.count <= generation.MAX_TEST_CASES:
        return fail(f"Count must be between 1 and {generation.MAX_TEST_CASES}.")
    if prompt.latest_version is None:
        return fail("The prompt has no version yet.")
    mix = {
        TestCaseCategory.NORMAL: form.mix_normal,
        TestCaseCategory.EDGE: form.mix_edge,
        TestCaseCategory.ADVERSARIAL: form.mix_adversarial,
    }
    if sum(max(0.0, w) for w in mix.values()) <= 0:
        return fail("At least one category needs a share above 0%.")

    try:
        client = await request.app.state.engines.get(engine)
        cases = await generation.generate(
            client, prompt, prompt.latest_version, form.count, mix, guidance=form.guidance
        )
    except EngineError as e:
        return fail(str(e))

    context["engine"] = engine
    context["cases"] = cases
    context["encoded"] = [
        json.dumps(
            {
                "name": c.name,
                "category": c.category.value,
                "inputs": c.inputs,
                "criteria": c.criteria,
                "expected_output": c.expected_output,
            },
            ensure_ascii=False,
        )
        for c in cases
    ]
    return templates.TemplateResponse(request, "prompts/_generated.html", context)


@router.post("/prompts/{prompt_id}/test-cases/bulk")
async def save_generated(request: Request, session: SessionDep, prompt_id: int) -> RedirectResponse:
    prompt = get_or_404(session, Prompt, prompt_id)
    form = await request.form()
    try:
        cases = [SavedCase.model_validate_json(str(raw)) for raw in form.getlist("case")]
    except ValidationError as e:
        raise HTTPException(status_code=400, detail="Invalid test case data") from e

    for case in cases:
        session.add(
            TestCase(
                prompt=prompt,
                name=case.name,
                category=case.category,
                inputs=case.inputs,
                criteria=case.criteria,
                expected_output=case.expected_output,
                source=TestCaseSource.GENERATED,
            )
        )
    session.commit()
    return redirect(f"/prompts/{prompt.id}/test-cases")
