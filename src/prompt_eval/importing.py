"""Import a whole project from one JSON file: prompts, their versions, scoring settings and
test cases. Every part is checked the way the UI checks it, and nothing is saved unless all of
it is valid.

The file looks like this (only `project.name`, a prompt's `name` and one version's `template`
are required):

    {
      "format": "prompt-eval/1",
      "project": {"name": "...", "description": "..."},
      "prompts": [{
        "name": "...", "task_description": "...", "output_format": "json",
        "output_schema": {...},
        "static_checks": [{"name": "parse", "weight": 40}, ...],
        "static_weight": 30,
        "judge_engine": "<an existing engine's name>",
        "judge_criteria": [{"name": "...", "description": "...", "weight": 100}],
        "judge_instructions": "...",
        "versions": [{"system_prompt": "", "template": "... {{var}} ...", "notes": ""}],
        "test_cases": [{"name": "...", "category": "normal", "inputs": {"var": "..."},
                        "criteria": "...", "expected_output": "..."}]
      }]
    }
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlmodel import Session, select

from prompt_eval import judge, static_checks
from prompt_eval.models import (
    Engine,
    OutputFormat,
    Project,
    Prompt,
    PromptVersion,
    TestCase,
    TestCaseCategory,
    TestCaseSource,
)
from prompt_eval.static_checks import CheckConfig
from prompt_eval.templating import find_variables

FORMAT = "prompt-eval/1"


class InvalidImport(Exception):
    """The file cannot be imported; `problems` says why, one line each."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


class CheckIn(BaseModel):
    name: str
    weight: float
    param: str = ""


class CriterionIn(BaseModel):
    name: str
    description: str = ""
    weight: float


class VersionIn(BaseModel):
    system_prompt: str = ""
    template: str = Field(min_length=1)
    notes: str = ""


class TestCaseIn(BaseModel):
    name: str = Field(min_length=1)
    category: TestCaseCategory = TestCaseCategory.NORMAL
    inputs: dict[str, str] = Field(default_factory=dict)
    criteria: str = ""
    expected_output: str = ""


class PromptIn(BaseModel):
    name: str = Field(min_length=1)
    task_description: str = ""
    output_format: OutputFormat = OutputFormat.TEXT
    output_schema: str = ""
    static_checks: list[CheckIn] | None = None
    static_weight: float = Field(default=30.0, ge=0, le=100)
    judge_engine: str | None = None
    judge_criteria: list[CriterionIn] | None = None
    judge_instructions: str = ""
    versions: list[VersionIn] = Field(min_length=1)
    test_cases: list[TestCaseIn] = Field(default_factory=list)

    @field_validator("output_schema", mode="before")
    @classmethod
    def schema_as_text(cls, value: Any) -> Any:
        # A schema may be given as an object; it is stored as the text the UI edits.
        return json.dumps(value, indent=2, ensure_ascii=False) if isinstance(value, dict) else value


class ProjectIn(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""


class ImportFile(BaseModel):
    format: Literal["prompt-eval/1"] = FORMAT
    project: ProjectIn
    prompts: list[PromptIn] = Field(default_factory=list)


def parse(text: str) -> ImportFile:
    try:
        return ImportFile.model_validate_json(text)
    except ValidationError as e:
        raise InvalidImport([_describe(error) for error in e.errors()]) from e


def import_project(session: Session, data: ImportFile) -> Project:
    """Adds the project with everything in it, or raises `InvalidImport` and adds nothing."""
    engines = {engine.name: engine for engine in session.exec(select(Engine)).all()}
    problems = [problem for prompt in data.prompts for problem in _problems(prompt, engines)]
    if problems:
        raise InvalidImport(problems)

    project = Project(name=data.project.name.strip(), description=data.project.description.strip())
    for prompt_in in data.prompts:
        project.prompts.append(_prompt(prompt_in, engines))
    session.add(project)
    session.commit()
    return project


def _prompt(data: PromptIn, engines: dict[str, Engine]) -> Prompt:
    judge_engine = engines.get(data.judge_engine) if data.judge_engine else None
    prompt = Prompt(
        name=data.name.strip(),
        task_description=data.task_description.strip(),
        output_format=data.output_format,
        output_schema=data.output_schema.strip(),
        static_checks=[c.model_dump() for c in data.static_checks] if data.static_checks else None,
        static_weight=data.static_weight,
        judge_engine_id=judge_engine.id if judge_engine else None,
        judge_criteria=[c.model_dump() for c in data.judge_criteria]
        if data.judge_criteria
        else None,
        judge_instructions=data.judge_instructions.strip(),
    )
    for number, version in enumerate(data.versions, start=1):
        prompt.versions.append(
            PromptVersion(
                version=number,
                system_prompt=version.system_prompt.strip(),
                template=version.template,
                notes=version.notes.strip(),
            )
        )
    for case in data.test_cases:
        prompt.test_cases.append(
            TestCase(
                name=case.name.strip(),
                category=case.category,
                inputs=case.inputs,
                criteria=case.criteria.strip(),
                expected_output=case.expected_output.strip(),
                source=TestCaseSource.MANUAL,
            )
        )
    return prompt


def _problems(data: PromptIn, engines: dict[str, Engine]) -> list[str]:
    """The same checks the scoring, judge and prompt pages apply, prefixed with the prompt."""
    problems = []
    if data.static_checks:
        configs = [CheckConfig(c.name, c.weight, c.param) for c in data.static_checks]
        problems += static_checks.validate_config(configs, data.output_format)
        uses_schema = any(c.name == "schema" for c in configs)
        if uses_schema and not static_checks.has_json_schema(data.output_schema):
            problems.append("Matches schema: the output_schema is not a JSON Schema object")
    if data.judge_criteria:
        criteria = [judge.Criterion(c.name, c.description, c.weight) for c in data.judge_criteria]
        problems += judge.validate_criteria(criteria)
    if data.judge_engine and data.judge_engine not in engines:
        problems.append(f"No engine named {data.judge_engine!r}: add it on the Engines page first")
    latest = data.versions[-1]
    variables = find_variables(latest.system_prompt, latest.template)
    for case in data.test_cases:
        missing = [name for name in variables if name not in case.inputs]
        if missing:
            problems.append(f"Test case {case.name!r} has no value for {', '.join(missing)}")
    return [f"{data.name}: {problem}" for problem in problems]


def _describe(error: Any) -> str:
    where = ".".join(str(part) for part in error["loc"]) or "file"
    return f"{where}: {error['msg']}"
