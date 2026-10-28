"""Database tables.

Prompt versions are immutable: editing a prompt's text creates a new version, so every run
can point at the exact text it was evaluated with.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship, SQLModel


def utcnow() -> datetime:
    return datetime.now(UTC)


class OutputFormat(StrEnum):
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    MARKDOWN = "markdown"
    TEXT = "text"


class TestCaseCategory(StrEnum):
    NORMAL = "normal"
    EDGE = "edge"
    ADVERSARIAL = "adversarial"


class TestCaseSource(StrEnum):
    MANUAL = "manual"
    GENERATED = "generated"


class EngineProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"  # LM Studio, Ollama, vLLM, ...


EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


class Engine(SQLModel, table=True):
    """A model endpoint plus its limits. API keys stay in `.env`, never in the database."""

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    provider: EngineProvider
    model: str
    # openai_compatible only; blank uses LMSTUDIO_BASE_URL from settings.
    base_url: str = ""
    max_tokens: int = 8192
    # anthropic only (the API no longer accepts temperature).
    effort: str = ""
    # openai_compatible only.
    temperature: float | None = None
    max_concurrency: int = 2
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    max_retries: int = 5
    timeout_seconds: float = 600.0
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Project(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: str = ""
    created_at: datetime = Field(default_factory=utcnow)

    prompts: list["Prompt"] = Relationship(back_populates="project", cascade_delete=True)


class Prompt(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True, ondelete="CASCADE")
    name: str
    # What the prompt should make the model do. Used to generate test cases and by the judge.
    task_description: str = ""
    output_format: OutputFormat = OutputFormat.TEXT
    # Optional JSON Schema (for JSON/YAML) or a free-form description of the expected shape.
    output_schema: str = ""
    created_at: datetime = Field(default_factory=utcnow)

    project: Project = Relationship(back_populates="prompts")
    versions: list["PromptVersion"] = Relationship(
        back_populates="prompt",
        cascade_delete=True,
        sa_relationship_kwargs={"order_by": "PromptVersion.version"},
    )
    test_cases: list["TestCase"] = Relationship(
        back_populates="prompt",
        cascade_delete=True,
        sa_relationship_kwargs={"order_by": "TestCase.id"},
    )

    @property
    def latest_version(self) -> "PromptVersion | None":
        return self.versions[-1] if self.versions else None


class PromptVersion(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    prompt_id: int = Field(foreign_key="prompt.id", index=True, ondelete="CASCADE")
    version: int
    system_prompt: str = ""
    # User message template; `{{name}}` placeholders are filled from a test case's inputs.
    template: str
    notes: str = ""
    created_at: datetime = Field(default_factory=utcnow)

    prompt: Prompt = Relationship(back_populates="versions")


class TestCase(SQLModel, table=True):
    __test__ = False  # not a pytest test class

    id: int | None = Field(default=None, primary_key=True)
    prompt_id: int = Field(foreign_key="prompt.id", index=True, ondelete="CASCADE")
    name: str
    inputs: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    # Optional reference answer the judge can compare against.
    expected_output: str = ""
    # What a good answer must (or must not) do, in plain language.
    criteria: str = ""
    category: TestCaseCategory = TestCaseCategory.NORMAL
    source: TestCaseSource = TestCaseSource.MANUAL
    created_at: datetime = Field(default_factory=utcnow)

    prompt: Prompt = Relationship(back_populates="test_cases")
