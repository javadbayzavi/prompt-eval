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
    __test__ = False  # not a pytest test class

    NORMAL = "normal"
    EDGE = "edge"
    ADVERSARIAL = "adversarial"


class TestCaseSource(StrEnum):
    __test__ = False

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
    # Static checks as [{name, weight, param}]; None means the defaults for the format.
    static_checks: list[dict] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    # Final score = static_weight% of the static score + the rest from the judge.
    static_weight: float = 30.0
    judge_engine_id: int | None = Field(default=None, foreign_key="engine.id", ondelete="SET NULL")
    # Judge criteria as [{name, description, weight}]; None means the defaults.
    judge_criteria: list[dict] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    judge_instructions: str = ""
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
    runs: list["Run"] = Relationship(
        back_populates="prompt",
        cascade_delete=True,
        sa_relationship_kwargs={"order_by": "Run.id.desc()"},
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


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"  # the server stopped while it was running

    @property
    def active(self) -> bool:
        return self in (RunStatus.QUEUED, RunStatus.RUNNING)


class ResultStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class Run(SQLModel, table=True):
    """One prompt version run on a set of test cases, on one or more engines."""

    id: int | None = Field(default=None, primary_key=True)
    prompt_id: int = Field(foreign_key="prompt.id", index=True, ondelete="CASCADE")
    prompt_version_id: int = Field(foreign_key="promptversion.id", ondelete="CASCADE")
    engine_ids: list[int] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    # Scoring settings copied from the prompt when the run is created (or rescored).
    output_format: OutputFormat = OutputFormat.TEXT
    output_schema: str = ""
    static_checks: list[dict] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    static_weight: float = 100.0
    judge_engine_id: int | None = Field(default=None, foreign_key="engine.id", ondelete="SET NULL")
    judge_criteria: list[dict] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    judge_instructions: str = ""
    status: RunStatus = RunStatus.QUEUED
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    prompt: Prompt = Relationship(back_populates="runs")
    version: PromptVersion = Relationship()
    results: list["RunResult"] = Relationship(
        back_populates="run",
        cascade_delete=True,
        # Created test case by test case, so id order groups each case's engines together.
        sa_relationship_kwargs={"order_by": "RunResult.id"},
    )


class RunResult(SQLModel, table=True):
    """One test case on one engine.

    Test case and engine details are copied in, so results stay readable after those
    are edited or deleted, and the rendered prompt is exactly what was sent.
    """

    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="run.id", index=True, ondelete="CASCADE")
    test_case_id: int | None = Field(
        default=None, foreign_key="testcase.id", index=True, ondelete="SET NULL"
    )
    engine_id: int | None = Field(default=None, foreign_key="engine.id", ondelete="SET NULL")

    test_case_name: str
    category: TestCaseCategory = TestCaseCategory.NORMAL
    inputs: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    criteria: str = ""
    expected_output: str = ""
    engine_name: str
    model: str
    system_text: str = ""
    user_text: str

    status: ResultStatus = ResultStatus.PENDING
    output: str = ""
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    attempts: int = 0
    error: str = ""
    finished_at: datetime | None = None

    # 0-100; None until scored (and for failed results).
    static_score: float | None = None
    static_checks: list[dict] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    judge_score: float | None = None
    # [{name, weight, score (0-10), reasoning}]
    judge_scores: list[dict] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    judge_summary: str = ""
    # What in the prompt likely caused the problems, according to the judge.
    judge_prompt_issues: str = ""
    judge_error: str = ""
    final_score: float | None = None

    run: Run = Relationship(back_populates="results")
