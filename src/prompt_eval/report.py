"""Aggregate a run's results into a report: per engine, per category, per check, per criterion."""

from dataclasses import dataclass, field

from prompt_eval.models import ResultStatus, Run, RunResult, TestCaseCategory


def avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


@dataclass
class EngineReport:
    name: str
    model: str
    total: int = 0
    failed: int = 0
    final: list[float] = field(default_factory=list)
    static: list[float] = field(default_factory=list)
    judge: list[float] = field(default_factory=list)
    by_category: dict[TestCaseCategory, list[float]] = field(default_factory=dict)
    latency_ms: list[int] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def avg_final(self) -> float | None:
        return avg(self.final)

    @property
    def avg_static(self) -> float | None:
        return avg(self.static)

    @property
    def avg_judge(self) -> float | None:
        return avg(self.judge)

    @property
    def avg_latency_ms(self) -> int | None:
        return int(sum(self.latency_ms) / len(self.latency_ms)) if self.latency_ms else None

    def category_avg(self, category: TestCaseCategory) -> float | None:
        return avg(self.by_category.get(category, []))


@dataclass
class Rate:
    passed: float = 0.0
    total: int = 0

    @property
    def percent(self) -> float | None:
        return round(self.passed / self.total * 100, 1) if self.total else None


@dataclass
class CheckStat:
    name: str
    label: str
    weight: float
    per_engine: dict[str, Rate] = field(default_factory=dict)


@dataclass
class CriterionStat:
    name: str
    weight: float
    per_engine: dict[str, list[float]] = field(default_factory=dict)

    def avg_for(self, engine: str) -> float | None:
        return avg(self.per_engine.get(engine, []))


@dataclass
class ResultReport:
    result: RunResult
    failed_checks: list[dict]

    @property
    def sort_key(self) -> tuple:
        # Errors first, then lowest final score.
        r = self.result
        score = r.final_score if r.final_score is not None else r.static_score
        return (r.status == ResultStatus.DONE, score if score is not None else -1)


@dataclass
class Report:
    run: Run
    engines: list[EngineReport]
    categories: list[TestCaseCategory]
    checks: list[CheckStat]
    criteria: list[CriterionStat]
    results: list[ResultReport]
    prompt_issues: list[tuple[str, str, str]]  # (test case, engine, issue)

    @property
    def has_judge(self) -> bool:
        return any(e.judge for e in self.engines)


def build_report(run: Run) -> Report:
    engines: dict[str, EngineReport] = {}
    checks: dict[str, CheckStat] = {}
    criteria: dict[str, CriterionStat] = {}
    categories: list[TestCaseCategory] = []
    results: list[ResultReport] = []
    issues: list[tuple[str, str, str]] = []

    for r in run.results:
        e = engines.setdefault(r.engine_name, EngineReport(r.engine_name, r.model))
        e.total += 1
        e.input_tokens += r.input_tokens
        e.output_tokens += r.output_tokens
        if r.category not in categories:
            categories.append(r.category)
        if r.status != ResultStatus.DONE:
            e.failed += 1
        else:
            e.latency_ms.append(r.latency_ms)
            if r.final_score is not None:
                e.final.append(r.final_score)
                e.by_category.setdefault(r.category, []).append(r.final_score)
            if r.static_score is not None:
                e.static.append(r.static_score)
            if r.judge_score is not None:
                e.judge.append(r.judge_score)

        for c in r.static_checks or []:
            stat = checks.setdefault(c["name"], CheckStat(c["name"], c["label"], c["weight"]))
            rate = stat.per_engine.setdefault(r.engine_name, Rate())
            rate.total += 1
            rate.passed += c["score"]
        for c in r.judge_scores or []:
            stat = criteria.setdefault(c["name"], CriterionStat(c["name"], c["weight"]))
            stat.per_engine.setdefault(r.engine_name, []).append(c["score"])
        if r.judge_prompt_issues:
            issues.append((r.test_case_name, r.engine_name, r.judge_prompt_issues))

        failed_checks = [c for c in r.static_checks or [] if c["score"] < 1]
        results.append(ResultReport(r, failed_checks))

    categories.sort(key=list(TestCaseCategory).index)
    results.sort(key=lambda rr: rr.sort_key)
    return Report(
        run=run,
        engines=list(engines.values()),
        categories=categories,
        checks=list(checks.values()),
        criteria=list(criteria.values()),
        results=results,
        prompt_issues=issues,
    )


@dataclass
class ComparisonRow:
    test_case: str
    engine: str
    before: float | None
    after: float | None

    @property
    def delta(self) -> float | None:
        if self.before is None or self.after is None:
            return None
        return round(self.after - self.before, 1)


def compare(before: Run, after: Run) -> tuple[list[ComparisonRow], list[ComparisonRow]]:
    """Per-engine and per-test-case final scores of two runs, matched by test case and engine."""

    def key(r: RunResult) -> tuple:
        return (r.test_case_id or r.test_case_name, r.engine_name)

    old = {key(r): r for r in before.results}
    new = {key(r): r for r in after.results}
    rows = []
    for k in list(old) + [k for k in new if k not in old]:
        a, b = old.get(k), new.get(k)
        name = (b or a).test_case_name
        rows.append(
            ComparisonRow(name, k[1], a.final_score if a else None, b.final_score if b else None)
        )
    rows.sort(key=lambda row: (row.delta is None, row.delta if row.delta is not None else 0))

    engine_names = list(dict.fromkeys(row.engine for row in rows))
    engine_rows = []
    for name in engine_names:
        before_scores = [r.final_score for r in before.results if r.engine_name == name]
        after_scores = [r.final_score for r in after.results if r.engine_name == name]
        engine_rows.append(
            ComparisonRow(
                "",
                name,
                avg([s for s in before_scores if s is not None]),
                avg([s for s in after_scores if s is not None]),
            )
        )
    return engine_rows, rows
