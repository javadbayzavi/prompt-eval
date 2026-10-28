"""Create runs and execute them in the background on the app's event loop."""

import asyncio
import logging

from sqlalchemy.engine import Engine as DbEngine
from sqlmodel import Session, select

from prompt_eval import judge, static_checks
from prompt_eval.engines import ChatRequest, EngineError, EngineRegistry
from prompt_eval.models import (
    Engine,
    PromptVersion,
    ResultStatus,
    Run,
    RunResult,
    RunStatus,
    TestCase,
    utcnow,
)
from prompt_eval.templating import render

log = logging.getLogger(__name__)

RETRYABLE = (ResultStatus.ERROR, ResultStatus.CANCELLED)


def create_run(
    session: Session,
    version: PromptVersion,
    engines: list[Engine],
    test_cases: list[TestCase],
) -> Run:
    """Create a run with one pending result per test case and engine (not started)."""
    prompt = version.prompt
    run = Run(
        prompt_id=version.prompt_id,
        prompt_version_id=version.id,
        engine_ids=[e.id for e in engines],
    )
    snapshot_static(run, prompt)
    snapshot_judge(run, prompt)
    for tc in test_cases:
        system_text = render(version.system_prompt, tc.inputs)
        user_text = render(version.template, tc.inputs)
        for engine in engines:
            run.results.append(
                RunResult(
                    test_case_id=tc.id,
                    engine_id=engine.id,
                    test_case_name=tc.name,
                    category=tc.category,
                    inputs=tc.inputs,
                    criteria=tc.criteria,
                    expected_output=tc.expected_output,
                    engine_name=engine.name,
                    model=engine.model,
                    system_text=system_text,
                    user_text=user_text,
                )
            )
    session.add(run)
    session.commit()
    return run


def snapshot_static(run: Run, prompt) -> None:
    run.output_format = prompt.output_format
    run.output_schema = prompt.output_schema
    run.static_checks = static_checks.configs_to_json(static_checks.prompt_config(prompt))
    run.static_weight = prompt.static_weight if prompt.judge_engine_id else 100.0


def snapshot_judge(run: Run, prompt) -> None:
    run.judge_engine_id = prompt.judge_engine_id
    run.judge_criteria = prompt.judge_criteria
    run.judge_instructions = prompt.judge_instructions
    run.static_weight = prompt.static_weight if prompt.judge_engine_id else 100.0


def rescore(session: Session, run: Run) -> int:
    """Re-apply the prompt's current static checks and weights; judge scores are kept."""
    snapshot_static(run, run.prompt)
    scored = 0
    for result in run.results:
        if result.status == ResultStatus.DONE:
            static_checks.score_result(result, run)
            result.final_score = judge.final_score(result, run)
            scored += 1
    session.commit()
    return scored


def clear_judgement(result: RunResult) -> None:
    result.judge_score = None
    result.judge_scores = None
    result.judge_summary = ""
    result.judge_prompt_issues = ""
    result.judge_error = ""


class RunManager:
    """Runs execute as asyncio tasks; each engine's limiter decides how many go at once."""

    def __init__(self, db: DbEngine, engines: EngineRegistry):
        self.db = db
        self.engines = engines
        self.tasks: dict[int, asyncio.Task] = {}

    def start(self, run_id: int) -> None:
        if run_id in self.tasks:
            return
        task = asyncio.create_task(self._execute(run_id), name=f"run-{run_id}")
        self.tasks[run_id] = task
        task.add_done_callback(lambda _: self.tasks.pop(run_id, None))

    def is_active(self, run_id: int) -> bool:
        return run_id in self.tasks

    def cancel(self, run_id: int) -> bool:
        task = self.tasks.get(run_id)
        if task is None:
            return False
        task.cancel()
        return True

    async def wait(self, run_id: int) -> None:
        if task := self.tasks.get(run_id):
            await asyncio.gather(task, return_exceptions=True)

    async def shutdown(self) -> None:
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def start_rejudge(self, run_id: int) -> bool:
        """Judge every finished result again with the prompt's current judge settings."""
        if run_id in self.tasks:
            return False
        task = asyncio.create_task(self._rejudge(run_id), name=f"rejudge-{run_id}")
        self.tasks[run_id] = task
        task.add_done_callback(lambda _: self.tasks.pop(run_id, None))
        return True

    def retry_failed(self, run_id: int) -> int:
        """Reset errored or cancelled results to pending and start the run again."""
        with Session(self.db) as session:
            run = session.get(Run, run_id)
            if run is None or run.id in self.tasks:
                return 0
            failed = [r for r in run.results if r.status in RETRYABLE]
            for result in failed:
                result.status = ResultStatus.PENDING
                result.error = ""
                clear_judgement(result)
            if failed:
                run.status = RunStatus.QUEUED
                run.finished_at = None
            session.commit()
        if failed:
            self.start(run_id)
        return len(failed)

    def recover_interrupted(self) -> None:
        """At startup: runs left active by a stopped server can't be resumed in-process."""
        with Session(self.db) as session:
            active = session.exec(
                select(Run).where(Run.status.in_([RunStatus.QUEUED, RunStatus.RUNNING]))
            ).all()
            for run in active:
                run.status = RunStatus.INTERRUPTED
                run.finished_at = utcnow()
                for result in run.results:
                    if result.status in (ResultStatus.PENDING, ResultStatus.RUNNING):
                        result.status = ResultStatus.CANCELLED
                        result.error = "Interrupted: the server stopped during the run"
            session.commit()

    async def _execute(self, run_id: int) -> None:
        with Session(self.db) as session:
            run = session.get(Run, run_id)
            if run is None:
                return
            run.status = RunStatus.RUNNING
            run.started_at = run.started_at or utcnow()
            pending = [r.id for r in run.results if r.status == ResultStatus.PENDING]
            session.commit()

        try:
            await asyncio.gather(*(self._run_one(result_id) for result_id in pending))
        except asyncio.CancelledError:
            self._finish(run_id, RunStatus.CANCELLED)
            raise
        self._finish(run_id, RunStatus.COMPLETED)

    async def _rejudge(self, run_id: int) -> None:
        with Session(self.db) as session:
            run = session.get(Run, run_id)
            if run is None:
                return
            snapshot_judge(run, run.prompt)
            run.status = RunStatus.RUNNING
            done = [r for r in run.results if r.status == ResultStatus.DONE]
            for result in done:
                clear_judgement(result)
                result.final_score = judge.final_score(result, run)
            session.commit()
            done_ids = [r.id for r in done]

        try:
            await asyncio.gather(*(self._judge(result_id) for result_id in done_ids))
        except asyncio.CancelledError:
            self._finish(run_id, RunStatus.CANCELLED)
            raise
        self._finish(run_id, RunStatus.COMPLETED)

    def _finish(self, run_id: int, status: RunStatus) -> None:
        with Session(self.db) as session:
            run = session.get(Run, run_id)
            if run is None:
                return
            for result in run.results:
                if result.status in (ResultStatus.PENDING, ResultStatus.RUNNING):
                    result.status = ResultStatus.CANCELLED
            run.status = status
            run.finished_at = utcnow()
            session.commit()

    async def _run_one(self, result_id: int) -> None:
        # Read what we need, then release the session while waiting on the model.
        with Session(self.db) as session:
            result = session.get(RunResult, result_id)
            engine = session.get(Engine, result.engine_id) if result.engine_id else None
            request = ChatRequest(system=result.system_text, user=result.user_text)
            if engine is None:
                self._save(session, result, error="The engine was deleted")
                return
            session.expunge(engine)
            result.status = ResultStatus.RUNNING
            session.commit()

        try:
            client = await self.engines.get(engine)
            response = await client.complete(request)
        except EngineError as e:
            with Session(self.db) as session:
                self._save(session, session.get(RunResult, result_id), error=str(e))
            return
        except asyncio.CancelledError:
            raise
        except Exception as e:  # keep one bad result from stopping the whole run
            log.exception("Result %s failed", result_id)
            with Session(self.db) as session:
                self._save(session, session.get(RunResult, result_id), error=f"Unexpected: {e!r}")
            return

        with Session(self.db) as session:
            result = session.get(RunResult, result_id)
            result.output = response.text
            result.stop_reason = response.stop_reason
            result.input_tokens = response.input_tokens
            result.output_tokens = response.output_tokens
            result.latency_ms = response.latency_ms
            result.attempts = response.attempts
            if response.refused:
                self._save(session, result, error="The model refused to answer")
                return
            static_checks.score_result(result, result.run)
            result.final_score = judge.final_score(result, result.run)
            self._save(session, result)
            needs_judge = result.run.judge_engine_id is not None

        if needs_judge:
            await self._judge(result_id)

    async def _judge(self, result_id: int) -> None:
        with Session(self.db) as session:
            result = session.get(RunResult, result_id)
            run = result.run
            engine = session.get(Engine, run.judge_engine_id) if run.judge_engine_id else None
            if engine is None:
                result.judge_error = "The judge engine was deleted"
                session.commit()
                return
            criteria = judge.criteria_from_json(run.judge_criteria)
            request = judge.build_request(result, run, run.prompt, criteria, run.judge_instructions)
            session.expunge(engine)

        try:
            client = await self.engines.get(engine)
            response = await client.complete(request)
            if response.refused:
                raise EngineError("The judge model refused to grade this output")
            verdict = judge.parse_verdict(response.text, criteria)
        except EngineError as e:
            with Session(self.db) as session:
                result = session.get(RunResult, result_id)
                result.judge_error = str(e)
                session.commit()
            return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("Judging result %s failed", result_id)
            with Session(self.db) as session:
                result = session.get(RunResult, result_id)
                result.judge_error = f"Unexpected: {e!r}"
                session.commit()
            return

        with Session(self.db) as session:
            result = session.get(RunResult, result_id)
            result.judge_score = verdict.score
            result.judge_scores = verdict.criteria_json()
            result.judge_summary = verdict.summary
            result.judge_prompt_issues = verdict.prompt_issues
            result.judge_error = ""
            result.final_score = judge.final_score(result, result.run)
            session.commit()

    @staticmethod
    def _save(session: Session, result: RunResult, *, error: str = "") -> None:
        result.status = ResultStatus.ERROR if error else ResultStatus.DONE
        result.error = error
        result.finished_at = utcnow()
        session.commit()
