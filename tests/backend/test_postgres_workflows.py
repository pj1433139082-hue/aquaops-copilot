from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event, func, inspect, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from aquaops.db.session import create_session_factory, session_scope
from aquaops.domain.models import AuditEvent, Case, Task
from aquaops.domain.services import TaskState, create_case, transition_persisted_task


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSTGRES_URL = os.getenv("AQUAOPS_TEST_POSTGRES_WORKFLOW_URL")
DESTRUCTIVE_TEST_ALLOWED = os.getenv("AQUAOPS_ALLOW_DESTRUCTIVE_MIGRATION_TEST") == "1"


def _guarded_factory() -> tuple[Config, sessionmaker[Session]]:
    assert POSTGRES_URL is not None
    database_name = make_url(POSTGRES_URL).database
    assert database_name is not None
    assert database_name.startswith("aquaops_workflow_test_")
    engine = create_engine(POSTGRES_URL)
    assert set(inspect(engine).get_table_names()).issubset({"alembic_version"})
    engine.dispose()

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", POSTGRES_URL)
    command.upgrade(config, "head")
    return config, create_session_factory(POSTGRES_URL)


@pytest.mark.skipif(
    POSTGRES_URL is None or not DESTRUCTIVE_TEST_ALLOWED,
    reason="requires an explicitly authorized isolated PostgreSQL workflow database",
)
def test_postgres_concurrent_case_create_returns_one_case_and_one_audit() -> None:
    config, factory = _guarded_factory()
    barrier = Barrier(2)

    def create(request_id: str) -> tuple[str, bool]:
        with session_scope(factory) as session:
            barrier.wait(timeout=5)
            case, created = create_case(
                session,
                owner_id="user-1",
                external_key="concurrent-key",
                title="Synthetic case",
                request_id=request_id,
            )
            return case.id, created

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(create, ["req-1", "req-2"]))

        assert len({case_id for case_id, _ in results}) == 1
        assert sorted(created for _, created in results) == [False, True]
        with factory() as session:
            assert session.scalar(select(func.count()).select_from(Case)) == 1
            assert session.scalar(select(func.count()).select_from(AuditEvent)) == 1
    finally:
        factory.kw["bind"].dispose()
        command.downgrade(config, "base")


@pytest.mark.skipif(
    POSTGRES_URL is None or not DESTRUCTIVE_TEST_ALLOWED,
    reason="requires an explicitly authorized isolated PostgreSQL workflow database",
)
def test_postgres_concurrent_task_transition_allows_one_audit() -> None:
    config, factory = _guarded_factory()
    with session_scope(factory) as session:
        task = Task(owner_id="user-1", state=TaskState.queued, request_id="req-create")
        session.add(task)
        session.flush()
        task_id = task.id

    barrier = Barrier(2)

    @event.listens_for(factory.kw["bind"], "before_cursor_execute")
    def synchronize_updates(
        _conn, _cursor, statement, _parameters, _context, _many
    ) -> None:
        if statement.lstrip().upper().startswith("UPDATE TASKS"):
            barrier.wait(timeout=5)

    def transition(target: TaskState) -> str:
        try:
            with session_scope(factory) as session:
                transition_persisted_task(
                    session,
                    task_id=task_id,
                    target=target,
                    actor_id="user-1",
                    request_id=f"req-{target}",
                )
            return "committed"
        except RuntimeError as exc:
            assert str(exc) == "concurrent task transition"
            return "conflict"

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(transition, [TaskState.running, TaskState.failed])
            )

        assert sorted(results) == ["committed", "conflict"]
        with factory() as session:
            assert session.scalar(select(func.count()).select_from(AuditEvent)) == 1
    finally:
        event.remove(factory.kw["bind"], "before_cursor_execute", synchronize_updates)
        factory.kw["bind"].dispose()
        command.downgrade(config, "base")
