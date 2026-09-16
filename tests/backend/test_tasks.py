from __future__ import annotations

from datetime import UTC, datetime, timedelta
import traceback
from unittest.mock import Mock, patch as mock_patch

import jwt
import pytest
from celery.exceptions import Retry
from fastapi.testclient import TestClient
from sqlalchemy import select

from aquaops.api.app import create_app
from aquaops.db.session import create_session_factory, session_scope
from aquaops.domain.models import AuditEvent, Base, Role, Task, User
from aquaops.domain.services import (
    TaskState,
    transition_persisted_task,
    transition_task,
)
from aquaops.tasks.jobs import (
    PublicIngestionTaskError,
    _finish_attempt,
    configure_worker_runtime,
    create_registered_public_ingestion_handler,
    ingest_public_document,
    run_public_ingestion,
)
from aquaops.tasks.celery_app import celery_app

from aquaops.config import Settings


JWT_SECRET = "synthetic-test-secret-that-is-at-least-32-bytes"


def _settings() -> Settings:
    return Settings(
        jwt_secret=JWT_SECRET,
        jwt_issuer="aquaops-test",
        jwt_audience="aquaops-api-test",
    )


def _token() -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "sub": "user-1",
            "roles": ["operator"],
            "iss": "aquaops-test",
            "aud": "aquaops-api-test",
            "iat": now,
            "exp": now + timedelta(minutes=5),
        },
        JWT_SECRET,
        algorithm="HS256",
    )


def test_task_cannot_skip_from_queued_to_succeeded() -> None:
    with pytest.raises(ValueError, match="queued -> succeeded"):
        transition_task(TaskState.queued, TaskState.succeeded)


def test_task_can_move_from_queued_to_running() -> None:
    assert transition_task(TaskState.queued, TaskState.running) is TaskState.running


@pytest.mark.parametrize("terminal", [TaskState.succeeded, TaskState.failed])
def test_task_can_finish_only_after_running(terminal: TaskState) -> None:
    assert transition_task(TaskState.running, terminal) is terminal
    with pytest.raises(ValueError, match=f"{terminal} ->"):
        transition_task(terminal, TaskState.running)


def test_persisted_task_transition_writes_one_audit_event(tmp_path) -> None:
    factory = create_session_factory(
        f"sqlite+pysqlite:///{(tmp_path / 'tasks.sqlite3').as_posix()}"
    )
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        task = Task(owner_id="user-1", state=TaskState.queued, request_id="req-create")
        session.add(task)
        session.flush()
        task_id = task.id

    with session_scope(factory) as session:
        transitioned = transition_persisted_task(
            session,
            task_id=task_id,
            target=TaskState.running,
            actor_id="user-1",
            request_id="req-transition",
        )
        assert transitioned.state == TaskState.running

    with factory() as session:
        audit = session.scalar(select(AuditEvent))
        assert audit is not None
        assert audit.actor_id == "user-1"
        assert audit.request_id == "req-transition"
    engine.dispose()


def test_task_route_returns_owned_task_and_invalid_transition_is_409(tmp_path) -> None:
    factory = create_session_factory(
        f"sqlite+pysqlite:///{(tmp_path / 'task-api.sqlite3').as_posix()}"
    )
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        session.add(
            User(id="user-1", username="task-user", roles=[Role(name="operator")])
        )
        task = Task(state=TaskState.queued, request_id="req-create", owner_id="user-1")
        session.add(task)
        session.flush()
        task_id = task.id

    client = TestClient(create_app(settings=_settings(), session_factory=factory))
    headers = {"Authorization": f"Bearer {_token()}"}
    fetched = client.get(f"/v1/tasks/{task_id}", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json() == {"id": task_id, "state": "queued", "retry_count": 0}

    rejected = client.post(
        f"/v1/tasks/{task_id}/transitions",
        headers={**headers, "X-Request-ID": "req-task-api"},
        json={"target": "succeeded"},
    )
    assert rejected.status_code == 409
    engine.dispose()


def test_ingestion_job_persists_running_then_succeeded(tmp_path) -> None:
    factory = create_session_factory(
        f"sqlite+pysqlite:///{(tmp_path / 'job-success.sqlite3').as_posix()}"
    )
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        task = Task(owner_id="user-1", state=TaskState.queued, request_id="req-job")
        session.add(task)
        session.flush()
        task_id = task.id

    observed_states: list[str] = []

    def ingest(source_id: str) -> None:
        assert source_id == "public-source-1"
        with factory() as session:
            stored = session.get(Task, task_id)
            assert stored is not None
            observed_states.append(stored.state)

    result = run_public_ingestion(
        factory,
        task_id=task_id,
        source_id="public-source-1",
        ingest=ingest,
        worker_token="celery-task-1",
    )

    assert result == {"task_id": task_id, "status": "succeeded"}
    assert observed_states == ["running"]
    with factory() as session:
        stored = session.get(Task, task_id)
        assert stored is not None
        assert stored.state == TaskState.succeeded
    engine.dispose()


def test_cache_invalidation_failure_cannot_roll_back_succeeded_task(tmp_path) -> None:
    factory = create_session_factory(
        f"sqlite+pysqlite:///{(tmp_path / 'job-cache-failure.sqlite3').as_posix()}"
    )
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        task = Task(owner_id="user-1", state=TaskState.queued, request_id="req-job")
        session.add(task)
        session.flush()
        task_id = task.id
    logged: list[dict[str, object]] = []

    result = run_public_ingestion(
        factory,
        task_id=task_id,
        source_id="public-source-1",
        ingest=lambda _: None,
        worker_token="celery-task-cache",
        invalidate_cache=lambda: (_ for _ in ()).throw(
            RuntimeError("canary-secret-redis-endpoint")
        ),
        log_event=logged.append,
    )

    assert result["status"] == "succeeded"
    assert logged == [
        {
            "event": "cache.invalidate_failed",
            "request_id": "req-job",
            "error_class": "RuntimeError",
        }
    ]
    assert "canary" not in str(logged)
    with factory() as session:
        stored = session.get(Task, task_id)
        assert stored is not None
        assert stored.state == TaskState.succeeded
    engine.dispose()


def test_ingestion_job_failure_persists_only_error_class_and_retry_count(
    tmp_path,
) -> None:
    factory = create_session_factory(
        f"sqlite+pysqlite:///{(tmp_path / 'job-failure.sqlite3').as_posix()}"
    )
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        task = Task(owner_id="user-1", state=TaskState.queued, request_id="req-job")
        session.add(task)
        session.flush()
        task_id = task.id

    def fail(_: str) -> None:
        raise TimeoutError("synthetic secret-bearing payload")

    with pytest.raises(TimeoutError, match="synthetic secret-bearing payload"):
        run_public_ingestion(
            factory,
            task_id=task_id,
            source_id="public-source-1",
            ingest=fail,
            retry_count=1,
            worker_token="celery-task-2",
            final_attempt=True,
        )

    with factory() as session:
        stored = session.get(Task, task_id)
        assert stored is not None
        assert stored.state == TaskState.failed
        assert stored.retry_count == 1
        assert stored.error_class == "TimeoutError"
        assert "secret" not in (stored.error_class or "")
    engine.dispose()


def test_ingestion_retry_stays_running_then_same_worker_can_succeed(tmp_path) -> None:
    factory = create_session_factory(
        f"sqlite+pysqlite:///{(tmp_path / 'job-retry.sqlite3').as_posix()}"
    )
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        task = Task(owner_id="user-1", state=TaskState.queued, request_id="req-job")
        session.add(task)
        session.flush()
        task_id = task.id

    def timeout(_: str) -> None:
        raise TimeoutError("retryable")

    with pytest.raises(TimeoutError):
        run_public_ingestion(
            factory,
            task_id=task_id,
            source_id="public-source-1",
            ingest=timeout,
            retry_count=0,
            worker_token="celery-task-stable",
            final_attempt=False,
        )

    with factory() as session:
        stored = session.get(Task, task_id)
        assert stored is not None
        assert stored.state == TaskState.running
        assert stored.retry_count == 1

    result = run_public_ingestion(
        factory,
        task_id=task_id,
        source_id="public-source-1",
        ingest=lambda _: None,
        retry_count=1,
        worker_token="celery-task-stable",
    )
    assert result["status"] == "succeeded"
    with factory() as session:
        audits = session.scalars(
            select(AuditEvent).order_by(AuditEvent.created_at)
        ).all()
        assert [audit.action for audit in audits] == [
            "task.started",
            "task.retry_scheduled",
            "task.succeeded",
        ]
    engine.dispose()


def test_non_retryable_ingestion_error_immediately_fails_task(tmp_path) -> None:
    factory = create_session_factory(
        f"sqlite+pysqlite:///{(tmp_path / 'job-nonretry.sqlite3').as_posix()}"
    )
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        task = Task(owner_id="user-1", state=TaskState.queued, request_id="req-job")
        session.add(task)
        session.flush()
        task_id = task.id

    with pytest.raises(ValueError, match="synthetic failure"):
        run_public_ingestion(
            factory,
            task_id=task_id,
            source_id="public-source-1",
            ingest=lambda _: (_ for _ in ()).throw(ValueError("synthetic failure")),
            worker_token="celery-task-nonretry",
        )

    with factory() as session:
        stored = session.get(Task, task_id)
        assert stored is not None
        assert stored.state == TaskState.failed
        assert stored.error_class == "ValueError"
    engine.dispose()


def test_duplicate_delivery_of_same_attempt_never_reexecutes_handler(tmp_path) -> None:
    factory = create_session_factory(
        f"sqlite+pysqlite:///{(tmp_path / 'job-same-attempt.sqlite3').as_posix()}"
    )
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        task = Task(
            owner_id="user-1",
            state=TaskState.running,
            request_id="req-job",
            worker_token="worker-one",
            worker_attempt=0,
            worker_lease_token="lease-one",
            worker_lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )
        session.add(task)
        session.flush()
        task_id = task.id
    handler = Mock()

    with pytest.raises(RuntimeError, match="attempt already processed"):
        run_public_ingestion(
            factory,
            task_id=task_id,
            source_id="public-source-1",
            ingest=handler,
            worker_token="worker-one",
            lease_token="lease-two",
            retry_count=0,
        )

    handler.assert_not_called()
    engine.dispose()


def test_expired_same_attempt_lease_can_recover_after_worker_crash(tmp_path) -> None:
    factory = create_session_factory(
        f"sqlite+pysqlite:///{(tmp_path / 'job-expired-lease.sqlite3').as_posix()}"
    )
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        task = Task(
            owner_id="user-1",
            state=TaskState.running,
            request_id="req-job",
            worker_token="worker-one",
            worker_attempt=0,
            worker_lease_token="dead-lease",
            worker_lease_expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
        session.add(task)
        session.flush()
        task_id = task.id
    handler = Mock()

    result = run_public_ingestion(
        factory,
        task_id=task_id,
        source_id="public-source-1",
        ingest=handler,
        worker_token="worker-one",
        lease_token="recovery-lease",
        retry_count=0,
    )

    assert result["status"] == "succeeded"
    handler.assert_called_once_with("public-source-1")
    engine.dispose()


def test_next_retry_cannot_preempt_active_recovered_attempt(tmp_path) -> None:
    factory = create_session_factory(
        f"sqlite+pysqlite:///{(tmp_path / 'job-cross-attempt-lease.sqlite3').as_posix()}"
    )
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        task = Task(
            owner_id="user-1",
            state=TaskState.running,
            request_id="req-job",
            worker_token="worker-one",
            worker_attempt=0,
            retry_count=1,
            worker_lease_token="recovered-attempt-zero",
            worker_lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )
        session.add(task)
        session.flush()
        task_id = task.id
    next_handler = Mock()

    with pytest.raises(RuntimeError, match="attempt already processed"):
        run_public_ingestion(
            factory,
            task_id=task_id,
            source_id="public-source-1",
            ingest=next_handler,
            worker_token="worker-one",
            lease_token="retry-one-lease",
            retry_count=1,
        )

    next_handler.assert_not_called()
    _finish_attempt(
        factory,
        task_id=task_id,
        worker_token="worker-one",
        lease_token="recovered-attempt-zero",
        target=TaskState.succeeded,
        action="task.succeeded",
        retry_count=0,
        error_class=None,
    )
    with factory() as session:
        stored = session.get(Task, task_id)
        assert stored is not None
        assert stored.state == TaskState.succeeded
        assert stored.retry_count == 1
    engine.dispose()


def test_redelivery_after_index_write_repairs_terminal_state_idempotently(
    tmp_path,
) -> None:
    factory = create_session_factory(
        f"sqlite+pysqlite:///{(tmp_path / 'job-post-upsert-crash.sqlite3').as_posix()}"
    )
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        task = Task(owner_id="user-1", state=TaskState.queued, request_id="req-job")
        session.add(task)
        session.flush()
        task_id = task.id
    indexed_points: set[str] = set()

    def idempotent_upsert(_: str) -> None:
        indexed_points.add("deterministic-point-id")

    with mock_patch(
        "aquaops.tasks.jobs._finish_attempt",
        side_effect=SystemExit("synthetic worker crash"),
    ):
        with pytest.raises(SystemExit, match="worker crash"):
            run_public_ingestion(
                factory,
                task_id=task_id,
                source_id="public-source-1",
                ingest=idempotent_upsert,
                worker_token="worker-one",
                lease_token="dead-lease",
            )

    with session_scope(factory) as session:
        stored = session.get(Task, task_id)
        assert stored is not None
        stored.worker_lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)

    result = run_public_ingestion(
        factory,
        task_id=task_id,
        source_id="public-source-1",
        ingest=idempotent_upsert,
        worker_token="worker-one",
        lease_token="recovery-lease",
    )

    assert result["status"] == "succeeded"
    assert indexed_points == {"deterministic-point-id"}
    with factory() as session:
        stored = session.get(Task, task_id)
        assert stored is not None
        assert stored.state == TaskState.succeeded
    engine.dispose()


def test_duplicate_delivery_with_different_worker_token_is_rejected(tmp_path) -> None:
    factory = create_session_factory(
        f"sqlite+pysqlite:///{(tmp_path / 'job-duplicate.sqlite3').as_posix()}"
    )
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        task = Task(
            owner_id="user-1",
            state=TaskState.running,
            request_id="req-job",
            worker_token="worker-one",
        )
        session.add(task)
        session.flush()
        task_id = task.id

    with pytest.raises(RuntimeError, match="task already claimed"):
        run_public_ingestion(
            factory,
            task_id=task_id,
            source_id="public-source-1",
            ingest=lambda _: None,
            worker_token="worker-two",
        )
    engine.dispose()


def test_celery_app_registers_the_durable_ingestion_task() -> None:
    celery_app.loader.import_default_modules()
    assert "aquaops.ingest_public_document" in celery_app.tasks


def test_celery_task_entry_runs_configured_handler_without_external_broker(
    tmp_path,
) -> None:
    database_url = (
        f"sqlite+pysqlite:///{(tmp_path / 'celery-entry.sqlite3').as_posix()}"
    )
    factory = create_session_factory(database_url)
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        task = Task(owner_id="user-1", state=TaskState.queued, request_id="req-job")
        session.add(task)
        session.flush()
        task_id = task.id
    handler = Mock()
    configure_worker_runtime(database_url=database_url, ingestion_handler=handler)

    result = ingest_public_document.apply(
        args=(task_id, "public-source-1"),
        task_id="11111111-1111-4111-8111-111111111111",
        throw=True,
    )

    assert result.successful()
    handler.assert_called_once_with("public-source-1")
    with factory() as session:
        stored = session.get(Task, task_id)
        assert stored is not None
        assert stored.state == TaskState.succeeded
    engine.dispose()


def test_celery_task_entry_redacts_handler_exception(tmp_path) -> None:
    database_url = (
        f"sqlite+pysqlite:///{(tmp_path / 'celery-redact.sqlite3').as_posix()}"
    )
    factory = create_session_factory(database_url)
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        task = Task(owner_id="user-1", state=TaskState.queued, request_id="req-job")
        session.add(task)
        session.flush()
        task_id = task.id

    def fail(_: str) -> None:
        raise ValueError("canary-secret-handler-message")

    configure_worker_runtime(database_url=database_url, ingestion_handler=fail)
    with pytest.raises(PublicIngestionTaskError) as captured:
        ingest_public_document.apply(
            args=(task_id, "public-source-1"),
            task_id="22222222-2222-4222-8222-222222222222",
            throw=True,
        )

    assert "canary" not in str(captured.value)
    assert ingest_public_document.ignore_result is True
    assert ingest_public_document.store_errors_even_if_ignored is False
    engine.dispose()


def test_celery_timeout_retry_has_no_original_exception_chain(tmp_path) -> None:
    database_url = (
        f"sqlite+pysqlite:///{(tmp_path / 'celery-timeout.sqlite3').as_posix()}"
    )
    factory = create_session_factory(database_url)
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        task = Task(owner_id="user-1", state=TaskState.queued, request_id="req-job")
        session.add(task)
        session.flush()
        task_id = task.id

    def timeout(_: str) -> None:
        raise TimeoutError("canary-secret-timeout")

    configure_worker_runtime(database_url=database_url, ingestion_handler=timeout)
    with pytest.raises(Retry) as captured:
        ingest_public_document.apply(
            args=(task_id, "public-source-1"),
            task_id="33333333-3333-4333-8333-333333333333",
            throw=True,
        )

    rendered = "".join(
        traceback.format_exception(
            type(captured.value), captured.value, captured.value.__traceback__
        )
    )
    assert "canary" not in rendered
    engine.dispose()


def test_registered_public_handler_upserts_only_selected_hash_locked_source(
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    class FakeEncoder:
        def encode_documents(self, texts: tuple[str, ...]):
            observed["texts"] = texts
            return tuple((0.0,) * 1024 for _ in texts)

    class FakeClient:
        def close(self) -> None:
            observed["closed"] = True

    class FakeStore:
        def __init__(self, client: object) -> None:
            observed["client"] = client

        def initialize(self) -> None:
            observed["initialized"] = True

        def upsert(self, chunks, vectors, *, topic: str, effective_date: str) -> None:
            observed["source_ids"] = {chunk.source_id for chunk in chunks}
            observed["vector_count"] = len(vectors)
            observed["topic"] = topic
            observed["effective_date"] = effective_date

    monkeypatch.setattr("aquaops.tasks.jobs.QdrantClient", lambda **_: FakeClient())
    monkeypatch.setattr("aquaops.tasks.jobs.QdrantPublicKnowledgeStore", FakeStore)
    handler = create_registered_public_ingestion_handler(
        qdrant_url="http://qdrant:6333",
        encoder=FakeEncoder(),
    )

    handler("epa-nutrient-removal")

    assert observed["source_ids"] == {"epa-nutrient-removal"}
    assert observed["vector_count"] == len(observed["texts"])
    assert observed["initialized"] is True
    assert observed["closed"] is True
