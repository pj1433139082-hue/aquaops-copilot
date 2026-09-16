from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
import logging
import re
from typing import Any
from uuid import uuid4

from celery.signals import worker_process_init
from qdrant_client import QdrantClient
from redis import Redis
from sqlalchemy import case, select, update
from sqlalchemy.orm import Session, sessionmaker

from aquaops.db.session import create_session_factory, session_scope
from aquaops.domain.models import AuditEvent, Task
from aquaops.domain.services import TaskState
from aquaops.cache.service import invalidate_index_cache
from aquaops.rag.demo_corpus import (
    DEMO_CORPUS_SHA256,
    load_verified_demo_documents,
)
from aquaops.observability.logging import event_payload
from aquaops.rag.local_models import LazyBgeM3Encoder, PublicModelSpec
from aquaops.rag.public_corpus import chunk_public_document
from aquaops.rag.store import QdrantPublicKnowledgeStore
from aquaops.tasks.celery_app import celery_app


MAX_INGESTION_RETRIES = 2
WORKER_LEASE_DURATION = timedelta(minutes=30)
_SAFE_SOURCE_ID = re.compile(r"[a-z][a-z0-9-]{2,63}")
_SAFE_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
_SAFE_WORKER_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_WORKER_SESSION_FACTORY: sessionmaker[Session] | None = None
_WORKER_INGESTION_HANDLER: Callable[[str], None] | None = None
_WORKER_CACHE_INVALIDATOR: Callable[[], None] | None = None
_WORKER_EVENT_LOGGER: Callable[[dict[str, Any]], None] | None = None
_LOGGER = logging.getLogger("aquaops.worker")


class PublicIngestionTaskError(RuntimeError):
    """Fixed worker failure that is safe for logs and result transports."""


def _validate_message_identifiers(*, task_id: str, source_id: str) -> None:
    if not _SAFE_UUID.fullmatch(task_id):
        raise PublicIngestionTaskError("public ingestion request is invalid")
    if not _SAFE_SOURCE_ID.fullmatch(source_id):
        raise PublicIngestionTaskError("public ingestion request is invalid")


def _validate_worker_token(worker_token: str) -> None:
    if not _SAFE_WORKER_TOKEN.fullmatch(worker_token):
        raise PublicIngestionTaskError("worker identity is invalid")


def _audit(task: Task, *, action: str) -> AuditEvent:
    return AuditEvent(
        action=action,
        request_id=task.request_id,
        actor_id=task.owner_id,
        entity_type="task",
        entity_id=task.id,
    )


def _claim_task(
    factory: sessionmaker[Session],
    *,
    task_id: str,
    worker_token: str,
    lease_token: str,
    retry_count: int,
) -> None:
    now = datetime.now(UTC)
    lease_deadline = now + WORKER_LEASE_DURATION
    with session_scope(factory) as session:
        claimed = session.execute(
            update(Task)
            .where(Task.id == task_id, Task.state == TaskState.queued)
            .values(
                state=TaskState.running,
                worker_token=worker_token,
                worker_attempt=retry_count,
                worker_lease_token=lease_token,
                worker_lease_expires_at=lease_deadline,
            )
        )
        if claimed.rowcount == 1:
            task = session.get(Task, task_id)
            assert task is not None
            session.add(_audit(task, action="task.started"))
            return
        task = session.scalar(select(Task).where(Task.id == task_id))
        if task is None:
            raise LookupError("task not found")
        if task.state == TaskState.running and task.worker_token == worker_token:
            advanced = session.execute(
                update(Task)
                .where(
                    Task.id == task_id,
                    Task.state == TaskState.running,
                    Task.worker_token == worker_token,
                    Task.worker_attempt < retry_count,
                    Task.retry_count == retry_count,
                    Task.worker_lease_expires_at <= now,
                )
                .values(
                    worker_attempt=retry_count,
                    worker_lease_token=lease_token,
                    worker_lease_expires_at=lease_deadline,
                )
                .execution_options(synchronize_session=False)
            )
            if advanced.rowcount == 1:
                return
            recovered = session.execute(
                update(Task)
                .where(
                    Task.id == task_id,
                    Task.state == TaskState.running,
                    Task.worker_token == worker_token,
                    Task.worker_attempt == retry_count,
                    Task.worker_lease_expires_at < now,
                )
                .values(
                    worker_lease_token=lease_token,
                    worker_lease_expires_at=lease_deadline,
                )
                .execution_options(synchronize_session=False)
            )
            if recovered.rowcount == 1:
                return
            raise RuntimeError("task attempt already processed")
        raise RuntimeError("task already claimed")


def _finish_attempt(
    factory: sessionmaker[Session],
    *,
    task_id: str,
    worker_token: str,
    lease_token: str,
    target: TaskState,
    action: str,
    retry_count: int,
    error_class: str | None,
    expected_attempt: int | None = None,
) -> None:
    expected_attempt = retry_count if expected_attempt is None else expected_attempt
    with session_scope(factory) as session:
        changed = session.execute(
            update(Task)
            .where(
                Task.id == task_id,
                Task.state == TaskState.running,
                Task.worker_token == worker_token,
                Task.worker_attempt == expected_attempt,
                Task.worker_lease_token == lease_token,
            )
            .values(
                state=target,
                retry_count=case(
                    (Task.retry_count > retry_count, Task.retry_count),
                    else_=retry_count,
                ),
                error_class=error_class,
                worker_lease_expires_at=(
                    datetime.now(UTC) if target == TaskState.running else None
                ),
            )
        )
        if changed.rowcount != 1:
            raise RuntimeError("task claim lost")
        task = session.get(Task, task_id)
        assert task is not None
        session.add(_audit(task, action=action))


def run_public_ingestion(
    factory: sessionmaker[Session],
    *,
    task_id: str,
    source_id: str,
    ingest: Callable[[str], None],
    worker_token: str,
    lease_token: str | None = None,
    retry_count: int = 0,
    final_attempt: bool = False,
    invalidate_cache: Callable[[], None] | None = None,
    log_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, str]:
    _validate_message_identifiers(task_id=task_id, source_id=source_id)
    _validate_worker_token(worker_token)
    resolved_lease_token = lease_token or worker_token
    _validate_worker_token(resolved_lease_token)
    _claim_task(
        factory,
        task_id=task_id,
        worker_token=worker_token,
        lease_token=resolved_lease_token,
        retry_count=retry_count,
    )
    try:
        ingest(source_id)
    except Exception as exc:
        retryable = isinstance(exc, TimeoutError) and not final_attempt
        if not retryable:
            _finish_attempt(
                factory,
                task_id=task_id,
                worker_token=worker_token,
                lease_token=resolved_lease_token,
                target=TaskState.failed,
                action="task.failed",
                retry_count=retry_count,
                error_class=type(exc).__name__[:120],
            )
        else:
            _finish_attempt(
                factory,
                task_id=task_id,
                worker_token=worker_token,
                lease_token=resolved_lease_token,
                target=TaskState.running,
                action="task.retry_scheduled",
                retry_count=retry_count + 1,
                expected_attempt=retry_count,
                error_class=None,
            )
        raise
    _finish_attempt(
        factory,
        task_id=task_id,
        worker_token=worker_token,
        lease_token=resolved_lease_token,
        target=TaskState.succeeded,
        action="task.succeeded",
        retry_count=retry_count,
        error_class=None,
    )
    if invalidate_cache is not None:
        try:
            invalidate_cache()
        except Exception as exc:
            if log_event is not None:
                try:
                    log_event(
                        event_payload(
                            "cache.invalidate_failed",
                            request_id=_request_id_for_task(factory, task_id),
                            error_class=type(exc).__name__[:120],
                        )
                    )
                except Exception:
                    pass
    return {"task_id": task_id, "status": "succeeded"}


def _request_id_for_task(factory: sessionmaker[Session], task_id: str) -> str:
    with factory() as session:
        request_id = session.scalar(select(Task.request_id).where(Task.id == task_id))
    if not isinstance(request_id, str):
        raise RuntimeError("task unavailable after commit")
    return request_id


def configure_worker_runtime(
    *,
    database_url: str,
    ingestion_handler: Callable[[str], None],
    cache_invalidator: Callable[[], None] | None = None,
    event_logger: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    global _WORKER_SESSION_FACTORY
    global _WORKER_INGESTION_HANDLER
    global _WORKER_CACHE_INVALIDATOR
    global _WORKER_EVENT_LOGGER
    _WORKER_SESSION_FACTORY = create_session_factory(database_url)
    _WORKER_INGESTION_HANDLER = ingestion_handler
    _WORKER_CACHE_INVALIDATOR = cache_invalidator
    _WORKER_EVENT_LOGGER = event_logger


def create_registered_public_ingestion_handler(
    *,
    qdrant_url: str,
    local_files_only: bool = True,
    encoder: Any | None = None,
) -> Callable[[str], None]:
    public_encoder = encoder or LazyBgeM3Encoder(
        spec=PublicModelSpec.default(local_files_only=local_files_only),
    )

    def ingest_registered_public_source(source_id: str) -> None:
        documents = load_verified_demo_documents()
        matches = [
            document for document in documents if document.source_id == source_id
        ]
        if len(matches) != 1:
            raise LookupError("registered public source unavailable")
        chunks = tuple(
            chunk_public_document(matches[0], max_tokens=450, overlap_tokens=40)
        )
        vectors = public_encoder.encode_documents(tuple(chunk.text for chunk in chunks))
        client = QdrantClient(url=qdrant_url)
        try:
            store = QdrantPublicKnowledgeStore(client)
            store.initialize()
            store.upsert(
                chunks,
                vectors,
                topic="water-operations",
                effective_date="2026-08-11",
            )
        finally:
            client.close()

    return ingest_registered_public_source


def _safe_worker_event_logger(payload: dict[str, Any]) -> None:
    _LOGGER.info("aquaops_event=%r", payload)


def _redis_cache_invalidator(redis_url: str) -> Callable[[], None]:
    def invalidate() -> None:
        client = Redis.from_url(redis_url, decode_responses=True)
        try:
            invalidate_index_cache(client, DEMO_CORPUS_SHA256[:16])
        finally:
            client.close()

    return invalidate


def ingest_registered_public_source(source_id: str) -> None:
    """Compatibility entry using production-safe local model resolution."""
    from aquaops.config import Settings

    settings = Settings()
    create_registered_public_ingestion_handler(
        qdrant_url=settings.qdrant_url,
        local_files_only=True,
    )(source_id)


@worker_process_init.connect
def initialize_worker_runtime(**_: object) -> None:
    from aquaops.config import Settings

    settings = Settings()
    configure_worker_runtime(
        database_url=settings.database_url,
        ingestion_handler=create_registered_public_ingestion_handler(
            qdrant_url=settings.qdrant_url,
            local_files_only=settings.model_local_files_only,
        ),
        cache_invalidator=_redis_cache_invalidator(settings.redis_url),
        event_logger=_safe_worker_event_logger,
    )


@celery_app.task(
    bind=True,
    name="aquaops.ingest_public_document",
    max_retries=MAX_INGESTION_RETRIES,
    ignore_result=True,
    store_errors_even_if_ignored=False,
)
def ingest_public_document(self, task_id: str, source_id: str) -> dict[str, str]:
    factory = _WORKER_SESSION_FACTORY
    handler = _WORKER_INGESTION_HANDLER
    if factory is None or not callable(handler):
        raise RuntimeError("worker runtime unavailable")
    worker_token = str(self.request.id or "")
    if not worker_token:
        raise RuntimeError("worker identity unavailable")
    retries = int(self.request.retries)
    try:
        return run_public_ingestion(
            factory,
            task_id=task_id,
            source_id=source_id,
            ingest=handler,
            worker_token=worker_token,
            lease_token=str(uuid4()),
            retry_count=retries,
            final_attempt=retries >= MAX_INGESTION_RETRIES,
            invalidate_cache=_WORKER_CACHE_INVALIDATOR,
            log_event=_WORKER_EVENT_LOGGER,
        )
    except TimeoutError:
        if retries < MAX_INGESTION_RETRIES:
            retry = self.retry(
                exc=PublicIngestionTaskError("public ingestion failed"),
                countdown=min(2**retries, 4),
                throw=False,
            )
            raise retry from None
        raise PublicIngestionTaskError("public ingestion failed") from None
    except Exception:
        raise PublicIngestionTaskError("public ingestion failed") from None
