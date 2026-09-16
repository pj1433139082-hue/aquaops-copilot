from __future__ import annotations

from enum import StrEnum

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from aquaops.domain.models import AuditEvent, Case, Task


class TaskState(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


ALLOWED_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.queued: frozenset({TaskState.running, TaskState.failed}),
    TaskState.running: frozenset({TaskState.succeeded, TaskState.failed}),
    TaskState.succeeded: frozenset(),
    TaskState.failed: frozenset(),
}


def transition_task(current: TaskState, target: TaskState) -> TaskState:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"{current} -> {target}")
    return target


def create_case(
    session: Session,
    *,
    owner_id: str,
    external_key: str,
    title: str,
    request_id: str,
) -> tuple[Case, bool]:
    existing = session.scalar(
        select(Case).where(
            Case.owner_id == owner_id,
            Case.external_key == external_key,
        )
    )
    if existing is not None:
        return existing, False

    case = Case(owner_id=owner_id, external_key=external_key, title=title)
    try:
        with session.begin_nested():
            session.add(case)
            session.flush()
    except IntegrityError:
        existing = session.scalar(
            select(Case).where(
                Case.owner_id == owner_id,
                Case.external_key == external_key,
            )
        )
        if existing is None:
            raise
        return existing, False
    session.add(
        AuditEvent(
            action="case.created",
            request_id=request_id,
            actor_id=owner_id,
            entity_type="case",
            entity_id=case.id,
        )
    )
    return case, True


def get_case_for_owner(session: Session, *, case_id: str, owner_id: str) -> Case | None:
    return session.scalar(
        select(Case).where(Case.id == case_id, Case.owner_id == owner_id)
    )


def transition_persisted_task(
    session: Session,
    *,
    task_id: str,
    target: TaskState,
    actor_id: str,
    request_id: str,
) -> Task:
    task = session.scalar(
        select(Task).where(Task.id == task_id, Task.owner_id == actor_id)
    )
    if task is None:
        raise LookupError("task not found")
    current = TaskState(task.state)
    validated_target = transition_task(current, target)
    result = session.execute(
        update(Task)
        .where(
            Task.id == task_id,
            Task.owner_id == actor_id,
            Task.state == current,
        )
        .values(state=validated_target)
    )
    if result.rowcount != 1:
        raise RuntimeError("concurrent task transition")
    task.state = validated_target
    session.add(
        AuditEvent(
            action="task.transitioned",
            request_id=request_id,
            actor_id=actor_id,
            entity_type="task",
            entity_id=task.id,
        )
    )
    return task


def get_task_for_owner(session: Session, *, task_id: str, owner_id: str) -> Task | None:
    return session.scalar(
        select(Task).where(Task.id == task_id, Task.owner_id == owner_id)
    )
