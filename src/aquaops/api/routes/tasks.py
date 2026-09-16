from __future__ import annotations

from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session, sessionmaker

from aquaops.db.session import session_scope
from aquaops.domain.models import Task
from aquaops.domain.services import (
    TaskState,
    get_task_for_owner,
    transition_persisted_task,
)
from aquaops.security.auth import MutationContext, Principal
from aquaops.security.dependencies import get_current_principal, require_case_write


router = APIRouter(prefix="/v1/tasks", tags=["tasks"])


class TaskTransition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    target: TaskState


def _factory(request: Request) -> sessionmaker[Session]:
    factory = getattr(request.app.state, "session_factory", None)
    if not isinstance(factory, sessionmaker):
        raise HTTPException(status_code=503, detail="database unavailable")
    return cast(sessionmaker[Session], factory)


def _response(task: Task) -> dict[str, str | int]:
    return {"id": task.id, "state": task.state, "retry_count": task.retry_count}


@router.get("/{task_id}")
def read_task(
    task_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> dict[str, str | int]:
    if "task:read" not in principal.permissions:
        raise HTTPException(status_code=403, detail="missing permission: task:read")
    with _factory(request)() as session:
        task = get_task_for_owner(session, task_id=task_id, owner_id=principal.user_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return _response(task)


@router.post("/{task_id}/transitions")
def transition_task_route(
    task_id: str,
    payload: TaskTransition,
    request: Request,
    context: Annotated[MutationContext, Depends(require_case_write)],
) -> dict[str, str | int]:
    try:
        with session_scope(_factory(request)) as session:
            task = transition_persisted_task(
                session,
                task_id=task_id,
                target=payload.target,
                actor_id=context.actor_id,
                request_id=context.request_id,
            )
            result = _response(task)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail="invalid task transition") from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=409, detail="concurrent task transition"
        ) from exc
    return result
