from __future__ import annotations

from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session, sessionmaker

from aquaops.db.session import session_scope
from aquaops.domain.models import Case
from aquaops.domain.services import create_case, get_case_for_owner
from aquaops.security.auth import MutationContext, Principal
from aquaops.security.dependencies import get_current_principal, require_case_write


router = APIRouter(prefix="/v1/cases", tags=["cases"])


class CaseCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    external_key: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=160)

    @field_validator("external_key", "title", mode="before")
    @classmethod
    def reject_untrimmed_values(cls, value: object) -> object:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("case fields must be non-empty trimmed strings")
        return value


def _factory(request: Request) -> sessionmaker[Session]:
    factory = getattr(request.app.state, "session_factory", None)
    if not isinstance(factory, sessionmaker):
        raise HTTPException(status_code=503, detail="database unavailable")
    return cast(sessionmaker[Session], factory)


def _response(case: Case) -> dict[str, str]:
    return {"id": case.id, "external_key": case.external_key, "title": case.title}


@router.post("")
def post_case(
    payload: CaseCreate,
    request: Request,
    response: Response,
    context: Annotated[MutationContext, Depends(require_case_write)],
) -> dict[str, str]:
    with session_scope(_factory(request)) as session:
        case, created = create_case(
            session,
            owner_id=context.actor_id,
            external_key=payload.external_key,
            title=payload.title,
            request_id=context.request_id,
        )
        result = _response(case)
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return result


@router.get("/{case_id}")
def read_case(
    case_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> dict[str, str]:
    if "case:read" not in principal.permissions:
        raise HTTPException(status_code=403, detail="missing permission: case:read")
    with _factory(request)() as session:
        case = get_case_for_owner(session, case_id=case_id, owner_id=principal.user_id)
        if case is None:
            raise HTTPException(status_code=404, detail="case not found")
        return _response(case)
