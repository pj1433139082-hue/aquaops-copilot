from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from aquaops.config import Settings
from aquaops.security.auth import (
    AuthenticationError,
    AuthenticationServiceError,
    MutationContext,
    Principal,
    RoleResolver,
    decode_access_token,
    require_permission,
)


_bearer = HTTPBearer(auto_error=False)


def get_current_principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid access token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    settings = cast(Settings, request.app.state.settings)
    resolver = cast(RoleResolver, getattr(request.app.state, "role_resolver", None))
    if resolver is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="authentication service unavailable",
        )
    try:
        return decode_access_token(
            credentials.credentials,
            secret=settings.jwt_secret,
            issuer=settings.jwt_issuer,
            audience=settings.jwt_audience,
            role_resolver=resolver,
        )
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid access token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except AuthenticationServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="authentication service unavailable",
        ) from exc


def require_case_write(
    request: Request,
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> MutationContext:
    try:
        require_permission(principal.permissions, "case:write")
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="missing permission: case:write",
        ) from exc
    return MutationContext(
        actor_id=principal.user_id,
        request_id=request.state.request_id,
        principal=principal,
    )
