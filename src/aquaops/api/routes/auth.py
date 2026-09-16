from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from aquaops.security.auth import Principal
from aquaops.security.dependencies import get_current_principal


router = APIRouter(prefix="/v1/auth", tags=["auth"])


@router.get("/me")
def read_current_principal(
    principal: Annotated[Principal, Depends(get_current_principal)],
) -> dict[str, object]:
    return {
        "user_id": principal.user_id,
        "roles": sorted(principal.roles),
        "permissions": sorted(principal.permissions),
    }
