from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import jwt


RoleResolver = Callable[[str], frozenset[str] | None]


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: str
    roles: frozenset[str]
    permissions: frozenset[str]


@dataclass(frozen=True, slots=True)
class MutationContext:
    actor_id: str
    request_id: str
    principal: Principal


class AuthenticationError(ValueError):
    """Fixed public authentication failure without token details."""


class AuthenticationServiceError(RuntimeError):
    """Authentication infrastructure failed independently of token validity."""


def require_permission(permissions: set[str] | frozenset[str], required: str) -> None:
    if required not in permissions:
        raise PermissionError(required)


def _validated_roles(value: Any) -> frozenset[str]:
    if not isinstance(value, list) or not value or len(value) > 16:
        raise AuthenticationError("invalid access token")
    if any(
        not isinstance(role, str) or not role or role != role.strip() or len(role) > 40
        for role in value
    ):
        raise AuthenticationError("invalid access token")
    return frozenset(value)


def decode_access_token(
    token: str,
    *,
    secret: str,
    issuer: str,
    audience: str,
    role_resolver: RoleResolver,
) -> Principal:
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            issuer=issuer,
            audience=audience,
            options={"require": ["sub", "roles", "iss", "aud", "iat", "exp"]},
        )
        subject = claims["sub"]
        if (
            not isinstance(subject, str)
            or not subject
            or subject != subject.strip()
            or len(subject) > 36
        ):
            raise AuthenticationError("invalid access token")
        claimed_roles = _validated_roles(claims["roles"])
        if not claimed_roles.issubset(ROLE_PERMISSIONS):
            raise AuthenticationError("invalid access token")
        try:
            current_roles = role_resolver(subject)
        except Exception as exc:
            raise AuthenticationServiceError(
                "authentication service unavailable"
            ) from exc
        if (
            not isinstance(current_roles, frozenset)
            or not current_roles
            or not current_roles.issubset(ROLE_PERMISSIONS)
            or current_roles != claimed_roles
        ):
            raise AuthenticationError("invalid access token")
        permissions = resolve_role_permissions(current_roles)
    except AuthenticationError:
        raise
    except AuthenticationServiceError:
        raise
    except Exception as exc:
        raise AuthenticationError("invalid access token") from exc

    return Principal(
        user_id=subject,
        roles=current_roles,
        permissions=permissions,
    )


ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "viewer": frozenset({"case:read", "task:read"}),
    "operator": frozenset({"case:read", "case:write", "task:read"}),
    "admin": frozenset({"case:read", "case:write", "task:read", "admin:write"}),
}


def resolve_role_permissions(roles: Iterable[str]) -> frozenset[str]:
    permissions: set[str] = set()
    for role in roles:
        permissions.update(ROLE_PERMISSIONS.get(role, frozenset()))
    return frozenset(permissions)
