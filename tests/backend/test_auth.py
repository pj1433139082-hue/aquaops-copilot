from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import jwt
import pytest
from alembic import command
from alembic.config import Config
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect

from aquaops.config import Settings
from aquaops.api.app import create_app
from aquaops.api.middleware import request_id_middleware
from aquaops.db.session import create_session_factory, session_scope
from aquaops.domain.models import Base, Role, User
from aquaops.security.auth import (
    AuthenticationError,
    AuthenticationServiceError,
    Principal,
    decode_access_token,
    require_permission,
)
from aquaops.security.dependencies import get_current_principal, require_case_write


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JWT_SECRET = "synthetic-test-secret-that-is-at-least-32-bytes"
JWT_ISSUER = "aquaops-test"
JWT_AUDIENCE = "aquaops-api-test"


def _token(
    *,
    subject: str = "user-1",
    roles: list[str] | None = None,
    expires_delta: timedelta = timedelta(minutes=5),
    secret: str = JWT_SECRET,
    **extra_claims: object,
) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "roles": roles or ["viewer"],
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "iat": now,
        "exp": now + expires_delta,
        **extra_claims,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def _settings() -> Settings:
    return Settings(
        jwt_secret=JWT_SECRET,
        jwt_issuer=JWT_ISSUER,
        jwt_audience=JWT_AUDIENCE,
    )


def test_viewer_cannot_create_cases() -> None:
    with pytest.raises(PermissionError, match="case:write"):
        require_permission(frozenset({"case:read"}), "case:write")


def test_operator_can_create_cases() -> None:
    require_permission(
        frozenset({"case:read", "case:write"}),
        "case:write",
    )


def test_token_permissions_are_resolved_server_side() -> None:
    token = _token(permissions=["case:write", "admin"])

    principal = decode_access_token(
        token,
        secret=JWT_SECRET,
        issuer=JWT_ISSUER,
        audience=JWT_AUDIENCE,
        role_resolver=lambda user_id: (
            frozenset({"viewer"}) if user_id == "user-1" else None
        ),
    )

    assert principal == Principal(
        user_id="user-1",
        roles=frozenset({"viewer"}),
        permissions=frozenset({"case:read", "task:read"}),
    )
    assert "case:write" not in principal.permissions


@pytest.mark.parametrize(
    "token",
    [
        pytest.param(_token(expires_delta=timedelta(seconds=-1)), id="expired"),
        pytest.param(
            _token(secret="wrong-secret-that-is-also-at-least-32-bytes"),
            id="bad-signature",
        ),
        pytest.param(_token(subject=" "), id="blank-subject"),
    ],
)
def test_invalid_access_token_is_rejected(token: str) -> None:
    with pytest.raises(AuthenticationError, match="invalid access token"):
        decode_access_token(
            token,
            secret=JWT_SECRET,
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            role_resolver=lambda _: frozenset({"viewer"}),
        )


def test_unknown_role_is_rejected_even_when_token_is_validly_signed() -> None:
    with pytest.raises(AuthenticationError, match="invalid access token"):
        decode_access_token(
            _token(roles=["unknown-role"]),
            secret=JWT_SECRET,
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            role_resolver=lambda _: frozenset({"unknown-role"}),
        )


def test_role_resolver_failure_is_not_reported_as_an_invalid_token() -> None:
    def unavailable(_: str) -> frozenset[str]:
        raise RuntimeError("synthetic database outage")

    with pytest.raises(
        AuthenticationServiceError, match="authentication service unavailable"
    ):
        decode_access_token(
            _token(),
            secret=JWT_SECRET,
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            role_resolver=unavailable,
        )


def test_fastapi_permission_dependency_uses_authenticated_actor_only() -> None:
    app = FastAPI()
    app.state.settings = _settings()
    current_roles = {
        "user-1": frozenset({"operator"}),
        "user-2": frozenset({"viewer"}),
    }
    app.state.role_resolver = lambda user_id: current_roles.get(user_id)
    app.middleware("http")(request_id_middleware)

    @app.get("/me")
    def me(principal: Principal = Depends(get_current_principal)) -> dict[str, str]:
        return {"actor_id": principal.user_id}

    @app.post("/cases")
    def create_case(
        context=Depends(require_case_write),
    ) -> dict[str, str]:
        return {"actor_id": context.actor_id, "request_id": context.request_id}

    client = TestClient(app)
    operator_headers = {"Authorization": f"Bearer {_token(roles=['operator'])}"}
    viewer_headers = {
        "Authorization": f"Bearer {_token(subject='user-2', roles=['viewer'])}"
    }

    assert client.get("/me", headers=operator_headers).json() == {"actor_id": "user-1"}
    allowed = client.post(
        "/cases",
        headers=operator_headers,
        json={"actor_id": "attacker-selected-user"},
    )
    assert allowed.status_code == 200
    assert allowed.json()["actor_id"] == "user-1"
    assert isinstance(allowed.json()["request_id"], str)
    assert client.post("/cases", headers=viewer_headers).status_code == 403
    assert client.post("/cases").status_code == 401

    current_roles["user-1"] = frozenset({"viewer"})
    assert client.post("/cases", headers=operator_headers).status_code == 401


def test_database_role_revocation_invalidates_an_existing_operator_token(
    tmp_path: Path,
) -> None:
    factory = create_session_factory(
        f"sqlite+pysqlite:///{(tmp_path / 'auth.sqlite3').as_posix()}"
    )
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        operator = Role(name="operator")
        viewer = Role(name="viewer")
        user = User(id="user-1", username="synthetic-user", roles=[operator])
        session.add_all([operator, viewer, user])

    client = TestClient(create_app(settings=_settings(), session_factory=factory))
    headers = {"Authorization": f"Bearer {_token(roles=['operator'])}"}
    assert client.get("/v1/auth/me", headers=headers).json()["permissions"] == [
        "case:read",
        "case:write",
        "task:read",
    ]

    with session_scope(factory) as session:
        user = session.get(User, "user-1")
        assert user is not None
        viewer = next(
            role for role in session.query(Role).all() if role.name == "viewer"
        )
        user.roles = [viewer]

    revoked = client.get("/v1/auth/me", headers=headers)
    assert revoked.status_code == 401
    assert revoked.json() == {"detail": "invalid access token"}
    engine.dispose()


def test_aquaops_auth_me_route_returns_only_server_resolved_identity() -> None:
    client = TestClient(
        create_app(
            settings=_settings(),
            role_resolver=lambda user_id: (
                frozenset({"operator"}) if user_id == "user-1" else None
            ),
        )
    )
    response = client.get(
        "/v1/auth/me",
        headers={
            "Authorization": f"Bearer {_token(roles=['operator'], permissions=['admin'])}"
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "user-1",
        "roles": ["operator"],
        "permissions": ["case:read", "case:write", "task:read"],
    }


def test_rbac_migration_upgrades_and_downgrades_without_dropping_audit(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "rbac-migration.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    assert {"users", "roles", "user_roles", "audit_events"}.issubset(
        inspect(engine).get_table_names()
    )
    engine.dispose()

    command.downgrade(config, "001_initial_domain")
    verification_engine = create_engine(database_url)
    assert "audit_events" in inspect(verification_engine).get_table_names()
    assert not {"users", "roles", "user_roles"}.intersection(
        inspect(verification_engine).get_table_names()
    )
    verification_engine.dispose()
