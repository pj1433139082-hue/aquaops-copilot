from __future__ import annotations

from pathlib import Path

import jwt
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, inspect, select

from aquaops.api.app import create_app
from aquaops.config import Settings
from aquaops.db.session import create_session_factory, session_scope
from aquaops.domain.models import AuditEvent, Base, Case, Role, User
from aquaops.domain.services import create_case


JWT_SECRET = "synthetic-test-secret-that-is-at-least-32-bytes"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _token(subject: str = "user-1") -> str:
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    return jwt.encode(
        {
            "sub": subject,
            "roles": ["operator"],
            "iss": "aquaops-test",
            "aud": "aquaops-api-test",
            "iat": now,
            "exp": now + timedelta(minutes=5),
        },
        JWT_SECRET,
        algorithm="HS256",
    )


def _settings() -> Settings:
    return Settings(
        jwt_secret=JWT_SECRET,
        jwt_issuer="aquaops-test",
        jwt_audience="aquaops-api-test",
    )


def test_repeated_external_key_returns_existing_case_and_one_audit(
    tmp_path: Path,
) -> None:
    factory = create_session_factory(
        f"sqlite+pysqlite:///{(tmp_path / 'cases.sqlite3').as_posix()}"
    )
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)

    with session_scope(factory) as session:
        first, first_created = create_case(
            session,
            owner_id="user-1",
            external_key="import-001",
            title="Synthetic incident",
            request_id="req-1",
        )
        first_id = first.id

    with session_scope(factory) as session:
        second, second_created = create_case(
            session,
            owner_id="user-1",
            external_key="import-001",
            title="Must not overwrite",
            request_id="req-2",
        )
        second_id = second.id

    with factory() as session:
        audit_count = session.scalar(select(func.count()).select_from(AuditEvent))
        stored = session.get(Case, first_id)

    assert first_created is True
    assert second_created is False
    assert second_id == first_id
    assert audit_count == 1
    assert stored is not None
    assert stored.title == "Synthetic incident"
    engine.dispose()


def test_case_external_key_is_scoped_per_owner(tmp_path: Path) -> None:
    factory = create_session_factory(
        f"sqlite+pysqlite:///{(tmp_path / 'case-owners.sqlite3').as_posix()}"
    )
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)

    with session_scope(factory) as session:
        first, _ = create_case(
            session,
            owner_id="user-1",
            external_key="same-key",
            title="First",
            request_id="req-1",
        )
        second, _ = create_case(
            session,
            owner_id="user-2",
            external_key="same-key",
            title="Second",
            request_id="req-2",
        )
        assert first.id != second.id
    engine.dispose()


def test_case_routes_use_authenticated_owner_and_preserve_idempotency(
    tmp_path: Path,
) -> None:
    factory = create_session_factory(
        f"sqlite+pysqlite:///{(tmp_path / 'case-api.sqlite3').as_posix()}"
    )
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with session_scope(factory) as session:
        session.add(
            User(
                id="user-1",
                username="synthetic-case-user",
                roles=[Role(name="operator")],
            )
        )
    client = TestClient(create_app(settings=_settings(), session_factory=factory))
    headers = {"Authorization": f"Bearer {_token()}", "X-Request-ID": "req-api-1"}
    payload = {
        "external_key": "api-import-1",
        "title": "Synthetic API case",
        "owner_id": "attacker-selected-owner",
    }

    rejected = client.post("/v1/cases", headers=headers, json=payload)
    assert rejected.status_code == 422

    payload.pop("owner_id")
    first = client.post("/v1/cases", headers=headers, json=payload)
    second = client.post("/v1/cases", headers=headers, json=payload)

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert "owner_id" not in first.json()
    case_id = first.json()["id"]
    fetched = client.get(
        f"/v1/cases/{case_id}",
        headers={"Authorization": f"Bearer {_token()}"},
    )
    assert fetched.status_code == 200
    assert fetched.json()["title"] == "Synthetic API case"

    with factory() as session:
        audit = session.scalar(select(AuditEvent))
        assert audit is not None
        assert audit.actor_id == "user-1"
        assert audit.request_id == "req-api-1"
    engine.dispose()


def test_case_task_migration_upgrades_and_downgrades_to_rbac(tmp_path: Path) -> None:
    database_path = tmp_path / "case-task-migration.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    assert {"cases", "agent_runs", "tasks"}.issubset(inspect(engine).get_table_names())
    task_columns = {column["name"] for column in inspect(engine).get_columns("tasks")}
    assert {"owner_id", "state", "request_id", "retry_count", "error_class"}.issubset(
        task_columns
    )
    engine.dispose()

    command.downgrade(config, "002_users_roles")
    verification_engine = create_engine(database_url)
    assert not {"cases", "agent_runs", "tasks"}.intersection(
        inspect(verification_engine).get_table_names()
    )
    assert {"audit_events", "users", "roles", "user_roles"}.issubset(
        inspect(verification_engine).get_table_names()
    )
    verification_engine.dispose()
