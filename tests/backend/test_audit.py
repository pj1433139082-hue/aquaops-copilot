from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from alembic import command
from alembic.config import Config
from pydantic import ValidationError
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import DBAPIError

from aquaops.db.session import create_session_factory, session_scope
from aquaops.domain.models import AuditEvent, Base, ImmutableAuditEventError
from aquaops.domain.schemas import AuditEventCreate
from aquaops.api.app import create_app
from aquaops.config import Settings
from aquaops.observability.logging import event_payload
from aquaops.observability.logging import latency_bucket_for
from aquaops.api.routes.ready import default_readiness_probes


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _event() -> AuditEvent:
    return AuditEvent(
        action="case.created",
        request_id="req-1",
        actor_id="user-1",
        entity_type="case",
        entity_id="case-1",
    )


def test_audit_event_keeps_request_id_and_actor_id_after_persistence() -> None:
    factory = create_session_factory("sqlite+pysqlite:///:memory:")
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)

    with session_scope(factory) as session:
        event = _event()
        session.add(event)
        session.flush()
        event_id = event.id

    with factory() as session:
        stored = session.scalar(select(AuditEvent).where(AuditEvent.id == event_id))

    assert stored is not None
    assert stored.request_id == "req-1"
    assert stored.actor_id == "user-1"
    assert stored.created_at is not None
    engine.dispose()


@pytest.mark.parametrize("operation", ["update", "delete"])
def test_persisted_audit_event_rejects_orm_mutation(operation: str) -> None:
    factory = create_session_factory("sqlite+pysqlite:///:memory:")
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)

    with session_scope(factory) as session:
        event = _event()
        session.add(event)
        session.flush()
        event_id = event.id

    with pytest.raises(ImmutableAuditEventError, match="immutable"):
        with session_scope(factory) as session:
            stored = session.get(AuditEvent, event_id)
            assert stored is not None
            if operation == "update":
                stored.action = "case.deleted"
            else:
                session.delete(stored)

    with factory() as session:
        stored = session.get(AuditEvent, event_id)
        assert stored is not None
        assert stored.action == "case.created"
    engine.dispose()


def test_session_scope_rolls_back_and_closes_on_failure() -> None:
    factory = create_session_factory("sqlite+pysqlite:///:memory:")
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)

    tracked_session = factory()
    tracked_session.close = Mock(wraps=tracked_session.close)  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="synthetic failure"):
        with session_scope(lambda: tracked_session) as session:  # type: ignore[arg-type]
            session.add(_event())
            session.flush()
            raise RuntimeError("synthetic failure")

    tracked_session.close.assert_called_once()

    with factory() as session:
        assert session.scalar(select(AuditEvent)) is None
    engine.dispose()


def test_audit_event_input_rejects_blank_fields() -> None:
    with pytest.raises(ValidationError):
        AuditEventCreate(
            action="case.created",
            request_id=" ",
            actor_id="user-1",
            entity_type="case",
            entity_id="case-1",
        )


def test_audit_event_input_is_strict_and_rejects_unknown_fields() -> None:
    valid = AuditEventCreate(
        action="case.created",
        request_id="req-1",
        actor_id="user-1",
        entity_type="case",
        entity_id="case-1",
    )

    assert valid.to_model().request_id == "req-1"

    with pytest.raises(ValidationError):
        AuditEventCreate(
            action="case.created",
            request_id="req-1",
            actor_id="user-1",
            entity_type="case",
            entity_id="case-1",
            raw_payload="must-not-be-stored",  # type: ignore[call-arg]
        )


def test_initial_migration_is_reversible_and_database_enforces_immutability(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "audit-migration.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    assert "audit_events" in inspect(engine).get_table_names()

    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO audit_events "
                "(id, action, request_id, actor_id, entity_type, entity_id, created_at) "
                "VALUES ('event-1', 'case.created', 'req-1', 'user-1', "
                "'case', 'case-1', CURRENT_TIMESTAMP)"
            )
        )

    with pytest.raises(DBAPIError, match="immutable"):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE audit_events SET action = 'case.deleted' "
                    "WHERE id = 'event-1'"
                )
            )

    with pytest.raises(DBAPIError, match="immutable"):
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM audit_events WHERE id = 'event-1'"))

    engine.dispose()
    command.downgrade(config, "base")
    verification_engine = create_engine(database_url)
    assert "audit_events" not in inspect(verification_engine).get_table_names()
    verification_engine.dispose()


def test_event_payload_contains_request_run_and_data_versions() -> None:
    payload = event_payload(
        "tool.finished",
        request_id="req-1",
        run_id="run-1",
        data_version="synthetic-v1",
    )
    assert payload == {
        "event": "tool.finished",
        "request_id": "req-1",
        "run_id": "run-1",
        "data_version": "synthetic-v1",
    }


def test_event_payload_rejects_unknown_or_sensitive_fields() -> None:
    with pytest.raises(TypeError):
        event_payload(
            "tool.failed",
            request_id="req-1",
            raw_exception="must not be logged",  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    ("elapsed_ms", "expected"),
    [
        (99, "under_100ms"),
        (100, "under_500ms"),
        (500, "under_2s"),
        (2_000, "over_2s"),
    ],
)
def test_latency_bucket_has_fixed_boundaries(elapsed_ms: float, expected: str) -> None:
    assert latency_bucket_for(elapsed_ms) == expected


def test_request_middleware_emits_fixed_redacted_completion_event() -> None:
    logged: list[dict[str, object]] = []
    app = create_app(event_logger=logged.append)
    with patch("aquaops.api.middleware.perf_counter", side_effect=[10.0, 10.125]):
        response = TestClient(app).get(
            "/health?raw_prompt=canary-secret",
            headers={
                "X-Request-ID": "req-observe-1",
                "Authorization": "Bearer canary-secret",
            },
        )

    assert response.status_code == 200
    assert logged == [
        {
            "event": "http.request_finished",
            "request_id": "req-observe-1",
            "latency_bucket": "under_500ms",
        }
    ]
    assert "canary" not in str(logged)


def test_observability_sink_failure_cannot_change_health_response() -> None:
    def broken_logger(_: dict[str, object]) -> None:
        raise RuntimeError("synthetic logging failure")

    response = TestClient(create_app(event_logger=broken_logger)).get("/health")

    assert response.status_code == 200


def test_invalid_request_id_is_replaced_before_response_and_log() -> None:
    logged: list[dict[str, object]] = []
    response = TestClient(create_app(event_logger=logged.append)).get(
        "/health", headers={"X-Request-ID": "canary secret\r\nforged: yes"}
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "canary secret\r\nforged: yes"
    assert "canary" not in str(logged)


@pytest.mark.parametrize("elapsed_ms", [-1.0, float("nan")])
def test_latency_bucket_rejects_invalid_measurements(elapsed_ms: float) -> None:
    with pytest.raises(ValueError, match="latency"):
        latency_bucket_for(elapsed_ms)


def test_production_rejects_default_jwt_secret() -> None:
    with pytest.raises(ValidationError, match="jwt_secret"):
        Settings(
            environment="production",
            database_url="postgresql+psycopg://postgres:5432/aquaops",
            _env_file=None,
        )


def test_default_test_settings_have_no_embedded_credentials() -> None:
    settings = Settings(_env_file=None)

    assert settings.database_url == "sqlite+pysqlite:///:memory:"
    assert settings.jwt_secret == ""


def test_settings_do_not_implicitly_load_a_project_dotenv() -> None:
    assert Settings.model_config.get("env_file") is None


@pytest.mark.parametrize("environment", ["development", "production"])
def test_deployed_environments_require_an_external_database_url(
    environment: str,
) -> None:
    with pytest.raises(ValidationError, match="database_url"):
        Settings(
            environment=environment,
            database_url="",
            jwt_secret="synthetic-test-secret-that-is-at-least-32-bytes",
            _env_file=None,
        )


@pytest.mark.parametrize("environment", ["development", "production"])
def test_deployed_environments_accept_external_credentials(environment: str) -> None:
    settings = Settings(
        environment=environment,
        database_url="postgresql+psycopg://postgres:5432/aquaops",
        jwt_secret="x" * 32,
        _env_file=None,
    )

    assert settings.environment == environment
    assert settings.database_url == "postgresql+psycopg://postgres:5432/aquaops"


@pytest.mark.parametrize("environment", ["PRODUCTION", " production "])
def test_production_environment_is_normalized_and_rejects_weak_secret(
    environment: str,
) -> None:
    with pytest.raises(ValidationError, match="jwt_secret"):
        Settings(
            environment=environment,
            database_url="postgresql+psycopg://postgres:5432/aquaops",
            jwt_secret="x" * 31,
            _env_file=None,
        )


def test_default_application_wires_real_readiness_probe_factory() -> None:
    with patch(
        "aquaops.api.app.default_readiness_probes",
        return_value={
            "database": lambda: True,
            "redis": lambda: True,
            "qdrant": lambda: True,
        },
    ) as probe_factory:
        response = TestClient(create_app()).get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    probe_factory.assert_called_once()


def test_default_application_wires_database_session_factory_from_settings() -> None:
    factory = MagicMock()
    with (
        patch(
            "aquaops.api.app.default_readiness_probes",
            return_value={
                "database": lambda: True,
                "redis": lambda: True,
                "qdrant": lambda: True,
            },
        ),
        patch("aquaops.api.app.create_session_factory", return_value=factory) as create,
    ):
        app = create_app(Settings(database_url="sqlite+pysqlite:///:memory:"))

    create.assert_called_once_with("sqlite+pysqlite:///:memory:")
    assert app.state.session_factory is factory
    assert callable(app.state.role_resolver)


def test_database_readiness_uses_bounded_connection_and_closes_engine() -> None:
    settings = Settings()
    engine = MagicMock()
    connection = MagicMock()
    connection.scalar.return_value = 1
    engine.connect.return_value.__enter__.return_value = connection

    with patch("aquaops.api.routes.ready.create_engine", return_value=engine) as create:
        probe = default_readiness_probes(settings)["database"]
        assert probe() is True

    assert create.call_args.kwargs["connect_args"] == {"connect_timeout": 1}
    engine.dispose.assert_called_once()


def test_ready_reports_dependencies_separately_without_raw_errors() -> None:
    probes = {
        "database": lambda: True,
        "redis": lambda: False,
        "qdrant": lambda: (_ for _ in ()).throw(RuntimeError("secret endpoint")),
    }
    client = TestClient(create_app(readiness_probes=probes))

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "dependencies": {
            "database": "ok",
            "redis": "unavailable",
            "qdrant": "unavailable",
        },
    }
    assert "secret" not in response.text
    assert client.get("/health").status_code == 200
