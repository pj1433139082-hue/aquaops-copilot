from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.engine import make_url


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POSTGRES_URL = os.getenv("AQUAOPS_TEST_POSTGRES_URL")
DESTRUCTIVE_TEST_ALLOWED = os.getenv("AQUAOPS_ALLOW_DESTRUCTIVE_MIGRATION_TEST") == "1"


@pytest.mark.skipif(
    POSTGRES_URL is None or not DESTRUCTIVE_TEST_ALLOWED,
    reason=(
        "set an isolated AQUAOPS_TEST_POSTGRES_URL and "
        "AQUAOPS_ALLOW_DESTRUCTIVE_MIGRATION_TEST=1"
    ),
)
def test_postgres_migration_enforces_immutable_audit_and_downgrades() -> None:
    assert POSTGRES_URL is not None
    database_name = make_url(POSTGRES_URL).database
    assert database_name is not None
    assert database_name.startswith("aquaops_task1_test_")

    preflight_engine = create_engine(POSTGRES_URL)
    assert inspect(preflight_engine).get_table_names() == []
    with preflight_engine.connect() as connection:
        existing_function_count = connection.scalar(
            text(
                "SELECT count(*) FROM pg_proc "
                "WHERE proname = 'aquaops_reject_audit_event_mutation'"
            )
        )
    assert existing_function_count == 0
    preflight_engine.dispose()

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", POSTGRES_URL)

    command.upgrade(config, "001_initial_domain")
    engine = create_engine(POSTGRES_URL)
    assert {"alembic_version", "audit_events"} == set(inspect(engine).get_table_names())

    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO audit_events "
                "(id, action, request_id, actor_id, entity_type, entity_id) "
                "VALUES ('event-1', 'case.created', 'req-1', 'user-1', "
                "'case', 'case-1')"
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
    verification_engine = create_engine(POSTGRES_URL)
    assert "audit_events" not in inspect(verification_engine).get_table_names()
    with verification_engine.connect() as connection:
        function_count = connection.scalar(
            text(
                "SELECT count(*) FROM pg_proc "
                "WHERE proname = 'aquaops_reject_audit_event_mutation'"
            )
        )
    assert function_count == 0
    verification_engine.dispose()
