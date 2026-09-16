"""Create the immutable audit-event foundation."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "001_initial_domain"
down_revision = None
branch_labels = None
depends_on = None


def _create_immutability_guards() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "sqlite":
        op.execute(
            """
            CREATE TRIGGER audit_events_no_update
            BEFORE UPDATE ON audit_events
            BEGIN
              SELECT RAISE(ABORT, 'audit_events are immutable');
            END
            """
        )
        op.execute(
            """
            CREATE TRIGGER audit_events_no_delete
            BEFORE DELETE ON audit_events
            BEGIN
              SELECT RAISE(ABORT, 'audit_events are immutable');
            END
            """
        )
    elif dialect == "postgresql":
        op.execute(
            """
            CREATE FUNCTION aquaops_reject_audit_event_mutation()
            RETURNS trigger AS $$
            BEGIN
              RAISE EXCEPTION 'audit_events are immutable';
            END;
            $$ LANGUAGE plpgsql
            """
        )
        op.execute(
            """
            CREATE TRIGGER audit_events_no_update_or_delete
            BEFORE UPDATE OR DELETE ON audit_events
            FOR EACH ROW EXECUTE FUNCTION aquaops_reject_audit_event_mutation()
            """
        )
    else:
        raise RuntimeError(f"unsupported database dialect: {dialect}")


def _drop_immutability_guards() -> None:
    dialect = op.get_bind().dialect.name
    if dialect == "sqlite":
        op.execute("DROP TRIGGER IF EXISTS audit_events_no_update")
        op.execute("DROP TRIGGER IF EXISTS audit_events_no_delete")
    elif dialect == "postgresql":
        op.execute(
            "DROP TRIGGER IF EXISTS audit_events_no_update_or_delete ON audit_events"
        )
        op.execute("DROP FUNCTION IF EXISTS aquaops_reject_audit_event_mutation()")
    else:
        raise RuntimeError(f"unsupported database dialect: {dialect}")


def upgrade() -> None:
    op.create_table(
        "audit_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("action", sa.String(length=80), nullable=False),
        sa.Column("request_id", sa.String(length=80), nullable=False),
        sa.Column("actor_id", sa.String(length=36), nullable=False),
        sa.Column("entity_type", sa.String(length=40), nullable=False),
        sa.Column("entity_id", sa.String(length=36), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_events_actor_id", "audit_events", ["actor_id"])
    op.create_index("ix_audit_events_request_id", "audit_events", ["request_id"])
    _create_immutability_guards()


def downgrade() -> None:
    _drop_immutability_guards()
    op.drop_index("ix_audit_events_request_id", table_name="audit_events")
    op.drop_index("ix_audit_events_actor_id", table_name="audit_events")
    op.drop_table("audit_events")
