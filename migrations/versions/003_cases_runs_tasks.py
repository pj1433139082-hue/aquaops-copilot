"""Add cases, agent runs, and durable task records."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "003_cases_runs_tasks"
down_revision = "002_users_roles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "cases",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=36), nullable=False),
        sa.Column("external_key", sa.String(length=80), nullable=False),
        sa.Column("title", sa.String(length=160), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "owner_id", "external_key", name="uq_cases_owner_external_key"
        ),
    )
    op.create_index("ix_cases_owner_id", "cases", ["owner_id"])
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("case_id", sa.String(length=36), nullable=False),
        sa.Column("request_id", sa.String(length=80), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["case_id"], ["cases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_runs_case_id", "agent_runs", ["case_id"])
    op.create_index("ix_agent_runs_request_id", "agent_runs", ["request_id"])
    op.create_table(
        "tasks",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("owner_id", sa.String(length=36), nullable=False),
        sa.Column("agent_run_id", sa.String(length=36), nullable=True),
        sa.Column("state", sa.String(length=20), nullable=False),
        sa.Column("request_id", sa.String(length=80), nullable=False),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("error_class", sa.String(length=120), nullable=True),
        sa.Column("worker_token", sa.String(length=80), nullable=True),
        sa.Column("worker_attempt", sa.Integer(), server_default="-1", nullable=False),
        sa.Column("worker_lease_token", sa.String(length=80), nullable=True),
        sa.Column("worker_lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["agent_run_id"], ["agent_runs.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_tasks_agent_run_id", "tasks", ["agent_run_id"])
    op.create_index("ix_tasks_owner_id", "tasks", ["owner_id"])
    op.create_index("ix_tasks_request_id", "tasks", ["request_id"])


def downgrade() -> None:
    op.drop_index("ix_tasks_request_id", table_name="tasks")
    op.drop_index("ix_tasks_owner_id", table_name="tasks")
    op.drop_index("ix_tasks_agent_run_id", table_name="tasks")
    op.drop_table("tasks")
    op.drop_index("ix_agent_runs_request_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_case_id", table_name="agent_runs")
    op.drop_table("agent_runs")
    op.drop_index("ix_cases_owner_id", table_name="cases")
    op.drop_table("cases")
