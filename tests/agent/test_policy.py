from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from aquaops.agent.policy import (
    PUBLIC_TOOL_REGISTRY,
    ToolBudget,
    ToolInvocation,
    ToolPolicyDecision,
    ToolTrace,
    authorize_public_tool_call,
    consume_issued_public_tool_grant,
    create_public_tool_budget_session,
    issue_public_tool_grant,
    public_tool_budget_snapshot,
)


def _history_arguments() -> dict[str, object]:
    return {
        "indicator": "nh4",
        "hours": 24,
        "end_at": "2025-01-02T00:00:00Z",
    }


def test_public_tool_registry_is_exact_immutable_and_read_only() -> None:
    assert tuple(PUBLIC_TOOL_REGISTRY) == (
        "retrieve_public_water_knowledge",
        "aggregate_public_history",
        "screen_public_anomaly",
    )
    assert all(
        spec.data_class == "public"
        and spec.risk == "read_only"
        and spec.read_only is True
        and spec.max_calls >= 1
        and spec.timeout_seconds >= 1
        for spec in PUBLIC_TOOL_REGISTRY.values()
    )

    with pytest.raises(TypeError):
        PUBLIC_TOOL_REGISTRY["drop_database"] = object()  # type: ignore[index]

    with pytest.raises(FrozenInstanceError):
        PUBLIC_TOOL_REGISTRY["aggregate_public_history"].max_calls = 99  # type: ignore[misc]


def test_policy_rejects_unallowlisted_or_capability_override_without_arguments_echo() -> (
    None
):
    rejected = authorize_public_tool_call(
        "drop_database", {"database": "private"}, ToolBudget.initial()
    )
    assert rejected.allowed is False
    assert rejected.code == "tool_not_allowlisted"
    assert not hasattr(rejected, "arguments")
    assert "private" not in repr(rejected)

    forged = authorize_public_tool_call(
        "aggregate_public_history",
        {**_history_arguments(), "risk": "write", "collection": "private"},
        ToolBudget.initial(),
    )
    assert forged.allowed is False
    assert forged.code == "invalid_tool_request"
    assert "private" not in repr(forged)


@pytest.mark.parametrize(
    "arguments",
    [
        {"indicator": "nh4", "hours": True, "end_at": "2025-01-02T00:00:00Z"},
        {"indicator": "nh4", "hours": 24, "end_at": "bad"},
        {
            "indicator": "nh4",
            "hours": 24,
            "end_at": "2025-01-02T00:00:00Z",
            "url": "https://bad",
        },
    ],
)
def test_policy_rejects_invalid_tool_arguments(arguments: dict[str, object]) -> None:
    decision = authorize_public_tool_call(
        "aggregate_public_history", arguments, ToolBudget.initial()
    )

    assert decision.allowed is False
    assert decision.code == "invalid_tool_request"


def test_tool_budget_is_strict_immutable_and_never_exceeds_system_maximum() -> None:
    with pytest.raises(ValidationError):
        ToolBudget(remaining_total=True, used_by_tool={})
    with pytest.raises(ValidationError):
        ToolBudget(remaining_total=7, used_by_tool={})

    initial = ToolBudget.initial()
    first = authorize_public_tool_call(
        "aggregate_public_history", _history_arguments(), initial
    )
    assert first.allowed is True
    assert first.budget.remaining_total == initial.remaining_total - 1
    assert initial.remaining_total == 6


def test_policy_enforces_total_and_per_tool_budget_without_mutating_prior_snapshot() -> (
    None
):
    budget = ToolBudget.initial()
    for _ in range(PUBLIC_TOOL_REGISTRY["aggregate_public_history"].max_calls):
        decision = authorize_public_tool_call(
            "aggregate_public_history", _history_arguments(), budget
        )
        assert decision.allowed is True
        budget = decision.budget

    exhausted_per_tool = authorize_public_tool_call(
        "aggregate_public_history", _history_arguments(), budget
    )
    assert exhausted_per_tool.allowed is False
    assert exhausted_per_tool.code == "tool_call_limit_reached"
    assert exhausted_per_tool.budget == budget

    for tool_name, arguments in (
        (
            "screen_public_anomaly",
            {
                "indicator": "nh4",
                "baseline_hours": 24,
                "end_at": "2025-01-02T00:00:00Z",
            },
        ),
        ("retrieve_public_water_knowledge", {"query": "公开水质知识"}),
        ("retrieve_public_water_knowledge", {"query": "公开水质知识"}),
        (
            "screen_public_anomaly",
            {
                "indicator": "nh4",
                "baseline_hours": 24,
                "end_at": "2025-01-02T00:00:00Z",
            },
        ),
    ):
        decision = authorize_public_tool_call(tool_name, arguments, budget)
        assert decision.allowed is True
        budget = decision.budget
    assert budget.remaining_total == 0

    rejected = authorize_public_tool_call(
        "screen_public_anomaly",
        {"indicator": "nh4", "baseline_hours": 24, "end_at": "2025-01-02T00:00:00Z"},
        budget,
    )
    assert rejected.allowed is False
    assert rejected.code == "tool_budget_exhausted"


def test_tool_trace_keeps_only_safe_summary_and_rejects_sensitive_payload_shapes() -> (
    None
):
    trace = ToolTrace.create(
        name="aggregate_public_history",
        decision_code="tool_completed",
        evidence_ids=("co-udlabs-wwtp-lpicm-2025:nh4:window",),
        elapsed_seconds=0.24,
        error_code=None,
    )
    assert trace.elapsed_bucket == "under_1s"
    assert trace.evidence_ids == ("co-udlabs-wwtp-lpicm-2025:nh4:window",)
    assert set(trace.model_dump()) == {
        "name",
        "decision_code",
        "evidence_ids",
        "elapsed_bucket",
        "error_code",
    }
    assert "0.24" not in repr(trace)

    with pytest.raises(ValidationError):
        ToolTrace(
            name="aggregate_public_history",
            decision_code="tool_completed",
            evidence_ids=("query=private",),
            elapsed_bucket="under_1s",
            error_code=None,
        )


def test_invocation_strictly_rejects_non_dict_arguments_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ToolInvocation(name="aggregate_public_history", arguments=True)
    with pytest.raises(ValidationError):
        ToolInvocation(
            name="aggregate_public_history",
            arguments=_history_arguments(),
            risk="write",
        )


def test_policy_decision_cannot_be_constructed_with_a_nonfixed_code_or_tool_name() -> (
    None
):
    with pytest.raises(ValidationError):
        ToolPolicyDecision(
            allowed=False,
            code="private_reason",
            tool_name="aggregate_public_history",
            budget=ToolBudget.initial(),
        )
    with pytest.raises(ValidationError):
        ToolPolicyDecision(
            allowed=False,
            code="tool_not_allowlisted",
            tool_name="drop_database",
            budget=ToolBudget.initial(),
        )


def test_only_policy_issued_identity_grant_can_consume_the_matching_budget() -> None:
    session = create_public_tool_budget_session()
    assert public_tool_budget_snapshot(session) is not None
    decision, grant = issue_public_tool_grant(
        "aggregate_public_history", _history_arguments(), session
    )

    assert decision.allowed is True
    assert grant is not None
    consumed = consume_issued_public_tool_grant(
        grant,
        expected_name="aggregate_public_history",
        session=session,
    )
    assert consumed is not None
    _, snapshot = consumed
    assert snapshot == decision.budget
    assert (
        consume_issued_public_tool_grant(
            grant,
            expected_name="aggregate_public_history",
            session=create_public_tool_budget_session(),
        )
        is None
    )


def test_trace_uses_only_fixed_decision_codes_and_safe_factory_drops_bad_ids() -> None:
    with pytest.raises(ValidationError):
        ToolTrace.create(
            name="aggregate_public_history",
            decision_code="write_complete",
            evidence_ids=(),
            elapsed_seconds=1,
            error_code=None,
        )

    trace = ToolTrace.create(
        name="aggregate_public_history",
        decision_code="tool_completed",
        evidence_ids=("query=private", "public:aggregate:1"),
        elapsed_seconds=1,
        error_code=None,
    )
    assert trace.evidence_ids == ("public:aggregate:1",)


def test_only_host_budget_session_can_issue_and_atomically_consume_tool_budget() -> (
    None
):
    session = create_public_tool_budget_session()
    for _ in range(2):
        decision, grant = issue_public_tool_grant(
            "aggregate_public_history", _history_arguments(), session
        )
        assert decision.allowed is True
        assert grant is not None
        snapshot = consume_issued_public_tool_grant(
            grant,
            expected_name="aggregate_public_history",
            session=session,
        )
        assert snapshot is not None
    assert public_tool_budget_snapshot(session).remaining_total == 4

    exhausted, grant = issue_public_tool_grant(
        "aggregate_public_history", _history_arguments(), session
    )
    assert exhausted.allowed is False
    assert exhausted.code == "tool_call_limit_reached"
    assert grant is None

    forged_budget = ToolBudget.initial().model_copy(update={"remaining_total": 0})
    rejected, forged_grant = issue_public_tool_grant(
        "screen_public_anomaly",
        {"indicator": "nh4", "baseline_hours": 24, "end_at": "2025-01-02T00:00:00Z"},
        forged_budget,
    )
    assert rejected.allowed is False
    assert rejected.code == "invalid_tool_budget"
    assert forged_grant is None
