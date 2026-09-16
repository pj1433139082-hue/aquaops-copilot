from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

from aquaops.agent.state import AgentRunCompleted, InMemoryAgentRunStore
from aquaops.agent.policy import ToolBudget
from aquaops.agent.state import initialize_safe_agent_state


def _completed_result() -> AgentRunCompleted:
    return AgentRunCompleted(
        run_id=uuid4(),
        request_id="trace-001",
        status="completed",
        answer="信息不足，无法基于已公开证据给出建议。",
        evidence=[],
        requires_human_review=True,
    )


def test_in_memory_run_store_evicts_the_oldest_result_at_capacity() -> None:
    store = InMemoryAgentRunStore(max_entries=2)
    first = _completed_result()
    second = _completed_result()
    third = _completed_result()

    store.save(first)
    store.save(second)
    store.save(third)

    assert store.get(first.run_id) is None
    assert store.get(second.run_id) == second
    assert store.get(third.run_id) == third


def test_in_memory_run_store_uses_the_production_default_capacity_of_100() -> None:
    store = InMemoryAgentRunStore()
    results = [_completed_result() for _ in range(101)]

    for result in results:
        assert store.save(result) is True

    assert store.get(results[0].run_id) is None
    assert store.get(results[-1].run_id) == results[-1]
    assert sum(store.get(result.run_id) is not None for result in results) == 100


def test_in_memory_run_store_handles_concurrent_save_and_get_without_corruption() -> (
    None
):
    capacity = 5
    results = [_completed_result() for _ in range(10)]
    store = InMemoryAgentRunStore(max_entries=capacity)
    start = Barrier(len(results))

    def save_and_read(
        result: AgentRunCompleted,
    ) -> tuple[bool, AgentRunCompleted | None]:
        start.wait(timeout=5)
        return store.save(result), store.get(result.run_id)

    with ThreadPoolExecutor(max_workers=len(results)) as executor:
        outcomes = list(executor.map(save_and_read, results))

    assert all(saved for saved, _ in outcomes)
    assert all(
        stored is None or stored.run_id in {result.run_id for result in results}
        for _, stored in outcomes
    )
    assert sum(store.get(result.run_id) is not None for result in results) == capacity


def test_initialize_safe_agent_state_ignores_model_supplied_budget_trace_and_unknown_fields() -> (
    None
):
    initialized = initialize_safe_agent_state(
        {
            "request_id": "safe-state",
            "mode": "operations",
            "question": "公开历史诊断 氨氮 24h 截止 2025-01-02T00:00:00Z",
            "tool_budget": ToolBudget.initial(),
            "tool_traces": ("forged",),
            "private_payload": {"values": [1]},
        }
    )

    assert initialized["tool_budget"] == ToolBudget.initial()
    assert initialized["tool_traces"] == ()
    assert initialized["coverage"] is None
    assert "private_payload" not in initialized
