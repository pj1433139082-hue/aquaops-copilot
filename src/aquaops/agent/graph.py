from time import perf_counter
from typing import Mapping, Protocol, TypedDict

from langgraph.graph import END, StateGraph

from aquaops.agent.coverage import (
    ABCPlan,
    CoverageGateResult,
    FixedAgentAnswer,
    coverage_gate,
    mint_verified_public_evidence,
    mint_verified_public_evidence_batch,
    render_fixed_agent_answer,
)
from aquaops.agent.policy import (
    ToolTrace,
    consume_issued_public_tool_grant,
    create_public_tool_budget_session,
    issue_public_tool_grant,
    public_tool_budget_snapshot,
)
from aquaops.agent.public_diagnostics import (
    expected_public_historical_rendering,
    parse_public_anomaly_request,
    parse_public_historical_request,
    render_public_historical_answer,
    run_public_historical_diagnostic,
)
from aquaops.agent.state import AgentState, initialize_safe_agent_state
from aquaops.rag.citations import build_grounded_answer
from aquaops.rag.hybrid import PublicEvidence, RetrievalResult


_EVIDENCE_FIELDS = frozenset({"chunk_id", "source_url", "source_version"})
_GRAPH_CONNECTED_PUBLIC_TOOLS = frozenset(
    {"aggregate_public_history", "screen_public_anomaly"}
)
_LEGACY_ABC_PLAN = ABCPlan.from_requirements(
    a="公开历史趋势或聚合证据",
    b="公开异常筛查证据",
    c="公开知识依据",
)


class PublicKnowledgeRetriever(Protocol):
    def retrieve(self, query: str, answer_limit: int = 6) -> RetrievalResult: ...


class OperationsGraphOutput(TypedDict, total=False):
    evidence: list[dict[str, str]]
    answer: str
    requires_human_review: bool
    tool_budget: object
    tool_traces: tuple[ToolTrace, ...]
    coverage: CoverageGateResult | None


class _OperationsInput(TypedDict, total=False):
    request_id: str
    mode: str
    question: str


def _safe_refusal(state: AgentState) -> AgentState:
    grounded = build_grounded_answer("", [])
    return {
        **state,
        "evidence": grounded.citations,
        "answer": grounded.answer,
        "requires_human_review": grounded.requires_human_review,
    }


def _with_trace(state: AgentState, trace: ToolTrace) -> AgentState:
    existing = state.get("tool_traces", ())
    if type(existing) is not tuple or not all(
        isinstance(item, ToolTrace) for item in existing
    ):
        existing = ()
    return {**state, "tool_traces": (*existing, trace)}


def _diagnostic_tool_request(
    question: str,
) -> tuple[str, dict[str, object]] | None:
    historical = parse_public_historical_request(question)
    if historical is not None:
        indicator, hours, end_at = historical
        return (
            "aggregate_public_history",
            {"indicator": indicator, "hours": hours, "end_at": end_at},
        )
    anomaly = parse_public_anomaly_request(question)
    if anomaly is not None:
        indicator, baseline_hours, end_at = anomaly
        return (
            "screen_public_anomaly",
            {
                "indicator": indicator,
                "baseline_hours": baseline_hours,
                "end_at": end_at,
            },
        )
    if question.startswith("公开知识检索 "):
        return (
            "retrieve_public_water_knowledge",
            {"query": question.removeprefix("公开知识检索 ").strip()},
        )
    return None


def _trace_from_evidence(
    name: str, evidence: list[dict[str, str]], started_at: float
) -> ToolTrace:
    evidence_ids = tuple(
        item.get("chunk_id")
        for item in evidence
        if isinstance(item, dict) and isinstance(item.get("chunk_id"), str)
    )
    return ToolTrace.create(
        name=name,
        decision_code="tool_completed",
        evidence_ids=evidence_ids,
        elapsed_seconds=perf_counter() - started_at,
        error_code=None,
    )


def _failed_trace(name: str, code: str, started_at: float) -> ToolTrace:
    return ToolTrace.create(
        name=name,
        decision_code=code,
        evidence_ids=(),
        elapsed_seconds=perf_counter() - started_at,
        error_code=None,
    )


def _is_valid_rendered_public_history(answer: object, evidence: object) -> bool:
    if not isinstance(answer, str) or not answer.strip():
        return False
    if not isinstance(evidence, list) or not evidence:
        return False
    for item in evidence:
        if not isinstance(item, dict) or set(item) != _EVIDENCE_FIELDS:
            return False
        if not all(
            isinstance(item[field], str) and item[field].strip()
            for field in _EVIDENCE_FIELDS
        ):
            return False
    return True


def build_abc_answer_from_public_evidence(
    *,
    plan: ABCPlan,
    evidence_by_dimension: Mapping[str, object],
    conclusions: Mapping[str, object],
) -> tuple[FixedAgentAnswer, str, CoverageGateResult]:
    """Production callable for the fixed A/B/C answer path.

    The host creates the coverage trace identifier inside ``coverage_gate``;
    callers can provide only public metadata references and dimension-specific
    conclusions.  This function intentionally accepts no model-supplied
    answer, trace identifier, tool policy, or mutable state.
    """

    coverage = coverage_gate(
        plan,
        evidence_by_dimension=evidence_by_dimension,
        conclusions=conclusions,
    )
    answer = FixedAgentAnswer.from_coverage(coverage)
    return answer, render_fixed_agent_answer(answer), coverage


def _empty_legacy_coverage() -> CoverageGateResult:
    return coverage_gate(
        _LEGACY_ABC_PLAN,
        evidence_by_dimension={},
        conclusions={},
    )


def _legacy_coverage_from_result(
    result: object,
    evidence: list[dict[str, str]],
    receipt: object,
) -> CoverageGateResult:
    """Map one deterministic legacy result to only its own A or B dimension."""

    source_kind = getattr(result, "source_kind", None)
    if source_kind not in {"public_historical", "public_historical_anomaly"}:
        return _empty_legacy_coverage()
    verified_evidence = tuple(
        mint_verified_public_evidence(
            receipt=receipt,
            evidence_id=item["chunk_id"],
            source_url=item["source_url"],
            source_version=item["source_version"],
        )
        for item in evidence
    )
    if source_kind == "public_historical":
        return coverage_gate(
            _LEGACY_ABC_PLAN,
            evidence_by_dimension={"A": verified_evidence},
            conclusions={"A": "公开历史聚合证据已完成确定性核验"},
        )
    if source_kind == "public_historical_anomaly":
        return coverage_gate(
            _LEGACY_ABC_PLAN,
            evidence_by_dimension={"B": verified_evidence},
            conclusions={"B": "公开异常筛查证据已完成确定性核验"},
        )
    return _empty_legacy_coverage()


def run_operations_graph(
    state: AgentState,
    *,
    retriever: PublicKnowledgeRetriever | None = None,
) -> AgentState:
    """Return only a rendered, aggregate-only public-history response when valid."""
    safe_state = initialize_safe_agent_state(state)
    budget_session = create_public_tool_budget_session()
    initial_budget = public_tool_budget_snapshot(budget_session)
    if initial_budget is None:
        return _safe_refusal(safe_state)
    safe_state = {
        **safe_state,
        "tool_budget": initial_budget,
        "coverage": _empty_legacy_coverage(),
    }
    if safe_state.get("mode") != "operations":
        return _safe_refusal(safe_state)

    question = safe_state.get("question")
    if not isinstance(question, str) or not question.strip():
        return _safe_refusal(safe_state)

    request = _diagnostic_tool_request(question)
    if request is None:
        return _safe_refusal(safe_state)
    tool_name, arguments = request
    started_at = perf_counter()
    available_tools = (
        _GRAPH_CONNECTED_PUBLIC_TOOLS
        if retriever is None
        else _GRAPH_CONNECTED_PUBLIC_TOOLS | {"retrieve_public_water_knowledge"}
    )
    decision, grant = issue_public_tool_grant(
        tool_name,
        arguments,
        budget_session,
        available_tool_names=frozenset(available_tools),
    )
    if not decision.allowed or grant is None:
        return _safe_refusal(
            _with_trace(safe_state, _failed_trace(tool_name, decision.code, started_at))
        )
    consumed = consume_issued_public_tool_grant(
        grant,
        expected_name=tool_name,
        session=budget_session,
    )
    if consumed is None:
        return _safe_refusal(
            _with_trace(
                safe_state,
                _failed_trace(tool_name, "tool_authorization_invalid", started_at),
            )
        )
    receipt, next_budget = consumed
    safe_state = {**safe_state, "tool_budget": next_budget}

    if tool_name == "retrieve_public_water_knowledge":
        return _run_public_retrieval(
            safe_state,
            arguments,
            receipt,
            retriever,
            started_at,
        )

    try:
        result = run_public_historical_diagnostic(question)
        if result is None or not result.ok:
            return _safe_refusal(
                _with_trace(
                    safe_state,
                    _failed_trace(tool_name, "tool_result_unavailable", started_at),
                )
            )
        rendered = render_public_historical_answer(result)
        expected_rendering = expected_public_historical_rendering(result)
        if not isinstance(rendered, tuple) or len(rendered) != 2:
            return _safe_refusal(
                _with_trace(
                    safe_state,
                    _failed_trace(tool_name, "tool_result_unavailable", started_at),
                )
            )
        answer, evidence = rendered
        if not _is_valid_rendered_public_history(answer, evidence):
            return _safe_refusal(
                _with_trace(
                    safe_state,
                    _failed_trace(tool_name, "tool_result_unavailable", started_at),
                )
            )
        if (answer, evidence) != expected_rendering:
            return _safe_refusal(
                _with_trace(
                    safe_state,
                    _failed_trace(tool_name, "tool_result_unavailable", started_at),
                )
            )
        coverage = _legacy_coverage_from_result(result, evidence, receipt)
    except Exception:
        return _safe_refusal(
            _with_trace(
                safe_state,
                _failed_trace(tool_name, "tool_result_unavailable", started_at),
            )
        )

    return {
        **_with_trace(
            safe_state, _trace_from_evidence(tool_name, evidence, started_at)
        ),
        "evidence": evidence,
        "answer": answer,
        "requires_human_review": True,
        "coverage": coverage,
    }


def _run_public_retrieval(
    state: AgentState,
    arguments: dict[str, object],
    receipt: object,
    retriever: PublicKnowledgeRetriever | None,
    started_at: float,
) -> AgentState:
    if retriever is None or set(arguments) != {"query"}:
        return _safe_refusal(
            _with_trace(
                state,
                _failed_trace(
                    "retrieve_public_water_knowledge", "tool_unavailable", started_at
                ),
            )
        )
    query = arguments["query"]
    if type(query) is not str:
        return _safe_refusal(state)
    try:
        result = retriever.retrieve(query, answer_limit=6)
        evidence = _validated_public_retrieval(result)
        verified = mint_verified_public_evidence_batch(
            receipt=receipt,
            references=tuple(
                (item.chunk_id, item.source_url, item.source_version)
                for item in evidence
            ),
        )
        snippet = evidence[0].text.strip().split("。", 1)[0].strip()[:180]
        conclusion = f"公开检索摘要：{snippet}。"
        fixed_answer, rendered, coverage = build_abc_answer_from_public_evidence(
            plan=_LEGACY_ABC_PLAN,
            evidence_by_dimension={"C": verified},
            conclusions={"C": conclusion},
        )
        if fixed_answer.next_safe_action != "collect_missing_public_evidence":
            raise ValueError("public retrieval coverage boundary is invalid")
        serialized = [
            {
                "chunk_id": item.chunk_id,
                "source_url": item.source_url,
                "source_version": item.source_version,
            }
            for item in evidence
        ]
    except Exception:
        return _safe_refusal(
            _with_trace(
                state,
                _failed_trace(
                    "retrieve_public_water_knowledge",
                    "tool_result_unavailable",
                    started_at,
                ),
            )
        )
    return {
        **_with_trace(
            state,
            _trace_from_evidence(
                "retrieve_public_water_knowledge", serialized, started_at
            ),
        ),
        "evidence": serialized,
        "answer": rendered,
        "requires_human_review": True,
        "coverage": coverage,
    }


def _validated_public_retrieval(result: object) -> tuple[PublicEvidence, ...]:
    if (
        type(result) is not RetrievalResult
        or type(result.evidence) is not tuple
        or not 1 <= len(result.evidence) <= 6
        or type(result.fused_candidate_ids) is not tuple
    ):
        raise ValueError("public retrieval result is invalid")
    seen: set[str] = set()
    for item in result.evidence:
        if (
            type(item) is not PublicEvidence
            or item.data_class != "public"
            or item.access_policy != "public_read"
            or len(item.chunk_id) != 64
            or any(character not in "0123456789abcdef" for character in item.chunk_id)
            or item.chunk_id in seen
            or item.chunk_id not in result.fused_candidate_ids
        ):
            raise ValueError("public retrieval evidence is invalid")
        seen.add(item.chunk_id)
    return result.evidence


def build_operations_graph(
    *, retriever: PublicKnowledgeRetriever | None = None
) -> object:
    def execute(state: AgentState) -> AgentState:
        return run_operations_graph(state, retriever=retriever)

    builder = StateGraph(
        AgentState,
        input_schema=_OperationsInput,
        output_schema=OperationsGraphOutput,
    )
    builder.add_node("operations", execute)
    builder.set_entry_point("operations")
    builder.add_edge("operations", END)
    return builder.compile()


operations_graph = build_operations_graph()
