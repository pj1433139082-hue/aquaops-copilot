"""Public-only composition root shared by the Agent, API, and MCP demo."""

from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aquaops.agent.graph import build_operations_graph
from aquaops.api.app import create_app
from aquaops.mcp.public_server import (
    PublicMcpDependencies,
    PublicMcpHandlers,
    create_public_handlers,
)
from aquaops.rag.demo_runtime import PublicDemoRuntime, build_public_demo_runtime
from aquaops.rag.demo_evaluation import (
    DemoEvaluationReport,
    evaluate_demo_variants,
    load_verified_demo_evaluation_suite,
)


_PUBLIC_SMOKE_QUESTION = "公开知识检索 紫外消毒效果下降如何排查"
_PUBLIC_SMOKE_QUERY = "紫外消毒效果下降如何排查"


@dataclass(frozen=True)
class PublicDemoComponents:
    runtime: PublicDemoRuntime
    operations_graph: object
    app: FastAPI
    mcp_handlers: PublicMcpHandlers


@dataclass(frozen=True)
class PublicDemoSmokeReport:
    ok: bool
    chunk_count: int
    agent_evidence_count: int
    api_evidence_count: int
    mcp_evidence_count: int
    citation_sets_match: bool
    abc_dimension_count: int
    human_review_required: bool


def compose_public_demo(
    *,
    runtime: PublicDemoRuntime | None = None,
    local_files_only: bool = False,
    reranker_backend: str = "torch",
) -> PublicDemoComponents:
    """Create all public demo entry points around one verified retriever instance."""
    if type(local_files_only) is not bool:
        raise TypeError("local_files_only must be a boolean")
    if runtime is not None and type(runtime) is not PublicDemoRuntime:
        raise TypeError("runtime must be an exact PublicDemoRuntime")
    if runtime is not None and reranker_backend != "torch":
        raise TypeError("an injected runtime cannot select a backend")
    public_runtime = runtime or build_public_demo_runtime(
        local_files_only=local_files_only,
        reranker_backend=reranker_backend,
    )
    operations_graph = build_operations_graph(retriever=public_runtime.retriever)
    app = create_app(operations_graph=operations_graph)
    mcp_handlers = create_public_handlers(
        PublicMcpDependencies(retriever=public_runtime.retriever)
    )
    return PublicDemoComponents(
        runtime=public_runtime,
        operations_graph=operations_graph,
        app=app,
        mcp_handlers=mcp_handlers,
    )


def evaluate_public_demo(components: PublicDemoComponents) -> DemoEvaluationReport:
    """Run all four fixed retrieval variants against the registered real-public suite."""
    if type(components) is not PublicDemoComponents:
        raise TypeError("components must be exact PublicDemoComponents")
    return evaluate_demo_variants(
        dict(components.runtime.variants),
        load_verified_demo_evaluation_suite(),
        k=5,
    )


def smoke_public_demo(components: PublicDemoComponents) -> PublicDemoSmokeReport:
    """Exercise Agent, ASGI API, and direct MCP handler without retaining the query."""
    if type(components) is not PublicDemoComponents:
        raise TypeError("components must be exact PublicDemoComponents")
    agent_state = components.operations_graph.invoke(
        {
            "request_id": "public-demo-smoke-agent",
            "mode": "operations",
            "question": _PUBLIC_SMOKE_QUESTION,
        }
    )
    api_response = TestClient(components.app).post(
        "/v1/agent/runs",
        headers={"X-Request-ID": "public-demo-smoke-api"},
        json={"mode": "operations", "question": _PUBLIC_SMOKE_QUESTION},
    )
    mcp_response = components.mcp_handlers.retrieve_public_water_knowledge(
        _PUBLIC_SMOKE_QUERY,
        limit=6,
    )
    if api_response.status_code != 200 or mcp_response.get("ok") is not True:
        raise RuntimeError("public demo smoke test failed safely")
    api_body = api_response.json()
    agent_evidence = agent_state.get("evidence")
    api_evidence = api_body.get("evidence")
    mcp_evidence = mcp_response.get("evidence")
    coverage = agent_state.get("coverage")
    if (
        type(agent_evidence) is not list
        or type(api_evidence) is not list
        or type(mcp_evidence) is not list
        or coverage is None
    ):
        raise RuntimeError("public demo smoke test failed safely")
    agent_ids = {item["chunk_id"] for item in agent_evidence}
    api_ids = {item["chunk_id"] for item in api_evidence}
    mcp_ids = {item["chunk_id"] for item in mcp_evidence}
    dimensions = getattr(coverage, "dimensions", ())
    human_review = (
        agent_state.get("requires_human_review") is True
        and api_body.get("requires_human_review") is True
    )
    citations_match = agent_ids == api_ids == mcp_ids
    return PublicDemoSmokeReport(
        ok=bool(
            agent_ids and citations_match and len(dimensions) == 3 and human_review
        ),
        chunk_count=components.runtime.chunk_count,
        agent_evidence_count=len(agent_evidence),
        api_evidence_count=len(api_evidence),
        mcp_evidence_count=len(mcp_evidence),
        citation_sets_match=citations_match,
        abc_dimension_count=len(dimensions),
        human_review_required=human_review,
    )
