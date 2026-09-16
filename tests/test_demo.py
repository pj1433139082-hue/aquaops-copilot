import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from aquaops.demo import (
    PublicDemoComponents,
    compose_public_demo,
    evaluate_public_demo,
    smoke_public_demo,
)
from aquaops.rag.demo_runtime import build_public_demo_runtime


class _Encoder:
    def encode_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._vector(text) for text in texts)

    def encode_query(self, query: str) -> list[float]:
        return list(self._vector(query))

    @staticmethod
    def _vector(text: str) -> tuple[float, ...]:
        vector = [0.0] * 1024
        vector[0] = 1.0 if "紫外" in text else 0.1
        vector[1] = 1.0 if "膜" in text else 0.1
        return tuple(vector)


class _Reranker:
    def rerank(self, query: str, candidates: list[object]) -> list[object]:
        return sorted(candidates, key=lambda item: "紫外" in item.text, reverse=True)


def _components() -> PublicDemoComponents:
    runtime = build_public_demo_runtime(
        client=QdrantClient(":memory:"),
        encoder=_Encoder(),
        reranker=_Reranker(),
    )
    return compose_public_demo(runtime=runtime)


def test_one_public_runtime_is_shared_by_agent_api_and_mcp() -> None:
    components = _components()
    question = "公开知识检索 紫外消毒效果下降如何排查"

    agent_state = components.operations_graph.invoke(
        {"request_id": "agent-demo", "mode": "operations", "question": question}
    )
    api_response = TestClient(components.app).post(
        "/v1/agent/runs",
        headers={"X-Request-ID": "api-demo"},
        json={"mode": "operations", "question": question},
    )
    mcp_response = components.mcp_handlers.retrieve_public_water_knowledge(
        "紫外消毒效果下降如何排查", limit=4
    )

    assert api_response.status_code == 200
    assert agent_state["evidence"]
    assert api_response.json()["evidence"]
    assert mcp_response["ok"] is True
    agent_ids = {item["chunk_id"] for item in agent_state["evidence"]}
    api_ids = {item["chunk_id"] for item in api_response.json()["evidence"]}
    mcp_ids = {item["chunk_id"] for item in mcp_response["evidence"]}
    assert agent_ids == api_ids
    assert mcp_ids <= agent_ids
    assert "人工复核：需要" in agent_state["answer"]


def test_public_demo_composition_has_no_private_module_dependency() -> None:
    module_path = Path(__file__).resolve().parents[1] / "src" / "aquaops" / "demo.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert all("private" not in module for module in imported)


def test_public_demo_evaluation_and_smoke_reports_are_aggregate_only() -> None:
    components = _components()

    evaluation = evaluate_public_demo(components)
    smoke = smoke_public_demo(components)

    assert len(evaluation.variants) == 4
    assert evaluation.case_count == 56
    assert smoke.ok is True
    assert smoke.chunk_count == 24
    assert smoke.agent_evidence_count >= 1
    assert smoke.api_evidence_count >= 1
    assert smoke.mcp_evidence_count >= 1
    assert smoke.citation_sets_match is True
    assert smoke.abc_dimension_count == 3
    assert smoke.human_review_required is True
    serialized = f"{evaluation!r}\n{smoke!r}".casefold()
    assert "公开知识检索" not in serialized
    assert "source_url" not in serialized
    assert "chunk_id" not in serialized
    assert "path" not in serialized


def test_injected_runtime_cannot_silently_ignore_reranker_backend_selection() -> None:
    runtime = build_public_demo_runtime(
        client=QdrantClient(":memory:"),
        encoder=_Encoder(),
        reranker=_Reranker(),
    )

    with pytest.raises(TypeError):
        compose_public_demo(runtime=runtime, reranker_backend="openvino")
