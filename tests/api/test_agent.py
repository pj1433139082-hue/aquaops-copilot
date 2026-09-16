from uuid import UUID

from fastapi.testclient import TestClient

from aquaops.api.app import create_app


class _InjectedOperationsGraph:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def invoke(self, state: dict[str, str]) -> dict[str, object]:
        self.calls.append(state)
        return {
            "answer": "A. 公开证据\nB. 引用来源\nC. 安全边界\n人工复核：需要",
            "evidence": [
                {
                    "chunk_id": "a" * 64,
                    "source_url": "https://epa.gov/public-guidance",
                    "source_version": "EPA-v1",
                }
            ],
            "requires_human_review": True,
        }


def test_create_app_uses_only_the_explicitly_injected_operations_graph() -> None:
    graph = _InjectedOperationsGraph()
    client = TestClient(create_app(operations_graph=graph))

    response = client.post(
        "/v1/agent/runs",
        headers={"X-Request-ID": "public-rag-api-demo"},
        json={
            "mode": "operations",
            "question": "公开知识检索 紫外消毒如何排查",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert UUID(body["run_id"])
    assert body["request_id"] == "public-rag-api-demo"
    assert body["requires_human_review"] is True
    assert body["evidence"][0]["chunk_id"] == "a" * 64
    assert graph.calls == [
        {
            "request_id": "public-rag-api-demo",
            "mode": "operations",
            "question": "公开知识检索 紫外消毒如何排查",
        }
    ]


def test_research_mode_never_calls_injected_operations_graph() -> None:
    graph = _InjectedOperationsGraph()
    response = TestClient(create_app(operations_graph=graph)).post(
        "/v1/agent/runs",
        json={"mode": "research", "question": "查询公开水务研究资料"},
    )

    assert response.status_code == 200
    assert graph.calls == []
