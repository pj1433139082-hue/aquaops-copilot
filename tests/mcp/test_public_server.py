"""Contract tests for the local stdio-only public MCP boundary.

All dependencies are deliberately synthetic so this suite never reads public
history files, starts Qdrant, downloads a model, or touches private data.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from aquaops.data.public_anomaly import PublicAnomalySummary
from aquaops.data.public_window import PublicWindowSummary
from aquaops.rag.hybrid import PublicEvidence, RetrievalResult


EXPECTED_TOOL_NAMES = (
    "retrieve_public_water_knowledge",
    "aggregate_public_history",
    "screen_public_anomaly",
)


def _public_summary() -> PublicWindowSummary:
    return PublicWindowSummary(
        source_id="co-udlabs-wwtp-lpicm-2025",
        source_url="https://zenodo.org/records/15285089",
        source_version="v1.0.0 (2025-04-26)",
        indicator="nh4",
        unit="mg/L",
        start="2025-01-01T00:00:00Z",
        end="2025-01-02T00:00:00Z",
        row_count=12,
        null_count=2,
        minimum=0.1,
        maximum=0.5,
        mean=0.3,
    )


def _public_anomaly() -> PublicAnomalySummary:
    return PublicAnomalySummary(
        source_id="co-udlabs-wwtp-lpicm-2025",
        source_url="https://zenodo.org/records/15285089",
        source_version="v1.0.0 (2025-04-26)",
        indicator="nh4",
        unit="mg/L",
        target_time="2025-01-02T00:00:00Z",
        baseline_start="2025-01-01T00:00:00Z",
        baseline_end="2025-01-02T00:00:00Z",
        history_count=50,
        history_null_count=2,
        baseline_median=0.3,
        baseline_scale=0.1,
        robust_score=4.0,
        threshold=3.5,
        method="rolling_median_mad",
        direction="above_baseline",
        is_anomaly=True,
        outcome="anomaly",
    )


def _public_retrieval() -> RetrievalResult:
    return RetrievalResult(
        evidence=(
            PublicEvidence(
                chunk_id="a" * 64,
                source_url="https://example.org/public-guidance",
                source_version="2026-01",
                text="This must never leave the MCP response.",
                score=0.42,
            ),
        ),
        fused_candidate_ids=("a" * 64,),
    )


@dataclass
class _FakeRetriever:
    result: RetrievalResult
    calls: list[tuple[str, int]]

    def retrieve(self, query: str, *, answer_limit: int) -> RetrievalResult:
        self.calls.append((query, answer_limit))
        return self.result


def _handlers(*, retriever: object | None = None):
    from aquaops.mcp.public_server import PublicMcpDependencies, create_public_handlers

    return create_public_handlers(
        PublicMcpDependencies(
            retriever=retriever,
            aggregate_public_history=lambda *_: _public_summary(),
            screen_public_anomaly=lambda *_: _public_anomaly(),
        )
    )


def _call_server_tool(
    server: object, tool_name: str, arguments: dict[str, object]
) -> dict[str, object]:
    raw_result = asyncio.run(server.call_tool(tool_name, arguments))  # type: ignore[attr-defined]
    if type(raw_result) is dict:
        return raw_result
    if (
        type(raw_result) is tuple
        and len(raw_result) == 2
        and type(raw_result[1]) is dict
    ):
        return raw_result[1]
    raise AssertionError("FastMCP must return the tool structured response")


def _server_with_non_reading_dependencies():
    from aquaops.mcp.public_server import (
        PublicMcpDependencies,
        create_public_mcp_server,
    )

    def must_not_run(*_: object) -> object:
        raise AssertionError("invalid FastMCP input must not reach a dependency")

    return create_public_mcp_server(
        PublicMcpDependencies(
            retriever=must_not_run,  # type: ignore[arg-type]
            aggregate_public_history=must_not_run,  # type: ignore[arg-type]
            screen_public_anomaly=must_not_run,  # type: ignore[arg-type]
        )
    )


def test_allowlisted_public_tool_names_are_exact_and_read_only() -> None:
    from aquaops.mcp.public_server import (
        PUBLIC_MCP_TOOL_NAMES,
        create_public_mcp_server,
        public_mcp_tool_names,
    )

    assert PUBLIC_MCP_TOOL_NAMES == EXPECTED_TOOL_NAMES
    assert public_mcp_tool_names() == EXPECTED_TOOL_NAMES
    assert all(
        forbidden not in PUBLIC_MCP_TOOL_NAMES
        for forbidden in ("write", "delete", "private", "sql", "path", "admin")
    )
    assert public_mcp_tool_names() == tuple(PUBLIC_MCP_TOOL_NAMES)
    assert create_public_mcp_server() is not None


@pytest.mark.parametrize("query", [None, True, " ", "x" * 513])
def test_retrieve_rejects_invalid_query_without_starting_retrieval(
    query: object,
) -> None:
    calls: list[tuple[str, int]] = []
    handlers = _handlers(retriever=_FakeRetriever(_public_retrieval(), calls))

    result = handlers.retrieve_public_water_knowledge(query)  # type: ignore[arg-type]

    assert result == {
        "ok": False,
        "error": {
            "code": "invalid_request",
            "message": "Request does not satisfy the public read-only contract.",
        },
    }
    assert calls == []


def test_retrieve_strips_text_and_does_not_construct_qdrant_or_a_model() -> None:
    calls: list[tuple[str, int]] = []
    handlers = _handlers(retriever=_FakeRetriever(_public_retrieval(), calls))

    result = handlers.retrieve_public_water_knowledge("ammonia guidance", limit=1)

    assert calls == [("ammonia guidance", 1)]
    assert result == {
        "ok": True,
        "evidence": [
            {
                "chunk_id": "a" * 64,
                "source_url": "https://example.org/public-guidance",
                "source_version": "2026-01",
                "score": 0.42,
            }
        ],
    }
    assert "text" not in repr(result).casefold()
    assert "qdrant" not in repr(result).casefold()


def test_retrieve_masks_dependency_errors_without_echoing_the_query() -> None:
    class _BrokenRetriever:
        def retrieve(self, query: str, *, answer_limit: int) -> RetrievalResult:
            raise RuntimeError(f"token=secret query={query} path=C:/private")

    result = _handlers(retriever=_BrokenRetriever()).retrieve_public_water_knowledge(
        "never echo this exact query"
    )

    assert result == {
        "ok": False,
        "error": {
            "code": "capability_unavailable",
            "message": "The requested public read capability is unavailable.",
        },
    }
    assert "never echo this exact query" not in repr(result)
    assert "secret" not in repr(result).casefold()
    assert "path" not in repr(result).casefold()


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        (
            "retrieve_public_water_knowledge",
            {"query": "public guidance", "limit": True},
        ),
        ("retrieve_public_water_knowledge", {"query": "public guidance", "limit": "1"}),
        (
            "aggregate_public_history",
            {"indicator": "nh4", "hours": True, "end_at": "2025-01-02T00:00:00Z"},
        ),
        (
            "aggregate_public_history",
            {"indicator": "nh4", "hours": "1", "end_at": "2025-01-02T00:00:00Z"},
        ),
        (
            "screen_public_anomaly",
            {
                "indicator": "nh4",
                "baseline_hours": True,
                "end_at": "2025-01-02T00:00:00Z",
            },
        ),
        (
            "screen_public_anomaly",
            {
                "indicator": "nh4",
                "baseline_hours": "1",
                "end_at": "2025-01-02T00:00:00Z",
            },
        ),
    ],
)
def test_fastmcp_rejects_coercible_numeric_values_with_fixed_typed_error(
    tool_name: str, arguments: dict[str, object]
) -> None:
    result = _call_server_tool(
        _server_with_non_reading_dependencies(), tool_name, arguments
    )

    assert result == {
        "ok": False,
        "error": {
            "code": "invalid_request",
            "message": "Request does not satisfy the public read-only contract.",
        },
    }


def test_fastmcp_passes_a_real_integer_to_the_retrieval_handler() -> None:
    calls: list[tuple[str, int]] = []
    from aquaops.mcp.public_server import (
        PublicMcpDependencies,
        create_public_mcp_server,
    )

    server = create_public_mcp_server(
        PublicMcpDependencies(
            retriever=_FakeRetriever(_public_retrieval(), calls),
            aggregate_public_history=lambda *_: _public_summary(),
            screen_public_anomaly=lambda *_: _public_anomaly(),
        )
    )

    result = _call_server_tool(
        server,
        "retrieve_public_water_knowledge",
        {"query": "public guidance", "limit": 1},
    )

    assert result["ok"] is True
    assert calls == [("public guidance", 1)]


@pytest.mark.parametrize(
    ("tool_name", "arguments", "canary"),
    [
        ("retrieve_public_water_knowledge", {"query": True}, None),
        (
            "retrieve_public_water_knowledge",
            {"query": {"CANARY_QUERY": "do-not-echo"}},
            "CANARY_QUERY",
        ),
        ("retrieve_public_water_knowledge", {}, None),
        (
            "aggregate_public_history",
            {"indicator": True, "hours": 1, "end_at": "2025-01-02T00:00:00Z"},
            None,
        ),
        (
            "aggregate_public_history",
            {
                "indicator": {"CANARY_INDICATOR": "do-not-echo"},
                "hours": 1,
                "end_at": "2025-01-02T00:00:00Z",
            },
            "CANARY_INDICATOR",
        ),
        (
            "aggregate_public_history",
            {"hours": 1, "end_at": "2025-01-02T00:00:00Z"},
            None,
        ),
        (
            "screen_public_anomaly",
            {
                "indicator": "nh4",
                "baseline_hours": 1,
                "end_at": True,
            },
            None,
        ),
        (
            "screen_public_anomaly",
            {
                "indicator": "nh4",
                "baseline_hours": 1,
                "end_at": ["CANARY_END_AT"],
            },
            "CANARY_END_AT",
        ),
        (
            "screen_public_anomaly",
            {"indicator": "nh4", "baseline_hours": 1},
            None,
        ),
        (
            "screen_public_anomaly",
            {"indicator": "nh4", "end_at": "2025-01-02T00:00:00Z"},
            None,
        ),
    ],
)
def test_fastmcp_rejects_untyped_or_missing_string_inputs_without_leaking_them(
    tool_name: str, arguments: dict[str, object], canary: str | None
) -> None:
    result = _call_server_tool(
        _server_with_non_reading_dependencies(), tool_name, arguments
    )

    assert result == {
        "ok": False,
        "error": {
            "code": "invalid_request",
            "message": "Request does not satisfy the public read-only contract.",
        },
    }
    rendered = repr(result)
    assert "ValidationError" not in rendered
    assert "Pydantic" not in rendered
    assert "https://" not in rendered
    if canary is not None:
        assert canary not in rendered


@pytest.mark.parametrize(
    ("indicator", "hours", "end_at"),
    [
        ("nh4", True, "2025-01-02T00:00:00Z"),
        ("nh4", 169, "2025-01-02T00:00:00Z"),
        ("password", 24, "2025-01-02T00:00:00Z"),
        ("nh4", 24, "2025-01-02T00:00:00+08:00"),
    ],
)
def test_aggregate_rejects_invalid_request_without_history_access(
    indicator: object, hours: object, end_at: object
) -> None:
    def should_not_run(*_: object) -> PublicWindowSummary:
        raise AssertionError("invalid input must not access history")

    from aquaops.mcp.public_server import PublicMcpDependencies, create_public_handlers

    result = create_public_handlers(
        PublicMcpDependencies(
            aggregate_public_history=should_not_run,
            screen_public_anomaly=lambda *_: _public_anomaly(),
        )
    ).aggregate_public_history(indicator, hours, end_at)  # type: ignore[arg-type]

    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_request"


def test_aggregate_adapts_only_the_existing_public_aggregate_contract() -> None:
    result = _handlers().aggregate_public_history("nh4", 24, "2025-01-02T00:00:00Z")

    assert result == {
        "ok": True,
        "aggregate": {
            "source_id": "co-udlabs-wwtp-lpicm-2025",
            "source_url": "https://zenodo.org/records/15285089",
            "source_version": "v1.0.0 (2025-04-26)",
            "indicator": "nh4",
            "unit": "mg/L",
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-01-02T00:00:00Z",
            "row_count": 12,
            "null_count": 2,
            "minimum": 0.1,
            "maximum": 0.5,
            "mean": 0.3,
        },
    }
    assert not {"values", "rows", "path"} & set(result["aggregate"])


def test_anomaly_adapts_only_the_existing_public_anomaly_contract() -> None:
    result = _handlers().screen_public_anomaly("nh4", 24, "2025-01-02T00:00:00Z")

    assert result["ok"] is True
    assert set(result) == {"ok", "anomaly"}
    assert result["anomaly"]["outcome"] == "anomaly"
    assert "is_anomaly" not in result["anomaly"]
    assert not {"values", "rows", "path", "text"} & set(result["anomaly"])


@pytest.mark.parametrize(
    ("source_url", "source_version", "canary"),
    [
        (
            "https://CANARY_USER:CANARY_PASSWORD@example.org/guidance",
            "2026-01",
            "CANARY_PASSWORD",
        ),
        (
            "https://example.org/guidance?token=CANARY_QUERY_TOKEN",
            "2026-01",
            "CANARY_QUERY_TOKEN",
        ),
        (
            "https://example.org/guidance#CANARY_FRAGMENT",
            "2026-01",
            "CANARY_FRAGMENT",
        ),
        (
            "https://example.org/guidance",
            "release-token-CANARY_VERSION",
            "CANARY_VERSION",
        ),
    ],
)
def test_retrieval_rejects_sensitive_source_metadata_without_leaking_it(
    source_url: str, source_version: str, canary: str
) -> None:
    evidence = PublicEvidence(
        chunk_id="a" * 64,
        source_url="https://example.org/guidance",
        source_version="2026-01",
        text="Never serialized.",
        score=0.42,
    )
    object.__setattr__(evidence, "source_url", source_url)
    object.__setattr__(evidence, "source_version", source_version)
    result = _handlers(
        retriever=_FakeRetriever(
            RetrievalResult(evidence=(evidence,), fused_candidate_ids=("a" * 64,)),
            [],
        )
    ).retrieve_public_water_knowledge("public guidance")

    assert result == {
        "ok": False,
        "error": {
            "code": "unsafe_public_result",
            "message": "The public dependency returned an unsafe result.",
        },
    }
    assert canary not in repr(result)


@pytest.mark.parametrize(
    "payload",
    [
        {"chunk_id": "a" * 64, "text": "leak"},
        {"chunk_id": "a" * 64, "sourceUrl": "https://leak.invalid"},
        {"chunk_id": "a" * 64, "score": float("nan")},
    ],
)
def test_structural_serializer_rejects_forbidden_keys_and_invalid_values(
    payload: dict[str, object],
) -> None:
    from aquaops.mcp.public_server import (
        PublicMcpContractError,
        serialize_public_payload,
    )

    with pytest.raises(PublicMcpContractError):
        serialize_public_payload(
            payload, allowed_fields={"chunk_id": str, "score": float}
        )


def test_module_entrypoint_declares_stdio_without_listening_on_a_port() -> None:
    from aquaops.mcp import public_server

    assert public_server.PUBLIC_MCP_TRANSPORT == "stdio"
    assert public_server.PUBLIC_MCP_LISTEN_PORTS == ()
    assert "uvicorn" not in public_server.__dict__
    assert "FastMCP" in public_server.__dict__


@pytest.mark.parametrize(
    ("source_url", "canary"),
    [
        ("https://127.0.0.1:8443/CANARY_LOOPBACK", "CANARY_LOOPBACK"),
        ("https://127.1/CANARY_SHORT_IPV4", "CANARY_SHORT_IPV4"),
        ("https://0177.0.0.1/CANARY_OCTAL_IPV4", "CANARY_OCTAL_IPV4"),
        ("https://0x7f.0.0.1/CANARY_HEX_IPV4", "CANARY_HEX_IPV4"),
        ("https://[::1]/CANARY_IPV6_LOOPBACK", "CANARY_IPV6_LOOPBACK"),
        ("https://10.2.3.4/CANARY_PRIVATE_10", "CANARY_PRIVATE_10"),
        ("https://192.168.2.3/CANARY_PRIVATE_192", "CANARY_PRIVATE_192"),
        ("https://169.254.169.254/CANARY_LINK_LOCAL", "CANARY_LINK_LOCAL"),
        ("https://224.0.0.1/CANARY_MULTICAST", "CANARY_MULTICAST"),
        ("https://240.0.0.1/CANARY_RESERVED", "CANARY_RESERVED"),
        ("https://0.0.0.0/CANARY_UNSPECIFIED", "CANARY_UNSPECIFIED"),
        ("https://localhost/CANARY_LOCALHOST", "CANARY_LOCALHOST"),
        ("https://api.internal/CANARY_INTERNAL", "CANARY_INTERNAL"),
        (
            "https://api.internal.example.org/CANARY_INTERNAL_LABEL",
            "CANARY_INTERNAL_LABEL",
        ),
        ("https://node.local/CANARY_LOCAL", "CANARY_LOCAL"),
    ],
)
def test_retrieval_rejects_local_or_non_public_source_hosts_without_leaking_them(
    source_url: str, canary: str
) -> None:
    evidence = PublicEvidence(
        chunk_id="a" * 64,
        source_url="https://example.org/guidance",
        source_version="2026-01",
        text="Never serialized.",
        score=0.42,
    )
    object.__setattr__(evidence, "source_url", source_url)
    result = _handlers(
        retriever=_FakeRetriever(
            RetrievalResult(evidence=(evidence,), fused_candidate_ids=("a" * 64,)),
            [],
        )
    ).retrieve_public_water_knowledge("public guidance")

    assert result == {
        "ok": False,
        "error": {
            "code": "unsafe_public_result",
            "message": "The public dependency returned an unsafe result.",
        },
    }
    assert canary not in repr(result)


def test_public_https_dns_source_remains_serializable() -> None:
    result = _handlers(
        retriever=_FakeRetriever(_public_retrieval(), [])
    ).retrieve_public_water_knowledge("public guidance")

    assert result["ok"] is True
    assert result["evidence"][0]["source_url"] == "https://example.org/public-guidance"


def test_public_tool_input_contracts_are_exact_and_immutable() -> None:
    from aquaops.mcp.public_server import PUBLIC_TOOL_INPUT_CONTRACTS

    assert tuple(PUBLIC_TOOL_INPUT_CONTRACTS) == EXPECTED_TOOL_NAMES
    assert PUBLIC_TOOL_INPUT_CONTRACTS["retrieve_public_water_knowledge"]["query"] == {
        "type": "string",
        "required": True,
        "min_length": 1,
        "max_length": 512,
    }
    assert PUBLIC_TOOL_INPUT_CONTRACTS["retrieve_public_water_knowledge"]["limit"] == {
        "type": "integer",
        "required": False,
        "minimum": 1,
        "maximum": 6,
        "default": 6,
    }
    assert PUBLIC_TOOL_INPUT_CONTRACTS["aggregate_public_history"]["end_at"] == {
        "type": "string",
        "required": True,
        "format": "canonical_utc_z",
    }
    assert PUBLIC_TOOL_INPUT_CONTRACTS["screen_public_anomaly"]["baseline_hours"] == {
        "type": "integer",
        "required": True,
        "minimum": 1,
        "maximum": 168,
    }
    with pytest.raises(TypeError):
        PUBLIC_TOOL_INPUT_CONTRACTS["retrieve_public_water_knowledge"]["query"][
            "type"
        ] = "number"  # type: ignore[index]
