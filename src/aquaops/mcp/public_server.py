"""Local stdio-only MCP boundary for approved public, aggregate-only evidence.

This module deliberately creates no HTTP application and does not construct a
Qdrant client, embedding model, or public-data reader at import time.  Tool
dependencies are injected so callers can keep deployment and data access
outside the MCP transport boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import ipaddress
from math import isfinite
import re
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, TypeAlias
from urllib.parse import urlsplit

from mcp.server.fastmcp import FastMCP
from pydantic import HttpUrl, TypeAdapter, ValidationError

from aquaops.agent.tools import PublicHistoricalAggregate, PublicHistoricalAnomaly
from aquaops.data.public_anomaly import PublicAnomalySummary, screen_public_anomaly
from aquaops.data.public_window import PublicWindowSummary, summarize_public_window
from aquaops.rag.hybrid import PublicEvidence, RetrievalResult


PUBLIC_MCP_TOOL_NAMES: tuple[str, ...] = (
    "retrieve_public_water_knowledge",
    "aggregate_public_history",
    "screen_public_anomaly",
)
PUBLIC_MCP_TRANSPORT = "stdio"
PUBLIC_MCP_LISTEN_PORTS: tuple[int, ...] = ()

_PUBLIC_INDICATORS = frozenset({"nh4", "cond", "q"})
_FORBIDDEN_KEY_PARTS = (
    "values",
    "rows",
    "path",
    "sql",
    "command",
    "token",
    "secret",
    "text",
    "query",
)
_KEY_NORMALIZER = re.compile(r"[^a-z0-9]+")
_SAFE_SOURCE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SENSITIVE_VERSION_MARKERS = (
    "token",
    "secret",
    "credential",
    "password",
    "passwd",
    "apikey",
    "key",
)
_LOCAL_OR_INTERNAL_HOSTNAMES = frozenset({"localhost", "local", "internal"})
_LOCAL_OR_INTERNAL_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".localdomain",
    ".lan",
    ".corp",
    ".home.arpa",
)
_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_HTTP_URL = TypeAdapter(HttpUrl)
JsonPrimitive: TypeAlias = str | int | float | bool | None
FieldType: TypeAlias = type[str] | type[int] | type[float] | type[bool]

_EVIDENCE_FIELDS: dict[str, FieldType] = {
    "chunk_id": str,
    "source_url": str,
    "source_version": str,
    "score": float,
}
_AGGREGATE_FIELDS: dict[str, FieldType] = {
    "source_id": str,
    "source_url": str,
    "source_version": str,
    "indicator": str,
    "unit": str,
    "start": str,
    "end": str,
    "row_count": int,
    "null_count": int,
    "minimum": float,
    "maximum": float,
    "mean": float,
}
_ANOMALY_FIELDS: dict[str, FieldType] = {
    "source_id": str,
    "source_url": str,
    "source_version": str,
    "indicator": str,
    "unit": str,
    "target_time": str,
    "baseline_start": str,
    "baseline_end": str,
    "history_count": int,
    "history_null_count": int,
    "baseline_median": float,
    "baseline_scale": float,
    "robust_score": float,
    "threshold": float,
    "method": str,
    "direction": str,
    "outcome": str,
}


def _freeze_tool_input_contracts(
    contracts: dict[str, dict[str, dict[str, object]]],
) -> Mapping[str, Mapping[str, Mapping[str, object]]]:
    return MappingProxyType(
        {
            tool_name: MappingProxyType(
                {
                    field_name: MappingProxyType(dict(field_contract))
                    for field_name, field_contract in fields.items()
                }
            )
            for tool_name, fields in contracts.items()
        }
    )


PUBLIC_TOOL_INPUT_CONTRACTS = _freeze_tool_input_contracts(
    {
        "retrieve_public_water_knowledge": {
            "query": {
                "type": "string",
                "required": True,
                "min_length": 1,
                "max_length": 512,
            },
            "limit": {
                "type": "integer",
                "required": False,
                "minimum": 1,
                "maximum": 6,
                "default": 6,
            },
        },
        "aggregate_public_history": {
            "indicator": {
                "type": "string",
                "required": True,
                "enum": ("nh4", "cond", "q"),
            },
            "hours": {
                "type": "integer",
                "required": True,
                "minimum": 1,
                "maximum": 168,
            },
            "end_at": {
                "type": "string",
                "required": True,
                "format": "canonical_utc_z",
            },
        },
        "screen_public_anomaly": {
            "indicator": {
                "type": "string",
                "required": True,
                "enum": ("nh4", "cond", "q"),
            },
            "baseline_hours": {
                "type": "integer",
                "required": True,
                "minimum": 1,
                "maximum": 168,
            },
            "end_at": {
                "type": "string",
                "required": True,
                "format": "canonical_utc_z",
            },
        },
    }
)


class PublicMcpContractError(ValueError):
    """A dependency result cannot cross the public MCP serialization boundary."""


class PublicRetriever(Protocol):
    """Small public retrieval interface; implementations stay outside MCP."""

    def retrieve(self, query: str, *, answer_limit: int) -> RetrievalResult: ...


@dataclass(frozen=True)
class PublicMcpDependencies:
    """Injected public-only capabilities; no default retrieval backend exists."""

    retriever: PublicRetriever | None = None
    aggregate_public_history: Callable[[str, int, str], PublicWindowSummary] = (
        summarize_public_window
    )
    screen_public_anomaly: Callable[[str, int, str], PublicAnomalySummary] = (
        screen_public_anomaly
    )


def public_mcp_tool_names() -> tuple[str, ...]:
    """Return the complete stable allowlist without inspecting SDK internals."""

    return PUBLIC_MCP_TOOL_NAMES


def _typed_error(code: str, message: str) -> dict[str, object]:
    return {"ok": False, "error": {"code": code, "message": message}}


def _invalid_request() -> dict[str, object]:
    return _typed_error(
        "invalid_request", "Request does not satisfy the public read-only contract."
    )


def _capability_unavailable() -> dict[str, object]:
    return _typed_error(
        "capability_unavailable", "The requested public read capability is unavailable."
    )


def _unsafe_public_result() -> dict[str, object]:
    return _typed_error(
        "unsafe_public_result", "The public dependency returned an unsafe result."
    )


def _is_canonical_utc_z(value: object) -> bool:
    if type(value) is not str or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError:
        return False
    return (
        parsed.tzinfo is not None
        and parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") == value
    )


def _is_valid_query(value: object) -> bool:
    return type(value) is str and bool(value.strip()) and len(value) <= 512


def _is_valid_limit(value: object) -> bool:
    return type(value) is int and 1 <= value <= 6


def _is_valid_history_request(indicator: object, hours: object, end_at: object) -> bool:
    return (
        type(indicator) is str
        and indicator in _PUBLIC_INDICATORS
        and type(hours) is int
        and 1 <= hours <= 168
        and _is_canonical_utc_z(end_at)
    )


def _normalized_key(key: object) -> str:
    if not isinstance(key, str) or not key:
        raise PublicMcpContractError("public payload contains a non-string key")
    return _KEY_NORMALIZER.sub("", key.casefold())


def _validate_field_value(value: object, expected_type: FieldType) -> JsonPrimitive:
    if value is None:
        if expected_type is float:
            return None
        raise PublicMcpContractError("public payload has an unexpected null")
    if expected_type is str:
        if type(value) is not str or not value.strip():
            raise PublicMcpContractError("public payload has an invalid string")
        return value
    if expected_type is int:
        if type(value) is not int or value < 0:
            raise PublicMcpContractError("public payload has an invalid integer")
        return value
    if expected_type is float:
        if type(value) not in (int, float) or not isfinite(float(value)):
            raise PublicMcpContractError("public payload has an invalid number")
        return float(value)
    if expected_type is bool and type(value) is bool:
        return value
    raise PublicMcpContractError("public payload has an invalid value type")


def serialize_public_payload(
    payload: object, *, allowed_fields: dict[str, FieldType]
) -> dict[str, JsonPrimitive]:
    """Copy only an exact scalar schema and reject sensitive key variants.

    The server never recursively converts arbitrary objects.  Each nested
    response section has a fixed schema, and this function copies only JSON
    scalar values approved by that schema.
    """

    if type(payload) is not dict:
        raise PublicMcpContractError("public payload must be a plain dictionary")
    for key in payload:
        normalized = _normalized_key(key)
        if any(part in normalized for part in _FORBIDDEN_KEY_PARTS):
            raise PublicMcpContractError("public payload contains a forbidden field")
    if set(payload) != set(allowed_fields):
        raise PublicMcpContractError("public payload fields do not match its schema")
    return {
        field_name: _validate_field_value(payload[field_name], expected_type)
        for field_name, expected_type in allowed_fields.items()
    }


def _serialize_aggregate(summary: object) -> dict[str, JsonPrimitive]:
    if type(summary) is not PublicWindowSummary:
        raise PublicMcpContractError("aggregate dependency returned an invalid result")
    try:
        aggregate = PublicHistoricalAggregate.model_validate(asdict(summary))
    except Exception as error:
        raise PublicMcpContractError(
            "aggregate dependency returned an invalid result"
        ) from error
    return serialize_public_payload(
        aggregate.model_dump(), allowed_fields=_AGGREGATE_FIELDS
    )


def _serialize_anomaly(
    summary: object, *, baseline_hours: int, end_at: str
) -> dict[str, JsonPrimitive]:
    if type(summary) is not PublicAnomalySummary:
        raise PublicMcpContractError("anomaly dependency returned an invalid result")
    summary_values = asdict(summary)
    try:
        anomaly = PublicHistoricalAnomaly.model_validate(
            {
                field_name: summary_values[field_name]
                for field_name in PublicHistoricalAnomaly.model_fields
            }
        )
        requested_end = datetime.fromisoformat(f"{end_at[:-1]}+00:00")
        expected_start = (
            (requested_end - timedelta(hours=baseline_hours))
            .isoformat()
            .replace("+00:00", "Z")
        )
        if (
            anomaly.target_time != end_at
            or anomaly.baseline_end != end_at
            or anomaly.baseline_start != expected_start
        ):
            raise ValueError("anomaly result does not match the requested window")
    except Exception as error:
        raise PublicMcpContractError(
            "anomaly dependency returned an invalid result"
        ) from error
    return serialize_public_payload(
        anomaly.model_dump(), allowed_fields=_ANOMALY_FIELDS
    )


def _serialize_retrieval(result: object) -> list[dict[str, JsonPrimitive]]:
    if type(result) is not RetrievalResult or type(result.evidence) is not tuple:
        raise PublicMcpContractError("retrieval dependency returned an invalid result")
    evidence: list[dict[str, JsonPrimitive]] = []
    seen_chunk_ids: set[str] = set()
    for item in result.evidence:
        if type(item) is not PublicEvidence:
            raise PublicMcpContractError(
                "retrieval dependency returned an invalid result"
            )
        if (
            item.data_class != "public"
            or item.access_policy != "public_read"
            or len(item.chunk_id) != 64
            or any(character not in "0123456789abcdef" for character in item.chunk_id)
            or not _is_safe_public_url(item.source_url)
            or not _is_safe_source_version(item.source_version)
            or item.chunk_id in seen_chunk_ids
        ):
            raise PublicMcpContractError(
                "retrieval dependency returned an invalid result"
            )
        seen_chunk_ids.add(item.chunk_id)
        evidence.append(
            serialize_public_payload(
                {
                    "chunk_id": item.chunk_id,
                    "source_url": item.source_url,
                    "source_version": item.source_version,
                    "score": item.score,
                },
                allowed_fields=_EVIDENCE_FIELDS,
            )
        )
    return evidence


def _is_safe_public_url(value: object) -> bool:
    if type(value) is not str or not value.strip():
        return False
    try:
        parsed = urlsplit(value)
        normalized_url = _HTTP_URL.validate_python(value)
        hostname = normalized_url.host
        _ = parsed.port
    except (ValidationError, ValueError):
        return False
    if (
        parsed.scheme != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return False
    hostname = hostname.casefold().rstrip(".")
    hostname_labels = hostname.split(".")
    if (
        not hostname
        or hostname in _LOCAL_OR_INTERNAL_HOSTNAMES
        or any(label in _LOCAL_OR_INTERNAL_HOSTNAMES for label in hostname_labels)
        or any(hostname.endswith(suffix) for suffix in _LOCAL_OR_INTERNAL_SUFFIXES)
    ):
        return False
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        # DNS resolution is intentionally not performed here.  Reject every IP
        # literal so a caller cannot repurpose this public-evidence boundary as
        # an SSRF primitive against public or non-public address space.
        return False
    return (
        "." in hostname
        and len(hostname) <= 253
        and all(_DNS_LABEL.fullmatch(label) for label in hostname_labels)
    )


def _is_safe_source_version(value: object) -> bool:
    if type(value) is not str or not _SAFE_SOURCE_VERSION.fullmatch(value):
        return False
    normalized = _KEY_NORMALIZER.sub("", value.casefold())
    return not any(marker in normalized for marker in _SENSITIVE_VERSION_MARKERS)


@dataclass(frozen=True)
class PublicMcpHandlers:
    """Direct, transport-neutral implementations of the three allowed tools."""

    dependencies: PublicMcpDependencies

    def retrieve_public_water_knowledge(
        self, query: object, limit: object = 6
    ) -> dict[str, object]:
        if not _is_valid_query(query) or not _is_valid_limit(limit):
            return _invalid_request()
        if self.dependencies.retriever is None:
            return _capability_unavailable()
        try:
            result = self.dependencies.retriever.retrieve(query, answer_limit=limit)
            evidence = _serialize_retrieval(result)
            if len(evidence) > limit:
                raise PublicMcpContractError("retrieval dependency exceeded its limit")
        except PublicMcpContractError:
            return _unsafe_public_result()
        except Exception:
            return _capability_unavailable()
        return {"ok": True, "evidence": evidence}

    def aggregate_public_history(
        self, indicator: object, hours: object, end_at: object
    ) -> dict[str, object]:
        if not _is_valid_history_request(indicator, hours, end_at):
            return _invalid_request()
        try:
            aggregate = _serialize_aggregate(
                self.dependencies.aggregate_public_history(indicator, hours, end_at)
            )
        except PublicMcpContractError:
            return _unsafe_public_result()
        except Exception:
            return _capability_unavailable()
        return {"ok": True, "aggregate": aggregate}

    def screen_public_anomaly(
        self, indicator: object, baseline_hours: object, end_at: object
    ) -> dict[str, object]:
        if not _is_valid_history_request(indicator, baseline_hours, end_at):
            return _invalid_request()
        try:
            anomaly = _serialize_anomaly(
                self.dependencies.screen_public_anomaly(
                    indicator, baseline_hours, end_at
                ),
                baseline_hours=baseline_hours,
                end_at=end_at,
            )
        except PublicMcpContractError:
            return _unsafe_public_result()
        except Exception:
            return _capability_unavailable()
        return {"ok": True, "anomaly": anomaly}


def create_public_handlers(
    dependencies: PublicMcpDependencies | None = None,
) -> PublicMcpHandlers:
    """Create testable public-only tool handlers without starting a transport."""

    return PublicMcpHandlers(dependencies or PublicMcpDependencies())


def create_public_mcp_server(
    dependencies: PublicMcpDependencies | None = None,
) -> FastMCP:
    """Build, but do not run, the exact three-tool local public MCP server."""

    handlers = create_public_handlers(dependencies)
    server = FastMCP(
        "AquaOps Public Read-Only",
        instructions=(
            "This local server exposes approved public evidence only. "
            "It has no private-data tools and no write operations."
        ),
        json_response=True,
    )

    @server.tool()
    def retrieve_public_water_knowledge(
        query: object = None, limit: object = 6
    ) -> dict[str, object]:
        """Retrieve public evidence only.

        Input contract: ``query`` is a required non-empty JSON string of at
        most 512 characters; ``limit`` is an optional JSON integer from 1 to
        6 (default 6).  Invalid raw JSON values return ``invalid_request``.
        """

        return handlers.retrieve_public_water_knowledge(query, limit)

    @server.tool()
    def aggregate_public_history(
        indicator: object = None, hours: object = None, end_at: object = None
    ) -> dict[str, object]:
        """Return an aggregate from the one approved public historical source.

        Input contract: ``indicator`` is one of ``nh4``, ``cond``, ``q``;
        ``hours`` is a required JSON integer from 1 to 168; and ``end_at`` is
        required canonical UTC ``Z`` text. Invalid raw JSON returns
        ``invalid_request``.
        """

        return handlers.aggregate_public_history(indicator, hours, end_at)

    @server.tool()
    def screen_public_anomaly(
        indicator: object = None,
        baseline_hours: object = None,
        end_at: object = None,
    ) -> dict[str, object]:
        """Screen one public historical target using aggregate-only statistics.

        Input contract: ``indicator`` is one of ``nh4``, ``cond``, ``q``;
        ``baseline_hours`` is a required JSON integer from 1 to 168; and
        ``end_at`` is required canonical UTC ``Z`` text. Invalid raw JSON
        returns ``invalid_request``.
        """

        return handlers.screen_public_anomaly(indicator, baseline_hours, end_at)

    return server


def main() -> None:
    """Run the public MCP server over local standard input/output only."""

    create_public_mcp_server().run(transport=PUBLIC_MCP_TRANSPORT)


if __name__ == "__main__":
    main()
