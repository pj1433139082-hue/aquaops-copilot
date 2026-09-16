"""Fail-closed policy boundary for the three public read-only MCP tools.

The policy layer deliberately receives raw tool requests but never returns or
persists their arguments.  It issues a new immutable budget snapshot only
after a request matches one of the immutable public MCP contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
import re
from threading import Lock
from types import MappingProxyType
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aquaops.mcp.public_server import PUBLIC_MCP_TOOL_NAMES, PUBLIC_TOOL_INPUT_CONTRACTS


MAX_TOTAL_TOOL_CALLS = 6
PolicyDecisionCode = Literal[
    "tool_allowed",
    "tool_unavailable",
    "tool_not_allowlisted",
    "invalid_tool_request",
    "invalid_tool_budget",
    "tool_budget_exhausted",
    "tool_call_limit_reached",
    "tool_authorization_invalid",
]
TraceDecisionCode = Literal[
    "tool_allowed",
    "tool_unavailable",
    "tool_not_allowlisted",
    "invalid_tool_request",
    "invalid_tool_budget",
    "tool_budget_exhausted",
    "tool_call_limit_reached",
    "tool_authorization_invalid",
    "tool_completed",
    "tool_result_unavailable",
]
_PUBLIC_INDICATORS = frozenset({"nh4", "cond", "q"})
_FORBIDDEN_ARGUMENT_NAMES = frozenset(
    {
        "access_policy",
        "collection",
        "data_class",
        "database",
        "path",
        "read_only",
        "risk",
        "sql",
        "timeout",
        "timeout_seconds",
        "token",
        "url",
    }
)
_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_EVIDENCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SENSITIVE_MARKERS = ("password", "private", "query", "secret", "token", "value")


@dataclass(frozen=True)
class ToolSpec:
    """Static capability facts; callers have no way to override them."""

    name: str
    data_class: Literal["public"] = "public"
    risk: Literal["read_only"] = "read_only"
    read_only: Literal[True] = True
    max_calls: int = 2
    timeout_seconds: int = 5

    @property
    def timeout(self) -> int:
        """A short alias kept for policy inspection without another mutable field."""

        return self.timeout_seconds


def _build_public_registry() -> Mapping[str, ToolSpec]:
    if tuple(PUBLIC_MCP_TOOL_NAMES) != (
        "retrieve_public_water_knowledge",
        "aggregate_public_history",
        "screen_public_anomaly",
    ):
        raise RuntimeError("public MCP allowlist changed without a policy review")
    if set(PUBLIC_MCP_TOOL_NAMES) != set(PUBLIC_TOOL_INPUT_CONTRACTS):
        raise RuntimeError("public MCP contracts and allowlist diverged")
    return MappingProxyType(
        {name: ToolSpec(name=name) for name in PUBLIC_MCP_TOOL_NAMES}
    )


PUBLIC_TOOL_REGISTRY: Mapping[str, ToolSpec] = _build_public_registry()
_GRANT_ISSUER = object()


class ToolInvocation(BaseModel):
    """The only accepted model/tool invocation shape, before contract checking."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(min_length=1, max_length=96)
    arguments: dict[str, object]

    @field_validator("arguments", mode="before")
    @classmethod
    def require_plain_argument_object(cls, value: object) -> object:
        if type(value) is not dict:
            raise ValueError("tool arguments must be a plain JSON object")
        return value


class ToolBudget(BaseModel):
    """Immutable system-issued call budget, never part of an agent request schema."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    remaining_total: int = Field(ge=0, le=MAX_TOTAL_TOOL_CALLS)
    used_by_tool: tuple[tuple[str, int], ...] = ()

    @field_validator("remaining_total", mode="before")
    @classmethod
    def reject_boolean_total(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("remaining tool budget must be an integer")
        return value

    @field_validator("used_by_tool", mode="before")
    @classmethod
    def require_immutable_usage_pairs(cls, value: object) -> object:
        if type(value) is not tuple:
            raise ValueError("tool usage must be an immutable tuple")
        return value

    @model_validator(mode="after")
    def validate_usage(self) -> "ToolBudget":
        names: set[str] = set()
        for name, count in self.used_by_tool:
            if name not in PUBLIC_TOOL_REGISTRY or name in names:
                raise ValueError(
                    "tool usage contains an unknown or duplicate capability"
                )
            if (
                type(count) is not int
                or not 0 <= count <= PUBLIC_TOOL_REGISTRY[name].max_calls
            ):
                raise ValueError("tool usage is outside its system limit")
            names.add(name)
        if MAX_TOTAL_TOOL_CALLS - self.remaining_total != sum(
            count for _, count in self.used_by_tool
        ):
            raise ValueError("tool budget snapshot is inconsistent")
        return self

    @classmethod
    def initial(cls) -> "ToolBudget":
        return cls(remaining_total=MAX_TOTAL_TOOL_CALLS, used_by_tool=())

    def calls_for(self, name: str) -> int:
        return dict(self.used_by_tool).get(name, 0)

    def consume(self, spec: ToolSpec) -> "ToolBudget":
        if spec.name not in PUBLIC_TOOL_REGISTRY:
            raise ValueError("unknown public tool")
        if self.remaining_total < 1 or self.calls_for(spec.name) >= spec.max_calls:
            raise ValueError("tool budget is exhausted")
        usage = dict(self.used_by_tool)
        usage[spec.name] = usage.get(spec.name, 0) + 1
        return ToolBudget(
            remaining_total=self.remaining_total - 1,
            used_by_tool=tuple(sorted(usage.items())),
        )


class ToolPolicyDecision(BaseModel):
    """Safe preflight result.  Raw arguments and dependency details never cross it."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    allowed: bool
    code: PolicyDecisionCode
    tool_name: str | None
    budget: ToolBudget

    @field_validator("tool_name")
    @classmethod
    def require_safe_tool_name(cls, value: str | None) -> str | None:
        if value is not None and value not in PUBLIC_TOOL_REGISTRY:
            raise ValueError("decision tool must be allowlisted")
        return value


@dataclass(frozen=True)
class _IssuedPublicToolGrant:
    """Identity-bound capability that never originates in model/request state."""

    issuer: object
    tool_name: str
    session: "_PublicToolBudgetSession"
    nonce: object


@dataclass(frozen=True)
class _ToolExecutionReceipt:
    """One-use proof that a host session actually consumed a read-only grant."""

    issuer: object
    tool_name: str
    session: "_PublicToolBudgetSession"
    nonce: object


class _PublicToolBudgetSession:
    """Host-only mutable authority; external budget snapshots are display-only."""

    def __init__(self, issuer: object) -> None:
        self._issuer = issuer
        self._budget = ToolBudget.initial()
        self._consumed_grant_nonces: set[object] = set()
        self._redeemed_receipt_nonces: set[object] = set()
        self._lock = Lock()


class ToolTrace(BaseModel):
    """Small auditable trace with no query, payload, path, row, or timing value."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str
    decision_code: TraceDecisionCode
    evidence_ids: tuple[str, ...] = ()
    elapsed_bucket: Literal["under_1s", "under_5s", "over_5s"]
    error_code: str | None = None

    @field_validator("name")
    @classmethod
    def require_allowlisted_name(cls, value: str) -> str:
        if value not in PUBLIC_TOOL_REGISTRY:
            raise ValueError("trace tool must be allowlisted")
        return value

    @field_validator("decision_code", "error_code")
    @classmethod
    def require_safe_code(cls, value: str | None) -> str | None:
        if value is not None and not _SAFE_CODE.fullmatch(value):
            raise ValueError("trace code is invalid")
        return value

    @field_validator("evidence_ids")
    @classmethod
    def require_safe_evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for evidence_id in value:
            normalized = evidence_id.casefold()
            if not _SAFE_EVIDENCE_ID.fullmatch(evidence_id) or any(
                marker in normalized for marker in _SENSITIVE_MARKERS
            ):
                raise ValueError("trace evidence id is unsafe")
        return value

    @classmethod
    def create(
        cls,
        *,
        name: str,
        decision_code: TraceDecisionCode,
        evidence_ids: tuple[str, ...] = (),
        elapsed_seconds: object,
        error_code: str | None,
    ) -> "ToolTrace":
        if type(elapsed_seconds) not in {int, float} or not isfinite(elapsed_seconds):
            raise ValueError("elapsed time must be finite")
        elapsed = float(elapsed_seconds)
        bucket: Literal["under_1s", "under_5s", "over_5s"]
        if elapsed < 1:
            bucket = "under_1s"
        elif elapsed < 5:
            bucket = "under_5s"
        else:
            bucket = "over_5s"
        safe_evidence_ids = tuple(
            evidence_id
            for evidence_id in evidence_ids
            if type(evidence_id) is str
            and _SAFE_EVIDENCE_ID.fullmatch(evidence_id)
            and not any(
                marker in evidence_id.casefold() for marker in _SENSITIVE_MARKERS
            )
        )
        return cls(
            name=name,
            decision_code=decision_code,
            evidence_ids=safe_evidence_ids,
            elapsed_bucket=bucket,
            error_code=error_code,
        )


def _decision(
    allowed: bool,
    code: PolicyDecisionCode,
    tool_name: str | None,
    budget: ToolBudget,
) -> ToolPolicyDecision:
    return ToolPolicyDecision(
        allowed=allowed,
        code=code,
        tool_name=tool_name,
        budget=budget,
    )


def _normalized_budget_snapshot(value: object) -> ToolBudget | None:
    """Reject subclasses and ``model_copy``/attribute updates that skip validation."""

    if type(value) is not ToolBudget:
        return None
    try:
        return ToolBudget.model_validate(value.model_dump())
    except Exception:
        return None


def create_public_tool_budget_session() -> _PublicToolBudgetSession:
    """Create one opaque host authority for a bounded graph execution."""

    return _PublicToolBudgetSession(_GRANT_ISSUER)


def _is_host_budget_session(value: object) -> bool:
    return type(value) is _PublicToolBudgetSession and value._issuer is _GRANT_ISSUER


def public_tool_budget_snapshot(session: object) -> ToolBudget | None:
    """Return a normalized display snapshot that cannot authorize a later call."""

    if not _is_host_budget_session(session):
        return None
    with session._lock:
        return _normalized_budget_snapshot(session._budget)


def _is_canonical_utc_z(value: object) -> bool:
    if type(value) is not str or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError:
        return False
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") == value


def _valid_public_arguments(name: str, arguments: dict[str, object]) -> bool:
    if set(arguments) & _FORBIDDEN_ARGUMENT_NAMES:
        return False
    if name == "retrieve_public_water_knowledge":
        if not set(arguments) <= {"query", "limit"} or "query" not in arguments:
            return False
        query = arguments["query"]
        limit = arguments.get("limit", 6)
        return (
            type(query) is str
            and 1 <= len(query.strip()) <= 512
            and type(limit) is int
            and 1 <= limit <= 6
        )

    hour_key = "hours" if name == "aggregate_public_history" else "baseline_hours"
    return (
        set(arguments) == {"indicator", hour_key, "end_at"}
        and type(arguments["indicator"]) is str
        and arguments["indicator"] in _PUBLIC_INDICATORS
        and type(arguments[hour_key]) is int
        and 1 <= arguments[hour_key] <= 168
        and _is_canonical_utc_z(arguments["end_at"])
    )


def authorize_public_tool_call(
    name: object,
    arguments: object,
    budget: object,
    *,
    available_tool_names: frozenset[str] | None = None,
) -> ToolPolicyDecision:
    """Fail closed before a dependency is invoked, returning no raw input data."""

    normalized_budget = _normalized_budget_snapshot(budget)
    safe_budget = normalized_budget or ToolBudget.initial()
    if type(name) is not str or name not in PUBLIC_TOOL_REGISTRY:
        return _decision(False, "tool_not_allowlisted", None, safe_budget)
    if available_tool_names is not None and name not in available_tool_names:
        return _decision(False, "tool_unavailable", name, safe_budget)
    try:
        invocation = ToolInvocation(name=name, arguments=arguments)
    except Exception:
        return _decision(False, "invalid_tool_request", name, safe_budget)
    spec = PUBLIC_TOOL_REGISTRY[name]
    if not (
        spec.data_class == "public"
        and spec.risk == "read_only"
        and spec.read_only is True
        and _valid_public_arguments(invocation.name, invocation.arguments)
    ):
        return _decision(False, "invalid_tool_request", name, safe_budget)
    if normalized_budget is None:
        return _decision(False, "invalid_tool_budget", name, safe_budget)
    if normalized_budget.remaining_total < 1:
        return _decision(False, "tool_budget_exhausted", name, normalized_budget)
    if normalized_budget.calls_for(name) >= spec.max_calls:
        return _decision(False, "tool_call_limit_reached", name, normalized_budget)
    return _decision(True, "tool_allowed", name, normalized_budget.consume(spec))


def issue_public_tool_grant(
    name: object,
    arguments: object,
    session: object,
    *,
    available_tool_names: frozenset[str] | None = None,
) -> tuple[ToolPolicyDecision, _IssuedPublicToolGrant | None]:
    """Issue an identity-bound capability only after all policy checks succeed."""

    if not _is_host_budget_session(session):
        return _decision(False, "invalid_tool_budget", None, ToolBudget.initial()), None
    with session._lock:
        decision = authorize_public_tool_call(
            name,
            arguments,
            session._budget,
            available_tool_names=available_tool_names,
        )
        if not decision.allowed or decision.tool_name is None:
            return decision, None
        return (
            decision,
            _IssuedPublicToolGrant(
                issuer=_GRANT_ISSUER,
                tool_name=decision.tool_name,
                session=session,
                nonce=object(),
            ),
        )


def consume_issued_public_tool_grant(
    grant: object,
    *,
    expected_name: str,
    session: object,
) -> tuple[_ToolExecutionReceipt, ToolBudget] | None:
    """Validate host continuity before a graph is allowed to invoke a dependency."""

    if (
        type(grant) is not _IssuedPublicToolGrant
        or grant.issuer is not _GRANT_ISSUER
        or grant.tool_name != expected_name
        or not _is_host_budget_session(session)
        or grant.session is not session
    ):
        return None
    with session._lock:
        if grant.nonce in session._consumed_grant_nonces:
            return None
        spec = PUBLIC_TOOL_REGISTRY.get(expected_name)
        if spec is None:
            return None
        try:
            next_budget = session._budget.consume(spec)
        except ValueError:
            return None
        session._budget = next_budget
        session._consumed_grant_nonces.add(grant.nonce)
        snapshot = _normalized_budget_snapshot(next_budget)
        if snapshot is None:
            return None
        return (
            _ToolExecutionReceipt(
                issuer=_GRANT_ISSUER,
                tool_name=expected_name,
                session=session,
                nonce=grant.nonce,
            ),
            snapshot,
        )


def _redeem_tool_execution_receipt(receipt: object) -> str | None:
    """Atomically bind one verified evidence item to one consumed tool execution."""

    if (
        type(receipt) is not _ToolExecutionReceipt
        or receipt.issuer is not _GRANT_ISSUER
        or not _is_host_budget_session(receipt.session)
    ):
        return None
    with receipt.session._lock:
        if (
            receipt.nonce not in receipt.session._consumed_grant_nonces
            or receipt.nonce in receipt.session._redeemed_receipt_nonces
        ):
            return None
        receipt.session._redeemed_receipt_nonces.add(receipt.nonce)
        return receipt.tool_name
