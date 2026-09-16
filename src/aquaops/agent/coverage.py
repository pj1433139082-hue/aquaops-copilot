"""Strict A/B/C evidence coverage gate and fixed safe answer renderer."""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import re
from typing import Literal, Mapping
from urllib.parse import urlsplit
from uuid import uuid4
import unicodedata

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    TypeAdapter,
    field_validator,
    model_validator,
)

from aquaops.agent.policy import _redeem_tool_execution_receipt


_DIMENSIONS: tuple[Literal["A", "B", "C"], ...] = ("A", "B", "C")
_HTTP_URL = TypeAdapter(HttpUrl)
_SAFE_EVIDENCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.:-]{0,255}$")
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._() -]{0,63}$")
_SAFE_TRACE_ID = re.compile(r"^[a-f0-9]{32}$")
_SENSITIVE_MARKERS = (
    "password",
    "private",
    "query",
    "secret",
    "token",
    "raw_value",
    "sql",
)
_LOCAL_HOSTS = frozenset({"localhost", "local", "internal"})
_LOCAL_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".corp", ".home.arpa")
_TOOL_ALLOWED_DIMENSIONS: Mapping[str, tuple[Literal["A", "B", "C"], ...]] = {
    "retrieve_public_water_knowledge": ("C",),
    "aggregate_public_history": ("A",),
    "screen_public_anomaly": ("B",),
}
_VERIFIED_EVIDENCE_ISSUER = object()


class CoverageValidationError(ValueError):
    """A caller attempted to bypass an A/B/C evidence boundary."""


def _public_https_url(value: object) -> bool:
    if type(value) is not str:
        return False
    try:
        parsed = _HTTP_URL.validate_python(value)
    except Exception:
        return False
    split = urlsplit(str(parsed))
    host = split.hostname
    if split.scheme != "https" or not host:
        return False
    normalized = host.casefold().rstrip(".")
    if normalized in _LOCAL_HOSTS or normalized.endswith(_LOCAL_SUFFIXES):
        return False
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _safe_text(value: object) -> bool:
    if type(value) is not str or not value.strip() or len(value) > 280:
        return False
    normalized = value.casefold()
    if any(marker in normalized for marker in _SENSITIVE_MARKERS):
        return False
    if any(character in value for character in "<>`[]*_#"):
        return False
    return not any(
        character in {"\n", "\r", "\u2028", "\u2029"}
        or unicodedata.category(character).startswith("C")
        or unicodedata.category(character) in {"Zl", "Zp"}
        for character in value
    )


class PublicEvidenceReference(BaseModel):
    """Public metadata only: never text, rows, values, paths, or source payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    evidence_id: str = Field(min_length=1, max_length=256)
    source_url: str
    source_version: str = Field(min_length=1, max_length=64)
    data_class: Literal["public"] = "public"
    access_policy: Literal["public_read"] = "public_read"

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        if not _SAFE_EVIDENCE_ID.fullmatch(value) or any(
            marker in value.casefold() for marker in _SENSITIVE_MARKERS
        ):
            raise ValueError("evidence id is not safe public metadata")
        return value

    @field_validator("source_url")
    @classmethod
    def validate_public_url(cls, value: str) -> str:
        if not _public_https_url(value):
            raise ValueError("evidence URL must be a public HTTPS source")
        return value

    @field_validator("source_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if not _SAFE_VERSION.fullmatch(value) or any(
            marker in value.casefold() for marker in _SENSITIVE_MARKERS
        ):
            raise ValueError("evidence version is unsafe")
        return value


@dataclass(frozen=True)
class _VerifiedEvidenceBinding:
    issuer: object
    tool_name: str
    reference_fingerprint: tuple[str, str, str, str, str]


@dataclass(frozen=True)
class VerifiedPublicEvidence:
    """Opaque provenance capability; only receipt redemption can create one."""

    _issuer: object
    _binding: _VerifiedEvidenceBinding
    _reference: PublicEvidenceReference


def _reference_fingerprint(
    reference: PublicEvidenceReference,
) -> tuple[str, str, str, str, str]:
    return (
        reference.evidence_id,
        reference.source_url,
        reference.source_version,
        reference.data_class,
        reference.access_policy,
    )


def mint_verified_public_evidence(
    *,
    receipt: object,
    evidence_id: str,
    source_url: str,
    source_version: str,
) -> VerifiedPublicEvidence:
    """Redeem one consumed tool execution into one trusted public evidence item."""

    return mint_verified_public_evidence_batch(
        receipt=receipt,
        references=((evidence_id, source_url, source_version),),
    )[0]


def mint_verified_public_evidence_batch(
    *,
    receipt: object,
    references: tuple[tuple[str, str, str], ...],
) -> tuple[VerifiedPublicEvidence, ...]:
    """Redeem one execution receipt into a bounded immutable evidence batch."""

    if type(references) is not tuple or not 1 <= len(references) <= 6:
        raise ValueError("public evidence batch must contain between one and six items")
    normalized: list[PublicEvidenceReference] = []
    seen: set[tuple[str, str, str]] = set()
    for item in references:
        if type(item) is not tuple or len(item) != 3:
            raise ValueError("public evidence batch item is invalid")
        reference = PublicEvidenceReference(
            evidence_id=item[0],
            source_url=item[1],
            source_version=item[2],
        )
        key = (reference.evidence_id, reference.source_url, reference.source_version)
        if key in seen:
            raise ValueError("public evidence batch contains duplicate evidence")
        seen.add(key)
        normalized.append(reference)

    tool_name = _redeem_tool_execution_receipt(receipt)
    if tool_name not in _TOOL_ALLOWED_DIMENSIONS:
        raise ValueError("receipt cannot mint public evidence")
    return tuple(
        VerifiedPublicEvidence(
            _issuer=_VERIFIED_EVIDENCE_ISSUER,
            _binding=_VerifiedEvidenceBinding(
                issuer=_VERIFIED_EVIDENCE_ISSUER,
                tool_name=tool_name,
                reference_fingerprint=_reference_fingerprint(reference),
            ),
            _reference=reference,
        )
        for reference in normalized
    )


class ABCRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dimension: Literal["A", "B", "C"]
    requirement: str = Field(min_length=1, max_length=280)

    @field_validator("requirement")
    @classmethod
    def validate_requirement(cls, value: str) -> str:
        if not _safe_text(value):
            raise ValueError("requirement must be safe display text")
        return value


class ABCPlan(BaseModel):
    """Exactly three immutable requirements in the user-visible A/B/C order."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dimensions: tuple[ABCRequirement, ABCRequirement, ABCRequirement]

    @model_validator(mode="after")
    def require_exact_abc_order(self) -> "ABCPlan":
        if tuple(item.dimension for item in self.dimensions) != _DIMENSIONS:
            raise ValueError("ABC plan dimensions must be exactly A, B, C in order")
        return self

    @classmethod
    def from_requirements(cls, *, a: str, b: str, c: str) -> "ABCPlan":
        return cls(
            dimensions=(
                ABCRequirement(dimension="A", requirement=a),
                ABCRequirement(dimension="B", requirement=b),
                ABCRequirement(dimension="C", requirement=c),
            )
        )


class DimensionCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dimension: Literal["A", "B", "C"]
    requirement: str
    evidence: tuple[PublicEvidenceReference, ...] = ()
    conclusion: str | None = None
    missing_reason: Literal["public_evidence_missing", "conclusion_missing"] | None = (
        None
    )

    @model_validator(mode="after")
    def require_evidence_or_explicit_missing_reason(self) -> "DimensionCoverage":
        if self.conclusion is None:
            if self.missing_reason is None:
                raise ValueError(
                    "missing conclusion requires an explicit missing reason"
                )
        elif not self.evidence or self.missing_reason is not None:
            raise ValueError("conclusion requires evidence and no missing reason")
        elif not _safe_text(self.conclusion):
            raise ValueError("conclusion must be safe display text")
        return self


class CoverageGateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dimensions: tuple[DimensionCoverage, DimensionCoverage, DimensionCoverage]
    complete: bool
    trace_id: str = Field(min_length=32, max_length=32)

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, value: str) -> str:
        if not _SAFE_TRACE_ID.fullmatch(value):
            raise ValueError("trace id must be a 32-character lowercase hex value")
        return value

    @model_validator(mode="after")
    def validate_complete_status(self) -> "CoverageGateResult":
        if tuple(item.dimension for item in self.dimensions) != _DIMENSIONS:
            raise ValueError("coverage dimensions must be exactly A, B, C in order")
        is_complete = all(item.conclusion is not None for item in self.dimensions)
        if self.complete != is_complete:
            raise ValueError("coverage completion flag is inconsistent")
        return self


class FixedAgentAnswer(BaseModel):
    """Only renderable final answer contract for the new A/B/C execution path."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    dimensions: tuple[DimensionCoverage, DimensionCoverage, DimensionCoverage]
    evidence: tuple[PublicEvidenceReference, ...]
    human_review_required: Literal[True] = True
    next_safe_action: Literal[
        "collect_missing_public_evidence", "human_review_before_any_operational_action"
    ]
    trace_id: str = Field(min_length=32, max_length=32)

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, value: str) -> str:
        if not _SAFE_TRACE_ID.fullmatch(value):
            raise ValueError("trace id must be a 32-character lowercase hex value")
        return value

    @model_validator(mode="after")
    def validate_fixed_answer(self) -> "FixedAgentAnswer":
        if tuple(item.dimension for item in self.dimensions) != _DIMENSIONS:
            raise ValueError("fixed answer dimensions must be A, B, C in order")
        expected: list[PublicEvidenceReference] = []
        seen: set[tuple[str, str, str]] = set()
        for dimension in self.dimensions:
            for evidence in dimension.evidence:
                key = (
                    evidence.evidence_id,
                    evidence.source_url,
                    evidence.source_version,
                )
                if key not in seen:
                    expected.append(evidence)
                    seen.add(key)
        if tuple(expected) != self.evidence:
            raise ValueError("fixed answer evidence must equal its dimension evidence")
        if all(item.conclusion is not None for item in self.dimensions):
            if self.next_safe_action != "human_review_before_any_operational_action":
                raise ValueError("complete evidence still requires human review")
        elif self.next_safe_action != "collect_missing_public_evidence":
            raise ValueError("incomplete evidence needs the safe collection action")
        return self

    @classmethod
    def from_coverage(cls, coverage: CoverageGateResult) -> "FixedAgentAnswer":
        coverage = _normalize_coverage_result(coverage)
        evidence: list[PublicEvidenceReference] = []
        seen: set[tuple[str, str, str]] = set()
        for dimension in coverage.dimensions:
            for item in dimension.evidence:
                key = (item.evidence_id, item.source_url, item.source_version)
                if key not in seen:
                    evidence.append(item)
                    seen.add(key)
        return cls(
            dimensions=coverage.dimensions,
            evidence=tuple(evidence),
            next_safe_action=(
                "human_review_before_any_operational_action"
                if coverage.complete
                else "collect_missing_public_evidence"
            ),
            trace_id=coverage.trace_id,
        )


def _evidence_sequence(
    value: object, dimension: str
) -> tuple[PublicEvidenceReference, ...]:
    if type(value) not in {list, tuple}:
        raise CoverageValidationError(
            f"{dimension} evidence sequence violates the dimension boundary"
        )
    references: list[PublicEvidenceReference] = []
    for raw_evidence in value:
        try:
            reference, allowed_dimensions = _normalize_verified_evidence(raw_evidence)
        except Exception as exc:
            raise CoverageValidationError(
                f"{dimension} evidence is not verified public evidence"
            ) from exc
        if dimension not in allowed_dimensions:
            raise CoverageValidationError(
                f"{dimension} evidence violates the producing tool dimension boundary"
            )
        references.append(reference)
    return tuple(references)


def _normalize_verified_evidence(
    evidence: object,
) -> tuple[PublicEvidenceReference, tuple[Literal["A", "B", "C"], ...]]:
    if (
        type(evidence) is not VerifiedPublicEvidence
        or evidence._issuer is not _VERIFIED_EVIDENCE_ISSUER
        or type(evidence._binding) is not _VerifiedEvidenceBinding
        or evidence._binding.issuer is not _VERIFIED_EVIDENCE_ISSUER
        or evidence._binding.tool_name not in _TOOL_ALLOWED_DIMENSIONS
        or type(evidence._reference) is not PublicEvidenceReference
    ):
        raise ValueError("evidence provenance capability is invalid")
    reference = PublicEvidenceReference.model_validate(evidence._reference.model_dump())
    if _reference_fingerprint(reference) != evidence._binding.reference_fingerprint:
        raise ValueError("evidence metadata does not match its provenance binding")
    return reference, _TOOL_ALLOWED_DIMENSIONS[evidence._binding.tool_name]


def _normalize_plan(plan: object) -> ABCPlan:
    if type(plan) is not ABCPlan:
        raise CoverageValidationError("coverage requires an exact ABC plan")
    try:
        return ABCPlan.model_validate(plan.model_dump())
    except Exception as exc:
        raise CoverageValidationError("ABC plan is not a strict safe model") from exc


def _normalize_coverage_result(coverage: object) -> CoverageGateResult:
    if type(coverage) is not CoverageGateResult:
        raise ValueError("coverage result must be an exact strict model")
    return CoverageGateResult.model_validate(coverage.model_dump())


def coverage_gate(
    plan: ABCPlan,
    *,
    evidence_by_dimension: Mapping[str, object],
    conclusions: Mapping[str, object],
) -> CoverageGateResult:
    """Create all A/B/C outcomes without ever inferring evidence across dimensions."""

    plan = _normalize_plan(plan)
    if not isinstance(evidence_by_dimension, Mapping) or not isinstance(
        conclusions, Mapping
    ):
        raise CoverageValidationError("coverage inputs must be mappings")
    unknown_keys = (set(evidence_by_dimension) | set(conclusions)) - set(_DIMENSIONS)
    if unknown_keys:
        raise CoverageValidationError("coverage input contains an unknown dimension")

    covered: list[DimensionCoverage] = []
    for requirement in plan.dimensions:
        dimension = requirement.dimension
        evidence = _evidence_sequence(
            evidence_by_dimension.get(dimension, ()), dimension
        )
        raw_conclusion = conclusions.get(dimension)
        if not evidence:
            covered.append(
                DimensionCoverage(
                    dimension=dimension,
                    requirement=requirement.requirement,
                    evidence=(),
                    conclusion=None,
                    missing_reason="public_evidence_missing",
                )
            )
        elif type(raw_conclusion) is not str or not _safe_text(raw_conclusion):
            covered.append(
                DimensionCoverage(
                    dimension=dimension,
                    requirement=requirement.requirement,
                    evidence=evidence,
                    conclusion=None,
                    missing_reason="conclusion_missing",
                )
            )
        else:
            covered.append(
                DimensionCoverage(
                    dimension=dimension,
                    requirement=requirement.requirement,
                    evidence=evidence,
                    conclusion=raw_conclusion,
                    missing_reason=None,
                )
            )
    dimensions = (covered[0], covered[1], covered[2])
    return CoverageGateResult(
        dimensions=dimensions,
        complete=all(item.conclusion is not None for item in dimensions),
        trace_id=uuid4().hex,
    )


def render_fixed_agent_answer(answer: FixedAgentAnswer) -> str:
    """Render only the fixed schema in A/B/C order; arbitrary dictionaries fail."""

    if type(answer) is not FixedAgentAnswer:
        raise TypeError("only a validated FixedAgentAnswer is renderable")
    answer = FixedAgentAnswer.model_validate(answer.model_dump())
    lines: list[str] = []
    for dimension in answer.dimensions:
        lines.append(f"{dimension.dimension}. {dimension.requirement}")
        if dimension.conclusion is None:
            lines.append(f"结论：缺失（{dimension.missing_reason}）")
        else:
            lines.append(f"结论：{dimension.conclusion}")
        evidence_text = ", ".join(item.evidence_id for item in dimension.evidence)
        lines.append(f"证据引用：{evidence_text if evidence_text else '无'}")
    lines.append("人工复核：需要")
    lines.append(f"下一安全动作：{answer.next_safe_action}")
    lines.append(f"追踪ID：{answer.trace_id}")
    return "\n".join(lines)
