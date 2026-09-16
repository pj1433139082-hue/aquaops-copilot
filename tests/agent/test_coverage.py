import pytest
from pydantic import ValidationError

from aquaops.agent.policy import (
    consume_issued_public_tool_grant,
    create_public_tool_budget_session,
    issue_public_tool_grant,
)
from aquaops.agent.coverage import (
    ABCPlan,
    CoverageValidationError,
    FixedAgentAnswer,
    PublicEvidenceReference,
    VerifiedPublicEvidence,
    mint_verified_public_evidence,
    coverage_gate,
    render_fixed_agent_answer,
)


def _plan() -> ABCPlan:
    return ABCPlan.from_requirements(
        a="A 维度：公开历史趋势",
        b="B 维度：公开异常筛查",
        c="C 维度：公开知识依据",
    )


def _evidence(identifier: str) -> VerifiedPublicEvidence:
    produced_by_tool = {
        "a": "aggregate_public_history",
        "b": "screen_public_anomaly",
        "c": "retrieve_public_water_knowledge",
    }[identifier.split(":")[1]]
    arguments = {
        "aggregate_public_history": {
            "indicator": "nh4",
            "hours": 24,
            "end_at": "2025-01-02T00:00:00Z",
        },
        "screen_public_anomaly": {
            "indicator": "nh4",
            "baseline_hours": 24,
            "end_at": "2025-01-02T00:00:00Z",
        },
        "retrieve_public_water_knowledge": {"query": "公开水质知识"},
    }[produced_by_tool]
    session = create_public_tool_budget_session()
    decision, grant = issue_public_tool_grant(produced_by_tool, arguments, session)
    assert decision.allowed is True and grant is not None
    consumed = consume_issued_public_tool_grant(
        grant, expected_name=produced_by_tool, session=session
    )
    assert consumed is not None
    receipt, _ = consumed
    return mint_verified_public_evidence(
        receipt=receipt,
        evidence_id=identifier,
        source_url="https://zenodo.org/records/15285089",
        source_version="v1.0.0 (2025-04-26)",
    )


def test_coverage_gate_never_lends_c_evidence_to_missing_a_or_b() -> None:
    result = coverage_gate(
        _plan(),
        evidence_by_dimension={"C": [_evidence("public:c:1")]},
        conclusions={"A": "不得借用 C", "B": "不得借用 C", "C": "仅 C 有依据"},
    )

    assert result.complete is False
    assert [item.dimension for item in result.dimensions] == ["A", "B", "C"]
    a, b, c = result.dimensions
    assert a.evidence == () and a.conclusion is None
    assert a.missing_reason == "public_evidence_missing"
    assert b.evidence == () and b.conclusion is None
    assert b.missing_reason == "public_evidence_missing"
    assert tuple(item.evidence_id for item in c.evidence) == ("public:c:1",)
    assert c.conclusion == "仅 C 有依据"
    assert c.missing_reason is None


@pytest.mark.parametrize(
    "reference",
    [
        {
            "evidence_id": "public:a:1",
            "source_url": "file:///secret.csv",
            "source_version": "v1",
            "produced_by_tool": "aggregate_public_history",
            "allowed_dimensions": ("A",),
        },
        {
            "evidence_id": "query=private",
            "source_url": "https://zenodo.org/records/15285089",
            "source_version": "v1",
            "produced_by_tool": "aggregate_public_history",
            "allowed_dimensions": ("A",),
        },
        {
            "evidence_id": "public:a:1",
            "source_url": "https://localhost/private",
            "source_version": "v1",
            "produced_by_tool": "aggregate_public_history",
            "allowed_dimensions": ("A",),
        },
        {
            "evidence_id": "public:a:1",
            "source_url": "https://zenodo.org/records/15285089",
            "source_version": "secret-token",
            "produced_by_tool": "aggregate_public_history",
            "allowed_dimensions": ("A",),
        },
    ],
)
def test_public_evidence_reference_rejects_non_public_or_malformed_metadata(
    reference: dict[str, str],
) -> None:
    with pytest.raises(ValidationError):
        PublicEvidenceReference.model_validate(reference)


def test_coverage_gate_rejects_cross_dimension_evidence_label() -> None:
    with pytest.raises(CoverageValidationError, match="dimension"):
        coverage_gate(
            _plan(),
            evidence_by_dimension={"A": {"C": [_evidence("public:c:1")]}},
            conclusions={"A": "错误嵌套"},
        )


def test_fixed_answer_schema_and_renderer_have_exact_abc_order_and_no_free_fields() -> (
    None
):
    gated = coverage_gate(
        _plan(),
        evidence_by_dimension={
            "A": [_evidence("public:a:1")],
            "B": [_evidence("public:b:1")],
            "C": [_evidence("public:c:1")],
        },
        conclusions={"A": "趋势已核验", "B": "筛查已核验", "C": "依据已核验"},
    )
    answer = FixedAgentAnswer.from_coverage(gated)

    assert answer.human_review_required is True
    assert answer.next_safe_action == "human_review_before_any_operational_action"
    assert [item.dimension for item in answer.dimensions] == ["A", "B", "C"]
    assert tuple(item.evidence_id for item in answer.evidence) == (
        "public:a:1",
        "public:b:1",
        "public:c:1",
    )
    assert answer.trace_id == gated.trace_id
    assert len(gated.trace_id) == 32
    rendered = render_fixed_agent_answer(answer)
    assert rendered.index("A.") < rendered.index("B.") < rendered.index("C.")
    assert "人工复核：需要" in rendered
    assert "下一安全动作：human_review_before_any_operational_action" in rendered

    with pytest.raises(ValidationError):
        FixedAgentAnswer.model_validate({**answer.model_dump(), "raw_values": [1]})
    with pytest.raises(TypeError):
        render_fixed_agent_answer(answer.model_dump())  # type: ignore[arg-type]


def test_incomplete_coverage_is_still_renderable_with_explicit_missing_reasons() -> (
    None
):
    gated = coverage_gate(
        _plan(),
        evidence_by_dimension={"C": [_evidence("public:c:1")]},
        conclusions={"C": "仅有公开知识依据"},
    )
    answer = FixedAgentAnswer.from_coverage(gated)
    rendered = render_fixed_agent_answer(answer)

    assert answer.next_safe_action == "collect_missing_public_evidence"
    assert "A. A 维度：公开历史趋势\n结论：缺失（public_evidence_missing）" in rendered
    assert "B. B 维度：公开异常筛查\n结论：缺失（public_evidence_missing）" in rendered
    assert "C. C 维度：公开知识依据\n结论：仅有公开知识依据" in rendered


def test_evidence_without_a_safe_conclusion_is_renderable_as_missing_not_an_exception() -> (
    None
):
    gated = coverage_gate(
        _plan(),
        evidence_by_dimension={"A": [_evidence("public:a:1")]},
        conclusions={},
    )

    a, b, c = gated.dimensions
    assert tuple(item.evidence_id for item in a.evidence) == ("public:a:1",)
    assert a.conclusion is None
    assert a.missing_reason == "conclusion_missing"
    assert b.missing_reason == "public_evidence_missing"
    assert c.missing_reason == "public_evidence_missing"
    assert FixedAgentAnswer.from_coverage(gated)


def test_coverage_revalidates_model_copy_and_rejects_wrong_tool_dimension() -> None:
    c_only_evidence = _evidence("public:c:1")
    with pytest.raises(CoverageValidationError, match="dimension"):
        coverage_gate(
            _plan(),
            evidence_by_dimension={"A": [c_only_evidence]},
            conclusions={"A": "不允许借用 C"},
        )

    forged_url = _evidence("public:a:1")
    object.__setattr__(forged_url._reference, "source_url", "https://localhost/private")
    with pytest.raises(CoverageValidationError, match="verified public"):
        coverage_gate(
            _plan(),
            evidence_by_dimension={"A": [forged_url]},
            conclusions={"A": "公开依据"},
        )

    forged_reference = _evidence("public:a:1")
    object.__setattr__(
        forged_reference,
        "_reference",
        PublicEvidenceReference(
            evidence_id="public:a:forged",
            source_url="https://zenodo.org/records/15285089",
            source_version="v1.0.0 (2025-04-26)",
        ),
    )
    with pytest.raises(CoverageValidationError, match="verified public"):
        coverage_gate(
            _plan(),
            evidence_by_dimension={"A": [forged_reference]},
            conclusions={"A": "公开依据"},
        )


def test_fixed_renderer_revalidates_model_copy_and_rejects_markup_or_line_separators() -> (
    None
):
    with pytest.raises(ValidationError):
        ABCPlan.from_requirements(a="<script>", b="公开 B", c="公开 C")
    with pytest.raises(ValidationError):
        ABCPlan.from_requirements(a="公开 A\u2028绕过", b="公开 B", c="公开 C")
    with pytest.raises(ValidationError):
        PublicEvidenceReference(
            evidence_id="public__markdown",
            source_url="https://zenodo.org/records/15285089",
            source_version="v1.0.0 (2025-04-26)",
        )

    gated = coverage_gate(
        _plan(),
        evidence_by_dimension={"A": [_evidence("public:a:1")]},
        conclusions={"A": "公开依据"},
    )
    answer = FixedAgentAnswer.from_coverage(gated)
    forged_dimension = answer.dimensions[0].model_copy(
        update={"conclusion": "<script>alert(1)</script>"}
    )
    forged_answer = answer.model_copy(
        update={"dimensions": (forged_dimension, *answer.dimensions[1:])}
    )
    with pytest.raises(ValidationError):
        render_fixed_agent_answer(forged_answer)


def test_coverage_rejects_plain_public_metadata_and_forged_provenance_strings() -> None:
    plain_reference = PublicEvidenceReference(
        evidence_id="public:a:1",
        source_url="https://zenodo.org/records/15285089",
        source_version="v1.0.0 (2025-04-26)",
    )
    with pytest.raises(CoverageValidationError, match="verified"):
        coverage_gate(
            _plan(),
            evidence_by_dimension={"A": [plain_reference]},
            conclusions={"A": "伪造来源"},
        )
    with pytest.raises(ValidationError):
        PublicEvidenceReference.model_validate(
            {
                **plain_reference.model_dump(),
                "produced_by_tool": "aggregate_public_history",
            }
        )
    with pytest.raises(ValueError, match="receipt"):
        mint_verified_public_evidence(
            receipt=object(),
            evidence_id="public:a:1",
            source_url="https://zenodo.org/records/15285089",
            source_version="v1.0.0 (2025-04-26)",
        )


def test_consumed_receipt_mints_once_and_its_tool_cannot_cross_dimensions() -> None:
    aggregate_evidence = _evidence("public:a:1")
    with pytest.raises(CoverageValidationError, match="dimension"):
        coverage_gate(
            _plan(),
            evidence_by_dimension={"B": [aggregate_evidence]},
            conclusions={"B": "不能跨工具维度"},
        )

    session = create_public_tool_budget_session()
    decision, grant = issue_public_tool_grant(
        "aggregate_public_history",
        {"indicator": "nh4", "hours": 24, "end_at": "2025-01-02T00:00:00Z"},
        session,
    )
    assert decision.allowed is True and grant is not None
    consumed = consume_issued_public_tool_grant(
        grant, expected_name="aggregate_public_history", session=session
    )
    assert consumed is not None
    receipt, _ = consumed
    mint_verified_public_evidence(
        receipt=receipt,
        evidence_id="public:a:2",
        source_url="https://zenodo.org/records/15285089",
        source_version="v1.0.0 (2025-04-26)",
    )
    with pytest.raises(ValueError, match="receipt"):
        mint_verified_public_evidence(
            receipt=receipt,
            evidence_id="public:a:3",
            source_url="https://zenodo.org/records/15285089",
            source_version="v1.0.0 (2025-04-26)",
        )
