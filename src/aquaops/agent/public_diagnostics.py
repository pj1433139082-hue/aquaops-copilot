"""Explicit, aggregate-only access to approved public historical monitoring."""

from __future__ import annotations

from datetime import datetime
import re

from aquaops.agent.tools import (
    PublicHistoricalAggregate,
    PublicHistoricalAnomaly,
    ToolResult,
    query_public_historical_window,
    screen_public_historical_anomaly,
)


_HISTORICAL_REQUEST_PATTERN = re.compile(
    r"公开历史诊断 "
    r"(?P<indicator>氨氮|(?i:nh4)|电导|(?i:cond)|流量|(?i:q)) "
    r"(?P<hours>[1-9][0-9]*)h 截止 "
    r"(?P<end_at>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z)",
)
_ANOMALY_REQUEST_PATTERN = re.compile(
    r"公开异常筛查 "
    r"(?P<indicator>氨氮|(?i:nh4)|电导|(?i:cond)|流量|(?i:q)) "
    r"(?P<hours>[1-9][0-9]*)h 截止 "
    r"(?P<end_at>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z)",
)
_INDICATOR_ALIASES = {
    "氨氮": "nh4",
    "nh4": "nh4",
    "电导": "cond",
    "cond": "cond",
    "流量": "q",
    "q": "q",
}


def parse_public_historical_request(question: object) -> tuple[str, int, str] | None:
    """Parse only a fully specified, bounded public-history request."""

    return _parse_explicit_public_request(question, _HISTORICAL_REQUEST_PATTERN)


def parse_public_anomaly_request(question: object) -> tuple[str, int, str] | None:
    """Parse only a fully specified public historical anomaly screen request."""

    return _parse_explicit_public_request(question, _ANOMALY_REQUEST_PATTERN)


def _parse_explicit_public_request(
    question: object, pattern: re.Pattern[str]
) -> tuple[str, int, str] | None:
    if not isinstance(question, str):
        return None
    match = pattern.fullmatch(question.strip())
    if match is None:
        return None

    hours = int(match.group("hours"))
    if not 1 <= hours <= 168:
        return None
    end_at = match.group("end_at")
    try:
        parsed_end_at = datetime.strptime(end_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    if parsed_end_at.strftime("%Y-%m-%dT%H:%M:%SZ") != end_at:
        return None
    return _INDICATOR_ALIASES[match.group("indicator").casefold()], hours, end_at


def run_public_historical_diagnostic(question: object) -> ToolResult | None:
    """Dispatch only explicit historical aggregate or anomaly-screen requests."""

    parsed_request = parse_public_historical_request(question)
    if parsed_request is not None:
        return query_public_historical_window(*parsed_request)
    parsed_anomaly_request = parse_public_anomaly_request(question)
    if parsed_anomaly_request is not None:
        return screen_public_historical_anomaly(*parsed_anomaly_request)
    return None


def _format_number(value: float | int | None) -> str:
    return "无有效值" if value is None else format(value, "g")


def _validated_public_historical_result(
    result: ToolResult,
) -> PublicHistoricalAggregate | PublicHistoricalAnomaly:
    try:
        if not isinstance(result, ToolResult) or not (
            result.ok and result.code == "ok"
        ):
            raise ValueError
        if result.source_kind == "public_historical":
            public_result: PublicHistoricalAggregate | PublicHistoricalAnomaly = (
                PublicHistoricalAggregate.model_validate(result.payload)
            )
        elif result.source_kind == "public_historical_anomaly":
            public_result = PublicHistoricalAnomaly.model_validate(result.payload)
        else:
            raise ValueError
        if result.data_version != public_result.source_version:
            raise ValueError
    except Exception:
        raise ValueError("invalid public historical result") from None
    return public_result


def expected_public_historical_rendering(
    result: ToolResult,
) -> tuple[str, list[dict[str, str]]]:
    """Build the only permitted answer and citations for validated public history."""

    public_result = _validated_public_historical_result(result)
    if isinstance(public_result, PublicHistoricalAnomaly):
        return _render_public_historical_anomaly(public_result)
    aggregate = public_result

    if (aggregate.minimum, aggregate.mean, aggregate.maximum) == (None, None, None):
        statistics = f"最小值、均值、最大值：该窗口无有效数值；单位 {aggregate.unit}"
    else:
        statistics = (
            f"最小值 {_format_number(aggregate.minimum)}，均值 {_format_number(aggregate.mean)}，"
            f"最大值 {_format_number(aggregate.maximum)}；单位 {aggregate.unit}"
        )
    answer = (
        f"公开历史入流监测聚合（指标 {aggregate.indicator}，窗口 "
        f"{aggregate.start} 至 {aggregate.end}）：样本数 {aggregate.row_count}，"
        f"缺失 {aggregate.null_count}，{statistics}。"
        "该结果为非实时、不可用于自动控制的公开历史数据，需人工复核。"
    )
    evidence = [
        {
            "chunk_id": (
                f"{aggregate.source_id}:{aggregate.indicator}:"
                f"{aggregate.start}:{aggregate.end}"
            ),
            "source_url": aggregate.source_url,
            "source_version": aggregate.source_version,
        }
    ]
    return answer, evidence


def _render_public_historical_anomaly(
    anomaly: PublicHistoricalAnomaly,
) -> tuple[str, list[dict[str, str]]]:
    context = (
        f"（指标 {anomaly.indicator}，基线 "
        f"{anomaly.baseline_start} 至 {anomaly.baseline_end}）"
    )
    if anomaly.outcome == "anomaly":
        if anomaly.robust_score is None:
            raise ValueError("invalid public historical result")
        answer = (
            f"公开历史异常筛查{context}：方法 {anomaly.method}，方向 "
            f"{anomaly.direction}，绝对稳健得分 {_format_number(abs(anomaly.robust_score))}，"
            f"筛查阈值 {_format_number(anomaly.threshold)}；结果为异常。"
        )
    elif anomaly.outcome == "normal":
        answer = (
            f"公开历史异常筛查{context}：方法 {anomaly.method}；"
            "未超过公开历史筛查阈值，不等同于正常运行结论。"
        )
    else:
        answer = (
            f"公开历史异常筛查{context}：方法 {anomaly.method}；"
            "历史证据不足以完成筛查。"
        )
    answer += "该结果为非实时、不可用于自动控制的公开历史数据，需人工复核。"
    evidence = [
        {
            "chunk_id": (
                f"{anomaly.source_id}:{anomaly.indicator}:anomaly:"
                f"{anomaly.baseline_start}:{anomaly.target_time}"
            ),
            "source_url": anomaly.source_url,
            "source_version": anomaly.source_version,
        }
    ]
    return answer, evidence


def render_public_historical_answer(
    result: ToolResult,
) -> tuple[str, list[dict[str, str]]]:
    """Render a safety-marked description from one public aggregate only."""

    return expected_public_historical_rendering(result)
