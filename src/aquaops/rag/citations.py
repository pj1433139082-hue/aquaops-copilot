from pydantic import BaseModel

from aquaops.rag.retrieve import RetrievedChunk


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[dict[str, str]]
    requires_human_review: bool


def build_grounded_answer(
    question: str,
    chunks: list[RetrievedChunk],
) -> GroundedAnswer:
    """Build a citation-gated answer from caller-supplied public evidence only."""
    if not chunks:
        return GroundedAnswer(
            answer="信息不足，无法基于已公开证据给出建议。",
            citations=[],
            requires_human_review=True,
        )

    citations = [
        {"chunk_id": chunk.chunk_id, "source_url": str(chunk.source_url)}
        for chunk in chunks
    ]
    return GroundedAnswer(
        answer=(f"已检索到 {len(chunks)} 条公开证据；请结合人工复核处理：{question}"),
        citations=citations,
        requires_human_review=True,
    )
