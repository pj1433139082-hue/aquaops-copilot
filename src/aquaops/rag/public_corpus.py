"""Deterministic chunking for caller-supplied public knowledge text."""

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Literal
from uuid import UUID, uuid5


class PublicCorpusError(ValueError):
    """Raised when a document is not eligible for public-corpus ingestion."""


@dataclass(frozen=True)
class PublicKnowledgeDocument:
    source_id: str
    title: str
    source_url: str
    source_version: str
    license_name: str
    text: str
    data_class: Literal["public"] = "public"
    access_policy: Literal["public_read"] = "public_read"


@dataclass(frozen=True)
class PublicKnowledgeChunk:
    chunk_id: str
    qdrant_point_id: str
    source_id: str
    source_url: str
    source_version: str
    parent_title: str
    section_path: tuple[str, ...]
    text: str
    data_class: Literal["public"] = "public"
    access_policy: Literal["public_read"] = "public_read"


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_TOKENS = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]")
_SENTENCES = re.compile(r".+?[.!?。！？]|.+$", re.DOTALL)
_QDRANT_NAMESPACE = UUID("ce3be424-4f91-55a3-8ed0-dd56dc95c39a")
_TokenSpan = tuple[str, int, int]


@dataclass(frozen=True)
class _Segment:
    tokens: tuple[_TokenSpan, ...]
    is_table: bool = False


def _tokens(text: str) -> list[str]:
    return _TOKENS.findall(text)


def _token_spans(text: str, offset: int = 0) -> tuple[_TokenSpan, ...]:
    return tuple(
        (match.group(), offset + match.start(), offset + match.end())
        for match in _TOKENS.finditer(text)
    )


def _is_table(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    return bool(lines) and (
        all(line.lstrip().startswith("|") for line in lines)
        or any(_TABLE_SEPARATOR.match(line) for line in lines)
    )


def _sections(text: str) -> list[tuple[tuple[str, ...], str]]:
    sections: list[tuple[tuple[str, ...], str]] = []
    hierarchy: list[str] = []
    body: list[str] = []
    current_path: tuple[str, ...] = ()

    def finish() -> None:
        nonlocal body
        content = "\n".join(body).strip()
        if content:
            sections.append((current_path, content))
        body = []

    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            finish()
            level, title = len(heading.group(1)), heading.group(2).strip()
            hierarchy[level - 1 :] = [title]
            current_path = tuple(hierarchy)
        else:
            body.append(line)
    finish()
    return sections


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for separator in re.finditer(r"\n[ \t]*\n", text):
        trimmed = _trim_span(text, start, separator.start())
        if trimmed[0] < trimmed[1]:
            spans.append(trimmed)
        start = separator.end()
    trimmed = _trim_span(text, start, len(text))
    if trimmed[0] < trimmed[1]:
        spans.append(trimmed)
    return spans


def _segments(section_text: str, max_tokens: int) -> list[_Segment]:
    result: list[_Segment] = []
    for start, end in _paragraph_spans(section_text):
        block = section_text[start:end]
        token_spans = _token_spans(block, start)
        if not token_spans:
            continue
        if _is_table(block):
            if len(token_spans) > max_tokens:
                raise PublicCorpusError("a Markdown table is larger than one chunk")
            result.append(_Segment(token_spans, is_table=True))
            continue
        if len(token_spans) <= max_tokens:
            result.append(_Segment(token_spans))
            continue
        for sentence in _SENTENCES.finditer(block):
            sentence_start, sentence_end = _trim_span(
                text=block, start=sentence.start(), end=sentence.end()
            )
            sentence_spans = _token_spans(
                block[sentence_start:sentence_end], start + sentence_start
            )
            if sentence_spans:
                result.append(_Segment(sentence_spans))
    return result


def _chunk_id(
    document: PublicKnowledgeDocument, path: tuple[str, ...], text: str
) -> str:
    stable_input = json.dumps(
        [document.source_id, document.source_version, list(path), text],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256(stable_input.encode("utf-8")).hexdigest()


def _qdrant_point_id(chunk_id: str) -> str:
    return str(uuid5(_QDRANT_NAMESPACE, chunk_id))


def chunk_public_document(
    document: PublicKnowledgeDocument,
    max_tokens: int = 360,
    overlap_tokens: int = 36,
) -> list[PublicKnowledgeChunk]:
    """Split a caller-provided, public-read document without any I/O."""
    if document.data_class != "public" or document.access_policy != "public_read":
        raise PublicCorpusError("only public, public_read documents may be chunked")
    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError(
            "max_tokens must be positive and overlap_tokens must be smaller"
        )
    if not document.text.strip():
        raise PublicCorpusError("document text is empty")

    output: list[PublicKnowledgeChunk] = []
    for path, section_text in _sections(document.text):
        current: list[_TokenSpan] = []
        current_has_table = False
        previous: tuple[_TokenSpan, ...] = ()
        previous_can_overlap = False

        def emit(tokens: list[_TokenSpan], has_table: bool) -> None:
            nonlocal previous, previous_can_overlap
            if not tokens:
                return
            chunk_text = section_text[tokens[0][1] : tokens[-1][2]]
            chunk_id = _chunk_id(document, path, chunk_text)
            output.append(
                PublicKnowledgeChunk(
                    chunk_id=chunk_id,
                    qdrant_point_id=_qdrant_point_id(chunk_id),
                    source_id=document.source_id,
                    source_url=document.source_url,
                    source_version=document.source_version,
                    parent_title=document.title,
                    section_path=path,
                    text=chunk_text,
                )
            )
            previous = tuple(tokens)
            previous_can_overlap = not has_table

        def overlap_prefix(segment: _Segment) -> list[_TokenSpan]:
            if not overlap_tokens or not previous_can_overlap:
                return []
            carry = list(previous[-overlap_tokens:])
            return carry if len(carry) + len(segment.tokens) <= max_tokens else []

        for segment in _segments(section_text, max_tokens):
            if len(segment.tokens) > max_tokens:
                if current:
                    emit(current, current_has_table)
                    current = []
                    current_has_table = False

                carry = (
                    list(previous[-overlap_tokens:])
                    if overlap_tokens and previous_can_overlap
                    else []
                )
                consumed = 0
                while consumed < len(segment.tokens):
                    prefix = carry if carry else []
                    capacity = max_tokens - len(prefix)
                    new_tokens = list(segment.tokens[consumed : consumed + capacity])
                    if not new_tokens:
                        break
                    emitted = prefix + new_tokens
                    emit(emitted, has_table=False)
                    consumed += len(new_tokens)
                    carry = list(previous[-overlap_tokens:]) if overlap_tokens else []
                continue

            if not current:
                current = overlap_prefix(segment) + list(segment.tokens)
                current_has_table = segment.is_table
                continue
            if len(current) + len(segment.tokens) <= max_tokens:
                current.extend(segment.tokens)
                current_has_table = current_has_table or segment.is_table
                continue

            emit(current, current_has_table)
            current = overlap_prefix(segment) + list(segment.tokens)
            current_has_table = segment.is_table

        emit(current, current_has_table)

    if not output:
        raise PublicCorpusError("no public knowledge chunks could be generated")
    return output
