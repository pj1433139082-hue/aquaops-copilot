from dataclasses import fields, replace
from uuid import UUID

import pytest

from aquaops.rag.public_corpus import (
    PublicCorpusError,
    PublicKnowledgeDocument,
    _tokens,
    chunk_public_document,
)


def public_document(text: str) -> PublicKnowledgeDocument:
    return PublicKnowledgeDocument(
        source_id="demo-source",
        title="Demo public guidance",
        source_url="https://example.test/guidance",
        source_version="2026-07",
        license_name="CC BY 4.0",
        text=text,
    )


def test_heading_metadata_and_no_sensitive_fields() -> None:
    chunks = chunk_public_document(
        public_document(
            "# Operations\n\nIntroductory public guidance.\n\n## Sampling\n\nCollect samples carefully."
        ),
        max_tokens=40,
    )

    assert [chunk.section_path for chunk in chunks] == [
        ("Operations",),
        ("Operations", "Sampling"),
    ]
    assert chunks[0].parent_title == "Demo public guidance"
    assert chunks[0].source_url == "https://example.test/guidance"
    assert chunks[0].data_class == "public"
    assert chunks[0].access_policy == "public_read"
    field_names = {field.name for field in fields(chunks[0])}
    assert field_names == {
        "chunk_id",
        "qdrant_point_id",
        "source_id",
        "source_url",
        "source_version",
        "parent_title",
        "section_path",
        "text",
        "data_class",
        "access_policy",
    }


@pytest.mark.parametrize(
    "document",
    [
        lambda: replace(public_document("Public text."), data_class="private"),
        lambda: replace(public_document("Public text."), access_policy="restricted"),
    ],
)
def test_rejects_non_public_access(document) -> None:
    with pytest.raises(PublicCorpusError):
        chunk_public_document(document())


@pytest.mark.parametrize("max_tokens, overlap_tokens", [(0, 0), (4, -1), (4, 4)])
def test_rejects_invalid_chunk_parameters(max_tokens: int, overlap_tokens: int) -> None:
    with pytest.raises(ValueError):
        chunk_public_document(
            public_document("Public text."), max_tokens, overlap_tokens
        )


def test_chunk_ids_are_stable_and_change_for_version_or_text() -> None:
    document = public_document("# Guidance\n\nUse clean containers.")
    original = chunk_public_document(document)[0]

    assert original.chunk_id == chunk_public_document(document)[0].chunk_id
    assert (
        original.chunk_id
        != chunk_public_document(replace(document, source_version="2026-08"))[
            0
        ].chunk_id
    )
    assert (
        original.chunk_id
        != chunk_public_document(
            replace(document, text="# Guidance\n\nUse sterile containers.")
        )[0].chunk_id
    )


def test_qdrant_point_id_is_a_stable_uuid_v5_derived_from_the_chunk_id() -> None:
    document = public_document("# Guidance\n\nUse clean containers.")
    original = chunk_public_document(document)[0]
    same_chunk = chunk_public_document(
        replace(
            document, title="Another title", source_url="https://example.test/other"
        )
    )[0]

    assert len(original.chunk_id) == 64
    assert int(original.chunk_id, 16)
    assert UUID(original.qdrant_point_id).version == 5
    assert original.qdrant_point_id == same_chunk.qdrant_point_id


def test_multiple_chunks_only_overlap_within_the_configured_budget() -> None:
    text = " ".join(f"token{i}" for i in range(20))
    chunks = chunk_public_document(
        public_document(text), max_tokens=6, overlap_tokens=2
    )

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        previous_tokens = _tokens(previous.text)
        current_tokens = _tokens(current.text)
        assert _longest_tail_prefix(previous_tokens, current_tokens) <= 2
        assert len(current_tokens) <= 6
        assert any(token not in previous_tokens for token in current_tokens)


def test_markdown_table_is_not_split_across_chunks() -> None:
    table = "| Metric | Value |\n| --- | --- |\n| pH | 7.2 |\n| Flow | 12 |"
    chunks = chunk_public_document(
        public_document(f"Before table.\n\n{table}\n\nAfter table."),
        max_tokens=30,
        overlap_tokens=0,
    )

    table_chunks = [chunk for chunk in chunks if "| Metric | Value |" in chunk.text]
    assert len(table_chunks) == 1
    assert "| Flow | 12 |" in table_chunks[0].text


def test_rejects_a_table_larger_than_one_chunk() -> None:
    table = "| Metric | Value |\n| --- | --- |\n| one | two |\n| three | four |"

    with pytest.raises(PublicCorpusError, match="table"):
        chunk_public_document(public_document(table), max_tokens=5, overlap_tokens=0)


@pytest.mark.parametrize("text", ["", "# Empty heading\n\n   "])
def test_rejects_empty_documents_or_sections(text: str) -> None:
    with pytest.raises(PublicCorpusError):
        chunk_public_document(public_document(text))


def test_nested_heading_allows_a_bodyless_parent_container() -> None:
    chunks = chunk_public_document(public_document("# Parent\n## Child\nActual body."))

    assert len(chunks) == 1
    assert chunks[0].section_path == ("Parent", "Child")
    assert chunks[0].text == "Actual body."


@pytest.mark.parametrize(
    ("text", "max_tokens", "expected"),
    [
        ("甲乙丙。丁戊己。", 6, ["甲乙丙。", "丁戊己。"]),
        ("one two.three four.", 4, ["one two.", "three four."]),
    ],
)
def test_long_text_prefers_chinese_and_english_sentence_boundaries(
    text: str, max_tokens: int, expected: list[str]
) -> None:
    chunks = chunk_public_document(
        public_document(text), max_tokens=max_tokens, overlap_tokens=0
    )

    assert [chunk.text for chunk in chunks] == expected


def test_window_overlap_never_emits_a_chunk_of_only_prior_overlap() -> None:
    chunks = chunk_public_document(
        public_document("one two.\n\nfive six seven eight nine ten eleven."),
        max_tokens=5,
        overlap_tokens=2,
    )

    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        previous_tokens = _tokens(previous.text)
        current_tokens = _tokens(current.text)
        assert any(token not in previous_tokens for token in current_tokens)
        overlap = _longest_tail_prefix(previous_tokens, current_tokens)
        assert overlap <= 2
        assert len(current_tokens) <= 5


def test_chinese_window_chunks_are_unmodified_source_substrings_with_overlap() -> None:
    text = "甲乙丙丁戊己庚辛"
    chunks = chunk_public_document(
        public_document(text), max_tokens=3, overlap_tokens=1
    )

    assert [chunk.text for chunk in chunks] == ["甲乙丙", "丙丁戊", "戊己庚", "庚辛"]
    assert all(chunk.text in text for chunk in chunks)


def test_cross_paragraph_chunks_keep_the_available_tail_overlap() -> None:
    chunks = chunk_public_document(
        public_document("one two.\n\nthree four."), max_tokens=5, overlap_tokens=2
    )

    assert len(chunks) == 2
    previous_tokens = _tokens(chunks[0].text)
    current_tokens = _tokens(chunks[1].text)
    assert _longest_tail_prefix(previous_tokens, current_tokens) == 2
    assert len(current_tokens) <= 5
    assert any(token not in previous_tokens for token in current_tokens)


def _longest_tail_prefix(previous: list[str], current: list[str]) -> int:
    for size in range(min(len(previous), len(current)), 0, -1):
        if previous[-size:] == current[:size]:
            return size
    return 0
