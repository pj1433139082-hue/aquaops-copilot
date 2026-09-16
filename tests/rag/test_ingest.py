from hashlib import sha256

from aquaops.rag.ingest import ingest_document
from aquaops.rag.models import SourceDocument


def test_ingest_rejects_a_document_without_license() -> None:
    source = SourceDocument(
        title="未授权文件",
        source_url="https://example.invalid",
        license_name="",
        text="x",
    )

    result = ingest_document(source)

    assert result.accepted is False
    assert result.reason == "missing_license"


def test_ingest_rejects_short_licensed_content() -> None:
    source = SourceDocument(
        title="短文本",
        source_url="https://example.invalid",
        license_name="CC BY 4.0",
        text="a" * 99,
    )

    result = ingest_document(source)

    assert result.accepted is False
    assert result.reason == "insufficient_content"
    assert result.content_hash is None


def test_ingest_accepts_licensed_content_with_a_stable_hash() -> None:
    text = "a" * 100
    source = SourceDocument(
        title="公开资料",
        source_url="https://example.invalid",
        license_name="CC BY 4.0",
        text=text,
    )

    result = ingest_document(source)

    assert result.accepted is True
    assert result.reason == "accepted"
    assert result.content_hash == sha256(text.encode("utf-8")).hexdigest()


def test_same_text_has_a_reproducible_content_hash() -> None:
    text = "可公开使用的合成文本。" * 10
    first = SourceDocument(
        title="版本一",
        source_url="https://example.invalid",
        license_name="CC BY 4.0",
        text=text,
    )
    second = SourceDocument(
        title="版本二",
        source_url="https://example.invalid",
        license_name="CC BY 4.0",
        text=text,
    )

    assert first.content_hash == second.content_hash
