"""Hash-locked loader for the reviewed public-only AquaOps demo corpus."""

from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from urllib.parse import urlsplit

from aquaops.rag.public_corpus import (
    PublicKnowledgeChunk,
    PublicKnowledgeDocument,
    chunk_public_document,
)


DEMO_CORPUS_DATASET_ID = "aquaops-public-water-demo-v1"
DEMO_CORPUS_SCHEMA_VERSION = "aquaops-public-water-demo-v1"
DEMO_CORPUS_SHA256 = "419241b71d695b2f9e1e8f411021e5169b2a61abfb6f18786fe7e7a9fa548a0a"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DEMO_CORPUS_PATH = (
    _PROJECT_ROOT / "data" / "rag" / "public-demo" / "corpus.json"
)
DEMO_CORPUS_PATH = Path(
    os.getenv("AQUAOPS_PUBLIC_DEMO_CORPUS_PATH", str(_DEFAULT_DEMO_CORPUS_PATH))
)
_DATASET_KEYS = {
    "schema_version",
    "dataset_id",
    "data_class",
    "access_policy",
    "documents",
}
_DOCUMENT_KEYS = {
    "source_id",
    "title",
    "source_url",
    "source_version",
    "accessed_on",
    "rights_note",
    "license_name",
    "data_class",
    "access_policy",
    "sections",
}
_SECTION_KEYS = {"title", "summary"}
_SOURCE_ID = re.compile(r"^[a-z][a-z0-9-]{2,63}$")
_SOURCE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_DATE = re.compile(r"^20\d{2}-\d{2}-\d{2}$")
_TOKENS = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]")
_FORBIDDEN_TEXT = (
    "private_data",
    "analytics-manifest.private",
    "normalized.duckdb",
    "d:\\",
    "c:\\users",
    "api_key",
    "password",
    "secret",
)


class DemoCorpusValidationError(ValueError):
    """The requested data is not the registered reviewed public corpus."""


@dataclass(frozen=True)
class VerifiedDemoCorpus:
    dataset_id: str
    schema_version: str
    documents: tuple[PublicKnowledgeDocument, ...]
    corpus_sha256: str
    _verification_token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._verification_token is not _CORPUS_TOKEN:
            raise DemoCorpusValidationError(
                "verified demo corpus must be loader-issued"
            )


_CORPUS_TOKEN = object()


def load_verified_demo_corpus(path: str | Path | None = None) -> VerifiedDemoCorpus:
    """Load only the canonical hash-registered public demo corpus."""
    _assert_registered_corpus_path()
    _assert_canonical_path(path)
    try:
        raw = DEMO_CORPUS_PATH.read_bytes()
    except OSError:
        raise DemoCorpusValidationError(
            "registered demo corpus could not be loaded"
        ) from None
    if sha256(raw).hexdigest() != DEMO_CORPUS_SHA256:
        raise DemoCorpusValidationError("registered demo corpus fingerprint is invalid")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise DemoCorpusValidationError(
            "registered demo corpus could not be decoded"
        ) from None
    documents = _parse_corpus(payload)
    return VerifiedDemoCorpus(
        dataset_id=DEMO_CORPUS_DATASET_ID,
        schema_version=DEMO_CORPUS_SCHEMA_VERSION,
        documents=documents,
        corpus_sha256=DEMO_CORPUS_SHA256,
        _verification_token=_CORPUS_TOKEN,
    )


def load_verified_demo_documents(
    path: str | Path | None = None,
) -> tuple[PublicKnowledgeDocument, ...]:
    return load_verified_demo_corpus(path).documents


def load_verified_demo_chunks() -> tuple[PublicKnowledgeChunk, ...]:
    corpus = load_verified_demo_corpus()
    chunks = tuple(
        chunk
        for document in corpus.documents
        for chunk in chunk_public_document(
            document,
            max_tokens=450,
            overlap_tokens=40,
        )
    )
    if len(chunks) < 24 or len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise DemoCorpusValidationError("registered demo corpus chunk set is invalid")
    if any(not 250 <= len(_TOKENS.findall(chunk.text)) <= 450 for chunk in chunks):
        raise DemoCorpusValidationError("registered demo corpus chunk size is invalid")
    return chunks


def _assert_canonical_path(path: str | Path | None) -> None:
    if path is None:
        return
    try:
        requested = Path(path)
        canonical = DEMO_CORPUS_PATH.resolve(strict=True)
        resolved = requested.resolve(strict=True)
    except (OSError, RuntimeError):
        raise DemoCorpusValidationError("demo corpus path must be canonical") from None
    if requested.is_symlink() or resolved != canonical:
        raise DemoCorpusValidationError("demo corpus path must be canonical")


def _assert_registered_corpus_path() -> None:
    if not DEMO_CORPUS_PATH.is_absolute():
        raise DemoCorpusValidationError("registered demo corpus path must be absolute")
    try:
        resolved = DEMO_CORPUS_PATH.resolve(strict=True)
    except (OSError, RuntimeError):
        raise DemoCorpusValidationError(
            "registered demo corpus could not be loaded"
        ) from None
    if DEMO_CORPUS_PATH.is_symlink() or resolved != DEMO_CORPUS_PATH:
        raise DemoCorpusValidationError("registered demo corpus path must be canonical")


def _parse_corpus(payload: object) -> tuple[PublicKnowledgeDocument, ...]:
    if type(payload) is not dict or set(payload) != _DATASET_KEYS:
        raise DemoCorpusValidationError("demo corpus schema is invalid")
    if (
        payload["schema_version"] != DEMO_CORPUS_SCHEMA_VERSION
        or payload["dataset_id"] != DEMO_CORPUS_DATASET_ID
        or payload["data_class"] != "public"
        or payload["access_policy"] != "public_read"
        or type(payload["documents"]) is not list
        or not 8 <= len(payload["documents"]) <= 32
    ):
        raise DemoCorpusValidationError("demo corpus header is invalid")

    documents: list[PublicKnowledgeDocument] = []
    source_ids: set[str] = set()
    for raw_document in payload["documents"]:
        document = _parse_document(raw_document)
        if document.source_id in source_ids:
            raise DemoCorpusValidationError("demo corpus source IDs must be unique")
        source_ids.add(document.source_id)
        documents.append(document)
    return tuple(documents)


def _parse_document(payload: object) -> PublicKnowledgeDocument:
    if type(payload) is not dict or set(payload) != _DOCUMENT_KEYS:
        raise DemoCorpusValidationError("demo corpus document schema is invalid")
    string_fields = (
        "source_id",
        "title",
        "source_url",
        "source_version",
        "accessed_on",
        "rights_note",
        "license_name",
    )
    if any(
        type(payload[field]) is not str or not payload[field].strip()
        for field in string_fields
    ):
        raise DemoCorpusValidationError("demo corpus document fields are invalid")
    if (
        not _SOURCE_ID.fullmatch(payload["source_id"])
        or not _SOURCE_VERSION.fullmatch(payload["source_version"])
        or not _DATE.fullmatch(payload["accessed_on"])
        or payload["data_class"] != "public"
        or payload["access_policy"] != "public_read"
        or "curated" not in payload["license_name"].casefold()
        or not _is_official_epa_url(payload["source_url"])
        or type(payload["sections"]) is not list
        or len(payload["sections"]) != 3
    ):
        raise DemoCorpusValidationError("demo corpus document contract is invalid")
    if len(payload["title"]) > 160 or len(payload["rights_note"]) > 500:
        raise DemoCorpusValidationError("demo corpus document metadata is too large")

    rendered_sections: list[str] = []
    for section in payload["sections"]:
        if type(section) is not dict or set(section) != _SECTION_KEYS:
            raise DemoCorpusValidationError("demo corpus section schema is invalid")
        title = section["title"]
        summary = section["summary"]
        if (
            type(title) is not str
            or not title.strip()
            or len(title) > 100
            or type(summary) is not str
            or not summary.strip()
            or not 250 <= len(_TOKENS.findall(summary)) <= 450
        ):
            raise DemoCorpusValidationError("demo corpus section is invalid")
        rendered_sections.append(f"## {title.strip()}\n{summary.strip()}")

    text = f"# {payload['title'].strip()}\n" + "\n".join(rendered_sections)
    inspected = "\n".join(str(payload[field]) for field in string_fields) + "\n" + text
    if any(marker in inspected.casefold() for marker in _FORBIDDEN_TEXT):
        raise DemoCorpusValidationError("demo corpus contains forbidden metadata")
    return PublicKnowledgeDocument(
        source_id=payload["source_id"],
        title=payload["title"].strip(),
        source_url=payload["source_url"],
        source_version=payload["source_version"],
        license_name=payload["license_name"],
        text=text,
    )


def _is_official_epa_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        return False
    return bool(
        parsed.scheme == "https"
        and hostname
        and (hostname == "epa.gov" or hostname.endswith(".epa.gov"))
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment
    )
