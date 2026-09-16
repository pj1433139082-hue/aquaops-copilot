from dataclasses import FrozenInstanceError
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

import aquaops.rag.demo_corpus as demo_corpus
from aquaops.rag.demo_corpus import (
    DEMO_CORPUS_DATASET_ID,
    DEMO_CORPUS_PATH,
    DemoCorpusValidationError,
    load_verified_demo_chunks,
    load_verified_demo_documents,
)


_TOKENS = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PUBLIC_SAFE_FORBIDDEN_TEXT = (
    "private_data",
    "analytics-manifest.private",
    "normalized.duckdb",
    "d:\\",
    "c:\\users",
    "api_key",
    "password",
    "secret",
)


def test_verified_demo_corpus_contains_only_reviewed_official_public_sources() -> None:
    documents = load_verified_demo_documents()

    assert DEMO_CORPUS_DATASET_ID == "aquaops-public-water-demo-v1"
    assert len(documents) >= 8
    assert len({document.source_id for document in documents}) == len(documents)
    assert all(document.data_class == "public" for document in documents)
    assert all(document.access_policy == "public_read" for document in documents)
    assert all(document.source_url.startswith("https://") for document in documents)
    assert all(
        document.source_url.split("/", 3)[2] == "epa.gov"
        or document.source_url.split("/", 3)[2].endswith(".epa.gov")
        for document in documents
    )
    assert all("curated" in document.license_name.casefold() for document in documents)


def test_verified_demo_chunks_are_deterministic_citation_bound_and_sized() -> None:
    first = load_verified_demo_chunks()
    second = load_verified_demo_chunks()

    assert first == second
    assert len(first) >= 24
    assert len({chunk.chunk_id for chunk in first}) == len(first)
    assert all(250 <= len(_TOKENS.findall(chunk.text)) <= 450 for chunk in first)
    assert all(chunk.data_class == "public" for chunk in first)
    assert all(chunk.access_policy == "public_read" for chunk in first)
    assert all(chunk.source_url.startswith("https://") for chunk in first)
    assert all(chunk.source_version for chunk in first)


def test_demo_corpus_loader_rejects_every_noncanonical_path_before_reading(
    tmp_path: Path,
) -> None:
    copied = tmp_path / DEMO_CORPUS_PATH.name
    copied.write_bytes(DEMO_CORPUS_PATH.read_bytes())

    with pytest.raises(DemoCorpusValidationError, match="canonical"):
        load_verified_demo_documents(copied)


def test_demo_corpus_loader_honors_the_hash_locked_runtime_path(
    tmp_path: Path,
) -> None:
    copied = tmp_path / DEMO_CORPUS_PATH.name
    copied.write_bytes(DEMO_CORPUS_PATH.read_bytes())
    environment = os.environ.copy()
    environment["AQUAOPS_PUBLIC_DEMO_CORPUS_PATH"] = str(copied.resolve())

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from aquaops.rag.demo_corpus import DEMO_CORPUS_PATH, "
                "load_verified_demo_documents; "
                "load_verified_demo_documents(); print(DEMO_CORPUS_PATH)"
            ),
        ],
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == copied.resolve()


def test_demo_corpus_loader_rejects_a_tampered_runtime_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tampered = tmp_path / DEMO_CORPUS_PATH.name
    tampered.write_bytes(DEMO_CORPUS_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(demo_corpus, "DEMO_CORPUS_PATH", tampered.resolve())

    with pytest.raises(DemoCorpusValidationError, match="fingerprint"):
        demo_corpus.load_verified_demo_documents()


def test_demo_corpus_loader_rejects_a_relative_runtime_path() -> None:
    environment = os.environ.copy()
    environment["AQUAOPS_PUBLIC_DEMO_CORPUS_PATH"] = "data/rag/public-demo/corpus.json"

    completed = _run_isolated_demo_loader(environment)

    assert completed.returncode != 0
    assert "path must be absolute" in completed.stderr


def test_demo_corpus_loader_rejects_a_registered_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered = (tmp_path / DEMO_CORPUS_PATH.name).resolve()
    registered.write_bytes(DEMO_CORPUS_PATH.read_bytes())
    original_is_symlink = Path.is_symlink

    def synthetic_is_symlink(path: Path) -> bool:
        if path == registered:
            return True
        return original_is_symlink(path)

    monkeypatch.setattr(demo_corpus, "DEMO_CORPUS_PATH", registered)
    monkeypatch.setattr(Path, "is_symlink", synthetic_is_symlink)

    with pytest.raises(DemoCorpusValidationError, match="path must be canonical"):
        demo_corpus.load_verified_demo_documents()


def test_demo_corpus_loader_rejects_non_public_metadata_even_when_hash_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.loads(DEMO_CORPUS_PATH.read_text(encoding="utf-8"))
    payload["documents"][0]["data_class"] = "private"
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    tampered = tmp_path / DEMO_CORPUS_PATH.name
    tampered.write_bytes(encoded)
    monkeypatch.setattr(demo_corpus, "DEMO_CORPUS_PATH", tampered.resolve())
    monkeypatch.setattr(demo_corpus, "DEMO_CORPUS_SHA256", sha256(encoded).hexdigest())

    with pytest.raises(DemoCorpusValidationError, match="contract"):
        demo_corpus.load_verified_demo_documents()


def _run_isolated_demo_loader(
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from aquaops.rag.demo_corpus import "
                "load_verified_demo_documents; load_verified_demo_documents()"
            ),
        ],
        cwd=_PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_demo_corpus_documents_are_immutable() -> None:
    document = load_verified_demo_documents()[0]

    with pytest.raises(FrozenInstanceError):
        document.text = "changed"  # type: ignore[misc]


def test_demo_corpus_forbidden_text_uses_only_public_safe_generic_markers() -> None:
    assert demo_corpus._FORBIDDEN_TEXT == _PUBLIC_SAFE_FORBIDDEN_TEXT


def test_demo_corpus_source_text_does_not_contain_private_or_path_metadata() -> None:
    documents = load_verified_demo_documents()

    serialized = "\n".join(
        f"{document.source_id}\n{document.title}\n{document.text}"
        for document in documents
    ).casefold()
    assert all(
        marker.casefold() not in serialized for marker in _PUBLIC_SAFE_FORBIDDEN_TEXT
    )
