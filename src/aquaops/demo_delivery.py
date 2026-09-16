"""Safe reproducibility document for the public-only AquaOps demo."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4

from aquaops.demo import PublicDemoSmokeReport
from aquaops.rag.demo_evaluation import DemoEvaluationReport
from aquaops.rag.demo_runtime import PublicDemoRuntime
from aquaops.rag.demo_runtime import (
    PUBLIC_DEMO_CANDIDATE_LIMIT,
    PUBLIC_DEMO_RERANK_LIMIT,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UV_LOCK_PATH = _PROJECT_ROOT / "uv.lock"
PUBLIC_DEMO_REPORT_PATH = (
    _PROJECT_ROOT / "artifacts" / "public-demo" / "public-demo-report.json"
)


@dataclass(frozen=True)
class PublicDemoDeliveryDocument:
    schema_version: str
    generated_at_utc: str
    corpus_sha256: str
    evaluation_sha256: str
    uv_lock_sha256: str
    embedding_model_id: str
    embedding_revision: str
    reranker_model_id: str
    reranker_revision: str
    reranker_backend_requested: str
    reranker_backend_active: str
    reranker_backend_fallback_used: bool
    reranker_backend_status_code: str
    chunk_count: int
    candidate_limit: int
    rerank_limit: int
    evaluation: DemoEvaluationReport
    smoke: PublicDemoSmokeReport


def build_public_demo_delivery_document(
    runtime: PublicDemoRuntime,
    evaluation: DemoEvaluationReport,
    smoke: PublicDemoSmokeReport,
    *,
    generated_at: datetime | None = None,
) -> PublicDemoDeliveryDocument:
    if (
        type(runtime) is not PublicDemoRuntime
        or type(evaluation) is not DemoEvaluationReport
        or type(smoke) is not PublicDemoSmokeReport
        or evaluation.corpus_sha256 != runtime.corpus_sha256
        or smoke.chunk_count != runtime.chunk_count
        or not smoke.ok
    ):
        raise ValueError("public demo delivery inputs are inconsistent")
    timestamp = generated_at or datetime.now(timezone.utc)
    if type(timestamp) is not datetime or timestamp.tzinfo is None:
        raise ValueError("delivery timestamp must be timezone-aware")
    try:
        uv_lock_sha256 = sha256(_UV_LOCK_PATH.read_bytes()).hexdigest()
    except OSError:
        raise ValueError("locked dependency manifest is unavailable") from None
    spec = runtime.model_spec
    backend = runtime.reranker_backend_status
    return PublicDemoDeliveryDocument(
        schema_version="aquaops-public-demo-report-v3",
        generated_at_utc=(
            timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        ),
        corpus_sha256=runtime.corpus_sha256,
        evaluation_sha256=evaluation.dataset_sha256,
        uv_lock_sha256=uv_lock_sha256,
        embedding_model_id=spec.embedding_model_id,
        embedding_revision=spec.embedding_revision,
        reranker_model_id=spec.reranker_model_id,
        reranker_revision=spec.reranker_revision,
        reranker_backend_requested=backend.requested_backend,
        reranker_backend_active=backend.active_backend,
        reranker_backend_fallback_used=backend.fallback_used,
        reranker_backend_status_code=backend.status_code,
        chunk_count=runtime.chunk_count,
        candidate_limit=PUBLIC_DEMO_CANDIDATE_LIMIT,
        rerank_limit=PUBLIC_DEMO_RERANK_LIMIT,
        evaluation=evaluation,
        smoke=smoke,
    )


def write_public_demo_delivery_document(
    document: PublicDemoDeliveryDocument,
) -> str:
    if type(document) is not PublicDemoDeliveryDocument:
        raise TypeError("document must be an exact PublicDemoDeliveryDocument")
    encoded = (
        json.dumps(
            asdict(document),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    destination = PUBLIC_DEMO_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".public-demo-report-{uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ValueError("public demo report could not be written") from None
    return sha256(encoded).hexdigest()
