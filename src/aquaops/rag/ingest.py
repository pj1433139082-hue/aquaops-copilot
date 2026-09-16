from aquaops.rag.models import IngestResult, SourceDocument


def ingest_document(source: SourceDocument) -> IngestResult:
    if not source.license_name.strip():
        return IngestResult(accepted=False, reason="missing_license")
    if len(source.text.strip()) < 100:
        return IngestResult(accepted=False, reason="insufficient_content")
    return IngestResult(
        accepted=True,
        reason="accepted",
        content_hash=source.content_hash,
    )
