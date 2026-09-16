from hashlib import sha256

from pydantic import BaseModel, HttpUrl


class SourceDocument(BaseModel):
    title: str
    source_url: HttpUrl
    license_name: str
    text: str
    source_version: str = "v1"

    @property
    def content_hash(self) -> str:
        return sha256(self.text.encode("utf-8")).hexdigest()


class IngestResult(BaseModel):
    accepted: bool
    reason: str
    content_hash: str | None = None
