from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aquaops.domain.models import AuditEvent


class AuditEventCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    action: str = Field(min_length=1, max_length=80)
    request_id: str = Field(min_length=1, max_length=80)
    actor_id: str = Field(min_length=1, max_length=36)
    entity_type: str = Field(min_length=1, max_length=40)
    entity_id: str = Field(min_length=1, max_length=36)

    @field_validator(
        "action", "request_id", "actor_id", "entity_type", "entity_id", mode="before"
    )
    @classmethod
    def reject_non_string_or_untrimmed_values(cls, value: object) -> object:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError("audit identifiers must be non-empty trimmed strings")
        return value

    def to_model(self) -> AuditEvent:
        return AuditEvent(**self.model_dump())
