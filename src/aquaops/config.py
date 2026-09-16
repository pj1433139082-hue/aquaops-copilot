from __future__ import annotations

from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from aquaops.rag.store import PUBLIC_COLLECTION


_TEST_DATABASE_URL = "sqlite+pysqlite:///:memory:"


class Settings(BaseSettings):
    environment: Literal["test", "development", "production"] = "test"
    app_name: str = "aquaops"
    database_url: str = _TEST_DATABASE_URL
    redis_url: str = "redis://redis:6379/0"
    qdrant_url: str = "http://qdrant:6333"
    public_qdrant_collection: str = PUBLIC_COLLECTION
    model_local_files_only: bool = True
    jwt_secret: str = ""
    jwt_issuer: str = "aquaops"
    jwt_audience: str = "aquaops-api"

    @field_validator("public_qdrant_collection")
    @classmethod
    def fixed_public_collection(cls, value: str) -> str:
        if value != PUBLIC_COLLECTION:
            raise ValueError(
                "public_qdrant_collection is fixed to the public collection"
            )
        return value

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def require_external_deployment_credentials(self) -> Settings:
        if self.environment in {"development", "production"} and (
            not self.database_url.strip() or self.database_url == _TEST_DATABASE_URL
        ):
            raise ValueError("database_url must be configured for deployment")
        if (
            self.environment in {"development", "production"}
            and len(self.jwt_secret.strip()) < 32
        ):
            raise ValueError("jwt_secret must be configured for deployment")
        return self

    model_config = SettingsConfigDict(env_prefix="AQUAOPS_")
