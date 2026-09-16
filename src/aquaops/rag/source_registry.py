from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


class PublicSource(BaseModel):
    source_id: str
    title: str
    source_url: HttpUrl
    source_kind: Literal["dataset", "knowledge_document", "simulation_model"]
    license_name: str
    license_status: Literal["verified", "review_required"]
    source_version: str
    access_mode: Literal["local_download_only", "reference_only"]
    repository_commit_allowed: Literal[False]
    attribution: str
    license_review_note: str | None = None
    license_evidence: str | None = None
    download_url: HttpUrl | None = None
    archive_md5: str | None = None
    permitted_paths: list[str] = Field(default_factory=list)
    verified_file_sha256: dict[str, str] = Field(default_factory=dict)

    @field_validator("archive_md5")
    @classmethod
    def validate_archive_md5(cls, archive_md5: str | None) -> str | None:
        if archive_md5 is None:
            return None
        if len(archive_md5) != 32 or any(
            character not in "0123456789abcdef" for character in archive_md5
        ):
            raise ValueError("archive_md5 must be 32 lowercase hexadecimal characters")
        return archive_md5

    @field_validator("permitted_paths")
    @classmethod
    def validate_permitted_paths(cls, permitted_paths: list[str]) -> list[str]:
        for permitted_path in permitted_paths:
            if not permitted_path:
                raise ValueError(
                    "permitted_paths must contain non-empty relative POSIX archive paths"
                )
            if permitted_path.startswith("/") or "\\" in permitted_path:
                raise ValueError(
                    "permitted_paths must contain relative POSIX archive paths"
                )

            path_parts = permitted_path.split("/")
            if any(part in {"", ".", ".."} for part in path_parts):
                raise ValueError("permitted_paths may not contain parent traversal")
            if any(":" in part for part in path_parts):
                raise ValueError(
                    "permitted_paths must contain relative POSIX archive paths"
                )
            if permitted_path in {"*", "**"}:
                raise ValueError("permitted_paths may not be full-path wildcards")
            if "*" in permitted_path and (
                not permitted_path.endswith("/**") or "*" in permitted_path[:-3]
            ):
                raise ValueError(
                    "permitted_paths may only use a terminal '/**' wildcard"
                )

        return permitted_paths

    @field_validator("verified_file_sha256")
    @classmethod
    def validate_verified_file_sha256(
        cls, verified_file_sha256: dict[str, str]
    ) -> dict[str, str]:
        for relative_path, sha256 in verified_file_sha256.items():
            if not relative_path:
                raise ValueError(
                    "verified_file_sha256 keys must be non-empty relative POSIX paths"
                )
            if relative_path.startswith("/") or "\\" in relative_path:
                raise ValueError(
                    "verified_file_sha256 keys must be relative POSIX paths"
                )
            path_parts = relative_path.split("/")
            if any(part in {"", ".", ".."} for part in path_parts):
                raise ValueError(
                    "verified_file_sha256 keys may not contain parent traversal"
                )
            if any(":" in part for part in path_parts):
                raise ValueError(
                    "verified_file_sha256 keys must be relative POSIX paths"
                )
            if "*" in relative_path:
                raise ValueError(
                    "verified_file_sha256 keys must name concrete file paths"
                )
            if len(sha256) != 64 or any(
                character not in "0123456789abcdef" for character in sha256
            ):
                raise ValueError(
                    "verified_file_sha256 values must be 64 lowercase hexadecimal "
                    "characters"
                )

        return verified_file_sha256


class PublicSourceRegistry(BaseModel):
    schema_version: Literal[1]
    sources: list[PublicSource]

    @model_validator(mode="after")
    def validate_sources(self) -> "PublicSourceRegistry":
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique")

        for source in self.sources:
            if source.access_mode == "local_download_only" and (
                source.license_status != "verified"
                or source.license_name.casefold() == "not_verified"
                or not source.license_name.strip()
            ):
                raise ValueError("local downloads require a verified license")
            if (
                source.access_mode == "local_download_only"
                and not source.permitted_paths
            ):
                raise ValueError("local downloads require non-empty permitted_paths")
            if (
                source.access_mode == "local_download_only"
                and not source.verified_file_sha256
            ):
                raise ValueError(
                    "local downloads require non-empty verified_file_sha256"
                )
            if source.access_mode == "local_download_only" and (
                source.download_url is None
                or source.archive_md5 is None
                or source.license_evidence is None
                or not source.license_evidence.strip()
            ):
                raise ValueError(
                    "local downloads require download_url, archive_md5, and "
                    "license_evidence"
                )
            for relative_path in source.verified_file_sha256:
                if not any(
                    _permitted_path_covers_file(permitted_path, relative_path)
                    for permitted_path in source.permitted_paths
                ):
                    raise ValueError(
                        "verified_file_sha256 paths must be covered by permitted_paths"
                    )

        return self


def load_public_source_registry(path: Path) -> list[PublicSource]:
    return PublicSourceRegistry.model_validate_json(
        path.read_text(encoding="utf-8")
    ).sources


def _permitted_path_covers_file(permitted_path: str, relative_path: str) -> bool:
    if permitted_path.endswith("/**"):
        directory = permitted_path[:-3]
        return relative_path.startswith(f"{directory}/")
    return permitted_path == relative_path
