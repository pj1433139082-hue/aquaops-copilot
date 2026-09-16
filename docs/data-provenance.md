# Public-document provenance

Only public documents with an explicit license, or contributor-created synthetic
documents, may be considered for ingestion. Never add real production SCADA data,
customer data, laboratory records, or internal materials to this repository.

For every actual import, record one completed provenance row. Use an ISO 8601 UTC
timestamp for `ingested_at`; never store a document when its license is missing or
its non-whitespace content is under 100 characters.

| title | source_url | license_name | source_version | content_hash | ingested_at |
| --- | --- | --- | --- | --- | --- |
| Fill during an actual import | Fill during an actual import | Fill during an actual import | Fill during an actual import | SHA-256 of UTF-8 text | ISO 8601 UTC timestamp |

The current ingestion function performs validation and produces deterministic metadata
only. It does not download, persist, index, or otherwise import document content.
