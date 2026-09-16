# Public data registry

The approved-source registry is [sources.json](sources.json). Its scope, initial
uses, licence-review status, and acquisition boundaries are documented in
[../../docs/public-data-sources.md](../../docs/public-data-sources.md).

Raw data is never committed to this repository. `data/public/raw` is reserved for
local, ignored acquisitions only. Do not add real production SCADA data, customer
information, laboratory materials, internal documents, or source downloads to Git.

The hash-locked, reviewed public RAG demo corpus is intentionally stored separately at
`data/rag/public-demo/corpus.json`. It contains original Chinese summaries and official
EPA citations rather than downloaded source files. Its loader, model revisions, real-public
evaluation set, and operator commands are documented in
[`docs/public-rag-demo.md`](../../docs/public-rag-demo.md).
