# Public candidate license and notices

## Status

This is the staging candidate's reproducible license record, not legal advice. The candidate's
original AquaOps source is offered under Apache-2.0 (see the repository-root `LICENSE`). Every
third-party dependency and public data source retains its own terms.

## Direct runtime dependencies

Versions are resolved from `uv.lock`; the license labels below are the local metadata review from
the locked environment:

| Dependency | Locked version | License metadata |
| --- | ---: | --- |
| fastapi | 0.139.2 | MIT |
| uvicorn | 0.51.0 | BSD-3-Clause |
| pydantic-settings | 2.14.2 | MIT |
| langgraph | 0.6.11 | MIT |
| qdrant-client | 1.18.0 | Apache-2.0 |
| sentence-transformers | 3.4.1 | Apache-2.0 |
| rank-bm25 | 0.2.2 | Apache-2.0 |
| mcp | 1.28.1 | MIT |
| xlrd | 2.0.2 | BSD |
| duckdb | 1.5.4 | MIT classifier |
| pyarrow | 19.0.1 | Apache Software License |
| sqlalchemy | 2.0.52 | MIT |
| alembic | 1.19.1 | MIT |
| psycopg | 3.3.4 | LGPL-3.0-only |
| PyJWT | 2.13.0 | MIT |
| redis | 5.3.1 | MIT |
| celery | 5.6.3 | BSD-3-Clause |

## Items requiring preservation or final review

- `psycopg`/`psycopg-binary` are LGPL-3.0-only. Any binary or container redistribution must
  preserve the applicable LGPL notices and allow the required relinking/redistribution rights.
- `certifi` is MPL-2.0; `orjson` is MPL-2.0 AND (Apache-2.0 OR MIT); and `tqdm` is MPL-2.0 AND
  MIT. Their notices remain part of the dependency distribution.
- `scipy` carries bundled notices in addition to its metadata classifiers; retain upstream notice
  files when the worker or optional model stack is redistributed.
- The optional `public-openvino` extra (`openvino`, `optimum`, `optimum-intel`) was not installed
  in the backend image. Package-level official review records OpenVINO 2026.3.0, Optimum 2.1.0
  and Optimum Intel 1.27.0 as Apache-2.0, with OpenVINO Telemetry 2025.2.0 also Apache-2.0.
  OpenVINO's bundled runtime/oneTBB/oneDNN attribution files still must be copied from the
  corresponding distribution before this extra is enabled or redistributed; see the evidence
  record `.planning/license-notices-review-2026-09-08.md`.
- The registered Co-UDlabs source is CC0-1.0 with the attribution recorded in
  `data/public/sources.json`; the public demo uses reviewed synthetic/derived content and never
  copies private plant data. Other registry entries remain reference-only until their licenses are
  directly verified.
- The fixed public model snapshots used by the demo are [`BAAI/bge-m3` at
  `5617a9f61b028005a4858fdac845db406aefb181`](https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181)
  and [`BAAI/bge-reranker-base` at
  `2cfc18c9415c912f9d8155881c133215df768a70`](https://huggingface.co/BAAI/bge-reranker-base/tree/2cfc18c9415c912f9d8155881c133215df768a70).
  Their exact Hugging Face model cards identify the model license as MIT. The snapshots stay outside
  Git, this staging tree and image layers; if they are redistributed, retain the model-card
  attribution and license reference.

## Reproduction and release gate

Before any external redistribution, install the complete locked dependency set in a clean staging
environment, collect the installed license/notice files, compare them with this record, and have a
human review the LGPL/MPL/bundled-notice obligations. External repository creation, commit/push and
image publication remain separate approvals.

## 2026-09-08 clean-image review

The E-drive backend image was inspected in a one-shot network-disabled container. The scan found
94 installed distributions (including the project package and base-image `pip`), with 91 carrying
license/notice/copyright file-name candidates. `langchain-core` and `langsmith` expose MIT metadata
without a matching file-name candidate; the project package is not a third-party notice. This is
package-level evidence only: it does not replace preserving bundled notices or a human legal review.
