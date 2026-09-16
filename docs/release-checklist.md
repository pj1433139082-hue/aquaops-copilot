# AquaOps public release checklist

Use this checklist before any public tag, image, or repository publication.

- [ ] Review Git history and the working tree for private plant data, internal architecture,
  credentials, raw prompts, database URLs, `.env` files, and generated model artifacts.
- [ ] Confirm `.env.example` contains blank secrets and no real search, JWT, or provider key.
- [ ] Run `uv sync --locked --all-groups`, Ruff check/format, and the full pytest suite.
- [ ] Confirm tests use synthetic fixtures and mock LLM/search providers; CI needs no external key.
- [ ] Run Alembic upgrade to `head` on an empty dedicated database and verify downgrade gates in
  their isolated destructive-test databases.
- [ ] Start Compose and require PostgreSQL, Redis, Qdrant, migration, API health and API readiness
  checks to pass; then remove only this project's containers and disposable volumes.
- [ ] Build the optional `model-worker` target from `requirements-worker.lock`; after the fixed
  public BGE-M3 revision is prepared, run one registered-public-source ingestion and verify the
  deterministic Qdrant upsert before removing only the disposable test collection/volumes.
- [ ] Review README data licenses, public/synthetic-only boundary, decision-support limitation,
  evidence links, and known limitations.
- [ ] Record exact test totals, image digests, evaluation report hash, and review date.
- [ ] Do not claim plant control, production performance, or private-data validation.
