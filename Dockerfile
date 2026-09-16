FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de AS backend-dependencies

COPY --from=ghcr.io/astral-sh/uv:0.5.30@sha256:bb74263127d6451222fe7f71b330edfb189ab1c98d7898df2401fbf4f272d9b9 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY requirements-backend.lock ./
RUN uv pip install --system --require-hashes -r requirements-backend.lock

FROM backend-dependencies AS runtime

COPY pyproject.toml ./pyproject.toml
# The staging tree contains only the allowlisted public aquaops package.
COPY src/aquaops ./src/aquaops
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini
COPY data/rag/public-demo ./data/rag/public-demo
RUN uv pip install --system --no-deps .

EXPOSE 8000

FROM runtime AS backend

RUN groupadd --system aquaops \
    && useradd --system --gid aquaops --create-home aquaops

USER aquaops

CMD ["uvicorn", "aquaops.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

FROM backend-dependencies AS worker-dependencies

COPY requirements-worker.lock ./
RUN uv pip install --system --require-hashes -r requirements-worker.lock

FROM worker-dependencies AS worker

COPY pyproject.toml ./pyproject.toml
# Keep the worker build on the same explicit public-source boundary.
COPY src/aquaops ./src/aquaops
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini
COPY data/rag/public-demo ./data/rag/public-demo
RUN uv pip install --system --no-deps .

RUN groupadd --system aquaops \
    && useradd --system --gid aquaops --create-home aquaops \
    && mkdir -p /home/aquaops/.cache/huggingface \
    && chown -R aquaops:aquaops /home/aquaops

USER aquaops

CMD ["celery", "-A", "aquaops.tasks.celery_app:celery_app", "worker", "--loglevel=INFO", "--concurrency=1"]
