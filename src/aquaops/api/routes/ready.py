from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

from fastapi import APIRouter, Request, Response, status
from qdrant_client import QdrantClient
from redis import Redis
from sqlalchemy import create_engine, text

from aquaops.config import Settings


router = APIRouter(tags=["health"])
DEPENDENCIES = ("database", "redis", "qdrant")


def default_readiness_probes(settings: Settings) -> dict[str, Callable[[], bool]]:
    def database() -> bool:
        engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 1},
        )
        try:
            with engine.connect() as connection:
                return connection.scalar(text("SELECT 1")) == 1
        finally:
            engine.dispose()

    def redis() -> bool:
        client = Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        try:
            return client.ping() is True
        finally:
            client.close()

    def qdrant() -> bool:
        client = QdrantClient(url=settings.qdrant_url, timeout=0.5)
        try:
            client.get_collections()
            return True
        finally:
            client.close()

    return {"database": database, "redis": redis, "qdrant": qdrant}


@router.get("/ready")
def readiness(request: Request, response: Response) -> dict[str, object]:
    probes = cast(
        Mapping[str, Callable[[], bool]],
        getattr(request.app.state, "readiness_probes", {}),
    )
    results: dict[str, str] = {}
    for name in DEPENDENCIES:
        probe = probes.get(name)
        try:
            healthy = probe is not None and probe() is True
        except Exception:
            healthy = False
        results[name] = "ok" if healthy else "unavailable"
    ready = all(value == "ok" for value in results.values())
    response.status_code = (
        status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return {"status": "ready" if ready else "not_ready", "dependencies": results}
