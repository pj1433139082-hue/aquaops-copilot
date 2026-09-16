from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from collections.abc import Callable
from typing import Any
from sqlalchemy.orm import Session, sessionmaker

from aquaops.agent.state import InMemoryAgentRunStore
from aquaops.api.middleware import request_id_middleware
from aquaops.api.routes.agent import router as agent_router
from aquaops.api.routes.auth import router as auth_router
from aquaops.api.routes.cases import router as cases_router
from aquaops.api.routes.health import router as health_router
from aquaops.api.routes.ready import router as ready_router
from aquaops.api.routes.tasks import router as tasks_router
from aquaops.config import Settings
from aquaops.api.routes.ready import default_readiness_probes
from aquaops.db.session import create_session_factory
from aquaops.domain.models import User
from aquaops.security.auth import RoleResolver


async def unhandled_exception_handler(request: Request, _: Exception) -> JSONResponse:
    return JSONResponse(
        content={"detail": "Internal Server Error"},
        headers={"X-Request-ID": request.state.request_id},
        status_code=500,
    )


def create_app(
    settings: Settings | None = None,
    *,
    operations_graph: object | None = None,
    role_resolver: RoleResolver | None = None,
    session_factory: sessionmaker[Session] | None = None,
    readiness_probes: dict[str, object] | None = None,
    event_logger: Callable[[dict[str, Any]], None] | None = None,
) -> FastAPI:
    app = FastAPI(title="AquaOps Copilot")
    resolved_settings = settings or Settings()
    resolved_session_factory = session_factory or create_session_factory(
        resolved_settings.database_url
    )
    app.state.settings = resolved_settings
    app.state.agent_run_store = InMemoryAgentRunStore()
    app.state.operations_graph = operations_graph
    app.state.session_factory = resolved_session_factory
    app.state.readiness_probes = (
        readiness_probes
        if readiness_probes is not None
        else default_readiness_probes(resolved_settings)
    )
    app.state.event_logger = event_logger
    if role_resolver is not None:
        app.state.role_resolver = role_resolver
    else:

        def database_role_resolver(user_id: str) -> frozenset[str] | None:
            with resolved_session_factory() as session:
                user = session.get(User, user_id)
                if user is None:
                    return None
                return frozenset(role.name for role in user.roles)

        app.state.role_resolver = database_role_resolver
    app.middleware("http")(request_id_middleware)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    app.include_router(health_router)
    app.include_router(ready_router)
    app.include_router(auth_router)
    app.include_router(cases_router)
    app.include_router(tasks_router)
    app.include_router(agent_router)
    return app
