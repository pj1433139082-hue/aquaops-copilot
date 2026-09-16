from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any
import re
from uuid import uuid4

from fastapi import Request, Response

from aquaops.observability.logging import event_payload, latency_bucket_for


_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,79}")


async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    supplied_request_id = request.headers.get("X-Request-ID")
    request.state.request_id = (
        supplied_request_id
        if supplied_request_id and _REQUEST_ID.fullmatch(supplied_request_id)
        else str(uuid4())
    )
    started = perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    logger: Callable[[dict[str, Any]], None] | None = getattr(
        request.app.state, "event_logger", None
    )
    if callable(logger):
        try:
            logger(
                event_payload(
                    "http.request_finished",
                    request_id=request.state.request_id,
                    latency_bucket=latency_bucket_for(
                        (perf_counter() - started) * 1_000
                    ),
                )
            )
        except Exception:
            pass
    return response
