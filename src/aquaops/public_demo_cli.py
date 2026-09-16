"""Fixed, aggregate-only command surface for the public AquaOps demo."""

from collections.abc import Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
from io import StringIO
import json
import os
import sys
import warnings

from aquaops.rag.demo_runtime import (
    PUBLIC_DEMO_CANDIDATE_LIMIT,
    PUBLIC_DEMO_RERANK_LIMIT,
)


_COMMANDS = frozenset({"prepare", "evaluate", "smoke"})
_OFFLINE_ENVIRONMENT_KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def compose_public_demo(**kwargs):
    from aquaops.demo import compose_public_demo as compose

    return compose(**kwargs)


def evaluate_public_demo(components):
    from aquaops.demo import evaluate_public_demo as evaluate

    return evaluate(components)


def smoke_public_demo(components):
    from aquaops.demo import smoke_public_demo as smoke

    return smoke(components)


def build_public_demo_delivery_document(*args):
    from aquaops.demo_delivery import build_public_demo_delivery_document as build

    return build(*args)


def write_public_demo_delivery_document(document):
    from aquaops.demo_delivery import write_public_demo_delivery_document as write

    return write(document)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one fixed public-only workflow and print one safe JSON object."""
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    parsed = _parse_arguments(arguments)
    if parsed is None:
        _emit({"error": "public_demo_request_invalid", "ok": False})
        return 2
    command, local_files_only, reranker_backend = parsed
    with (
        _model_offline_environment(local_files_only),
        warnings.catch_warnings(),
        redirect_stdout(StringIO()),
        redirect_stderr(StringIO()),
    ):
        warnings.simplefilter("ignore")
        components = None
        try:
            components = compose_public_demo(
                local_files_only=local_files_only,
                reranker_backend=reranker_backend,
            )
            if command == "prepare":
                payload = _prepare_payload(components.runtime)
            elif command == "smoke":
                payload = asdict(smoke_public_demo(components))
                payload["reranker_backend"] = _backend_payload(components.runtime)
            else:
                evaluation = evaluate_public_demo(components)
                smoke = smoke_public_demo(components)
                document = build_public_demo_delivery_document(
                    components.runtime,
                    evaluation,
                    smoke,
                )
                report_sha256 = write_public_demo_delivery_document(document)
                payload = {
                    "ok": True,
                    "case_count": evaluation.case_count,
                    "k": evaluation.k,
                    "candidate_limit": PUBLIC_DEMO_CANDIDATE_LIMIT,
                    "rerank_limit": PUBLIC_DEMO_RERANK_LIMIT,
                    "corpus_sha256": evaluation.corpus_sha256,
                    "evaluation_sha256": evaluation.dataset_sha256,
                    "report_sha256": report_sha256,
                    "variants": [asdict(item) for item in evaluation.variants],
                    "smoke": asdict(smoke),
                    "reranker_backend": _backend_payload(components.runtime),
                }
        except Exception:
            payload = {"error": "public_demo_failed", "ok": False}
            exit_code = 1
        else:
            exit_code = 0
        finally:
            if components is not None:
                try:
                    components.runtime.close()
                except Exception:
                    payload = {"error": "public_demo_failed", "ok": False}
                    exit_code = 1
    _emit(payload)
    return exit_code


def _parse_arguments(
    arguments: tuple[str, ...],
) -> tuple[str, bool, str] | None:
    if not 1 <= len(arguments) <= 3 or arguments[0] not in _COMMANDS:
        return None
    options = arguments[1:]
    if len(set(options)) != len(options):
        return None
    allowed = {
        "--offline",
        "--reranker-backend=torch",
        "--reranker-backend=openvino",
    }
    if any(option not in allowed for option in options):
        return None
    backend_options = [
        option for option in options if option.startswith("--reranker-backend=")
    ]
    if len(backend_options) > 1:
        return None
    backend = backend_options[0].split("=", 1)[1] if backend_options else "torch"
    return arguments[0], "--offline" in options, backend


@contextmanager
def _model_offline_environment(local_files_only: bool):
    previous = {key: os.environ.get(key) for key in _OFFLINE_ENVIRONMENT_KEYS}
    if local_files_only:
        for key in _OFFLINE_ENVIRONMENT_KEYS:
            os.environ[key] = "1"
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _prepare_payload(runtime) -> dict[str, object]:
    spec = runtime.model_spec
    return {
        "ok": True,
        "chunk_count": runtime.chunk_count,
        "candidate_limit": PUBLIC_DEMO_CANDIDATE_LIMIT,
        "rerank_limit": PUBLIC_DEMO_RERANK_LIMIT,
        "corpus_sha256": runtime.corpus_sha256,
        "embedding_model_id": spec.embedding_model_id,
        "embedding_revision": spec.embedding_revision,
        "reranker_model_id": spec.reranker_model_id,
        "reranker_revision": spec.reranker_revision,
        "reranker_backend": _backend_payload(runtime),
    }


def _backend_payload(runtime) -> dict[str, object]:
    backend = runtime.reranker_backend_status
    return {
        "requested": backend.requested_backend,
        "active": backend.active_backend,
        "fallback_used": backend.fallback_used,
        "status_code": backend.status_code,
    }


def _emit(payload: dict[str, object]) -> None:
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
