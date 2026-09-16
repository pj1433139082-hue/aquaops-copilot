from pathlib import Path
import os
import subprocess
import sys

from sqlalchemy import create_engine, inspect
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_compose_defines_healthy_backend_stack() -> None:
    compose = (PROJECT_ROOT / "compose.yml").read_text(encoding="utf-8")
    for service in ("postgres:", "redis:", "qdrant:", "migrate:", "api:", "worker:"):
        assert service in compose
    assert compose.count("healthcheck:") >= 4
    assert "condition: service_healthy" in compose
    assert "condition: service_completed_successfully" in compose


def test_compose_routes_every_database_consumer_to_postgresql() -> None:
    compose = yaml.safe_load((PROJECT_ROOT / "compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    database_urls = {
        name: services[name]["environment"]["AQUAOPS_DATABASE_URL"]
        for name in ("migrate", "api", "worker")
    }

    assert len(set(database_urls.values())) == 1
    assert database_urls["migrate"].startswith("postgresql+psycopg://")
    assert "@postgres:5432/aquaops" in database_urls["migrate"]
    assert "sqlite" not in database_urls["migrate"].casefold()


def test_ci_is_key_free_and_runs_locked_full_verification() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "uv sync --locked --all-groups" in workflow
    assert "uv run --locked ruff check" in workflow
    assert "uv run --locked pytest" in workflow
    assert "SEARCH_API_KEY" not in workflow
    assert "secrets." not in workflow


def test_release_checklist_and_environment_template_are_safe() -> None:
    checklist = (PROJECT_ROOT / "docs" / "release-checklist.md").read_text(
        encoding="utf-8"
    )
    template = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "Git history" in checklist
    assert "mock" in checklist.casefold()
    assert "known limitations" in checklist.casefold()
    assert "AQUAOPS_SEARCH_API_KEY=" in template
    assert "AQUAOPS_JWT_SECRET=" in template


def test_container_includes_only_the_registered_public_demo_corpus() -> None:
    compose = (PROJECT_ROOT / "compose.yml").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "COPY data/rag/public-demo ./data/rag/public-demo" in dockerfile
    assert "COPY migrations ./migrations" in dockerfile
    assert "requirements-backend.lock" in dockerfile
    assert (
        "AQUAOPS_PUBLIC_DEMO_CORPUS_PATH: /app/data/rag/public-demo/corpus.json"
        in compose
    )
    assert "data/public/" in dockerignore
    assert "data/rag/evals/" in dockerignore


def test_container_build_context_excludes_all_private_and_generated_surfaces() -> None:
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    for pattern in (
        "data/private/",
        "_private_local/",
        "reports/private/",
        "*.private.json",
        "*.private.json.tmp",
        "artifacts/",
    ):
        assert pattern in dockerignore


def test_model_worker_has_a_locked_runtime_and_explicit_public_model_profile() -> None:
    compose = (PROJECT_ROOT / "compose.yml").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    worker_lock = (PROJECT_ROOT / "requirements-worker.lock").read_text(
        encoding="utf-8"
    )
    assert "target: worker" in compose
    assert 'profiles: ["model-worker"]' in compose
    assert "huggingface-model-cache:/home/aquaops/.cache/huggingface" in compose
    assert "FROM worker-dependencies AS worker" in dockerfile
    assert "COPY requirements-worker.lock ./" in dockerfile
    for dependency in ("sentence-transformers==", "torch==", "transformers=="):
        assert dependency in worker_lock


def test_backend_and_worker_run_as_a_dedicated_unprivileged_user() -> None:
    compose = (PROJECT_ROOT / "compose.yml").read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.count("USER aquaops") == 2
    assert dockerfile.count("useradd --system") == 2
    assert "huggingface-model-cache:/home/aquaops/.cache/huggingface" in compose
    assert "huggingface-model-cache:/root/" not in compose


def test_worker_model_dependencies_are_cached_before_application_source() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    worker_dependencies = dockerfile.index(
        "RUN uv pip install --system --require-hashes -r requirements-worker.lock"
    )
    source_markers = ("COPY src/aquaops ./src/aquaops", "COPY src ./src")
    worker_source = max(
        (dockerfile.rfind(marker) for marker in source_markers),
        default=-1,
    )
    assert worker_source >= 0, "worker image must copy the public application source"

    assert worker_dependencies < worker_source


def test_ci_builds_and_import_checks_the_key_free_backend_image() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    assert "docker build --target backend" in workflow
    assert "docker run --rm aquaops-backend:ci python -c" in workflow
    assert "docker build --target worker" in workflow
    assert "import sentence_transformers, torch, transformers" in workflow


def test_alembic_cli_uses_the_configured_environment_database(tmp_path: Path) -> None:
    database_path = tmp_path / "compose-migration.sqlite3"
    environment = os.environ.copy()
    environment["AQUAOPS_DATABASE_URL"] = (
        f"sqlite+pysqlite:///{database_path.as_posix()}"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "alembic",
            "-c",
            str(PROJECT_ROOT / "alembic.ini"),
            "upgrade",
            "head",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    engine = create_engine(environment["AQUAOPS_DATABASE_URL"])
    try:
        assert {
            "alembic_version",
            "audit_events",
            "users",
            "roles",
            "cases",
            "agent_runs",
            "tasks",
        } <= set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
