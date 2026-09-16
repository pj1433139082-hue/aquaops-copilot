from pathlib import Path
import re
import tomllib

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QDRANT_IMAGE = "qdrant/qdrant:v1.18.2@sha256:75eab8c4ba42096724fdcfde8b4de0b5713d529dde32f285a1f86fdcb2c9e50c"
PYTHON_IMAGE = "python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de"
UV_IMAGE = "ghcr.io/astral-sh/uv:0.5.30@sha256:bb74263127d6451222fe7f71b330edfb189ab1c98d7898df2401fbf4f272d9b9"
_SECRET_ENVIRONMENT_MARKERS = ("token", "secret", "password", "passwd", "apikey")
_PUBLIC_DEPLOYMENT_CONFIGS = (
    "compose.yml",
    "Dockerfile",
    ".env.example",
    ".dockerignore",
)
_PUBLIC_PROCESS_ENTRYPOINTS = (
    "compose.yml",
    "Dockerfile",
    "src/aquaops/api/app.py",
    "src/aquaops/api/routes/agent.py",
    "src/aquaops/mcp/public_server.py",
)
_CONVERSION_SURFACE_MARKERS = (
    "converted",
    "conversions.private.json",
    "convert-xls",
    "conversion_",
)
_NORMALIZATION_SURFACE_MARKERS = (
    "normalized",
    "schema-candidates.private.json",
    "schema-mapping.private.json",
    "normalization-manifest.private.json",
    "normalized.duckdb",
    "propose-schema",
    "approve-schema",
    "normalize-csv",
    "verify-normalized",
)
_NORMALIZED_DIRECTORY_REFERENCE = re.compile(
    r"""(?x)
    (?:
        (?<![a-z0-9_])normalized(?=[./\\])
        | \bpath\s*\(\s*["']normalized["']\s*\)
        | \bnormalized_dir\b
        | (?:^|[=:(,\[])\s*["']normalized["'](?=\s*(?:$|[,)\]\n]))
    )
    """
)


def test_compose_pins_qdrant_and_keeps_public_dependencies_internal() -> None:
    compose = (PROJECT_ROOT / "compose.yml").read_text(encoding="utf-8")
    services = yaml.safe_load(compose)["services"]
    qdrant_service = services["qdrant"]

    assert qdrant_service["image"] == QDRANT_IMAGE
    _assert_compose_service_security(services)
    assert services["api"]["command"] == [
        "uvicorn",
        "aquaops.api.app:create_app",
        "--factory",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]


def test_compose_starts_only_the_public_backend_stack_not_public_mcp() -> None:
    compose = (PROJECT_ROOT / "compose.yml").read_text(encoding="utf-8")
    services = yaml.safe_load(compose)["services"]

    assert set(services) == {
        "postgres",
        "redis",
        "qdrant",
        "migrate",
        "api",
        "worker",
    }
    assert "aquaops.mcp.public_server" not in compose
    assert services["qdrant"].get("ports") is None
    assert services["api"]["ports"] == ["8000:8000"]


@pytest.mark.parametrize(
    "unsafe_build",
    [
        {"context": ".."},
        {"context": "C:/unsafe-parent-directory"},
    ],
)
def test_compose_security_rejects_parent_or_absolute_api_build_context(
    unsafe_build: dict[str, str],
) -> None:
    with pytest.raises(AssertionError, match="build context must be project root"):
        _assert_compose_service_security(
            {
                "api": {
                    "build": {"target": "backend", **unsafe_build},
                    "ports": ["8000:8000"],
                }
            }
        )


def test_public_deployment_and_ci_configuration_has_no_private_adapter_references() -> (
    None
):
    configurations = [PROJECT_ROOT / name for name in _PUBLIC_DEPLOYMENT_CONFIGS]
    workflows = PROJECT_ROOT / ".github" / "workflows"
    if workflows.is_dir():
        configurations.extend(sorted(workflows.glob("*.y*ml")))

    for configuration in configurations:
        contents = configuration.read_text(encoding="utf-8")
        if configuration.name == ".dockerignore":
            # The build-context denylist must name the private surfaces it excludes.
            continue
        assert "_private_local" not in contents.casefold()
        assert "private-data" not in contents.casefold()
        assert re.search(r"\bprivate_[a-z0-9_]*", contents, re.IGNORECASE) is None
        _assert_no_conversion_surface(contents)
        _assert_no_normalization_surface(contents)


def test_public_fastapi_and_mcp_entrypoints_do_not_import_private_adapter() -> None:
    for relative_path in _PUBLIC_PROCESS_ENTRYPOINTS:
        contents = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert "aquaops.private_data" not in contents
        assert "_private_local" not in contents.casefold()
        assert re.search(r"\bprivate_[a-z0-9_]*", contents, re.IGNORECASE) is None
        _assert_no_conversion_surface(contents)
        _assert_no_normalization_surface(contents)


@pytest.mark.parametrize(
    "contents",
    [
        'output_directory = Path("normalized")',
        'output_directory = "normalized"',
        "NORMALIZED_DIR = output_root",
        "normalized_dir = output_root",
    ],
)
def test_public_isolation_rejects_normalized_directory_references(
    contents: str,
) -> None:
    with pytest.raises(AssertionError):
        _assert_no_normalization_surface(contents)


def test_public_isolation_allows_unrelated_normalized_prose() -> None:
    _assert_no_normalization_surface(
        'description = "A normalized public response uses safe aggregate fields."'
    )


def test_qdrant_server_minor_is_compatible_with_locked_client() -> None:
    lock = (PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8")
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    client_version = re.search(
        r'(?ms)^name = "qdrant-client"\nversion = "(\d+)\.(\d+)\.\d+"', lock
    )
    project_specification = next(
        dependency
        for dependency in project["project"]["dependencies"]
        if dependency.startswith("qdrant-client")
    )
    specification_version = re.search(
        r">=(\d+)\.(\d+),<(\d+)\.(\d+)", project_specification
    )
    server_version = re.search(
        r":v(\d+)\.(\d+)\.\d+@sha256:[0-9a-f]{64}$", QDRANT_IMAGE
    )

    assert client_version is not None
    assert specification_version is not None
    assert server_version is not None
    assert project_specification == "qdrant-client>=1.18,<1.19"
    assert client_version.group(1) == server_version.group(1)
    assert abs(int(client_version.group(2)) - int(server_version.group(2))) <= 1
    assert client_version.group(1, 2) == specification_version.group(1, 2)
    assert specification_version.group(3, 4) == ("1", "19")


@pytest.mark.parametrize(
    "unsafe_qdrant",
    [
        "network_mode: host",
        "volumes:\n      - /qdrant/storage",
        "volumes:\n      - ./private:/qdrant/storage",
        "volumes_from:\n      - external-storage",
        "extends:\n      service: external-qdrant",
    ],
)
def test_compose_security_rejects_host_network_or_any_qdrant_storage_mount(
    unsafe_qdrant: str,
) -> None:
    service = yaml.safe_load(f"image: {QDRANT_IMAGE}\n{unsafe_qdrant}\n")

    with pytest.raises(AssertionError):
        _assert_compose_service_security({"qdrant": service})


def test_compose_security_rejects_unsafe_api_service() -> None:
    with pytest.raises(AssertionError, match="API must expose only 8000"):
        _assert_compose_service_security(
            {
                "api": {
                    "build": {"context": ".", "target": "backend"},
                    "ports": ["8001:8001"],
                }
            }
        )


@pytest.mark.parametrize(
    "unsafe_service",
    [
        {"privileged": True},
        {"environment": {"AQUAOPS_SEARCH_API_KEY": "hard-coded-token"}},
        {"environment": ["SERVICE_TOKEN=hard-coded-token"]},
        {"environment": {"PRIVATE_ROOT": "/unsafe-private-volume"}},
    ],
)
def test_compose_security_rejects_privilege_or_hard_coded_secret(
    unsafe_service: dict[str, object],
) -> None:
    with pytest.raises(AssertionError):
        _assert_compose_service_security(
            {
                "api": {
                    "build": {"context": ".", "target": "backend"},
                    "ports": ["8000:8000"],
                    **unsafe_service,
                }
            }
        )


def _assert_compose_service_security(services: dict[str, dict[str, object]]) -> None:
    allowed_named_volumes = {
        "postgres": {"postgres-data:/var/lib/postgresql/data"},
        "qdrant": {"qdrant-data:/qdrant/storage"},
        "worker": {"huggingface-model-cache:/home/aquaops/.cache/huggingface"},
    }
    for name, service in services.items():
        assert name != "private-data"
        assert service.get("network_mode") != "host"
        assert service.get("privileged", False) is False
        volumes = service.get("volumes", [])
        assert type(volumes) is list
        assert set(volumes) <= allowed_named_volumes.get(name, set())
        assert "volumes_from" not in service
        assert "extends" not in service
        assert "_private_local" not in str(service).casefold()
        assert "private" not in str(service).lower()
        _assert_no_conversion_surface(str(service))
        _assert_no_normalization_surface(str(service))
        _assert_no_hard_coded_secret_environment(service.get("environment"))
        if name in {"api", "migrate", "worker"}:
            build = service.get("build")
            assert type(build) is dict
            assert set(build) == {"context", "target"}
            assert build["context"] == ".", "build context must be project root"
            assert build["target"] == ("worker" if name == "worker" else "backend")
        if name == "api":
            assert service.get("ports") == ["8000:8000"], "API must expose only 8000"
        else:
            assert not service.get("ports")


def _assert_no_hard_coded_secret_environment(environment: object) -> None:
    if environment is None:
        return
    if type(environment) is dict:
        assignments = environment.items()
    elif type(environment) is list:
        assignments = (
            (item.partition("=")[0], item.partition("=")[2])
            for item in environment
            if type(item) is str
        )
        assert len(environment) == sum(type(item) is str for item in environment)
    else:
        raise AssertionError("Compose environment must be a mapping or a list")

    for key, value in assignments:
        assert type(key) is str
        assert not key.casefold().startswith("private_")
        normalized_key = re.sub(r"[^a-z0-9]+", "", key.casefold())
        if any(marker in normalized_key for marker in _SECRET_ENVIRONMENT_MARKERS):
            assert value in (None, "") or (
                type(value) is str and value.startswith("${") and value.endswith("}")
            )


def _assert_no_conversion_surface(contents: str) -> None:
    normalized = contents.casefold()
    for marker in _CONVERSION_SURFACE_MARKERS:
        assert marker not in normalized


def _assert_no_normalization_surface(contents: str) -> None:
    normalized = contents.casefold()
    for marker in _NORMALIZATION_SURFACE_MARKERS:
        if marker == "normalized":
            assert _NORMALIZED_DIRECTORY_REFERENCE.search(normalized) is None
            continue
        assert marker not in normalized


def test_dockerfile_pins_images_and_uses_locked_uv_commands() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert f"FROM {PYTHON_IMAGE}" in dockerfile
    assert f"COPY --from={UV_IMAGE} /uv /uvx /bin/" in dockerfile
    assert "COPY requirements-backend.lock ./" in dockerfile
    assert (
        "uv pip install --system --require-hashes -r requirements-backend.lock"
        in dockerfile
    )
    assert "uv pip install --system --no-deps ." in dockerfile
    assert '"uvicorn", "aquaops.api.app:create_app"' in dockerfile
    assert '"uv", "run", "--locked", "--no-sync", "uvicorn"' not in dockerfile


def test_gitignore_protects_local_environment_overrides() -> None:
    ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in ignored
    assert ".env.*" in ignored
    assert "!.env.example" in ignored
    assert "*.private.json" in ignored
    assert "*.private.json.tmp" in ignored
    assert "_private_local/" in ignored
    assert "reports/private/" in ignored


def test_environment_template_has_no_secret_default_values() -> None:
    template = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    for line in template.splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        assert separator == "="
        normalized_key = re.sub(r"[^a-z0-9]+", "", key.casefold())
        if any(marker in normalized_key for marker in _SECRET_ENVIRONMENT_MARKERS):
            assert value == ""


def test_readme_records_image_digest_provenance() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "2026-07-19" in readme
    assert "Docker Hub Registry v2" in readme
    assert "GitHub Container Registry v2" in readme
    assert QDRANT_IMAGE in readme
    assert "digest fixes the image content" in readme
    assert PYTHON_IMAGE in readme
    assert UV_IMAGE in readme


def test_public_candidate_license_and_build_boundaries_are_explicit() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    license_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "Apache-2.0" in readme
    assert "docs/public-license-notices.md" in readme
    assert "SECURITY.md" in readme
    assert "Apache License" in license_text
    assert (PROJECT_ROOT / "docs" / "public-license-notices.md").is_file()
    assert (PROJECT_ROOT / "SECURITY.md").is_file()
    wheel_target = project["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert "only-include" in wheel_target
    assert "packages" not in wheel_target
    assert "COPY src ./src" not in dockerfile
    assert dockerfile.count("COPY src/aquaops ./src/aquaops") == 2


def test_readme_documents_public_only_manual_entrypoints() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "docker compose up -d" in readme
    assert "docker compose --profile model-worker up --build -d" in readme
    assert "uv run --locked python -m aquaops.mcp.public_server" in readme
    assert "test_public_corpus.py" in readme
    assert "test_hybrid.py" in readme
    assert "test_evaluation.py" in readme
    assert "不得指向远程/生产 Qdrant" in readme
    assert "当前公开服务没有写入、删除、控制或私有数据工具" in readme
    # The public staging tree intentionally excludes private operational docs.
    assert "docs/private-local-data.md" not in readme
    assert "docs/private-analytics-agent.md" not in readme


def test_public_diagnostic_demo_uses_locked_fastapi_factory_command() -> None:
    demo = (PROJECT_ROOT / "docs" / "public-historical-diagnostic-demo.md").read_text(
        encoding="utf-8"
    )

    assert (
        "uv run --locked uvicorn aquaops.api.app:create_app --factory --reload" in demo
    )
    assert "aquaops.api.app:app" not in demo


def test_opt_in_contract_test_requires_two_local_write_flags_and_cleans_up() -> None:
    contract = (
        PROJECT_ROOT / "tests" / "integration" / "test_qdrant_contract.py"
    ).read_text(encoding="utf-8")

    assert "AQUAOPS_RUN_QDRANT_CONTRACT" in contract
    assert "AQUAOPS_QDRANT_CONTRACT_ALLOW_WRITE" in contract
    assert "localhost" in contract
    assert "127.0.0.1" in contract
    assert "delete_collection" in contract
    assert "uuid4" in contract
