from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_openvino_dependencies_are_optional_and_exactly_pinned() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert project["project"]["optional-dependencies"]["public-openvino"] == [
        "openvino==2026.3.0",
        "optimum==2.1.0",
        "optimum-intel==1.27.0",
    ]
    assert all(
        "openvino" not in dependency and "optimum" not in dependency
        for dependency in project["project"]["dependencies"]
    )
