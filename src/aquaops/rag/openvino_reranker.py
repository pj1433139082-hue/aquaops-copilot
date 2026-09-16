"""Verified, offline-only OpenVINO adapter for the fixed public reranker."""

from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
from io import StringIO
import logging
from math import isfinite
from pathlib import Path
from typing import Protocol
import warnings


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_OPENVINO_ARTIFACT_PATH = (
    _PROJECT_ROOT
    / "artifacts"
    / "public-demo"
    / "backend-benchmark"
    / "models"
    / "openvino"
)
PUBLIC_OPENVINO_ARTIFACT_FILES = {
    "config_sentence_transformers.json": "ed7d3dd12f6cbac151266a1ae160e9333b9cd1e5797fb00b0fa44bd946679842",
    "config.json": "b654d6598b95be4656a4eefd389695542aa4bf30c7eb24378f2e9da8abcfcaa5",
    "modules.json": "d14691fd903a7121e75b7cf346528a2aa440a6135205a6add060b8f07e8b738d",
    "openvino/openvino_model.bin": "f9955f20ea1991bf91e0e74f3242b395eee5695d575ffd279cc9287c9c60c48f",
    "openvino/openvino_model.xml": "22cc149d009bf1baa5a65594898ea6e03e35c6725576d797bd519773be06ad7f",
    "sentence_bert_config.json": "dd5041705e857af983711291644d950d98f590e59b9152043d24b306a929842c",
    "special_tokens_map.json": "152f77ffe3cc174ee82dfc88eed567a04152a7ed214818ec74c99548fbf347c8",
    "tokenizer_config.json": "410d349ad6778e60273579081da479cf72b4a979de111d597bdb805b9afc6bab",
    "tokenizer.json": "14917dd757b81bc44d4af6b028367351702656670c1954e055dabdfcf21593cf",
}
_MAX_ARTIFACT_FILE_BYTES = 1_200_000_000
_HASH_BUFFER_BYTES = 1024 * 1024


class PublicOpenVinoArtifactError(ValueError):
    """The fixed public OpenVINO artifact or runtime violated its contract."""


class _Tokenizer(Protocol):
    def __call__(
        self, first: list[str], second: list[str], **kwargs: object
    ) -> Mapping[str, object]: ...


class _Model(Protocol):
    def __call__(self, **inputs: object) -> object: ...


class OpenVinoPublicRerankerBackend:
    """Small CrossEncoder-compatible wrapper around one verified OV model."""

    def __init__(self, *, tokenizer: _Tokenizer, model: _Model) -> None:
        self._tokenizer = tokenizer
        self._model = model

    def predict(self, pairs: list[list[str]], **kwargs: object) -> list[float]:
        if kwargs != {"show_progress_bar": False}:
            raise PublicOpenVinoArtifactError("public OpenVINO request is invalid")
        if (
            type(pairs) is not list
            or not 1 <= len(pairs) <= 20
            or any(
                type(pair) is not list
                or len(pair) != 2
                or any(type(text) is not str or not text for text in pair)
                for pair in pairs
            )
        ):
            raise PublicOpenVinoArtifactError("public OpenVINO request is invalid")
        try:
            inputs = self._tokenizer(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            output = self._model(**dict(inputs))
            logits = output.logits
            if hasattr(logits, "detach"):
                logits = logits.detach()
            if hasattr(logits, "cpu"):
                logits = logits.cpu()
            if hasattr(logits, "numpy"):
                logits = logits.numpy()
            values = logits.reshape(-1).tolist()
        except PublicOpenVinoArtifactError:
            raise
        except Exception as error:
            raise PublicOpenVinoArtifactError(
                "public OpenVINO inference is unavailable"
            ) from error
        if (
            type(values) is not list
            or len(values) != len(pairs)
            or any(
                type(value) not in (int, float) or not isfinite(float(value))
                for value in values
            )
        ):
            raise PublicOpenVinoArtifactError("public OpenVINO output is invalid")
        return [float(value) for value in values]


def verify_openvino_artifact(
    root: Path,
    expected_files: Mapping[str, str] = PUBLIC_OPENVINO_ARTIFACT_FILES,
) -> None:
    """Verify the exact regular-file set before optional runtime imports."""
    if not isinstance(root, Path) or type(expected_files) is not dict:
        raise PublicOpenVinoArtifactError("public OpenVINO artifact is invalid")
    try:
        if not root.is_dir() or root.is_symlink():
            raise PublicOpenVinoArtifactError("public OpenVINO artifact is unavailable")
        actual_files: dict[str, Path] = {}
        for item in root.rglob("*"):
            if item.is_symlink():
                raise PublicOpenVinoArtifactError("public OpenVINO artifact is invalid")
            if item.is_file():
                relative = item.relative_to(root).as_posix()
                actual_files[relative] = item
        if set(actual_files) != set(expected_files):
            raise PublicOpenVinoArtifactError("public OpenVINO artifact is invalid")
        for relative, expected_sha256 in expected_files.items():
            item = actual_files[relative]
            size = item.stat().st_size
            if not 0 < size <= _MAX_ARTIFACT_FILE_BYTES:
                raise PublicOpenVinoArtifactError("public OpenVINO artifact is invalid")
            digest = sha256()
            with item.open("rb") as stream:
                while buffer := stream.read(_HASH_BUFFER_BYTES):
                    digest.update(buffer)
            if digest.hexdigest() != expected_sha256:
                raise PublicOpenVinoArtifactError("public OpenVINO artifact is invalid")
    except PublicOpenVinoArtifactError:
        raise
    except Exception as error:
        raise PublicOpenVinoArtifactError(
            "public OpenVINO artifact is unavailable"
        ) from error


def load_verified_openvino_reranker() -> OpenVinoPublicRerankerBackend:
    """Load only the fixed offline artifact on CPU after full verification."""
    verify_openvino_artifact(PUBLIC_OPENVINO_ARTIFACT_PATH)
    try:
        with (
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            previous_disable = logging.root.manager.disable
            logging.disable(logging.CRITICAL)
            try:
                from optimum.intel.openvino import OVModelForSequenceClassification
                from transformers import AutoConfig, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    PUBLIC_OPENVINO_ARTIFACT_PATH,
                    local_files_only=True,
                    trust_remote_code=False,
                )
                config = AutoConfig.from_pretrained(
                    PUBLIC_OPENVINO_ARTIFACT_PATH,
                    local_files_only=True,
                    trust_remote_code=False,
                )
                model = OVModelForSequenceClassification.from_pretrained(
                    PUBLIC_OPENVINO_ARTIFACT_PATH,
                    config=config,
                    subfolder="openvino",
                    file_name="openvino_model.xml",
                    export=False,
                    local_files_only=True,
                    trust_remote_code=False,
                    device="CPU",
                    compile=True,
                )
            finally:
                logging.disable(previous_disable)
    except Exception as error:
        raise PublicOpenVinoArtifactError(
            "public OpenVINO runtime is unavailable"
        ) from error
    return OpenVinoPublicRerankerBackend(tokenizer=tokenizer, model=model)
