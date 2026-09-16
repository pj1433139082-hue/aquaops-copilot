from hashlib import sha256
from pathlib import Path
import logging
import sys
from types import ModuleType
import warnings

import pytest

from aquaops.rag.openvino_reranker import (
    OpenVinoPublicRerankerBackend,
    PublicOpenVinoArtifactError,
    verify_openvino_artifact,
    load_verified_openvino_reranker,
)


def _write_artifact(root: Path, files: dict[str, bytes]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, content in files.items():
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        hashes[name] = sha256(content).hexdigest()
    return hashes


def test_openvino_artifact_requires_the_exact_regular_file_manifest(
    tmp_path: Path,
) -> None:
    files = {"config.json": b"config", "openvino/model.xml": b"xml"}
    hashes = _write_artifact(tmp_path, files)

    verify_openvino_artifact(tmp_path, hashes)

    (tmp_path / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(PublicOpenVinoArtifactError):
        verify_openvino_artifact(tmp_path, hashes)


def test_openvino_artifact_rejects_changed_content_and_missing_root(
    tmp_path: Path,
) -> None:
    hashes = _write_artifact(tmp_path, {"config.json": b"expected"})
    (tmp_path / "config.json").write_bytes(b"changed")

    with pytest.raises(PublicOpenVinoArtifactError):
        verify_openvino_artifact(tmp_path, hashes)
    with pytest.raises(PublicOpenVinoArtifactError):
        verify_openvino_artifact(tmp_path / "missing", hashes)


class _Array:
    def __init__(self, values):
        self._values = values

    def reshape(self, *_shape):
        return self

    def tolist(self):
        return self._values


class _Output:
    def __init__(self, values):
        self.logits = _Array(values)


class _Tokenizer:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, first, second, **kwargs):
        self.calls.append((first, second, kwargs))
        return {"input_ids": [[1], [2]]}


class _Model:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def __call__(self, **inputs):
        self.calls.append(inputs)
        return _Output(self.values)


def test_openvino_backend_tokenizes_fixed_pairs_and_returns_flat_finite_scores() -> (
    None
):
    tokenizer = _Tokenizer()
    model = _Model([0.1, 0.9])
    backend = OpenVinoPublicRerankerBackend(tokenizer=tokenizer, model=model)

    result = backend.predict(
        [["查询一", "公开文本一"], ["查询二", "公开文本二"]],
        show_progress_bar=False,
    )

    assert result == [0.1, 0.9]
    assert tokenizer.calls == [
        (
            ["查询一", "查询二"],
            ["公开文本一", "公开文本二"],
            {
                "padding": True,
                "truncation": True,
                "max_length": 512,
                "return_tensors": "np",
            },
        )
    ]
    assert model.calls == [{"input_ids": [[1], [2]]}]


@pytest.mark.parametrize("values", [[0.1], [float("nan"), 0.2]])
def test_openvino_backend_rejects_malformed_scores(values) -> None:
    backend = OpenVinoPublicRerankerBackend(
        tokenizer=_Tokenizer(),
        model=_Model(values),
    )

    with pytest.raises(PublicOpenVinoArtifactError):
        backend.predict(
            [["查询一", "公开文本一"], ["查询二", "公开文本二"]],
            show_progress_bar=False,
        )


def test_openvino_backend_rejects_unexpected_prediction_options() -> None:
    backend = OpenVinoPublicRerankerBackend(
        tokenizer=_Tokenizer(),
        model=_Model([0.1]),
    )

    with pytest.raises(PublicOpenVinoArtifactError):
        backend.predict([["查询", "公开文本"]], batch_size=99)


def test_openvino_loader_uses_complete_root_with_fixed_ir_subfolder(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return _Tokenizer()

    class _AutoConfig:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return object()

    class _OVModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["path"] = path
            calls["kwargs"] = kwargs
            return _Model([0.5])

    optimum = ModuleType("optimum")
    intel = ModuleType("optimum.intel")
    openvino = ModuleType("optimum.intel.openvino")
    openvino.OVModelForSequenceClassification = _OVModel
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = _AutoTokenizer
    transformers.AutoConfig = _AutoConfig
    monkeypatch.setitem(sys.modules, "optimum", optimum)
    monkeypatch.setitem(sys.modules, "optimum.intel", intel)
    monkeypatch.setitem(sys.modules, "optimum.intel.openvino", openvino)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        "aquaops.rag.openvino_reranker.verify_openvino_artifact",
        lambda _path: None,
    )

    backend = load_verified_openvino_reranker()

    assert isinstance(backend, OpenVinoPublicRerankerBackend)
    assert calls["path"].name == "openvino"
    assert calls["path"].parent.name == "models"
    assert calls["kwargs"]["subfolder"] == "openvino"
    assert calls["kwargs"]["file_name"] == "openvino_model.xml"
    assert calls["kwargs"]["export"] is False


def test_openvino_loader_suppresses_third_party_path_output(
    monkeypatch,
    capsys,
    caplog,
    recwarn,
) -> None:
    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            print(f"warning path={path}", file=sys.stderr)
            logging.getLogger("transformers").warning("tokenizer path=%s", path)
            warnings.warn(f"tokenizer path={path}", SyntaxWarning, stacklevel=1)
            return _Tokenizer()

    class _AutoConfig:
        @staticmethod
        def from_pretrained(path, **kwargs):
            print(f"config path={path}")
            return object()

    class _OVModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            print(f"model path={path}", file=sys.stderr)
            return _Model([0.5])

    optimum = ModuleType("optimum")
    intel = ModuleType("optimum.intel")
    openvino = ModuleType("optimum.intel.openvino")
    openvino.OVModelForSequenceClassification = _OVModel
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = _AutoTokenizer
    transformers.AutoConfig = _AutoConfig
    monkeypatch.setitem(sys.modules, "optimum", optimum)
    monkeypatch.setitem(sys.modules, "optimum.intel", intel)
    monkeypatch.setitem(sys.modules, "optimum.intel.openvino", openvino)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(
        "aquaops.rag.openvino_reranker.verify_openvino_artifact",
        lambda _path: None,
    )

    load_verified_openvino_reranker()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert caplog.records == []
    assert list(recwarn) == []
    assert logging.root.manager.disable == logging.NOTSET
