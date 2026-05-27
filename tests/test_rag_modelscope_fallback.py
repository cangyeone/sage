from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).parent.parent
WEB_APP_DIR = PROJECT_ROOT / "web_app"
for path in (str(WEB_APP_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import rag_backends


def test_modelscope_download_uses_baai_bge_m3_model_id(tmp_path, monkeypatch):
    calls = []
    fake_modelscope = types.ModuleType("modelscope")

    def fake_snapshot_download(model_id, local_dir):
        calls.append((model_id, local_dir))
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        return local_dir

    fake_modelscope.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)

    target = rag_backends._download_bge_m3_from_modelscope(tmp_path / "bge-m3")

    assert target == tmp_path / "bge-m3"
    assert calls == [("BAAI/bge-m3", str(tmp_path / "bge-m3"))]


def test_embedding_model_tries_modelscope_after_default_hf_import_failures(tmp_path):
    class DummyOnnx:
        pass

    def fake_import(name, *args, **kwargs):
        if name in {"FlagEmbedding", "sentence_transformers"}:
            raise ImportError(f"{name} unavailable")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    model = rag_backends.EmbeddingModel()
    local_dir = tmp_path / "bge-m3"

    with patch("rag_backends.get_embedding_model_path", return_value="BAAI/bge-m3"), \
         patch("rag_backends._download_bge_m3_from_modelscope", return_value=local_dir) as download, \
         patch("rag_backends._load_local_onnx_bge_m3", return_value=DummyOnnx()), \
         patch("builtins.__import__", side_effect=fake_import):
        model._load()

    download.assert_called_once()
    assert model.backend == "onnx-modelscope"
    assert isinstance(model._model, DummyOnnx)
