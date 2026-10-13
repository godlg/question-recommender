"""P7 embedding 层测试:全部用假实现 / monkeypatch,**不依赖任何真模型**。"""
import json

import numpy as np
import pytest

from src.embedder import (
    EmbedderError,
    FakeEmbedder,
    LocalONNXEmbedder,
    OpenAICompatEmbedder,
    get_embedder,
    l2_normalize,
)


def test_fake_embedder_shape_and_normalized():
    mat = FakeEmbedder(dim=32).encode(["分数 加减 通分", "分数 加减 通分 通分"])
    assert mat.shape == (2, 32)
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0)


def test_fake_embedder_similarity_reflects_shared_tokens():
    emb = FakeEmbedder(dim=64)
    a, b, c = emb.encode(
        ["分数加减法要先通分", "分数加减法要通分再算", "今天天气不错适合出门散步"]
    )
    assert float(a @ b) > float(a @ c)


def test_fake_embedder_tiny_dim_does_not_crash():
    mat = FakeEmbedder(dim=2).encode(["甲乙丙丁", "戊己庚辛"])
    assert mat.shape == (2, 2)


def test_fake_embedder_rejects_non_positive_dim():
    with pytest.raises(EmbedderError):
        FakeEmbedder(dim=0)


def test_l2_normalize_zero_vector_is_safe():
    assert not np.isnan(l2_normalize(np.zeros((2, 3)))).any()


def test_get_embedder_defaults_to_none(monkeypatch):
    monkeypatch.delenv("EMBED_PROVIDER", raising=False)
    assert get_embedder() is None


def test_get_embedder_none_and_unknown_provider(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "none")
    assert get_embedder() is None
    monkeypatch.setenv("EMBED_PROVIDER", "这东西不存在")
    assert get_embedder() is None


def test_get_embedder_fake(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "fake")
    assert isinstance(get_embedder(), FakeEmbedder)


def test_get_embedder_onnx_missing_dir_returns_none(monkeypatch, tmp_path):
    """模型目录不存在时必须降级为 None,不抛异常。"""
    monkeypatch.setenv("EMBED_PROVIDER", "onnx")
    monkeypatch.setenv("EMBED_ONNX_DIR", str(tmp_path / "不存在的目录"))
    assert get_embedder() is None


def test_get_embedder_openai_missing_env_returns_none(monkeypatch):
    monkeypatch.setenv("EMBED_PROVIDER", "openai")
    for key in ("EMBED_BASE_URL", "EMBED_MODEL", "EMBED_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert get_embedder() is None


def test_openai_compat_sends_user_agent_and_parses(monkeypatch):
    captured = {}

    class _Resp:
        def read(self):
            return json.dumps({"data": [{"index": 0, "embedding": [3.0, 4.0]}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        captured["headers"] = {k.lower(): v for k, v in dict(req.headers).items()}
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr("src.embedder.urllib.request.urlopen", fake_urlopen)
    emb = OpenAICompatEmbedder(base_url="http://x/v1", model="m", api_key="k")
    out = emb.encode(["你好"])

    assert captured["body"]["input"] == ["你好"]
    assert captured["body"]["model"] == "m"
    assert "Mozilla" in captured["headers"].get("user-agent", "")  # 不是 urllib 默认 UA
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_openai_compat_requires_all_env(monkeypatch):
    monkeypatch.delenv("EMBED_BASE_URL", raising=False)
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    monkeypatch.delenv("EMBED_API_KEY", raising=False)
    with pytest.raises(EmbedderError):
        OpenAICompatEmbedder()


def test_onnx_embedder_raises_when_model_missing(tmp_path):
    with pytest.raises(EmbedderError):
        LocalONNXEmbedder(onnx_dir=tmp_path)
