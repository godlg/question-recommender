"""P7 · 可插拔 embedding(不绑定任何具体模型)。

定义 :class:`Embedder` 协议(``encode`` 返回 **已 L2 归一化** 的向量),
并提供两种真实实现(均从环境变量读配置,默认不启用)与一个测试用假实现:

  - :class:`OpenAICompatEmbedder` —— 任何 OpenAI 兼容 ``/embeddings`` 端点
    (``EMBED_BASE_URL`` / ``EMBED_MODEL`` / ``EMBED_API_KEY``)。
    ⚠️ 请求头必须带 ``User-Agent``:本机 WAF 会拦 urllib 默认 UA(见
    :mod:`src.llm_client`)。
  - :class:`LocalONNXEmbedder` —— 本地 ONNX 模型目录 ``EMBED_ONNX_DIR``
    (目录内需有 ``model.onnx`` + ``tokenizer.json``),用 ``onnxruntime`` +
    ``tokenizers`` 批推理 + mean pooling。
  - :class:`FakeEmbedder` —— **词袋语义向量**(按 token 计数构造,非哈希),
    维度可配,供测试与离线 demo 显式启用。

:func:`get_embedder` 按 ``EMBED_PROVIDER``(``openai`` / ``onnx`` / ``fake`` /
未设或 ``none``)选择;**任何初始化失败都记 warning 后返回 ``None``,绝不抛异常**。
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Protocol, runtime_checkable

import jieba
import numpy as np

from src.llm_client import DEFAULT_USER_AGENT

logger = logging.getLogger(__name__)


class EmbedderError(RuntimeError):
    """embedding 初始化 / 调用失败时抛出(上层负责降级)。"""


@runtime_checkable
class Embedder(Protocol):
    """embedding 协议。``encode`` 返回 **(n, dim) 且已 L2 归一化** 的矩阵。"""

    dim: int
    name: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化;零向量保持零(避免除 0)。"""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class FakeEmbedder:
    """词袋语义向量:**按 token 计数构造**,绝不用哈希。

    每个出现过的 token 依首次出现顺序分到一个维度(超出维度后按 ``dim`` 取模),
    向量即各维计数再做 L2 归一化。共享 token 的文本余弦相似度高,能真实反映召回。
    """

    def __init__(self, dim: int = 64, name: str | None = None) -> None:
        if dim <= 0:
            raise EmbedderError("dim 必须为正整数")
        self.dim = int(dim)
        self.name = name or f"fake-{self.dim}"
        self._vocab: dict[str, int] = {}

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [t for t in jieba.lcut(text or "") if t.strip()]

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            for tok in self._tokens(text):
                idx = self._vocab.get(tok)
                if idx is None:
                    idx = len(self._vocab) % self.dim
                    self._vocab[tok] = idx
                mat[i, idx] += 1.0
        return l2_normalize(mat)


class OpenAICompatEmbedder:
    """任何 OpenAI 兼容 ``POST {base}/embeddings`` 端点。

    配置来自 ``EMBED_BASE_URL`` / ``EMBED_MODEL`` / ``EMBED_API_KEY``,任一缺失即
    抛 :class:`EmbedderError`(由 :func:`get_embedder` 兜底为 ``None``)。
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        user_agent: str | None = None,
    ) -> None:
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        base = (base_url or os.environ.get("EMBED_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("EMBED_MODEL", "")
        self.api_key = api_key or os.environ.get("EMBED_API_KEY", "")
        missing = [
            name
            for name, value in (
                ("EMBED_BASE_URL", base),
                ("EMBED_MODEL", self.model),
                ("EMBED_API_KEY", self.api_key),
            )
            if not value
        ]
        if missing:
            raise EmbedderError(
                "缺少 embedding 配置: " + ", ".join(missing) + " —— 请设置对应环境变量"
            )
        if not base.endswith("/embeddings"):
            base = f"{base}/embeddings"
        self.base_url = base
        self.timeout = timeout
        self.dim = 0  # 首次调用后从响应推断
        self.name = f"openai:{self.model}"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        payload = {"model": self.model, "input": list(texts)}
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - 统一包成 EmbedderError
            raise EmbedderError(f"embedding 调用失败: {exc}") from exc
        try:
            data = sorted(body["data"], key=lambda d: d.get("index", 0))
            vecs = np.array([d["embedding"] for d in data], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbedderError(f"embedding 返回格式异常: {body!r}") from exc
        if vecs.ndim != 2 or vecs.shape[0] != len(texts):
            raise EmbedderError(f"embedding 返回条数不匹配: {vecs.shape} vs {len(texts)}")
        self.dim = int(vecs.shape[1])
        return l2_normalize(vecs)


class LocalONNXEmbedder:
    """本地 ONNX 模型(``model.onnx`` + ``tokenizer.json``),mean pooling + L2。

    不依赖 torch / transformers / fastembed / sentence-transformers;只用
    ``onnxruntime`` + ``tokenizers``。模型目录不存在或依赖缺失即抛
    :class:`EmbedderError`。
    """

    _DEFAULT_LIMIT = 512

    def __init__(
        self,
        onnx_dir: str | Path | None = None,
        limit: int = _DEFAULT_LIMIT,
    ) -> None:
        raw = str(onnx_dir or os.environ.get("EMBED_ONNX_DIR", "")).strip()
        if not raw:
            raise EmbedderError("未配置 EMBED_ONNX_DIR")
        directory = Path(raw)
        model_path = directory / "model.onnx"
        tok_path = directory / "tokenizer.json"
        if not model_path.is_file():
            raise EmbedderError(f"ONNX 模型不存在: {model_path}")
        if not tok_path.is_file():
            raise EmbedderError(f"tokenizer.json 不存在: {tok_path}")
        try:
            import onnxruntime as ort  # noqa: PLC0415 - 惰性导入,缺失时降级
            from tokenizers import Tokenizer  # noqa: PLC0415
        except ImportError as exc:
            raise EmbedderError(
                "缺少依赖 onnxruntime / tokenizers,无法使用本地 ONNX embedding"
            ) from exc
        self.limit = limit
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        if hasattr(self.tokenizer, "enable_truncation"):
            self.tokenizer.enable_truncation(max_length=limit)
        if hasattr(self.tokenizer, "enable_padding"):
            self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self.session.get_inputs()}
        self.dim = self._infer_dim()
        self.name = f"onnx:{directory.name}"

    def _infer_dim(self) -> int:
        shape = self.session.get_outputs()[0].shape
        for val in reversed(shape):
            if isinstance(val, int) and val > 0:
                return int(val)
        return 0

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        encodings = self.tokenizer.encode_batch(list(texts))
        max_len = max(len(e.ids) for e in encodings) or 1
        input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        attention = np.zeros((len(encodings), max_len), dtype=np.int64)
        for i, e in enumerate(encodings):
            n = len(e.ids)
            input_ids[i, :n] = e.ids
            attention[i, :n] = e.attention_mask or [1] * n
        feeds: dict[str, np.ndarray] = {}
        if "input_ids" in self._input_names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = attention
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        try:
            outputs = self.session.run(None, feeds)
        except Exception as exc:  # noqa: BLE001
            raise EmbedderError(f"ONNX 推理失败: {exc}") from exc
        hidden = np.asarray(outputs[0], dtype=np.float64)
        mask = attention.astype(np.float64)[..., None]
        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        pooled = summed / counts
        self.dim = int(pooled.shape[1])
        return l2_normalize(pooled)


_PROVIDERS = ("openai", "onnx", "fake")


def get_embedder(provider: str | None = None) -> Embedder | None:
    """按 ``EMBED_PROVIDER`` 构造 embedder;未配置或任何失败 → ``None``(不抛)。"""
    if provider is None:
        provider = os.environ.get("EMBED_PROVIDER", "")
    provider = (provider or "").strip().lower()
    if provider in ("", "none"):
        return None
    if provider not in _PROVIDERS:
        logger.warning("未知 EMBED_PROVIDER=%r,按未启用处理", provider)
        return None
    try:
        if provider == "fake":
            return FakeEmbedder()
        if provider == "onnx":
            return LocalONNXEmbedder()
        return OpenAICompatEmbedder()
    except Exception as exc:  # noqa: BLE001 - 初始化失败一律降级
        logger.warning("embedder(%s) 初始化失败,已降级: %s", provider, exc)
        return None
