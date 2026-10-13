"""P6/P7 · 材料检索(可插拔:词法 / 向量 / 混合)。

三种检索器,返回结构一致(``[{file, skill, text, score}]``),可直接互换:

  - :class:`KeywordRetriever` —— ``jieba`` 分词 + ``rank_bm25`` 词法排序(P6 默认)。
  - :class:`VectorRetriever`  —— 用 :class:`~src.embedder.Embedder` 做余弦相似度排序。
  - :class:`HybridRetriever`  —— 词法 + 向量,RRF(``rrf_k=60``)融合。

检索策略(**关键**,三种检索器统一):

  - chunk 有 ``skill`` 标签 → **先按知识点精确过滤,再在候选内排序**;
  - chunk 无 ``skill`` 标签(材料没按知识点组织)→ 退化为**全库检索**。

模式由 :func:`resolve_mode` 决定:``keyword``(**默认**)/ ``vector`` / ``hybrid``;
**embedder 为 None 时 ``vector`` / ``hybrid`` 自动降级为 ``keyword``**(不报错)。
"""
from __future__ import annotations

import os
from typing import Protocol

import jieba
from rank_bm25 import BM25Okapi

from src.embedder import Embedder

MODES = ("keyword", "vector", "hybrid")
RRF_K = 60


class Retriever(Protocol):
    def search(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 3,
        skill: str | None = None,
    ) -> list[dict]: ...


def tokenize(text: str) -> list[str]:
    """jieba 分词,过滤纯空白。"""
    return [t for t in jieba.lcut(text or "") if t.strip()]


def _chunk_skill(chunk: dict) -> str:
    return str(chunk.get("skill", "") or "").strip()


def _is_untagged(chunk: dict) -> bool:
    """该 chunk 是否「没按知识点组织」(可经全库回退被召回)。

    新索引带 ``structured`` 标记(见 :mod:`src.materials`):``False`` 即无结构
    (纯文本材料,``skill`` 只是文件名回落)。老索引无该标记时,退回「``skill``
    为空」的旧语义,保持兼容。
    """
    if "structured" in chunk:
        return chunk.get("structured") is not True
    return not _chunk_skill(chunk)


def filter_by_skill(chunks: list[dict], skill: str | None) -> list[dict]:
    """skill 精确过滤 + 无结构回退。

    - 未给 ``skill`` → 全量;
    - 给了 ``skill`` 且有**精确匹配** → 只留精确匹配(防止串知识点);
    - 给了 ``skill`` 但无匹配 → 回退到**无结构 chunk**(材料没按知识点组织的
      场景,含 ``##`` 之外的纯文本),没有任何无结构 chunk 时返回空。
    """
    if not skill:
        return list(chunks)
    target = skill.strip()
    exact = [c for c in chunks if _chunk_skill(c) == target]
    if exact:
        return exact
    return [c for c in chunks if _is_untagged(c)]


def _hit(chunk: dict, score: float) -> dict:
    return {
        "file": chunk.get("file", ""),
        "skill": chunk.get("skill", ""),
        "text": chunk.get("text", ""),
        "score": float(score),
    }


class KeywordRetriever:
    """默认检索器:skill 精确过滤(带无标签回退)+ BM25 词法排序。"""

    mode = "keyword"

    def search(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 3,
        skill: str | None = None,
    ) -> list[dict]:
        if not chunks or top_k <= 0:
            return []
        candidates = filter_by_skill(chunks, skill)
        if not candidates:
            return []

        corpus = [tokenize(c.get("text", "")) for c in candidates]
        if not any(corpus):
            # 全为空文本:退回原始顺序,分数置 0
            return [_hit(c, 0.0) for c in candidates[:top_k]]

        # 保证每个候选至少 1 个 token,避免 BM25 空文档报错
        safe_corpus = [toks if toks else ["\u200b"] for toks in corpus]
        bm25 = BM25Okapi(safe_corpus)
        q_tokens = tokenize(query)
        if not q_tokens:
            return [_hit(c, 0.0) for c in candidates[:top_k]]
        scores = bm25.get_scores(q_tokens)

        order = sorted(
            range(len(candidates)), key=lambda i: (-float(scores[i]), i)
        )
        return [_hit(candidates[i], float(scores[i])) for i in order[:top_k]]


class VectorRetriever:
    """向量检索:embedder 编码 query 与候选文本,按余弦相似度排序。"""

    mode = "vector"

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def search(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 3,
        skill: str | None = None,
    ) -> list[dict]:
        if not chunks or top_k <= 0:
            return []
        candidates = filter_by_skill(chunks, skill)
        if not candidates:
            return []
        texts = [query] + [str(c.get("text", "")) for c in candidates]
        mat = self.embedder.encode(texts)
        q = mat[0]
        docs = mat[1:]
        sims = docs @ q  # 两侧均已 L2 归一化 → 点积即余弦
        order = sorted(range(len(candidates)), key=lambda i: (-float(sims[i]), i))
        return [_hit(candidates[i], float(sims[i])) for i in order[:top_k]]


def _key(hit: dict) -> tuple[str, str, str]:
    return (str(hit.get("file", "")), str(hit.get("skill", "")), str(hit.get("text", "")))


def rrf_fuse(ranked_lists: list[list[dict]], top_k: int, rrf_k: int = RRF_K) -> list[dict]:
    """RRF 融合多路有序结果:``score = Σ 1 / (rrf_k + rank)``(rank 从 1 起)。"""
    scores: dict[tuple, float] = {}
    first: dict[tuple, dict] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            k = _key(hit)
            scores[k] = scores.get(k, 0.0) + 1.0 / (rrf_k + rank)
            first.setdefault(k, hit)
    order = sorted(scores, key=lambda k: -scores[k])
    return [{**first[k], "score": scores[k]} for k in order[:top_k]]


class HybridRetriever:
    """混合检索:词法 + 向量各取候选,RRF 融合。"""

    mode = "hybrid"

    def __init__(
        self,
        embedder: Embedder,
        keyword: KeywordRetriever | None = None,
        rrf_k: int = RRF_K,
    ) -> None:
        self.embedder = embedder
        self.keyword = keyword or KeywordRetriever()
        self.vector = VectorRetriever(embedder)
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 3,
        skill: str | None = None,
    ) -> list[dict]:
        if not chunks or top_k <= 0:
            return []
        depth = max(top_k * 2, 10)
        kw = self.keyword.search(query, chunks, top_k=depth, skill=skill)
        vec = self.vector.search(query, chunks, top_k=depth, skill=skill)
        if not vec:
            return kw[:top_k]
        if not kw:
            return vec[:top_k]
        return rrf_fuse([kw, vec], top_k=top_k, rrf_k=self.rrf_k)


def resolve_mode(mode: str | None = None, embedder: Embedder | None = None) -> str:
    """解析实际生效的检索模式(embedder 缺失时 vector/hybrid 自动降级 keyword)。"""
    if mode is None:
        mode = os.environ.get("RETRIEVAL_MODE", "keyword")
    mode = (mode or "keyword").strip().lower()
    if mode not in MODES:
        mode = "keyword"
    if embedder is None and mode in ("vector", "hybrid"):
        return "keyword"
    return mode


def build_retriever(
    mode: str | None = None, embedder: Embedder | None = None
) -> tuple[Retriever, str]:
    """按模式构造检索器,返回 ``(retriever, 实际生效模式)``。"""
    actual = resolve_mode(mode, embedder)
    if actual == "vector" and embedder is not None:
        return VectorRetriever(embedder), actual
    if actual == "hybrid" and embedder is not None:
        return HybridRetriever(embedder), actual
    return KeywordRetriever(), "keyword"


def search_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 3,
    skill: str | None = None,
    retriever: Retriever | None = None,
) -> list[dict]:
    """便捷函数:用给定(或默认)检索器检索。"""
    r = retriever or KeywordRetriever()
    return r.search(query, chunks, top_k=top_k, skill=skill)
