"""P6 · 材料检索(可插拔)。

定义 :class:`Retriever` 协议,默认实现 :class:`KeywordRetriever`:
``jieba`` 分词 + ``rank_bm25`` 词法排序。

检索思路与推荐链路一致:**精确条件优先**(``skill`` 非空时先按知识点精确过滤),
**词法/语义只用于同知识点内的排序**。

扩展点(本任务不实现向量):
    class VectorRetriever:
        def __init__(self, embedder, materials): ...
        def search(self, query, chunks, top_k, skill=None):
            # 1) 同样先 skill 精确过滤
            # 2) 用 embedder(query) 与 embedder(chunk.text) 的余弦相似度排序
            # 3) 返回同结构 [{file, skill, text, score}]
            ...
只要保持返回结构一致,即可直接替换 KeywordRetriever,无需改动 Explainer / 接口。
"""
from __future__ import annotations

from typing import Protocol

import jieba
from rank_bm25 import BM25Okapi


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


class KeywordRetriever:
    """默认检索器:skill 精确过滤 + BM25 词法排序。"""

    def search(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 3,
        skill: str | None = None,
    ) -> list[dict]:
        if not chunks or top_k <= 0:
            return []
        candidates = self._filter_by_skill(chunks, skill)
        if not candidates:
            return []

        corpus = [tokenize(c.get("text", "")) for c in candidates]
        if not any(corpus):
            # 全为空文本:退回原始顺序,分数置 0
            return [self._hit(c, 0.0) for c in candidates[:top_k]]

        # 保证每个候选至少 1 个 token,避免 BM25 空文档报错
        safe_corpus = [toks if toks else ["\u200b"] for toks in corpus]
        bm25 = BM25Okapi(safe_corpus)
        q_tokens = tokenize(query)
        if not q_tokens:
            return [self._hit(c, 0.0) for c in candidates[:top_k]]
        scores = bm25.get_scores(q_tokens)

        order = sorted(
            range(len(candidates)), key=lambda i: (-float(scores[i]), i)
        )
        return [self._hit(candidates[i], float(scores[i])) for i in order[:top_k]]

    @staticmethod
    def _filter_by_skill(chunks: list[dict], skill: str | None) -> list[dict]:
        if not skill:
            return list(chunks)
        exact = [c for c in chunks if str(c.get("skill", "")).strip() == skill.strip()]
        return exact

    @staticmethod
    def _hit(chunk: dict, score: float) -> dict:
        return {
            "file": chunk.get("file", ""),
            "skill": chunk.get("skill", ""),
            "text": chunk.get("text", ""),
            "score": score,
        }


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
