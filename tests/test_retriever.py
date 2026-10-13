"""P7 检索层测试:词法 / 向量 / 混合 + 模式降级。全部用 FakeEmbedder。"""
from src.embedder import FakeEmbedder
from src.retriever import (
    KeywordRetriever,
    VectorRetriever,
    build_retriever,
    filter_by_skill,
    resolve_mode,
    rrf_fuse,
    search_chunks,
)


def chunk(text, skill="", structured=None, file="m.md"):
    c = {"text": text, "skill": skill, "file": file}
    if structured is not None:
        c["structured"] = structured
    return c


CHUNKS = [
    chunk("分数加减法要先通分,化成同分母再运算。", "分数运算", True),
    chunk("一元一次方程:移项、合并同类项求解。", "一元一次方程", True),
    chunk("今天天气不错,适合出门散步放松一下。", "", False),  # 没按知识点组织
]


# ---------- 过滤 ----------


def test_filter_without_skill_returns_all():
    assert len(filter_by_skill(CHUNKS, None)) == 3


def test_filter_exact_match_only():
    got = filter_by_skill(CHUNKS, "分数运算")
    assert [c["skill"] for c in got] == ["分数运算"]


def test_filter_falls_back_to_untagged_when_no_match():
    """指定了库里没有的知识点 → 回退到「没按知识点组织」的 chunk。"""
    got = filter_by_skill(CHUNKS, "几何周长")
    assert got and all(c["skill"] == "" for c in got)


def test_filter_empty_when_all_structured_and_no_match():
    structured_only = [CHUNKS[0], CHUNKS[1]]
    assert filter_by_skill(structured_only, "几何周长") == []


# ---------- 三种检索器 ----------


def test_keyword_retriever_finds_relevant_chunk():
    hits = KeywordRetriever().search("通分 同分母", CHUNKS, top_k=2)
    assert hits[0]["skill"] == "分数运算"


def test_vector_retriever_finds_relevant_chunk():
    hits = VectorRetriever(FakeEmbedder(dim=64)).search("通分 同分母", CHUNKS, top_k=2)
    assert hits[0]["skill"] == "分数运算"


def test_vector_retriever_reaches_untagged_chunks():
    """材料没按知识点组织时,无结构 chunk 也要能被检索到。"""
    hits = VectorRetriever(FakeEmbedder(dim=64)).search("天气 散步", CHUNKS, top_k=3)
    assert any(h["skill"] == "" for h in hits)


def test_rrf_fuse_keeps_both_routes():
    a = [
        {"file": "a", "skill": "s", "text": "x", "score": 1.0},
        {"file": "b", "skill": "s", "text": "y", "score": 0.5},
    ]
    b = [
        {"file": "b", "skill": "s", "text": "y", "score": 9.0},
        {"file": "a", "skill": "s", "text": "x", "score": 1.0},
    ]
    fused = rrf_fuse([a, b], top_k=2)
    assert {h["file"] for h in fused} == {"a", "b"}


def test_hybrid_retriever_ranks_relevant_first():
    from src.retriever import HybridRetriever

    hits = HybridRetriever(FakeEmbedder(dim=64)).search("通分 同分母", CHUNKS, top_k=3)
    assert hits[0]["skill"] == "分数运算"


# ---------- 模式解析与降级 ----------


def test_resolve_mode_downgrades_without_embedder(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_MODE", raising=False)
    assert resolve_mode("vector", None) == "keyword"
    assert resolve_mode("hybrid", None) == "keyword"
    assert resolve_mode("vector", FakeEmbedder()) == "vector"


def test_resolve_mode_defaults_to_keyword(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_MODE", raising=False)
    assert resolve_mode(None, FakeEmbedder()) == "keyword"


def test_resolve_mode_unknown_falls_back():
    assert resolve_mode("乱写的", FakeEmbedder()) == "keyword"


def test_build_retriever_downgrade_returns_keyword():
    retriever, actual = build_retriever("vector", None)
    assert isinstance(retriever, KeywordRetriever)
    assert actual == "keyword"


def test_build_retriever_vector_with_embedder():
    retriever, actual = build_retriever("vector", FakeEmbedder())
    assert isinstance(retriever, VectorRetriever)
    assert actual == "vector"


def test_search_chunks_default_is_keyword():
    hits = search_chunks("通分", CHUNKS, top_k=1)
    assert len(hits) == 1
