"""P6:讲解材料库 + 检索测试(全部离线,用内置样例材料文本)。"""
from __future__ import annotations

import json

import pytest

from src.materials import MaterialError, MaterialStore, split_markdown
from src.retriever import KeywordRetriever, search_chunks

MD = (
    "# 数学\n"
    "\n"
    "## 一元一次方程\n"
    "移项要变号。2x + 3 = 7 解得 x = 2。\n"
    "\n"
    "## 分数运算\n"
    "异分母先通分。1/2 + 1/3 = 5/6。\n"
    "\n"
    "## 几何周长\n"
    "长方形周长 = (长 + 宽) × 2。\n"
)

PLAIN = "这是一段没有小标题的纯文本讲解,用来验证按文件名做知识点名。"


def _store(tmp_path) -> MaterialStore:
    return MaterialStore(tmp_path / "materials", tmp_path / "materials_index.json")


# --- 切分 -----------------------------------------------------------------


def test_split_by_h2_headings():
    chunks = split_markdown(MD, fallback_skill="文件名")
    assert [c["skill"] for c in chunks] == ["一元一次方程", "分数运算", "几何周长"]
    assert "移项要变号" in chunks[0]["text"]
    assert "通分" in chunks[1]["text"]


def test_split_plain_text_uses_filename():
    chunks = split_markdown(PLAIN, fallback_skill="纯文本讲解")
    assert len(chunks) == 1
    assert chunks[0]["skill"] == "纯文本讲解"
    assert "纯文本讲解" in chunks[0]["text"]


def test_split_empty_text_returns_nothing():
    assert split_markdown("   \n  ", fallback_skill="x") == []


def test_add_markdown_uses_heading_as_skill(tmp_path):
    store = _store(tmp_path)
    new = store.add_from_bytes("数学知识点讲解.md", MD.encode("utf-8"))
    assert len(new) == 3
    assert store.skills() == ["一元一次方程", "分数运算", "几何周长"]
    assert store.files() == ["数学知识点讲解.md"]


def test_add_plain_txt_uses_filename_as_skill(tmp_path):
    store = _store(tmp_path)
    store.add_from_bytes("概率笔记.txt", PLAIN.encode("utf-8"))
    assert store.skills() == ["概率笔记"]


def test_reject_unsupported_suffix(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(MaterialError):
        store.add_from_bytes("notes.pdf", b"%PDF-1.4")


def test_reject_empty_file(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(MaterialError):
        store.add_from_bytes("empty.md", b"   ")


def test_original_file_written_to_disk(tmp_path):
    store = _store(tmp_path)
    store.add_from_bytes("数学知识点讲解.md", MD.encode("utf-8"))
    saved = store.dir / "数学知识点讲解.md"
    assert saved.exists()
    assert saved.read_text(encoding="utf-8") == MD


# --- 持久化 ---------------------------------------------------------------


def test_save_then_reload_consistent(tmp_path):
    store = _store(tmp_path)
    store.add_from_bytes("数学知识点讲解.md", MD.encode("utf-8"))
    store.save()

    reloaded = _store(tmp_path)
    reloaded.load_all()
    assert reloaded.chunks == store.chunks
    assert reloaded.skills() == store.skills()


def test_load_all_scans_dir_without_index(tmp_path):
    """索引缺失时也能扫材料目录重建 chunk。"""
    store = _store(tmp_path)
    store.add_from_bytes("复习.txt", PLAIN.encode("utf-8"))
    store.save()
    store.index_path.unlink()  # 删掉索引

    fresh = _store(tmp_path)
    fresh.load_all()
    assert fresh.skills() == ["复习"]


def test_reupload_same_file_replaces_chunks(tmp_path):
    store = _store(tmp_path)
    store.add_from_bytes("m.md", MD.encode("utf-8"))
    store.add_from_bytes("m.md", "## 新知识点\n只有一节。".encode("utf-8"))
    assert store.skills() == ["新知识点"]
    assert store.files() == ["m.md"]


def test_summary_shape(tmp_path):
    store = _store(tmp_path)
    store.add_from_bytes("数学知识点讲解.md", MD.encode("utf-8"))
    s = store.summary()
    assert s["n_chunks"] == 3
    assert set(s["skills"]) == {"一元一次方程", "分数运算", "几何周长"}
    assert s["files"] == ["数学知识点讲解.md"]


# --- 检索 -----------------------------------------------------------------


def test_search_filters_by_skill(tmp_path):
    store = _store(tmp_path)
    store.add_from_bytes("数学知识点讲解.md", MD.encode("utf-8"))
    hits = KeywordRetriever().search("1/2 + 1/3", store.chunks, top_k=3, skill="分数运算")
    assert hits
    assert all(h["skill"] == "分数运算" for h in hits)


def test_search_same_skill_ranks_relevant_first(tmp_path):
    store = _store(tmp_path)
    store.add_from_bytes("数学知识点讲解.md", MD.encode("utf-8"))
    hits = KeywordRetriever().search("移项 变号 方程", store.chunks, top_k=3, skill=None)
    assert hits[0]["skill"] == "一元一次方程"


def test_search_empty_chunks_returns_empty():
    assert KeywordRetriever().search("任意", [], top_k=3, skill=None) == []


def test_search_skill_with_no_match_returns_empty(tmp_path):
    store = _store(tmp_path)
    store.add_from_bytes("数学知识点讲解.md", MD.encode("utf-8"))
    assert KeywordRetriever().search("x", store.chunks, top_k=3, skill="不存在的知识点") == []


def test_search_result_shape(tmp_path):
    store = _store(tmp_path)
    store.add_from_bytes("数学知识点讲解.md", MD.encode("utf-8"))
    hit = search_chunks("周长", store.chunks, top_k=1, skill="几何周长")[0]
    assert set(hit) == {"file", "skill", "text", "score"}
    assert hit["file"] == "数学知识点讲解.md"


def test_sample_materials_align_with_bank_skills():
    """内置样例材料的 5 节必须覆盖 sample_bank.csv 的全部知识点。"""
    import csv
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    raw = (root / "data" / "sample_materials" / "数学知识点讲解.md").read_bytes()
    md_skills = {c["skill"] for c in split_markdown(raw.decode("utf-8"), "数学知识点讲解")}
    with (root / "data" / "sample_bank.csv").open(encoding="utf-8") as f:
        bank_skills = {row["skill"] for row in csv.DictReader(f)}
    assert bank_skills <= md_skills
    assert len(md_skills) == 5


def test_index_json_written(tmp_path):
    store = _store(tmp_path)
    store.add_from_bytes("数学知识点讲解.md", MD.encode("utf-8"))
    store.save()
    data = json.loads(store.index_path.read_text(encoding="utf-8"))
    assert len(data["chunks"]) == 3
