"""P6 · 讲解材料库:上传 / 切分 / 落盘。

支持 ``.md`` / ``.txt``。切分规则:

  - 有 Markdown 二级标题 ``## 标题`` → **按标题切 chunk,标题即知识点名**
    (与题库 ``skill`` 字段对齐);
  - 没有 ``##`` 的纯文本 → 整个文件作为 1 个 chunk,知识点名取**文件名(去扩展名)**。

落盘:

  - ``data/materials/<原文件名>``     保留原文
  - ``data/materials_index.json``     chunk 索引

chunk 结构: ``{"file": str, "skill": str, "text": str, "structured": bool}``。

``structured`` 标记该 chunk 是否来自 ``##`` 分节:

  - ``True``  —— 来自 ``## 标题``,``skill`` 是真实知识点名;
  - ``False`` —— 来自没有 ``##`` 的纯文本(``skill`` 只是文件名回落),
    检索时视为**无结构材料**,可经全库回退被召回。

P7 向量缓存:落盘时可选写入 ``embedding``(list[float])与 ``embedder_name``;重载时
若 ``embedder_name`` 与当前 embedder 一致且字段齐全则**直接复用,不再重算**。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.embedder import Embedder

ALLOWED_SUFFIXES = (".md", ".txt")
_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "ISO-8859-15")

_HEADING = re.compile(r"^##[ \t]+(.+?)\s*$", re.MULTILINE)


class MaterialError(ValueError):
    """材料格式错误(扩展名不支持 / 为空 / 无法解码),对应 HTTP 400。"""


def _decode(content: bytes) -> str:
    for enc in _ENCODINGS:
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    raise MaterialError("无法识别文件编码(支持 UTF-8 / GBK)")


def _basename(name: str) -> str:
    # 只取文件名,避免上传路径穿越
    return Path(name.replace("\\", "/")).name


def split_markdown(text: str, fallback_skill: str) -> list[dict]:
    """把文本切成 chunk。返回 ``[{"skill", "text", "structured"}]``(不含 file)。

    ``structured`` 标记该 chunk 是否有真实知识点结构(来自 ``##`` 分节)。
    """
    matches = list(_HEADING.finditer(text))
    if not matches:
        body = text.strip()
        return [{"skill": fallback_skill, "text": body, "structured": False}] if body else []
    chunks: list[dict] = []
    for i, m in enumerate(matches):
        skill = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        chunks.append({"skill": skill, "text": body, "structured": True})
    return chunks


@dataclass
class MaterialStore:
    """材料库。``chunks`` 是内存里的 chunk 索引,``save()`` 落盘。

    P7:可选持有 ``embedder``。持有且可用时,``save()`` 会把每个 chunk 的
    ``embedding`` 与 ``embedder_name`` 一并落盘;``load_all()`` 若发现
    ``embedder_name`` 与当前 embedder 一致且向量齐全则**直接复用(不重算)**。
    """

    dir: Path
    index_path: Path
    chunks: list[dict] = field(default_factory=list)
    embedder: "Embedder | None" = None

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def skills(self) -> list[str]:
        seen: list[str] = []
        for c in self.chunks:
            s = c.get("skill", "")
            if s and s not in seen:
                seen.append(s)
        return seen

    def files(self) -> list[str]:
        seen: list[str] = []
        for c in self.chunks:
            f = c.get("file", "")
            if f and f not in seen:
                seen.append(f)
        return seen

    def _embed_texts(self, texts: list[str]) -> list[list[float]] | None:
        """用当前 embedder 算出向量(已 L2 归一化);无 embedder 或失败返回 None。"""
        if self.embedder is None or not texts:
            return None
        try:
            mat = self.embedder.encode(texts)
        except Exception:  # noqa: BLE001 - 向量缓存失败不影响入库
            return None
        return [[float(x) for x in row] for row in mat]

    def _cache_embeddings(self) -> None:
        """给缺向量的 chunk 补算(供落盘复用)。无 embedder 时保持不写字段。"""
        if self.embedder is None:
            return
        name = self.embedder.name
        todo = [
            c
            for c in self.chunks
            if c.get("embedder_name") != name or not isinstance(c.get("embedding"), list)
        ]
        if not todo:
            return
        vecs = self._embed_texts([str(c.get("text", "")) for c in todo])
        if vecs is None:
            return
        for c, vec in zip(todo, vecs):
            c["embedding"] = vec
            c["embedder_name"] = name

    def add_from_bytes(self, name: str, raw: bytes) -> list[dict]:
        """解析并入库一个材料文件,返回**本次新增**的 chunk。"""
        filename = _basename(name or "")
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise MaterialError(
                f"不支持的文件类型 {suffix or '(无扩展名)'};只支持 {'/'.join(ALLOWED_SUFFIXES)}"
            )
        if not raw or not raw.strip():
            raise MaterialError("文件为空")
        text = _decode(raw)
        fallback = Path(filename).stem
        parsed = split_markdown(text, fallback)
        if not parsed:
            raise MaterialError("文件没有可用的正文内容")
        new_chunks = [
            {
                "file": filename,
                "skill": c["skill"],
                "text": c["text"],
                "structured": bool(c.get("structured", False)),
            }
            for c in parsed
        ]
        if self.embedder is not None:
            vecs = self._embed_texts([c["text"] for c in new_chunks])
            if vecs is not None:
                for c, vec in zip(new_chunks, vecs):
                    c["embedding"] = vec
                    c["embedder_name"] = self.embedder.name
        # 同一文件名重复上传 → 覆盖旧 chunk
        self.chunks = [c for c in self.chunks if c.get("file") != filename]
        self.chunks.extend(new_chunks)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / filename).write_bytes(raw)
        return new_chunks

    def save(self) -> Path:
        """把 chunk 索引写入 ``index_path``(有 embedder 时补齐向量缓存)。"""
        self._cache_embeddings()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": self.chunks}
        self.index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.index_path

    def load_all(self) -> list[dict]:
        """从磁盘重建 chunk 索引(优先 index;缺失时扫描材料目录)。

        已有向量缓存的 chunk 直接复用;``embedder_name`` 变化或字段缺失的
        chunk 会在持有 embedder 时按需补算。
        """
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                chunks = data.get("chunks", []) if isinstance(data, dict) else data
            except (json.JSONDecodeError, OSError):
                chunks = []
            self.chunks = [c for c in chunks if isinstance(c, dict) and c.get("text") is not None]
            self._cache_embeddings()
            return self.chunks
        self.chunks = []
        if self.dir.exists():
            for path in sorted(self.dir.iterdir()):
                if path.suffix.lower() in ALLOWED_SUFFIXES and path.is_file():
                    self.add_from_bytes(path.name, path.read_bytes())
        return self.chunks

    def summary(self) -> dict:
        return {
            "n_chunks": self.n_chunks,
            "skills": self.skills(),
            "files": self.files(),
        }
