"""P6 · 讲解材料库:上传 / 切分 / 落盘。

支持 ``.md`` / ``.txt``。切分规则:

  - 有 Markdown 二级标题 ``## 标题`` → **按标题切 chunk,标题即知识点名**
    (与题库 ``skill`` 字段对齐);
  - 没有 ``##`` 的纯文本 → 整个文件作为 1 个 chunk,知识点名取**文件名(去扩展名)**。

落盘:

  - ``data/materials/<原文件名>``     保留原文
  - ``data/materials_index.json``     chunk 索引

chunk 结构: ``{"file": str, "skill": str, "text": str}``。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

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
    """把文本切成 chunk。返回 ``[{"skill", "text"}]``(不含 file)。"""
    matches = list(_HEADING.finditer(text))
    if not matches:
        body = text.strip()
        return [{"skill": fallback_skill, "text": body}] if body else []
    chunks: list[dict] = []
    for i, m in enumerate(matches):
        skill = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        chunks.append({"skill": skill, "text": body})
    return chunks


@dataclass
class MaterialStore:
    """材料库。``chunks`` 是内存里的 chunk 索引,``save()`` 落盘。"""

    dir: Path
    index_path: Path
    chunks: list[dict] = field(default_factory=list)

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
        new_chunks = [{"file": filename, "skill": c["skill"], "text": c["text"]} for c in parsed]
        # 同一文件名重复上传 → 覆盖旧 chunk
        self.chunks = [c for c in self.chunks if c.get("file") != filename]
        self.chunks.extend(new_chunks)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / filename).write_bytes(raw)
        return new_chunks

    def save(self) -> Path:
        """把 chunk 索引写入 ``index_path``。"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": self.chunks}
        self.index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return self.index_path

    def load_all(self) -> list[dict]:
        """从磁盘重建 chunk 索引(优先 index;缺失时扫描材料目录)。"""
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                chunks = data.get("chunks", []) if isinstance(data, dict) else data
            except (json.JSONDecodeError, OSError):
                chunks = []
            self.chunks = [c for c in chunks if isinstance(c, dict) and c.get("text") is not None]
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
