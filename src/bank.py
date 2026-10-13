"""P5:自定义题库解析 + 存储(纯本地,无 LLM)。

标准题库格式(CSV / Excel,列名大小写不敏感):

| 列 | 说明 |
|---|---|
| ``stem`` | 题干 |
| ``answer`` | 正确答案(有 options 时必须属于 options) |
| ``skill`` | 知识点名 |
| ``options`` | 选项,用 ``|`` 分隔(如 ``x=3|x=5``);可为空(填空题) |
| ``difficulty`` | 0~1,越大越容易;可空 → 默认 0.5 |

``question_id`` 由本模块生成(从 1 递增)。缺必需列 → 抛 :class:`BankError`
(对应 HTTP 400);单行坏(如 answer 不在 options 里)→ 跳过并计入 ``n_skipped``。

新题库没有历史作答,BKT 用 :data:`DEFAULT_PARAMS`(L0=0.3, T=0.2, G=0.2, S=0.1)。
"""
from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.bkt import BKTParams
from src.recommender import QuestionPool

REQUIRED_COLUMNS = ("stem", "answer", "skill", "options", "difficulty")
DEFAULT_DIFFICULTY = 0.5
BANK_FILENAME = "bank.jsonl"

# 新题库无历史作答 → BKT 默认参数(README P5 章节说明)
DEFAULT_PARAMS = BKTParams(l0=0.3, t=0.2, g=0.2, s=0.1)

_ENCODINGS = ("utf-8-sig", "gb18030", "ISO-8859-15")


class BankError(ValueError):
    """题库格式错误(缺列 / 无法解析),对应 HTTP 400。"""


class _BadRow(Exception):
    """单行坏数据,跳过并计数。"""


@dataclass
class Bank:
    questions: list[dict] = field(default_factory=list)
    skill_names: dict[int, str] = field(default_factory=dict)
    skill_to_id: dict[str, int] = field(default_factory=dict)
    source_name: str = ""
    n_skipped: int = 0

    @property
    def n_questions(self) -> int:
        return len(self.questions)

    @property
    def n_skills(self) -> int:
        return len(self.skill_names)

    def params(self) -> dict[int, BKTParams]:
        """所有知识点共用同一套默认 BKT 参数。"""
        return {sid: DEFAULT_PARAMS for sid in self.skill_names}

    def pool(self) -> QuestionPool:
        """把题库变成推荐用的 QuestionPool(problem_id = question_id)。"""
        pids = np.array([q["question_id"] for q in self.questions], dtype=int)
        skills = np.array(
            [self.skill_to_id[q["skill"]] for q in self.questions], dtype=int
        )
        difficulty = np.array([q["difficulty"] for q in self.questions], dtype=float)
        pool = QuestionPool(problem_id=pids, skill=skills, difficulty=difficulty)
        pool.skill_lookup = dict(zip(pids.tolist(), skills.tolist()))
        for sid in np.unique(skills):
            pool.by_skill[int(sid)] = pids[skills == sid]
        return pool

    def question(self, question_id: int) -> dict | None:
        for q in self.questions:
            if q["question_id"] == int(question_id):
                return q
        return None


def _clean_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            # pandas 把 "2" 读成 2.0 时,还原成 "2" 以便与字符串选项比较
            return str(int(value))
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def _parse_options(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        if isinstance(value, float) and math.isnan(value):
            return []
        items = str(value).split("|")
    return [s for s in (_clean_str(it) for it in items) if s]


def _parse_difficulty(value) -> float:
    if value is None:
        return DEFAULT_DIFFICULTY
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none"}:
        return DEFAULT_DIFFICULTY
    try:
        d = float(value)
    except (TypeError, ValueError):
        return DEFAULT_DIFFICULTY
    if math.isnan(d):
        return DEFAULT_DIFFICULTY
    return min(1.0, max(0.0, d))


def _row_to_question(rec: dict) -> dict:
    stem = _clean_str(rec.get("stem"))
    answer = _clean_str(rec.get("answer"))
    skill = _clean_str(rec.get("skill"))
    if not stem or not answer or not skill:
        raise _BadRow
    options = _parse_options(rec.get("options"))
    if options and answer not in options:
        raise _BadRow
    return {
        "stem": stem,
        "answer": answer,
        "options": options,
        "skill": skill,
        "difficulty": _parse_difficulty(rec.get("difficulty")),
    }


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: str(c).strip().lower() for c in df.columns})


def parse_frame(df: pd.DataFrame, source_name: str = "") -> Bank:
    """解析标准题库 DataFrame → :class:`Bank`。缺列抛 :class:`BankError`。"""
    df = _normalize_columns(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise BankError("缺少必需列:" + "、".join(missing))

    bank = Bank(source_name=source_name)
    for rec in df.to_dict("records"):
        try:
            q = _row_to_question(rec)
        except _BadRow:
            bank.n_skipped += 1
            continue
        skill = q["skill"]
        if skill not in bank.skill_to_id:
            sid = len(bank.skill_to_id) + 1
            bank.skill_to_id[skill] = sid
            bank.skill_names[sid] = skill
        q["question_id"] = len(bank.questions) + 1
        bank.questions.append(q)
    return bank


def _decode(content: bytes) -> str:
    for enc in _ENCODINGS:
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    raise BankError("无法识别文件编码(支持 UTF-8 / GBK)")


def _read_bytes(filename: str | None, content: bytes) -> pd.DataFrame:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        try:
            return pd.read_excel(io.BytesIO(content), engine="openpyxl")
        except Exception as exc:  # noqa: BLE001 - 统一转成 400
            raise BankError(f"无法解析 Excel 文件:{exc}") from exc
    text = _decode(content)
    try:
        return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except Exception as exc:  # noqa: BLE001
        raise BankError(f"无法解析 CSV 文件:{exc}") from exc


def parse_upload(
    filename: str | None, content: bytes, source_name: str | None = None
) -> Bank:
    """从上传的字节流解析题库。格式错 / 缺列抛 :class:`BankError`。"""
    if not content or not content.strip():
        raise BankError("文件为空")
    df = _read_bytes(filename, content)
    return parse_frame(df, source_name=source_name or (filename or "upload"))


def save_bank(bank: Bank, path: str | Path) -> Path:
    """落盘为 JSONL:每行一道题的标准 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in bank.questions:
            rec = dict(q)
            rec["source_name"] = bank.source_name
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def load_bank(path: str | Path) -> Bank | None:
    """从 JSONL 重载题库;文件不存在返回 None。"""
    path = Path(path)
    if not path.exists():
        return None
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        return None
    source = str(records[0].get("source_name", path.name))
    return parse_frame(pd.DataFrame(records), source_name=source)
