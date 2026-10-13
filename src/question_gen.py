"""P3 · 用 LLM 生成变式题,并以知识点样例题信息作为 RAG 上下文。

数据集(ASSISTments 2009-2010 skill builder)**没有题干文本**,只有
``problem_id / skill_id / skill_name / correct``。所以:

  - 生成变式题的输入是 **``skill_name``**(知识点名),不是原题题干;
  - RAG 检索的是该知识点的**样例题统计信息**(题目数、尝试数、全局正确率
    的 min/mean/max、代表性 problem_id),作为生成时的上下文。

LLM 输出一律走**结构化 JSON**,用 ``json.loads`` 解析;解析失败会重试,
最终仍失败则抛 ``GenerationError``,绝不静默吞掉。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data import ENCODING, find_raw
from src.llm_client import LLMClient, strip_code_fence

REQUIRED_FIELDS = ("stem", "options", "answer", "skill_name", "difficulty_hint")
_MAX_SAMPLE_PROBLEMS = 5


class GenerationError(RuntimeError):
    """LLM 未能返回可解析的结构化题目时抛出。"""


@dataclass(frozen=True)
class SkillContext:
    """某个知识点的样例题信息,拼成 RAG 上下文。"""

    skill_name: str
    n_problems: int
    n_attempts: int
    difficulty_mean: float
    difficulty_min: float
    difficulty_max: float
    sample_problem_ids: tuple[int, ...] = ()

    def to_prompt(self) -> str:
        """渲染成给 LLM 的上下文文本(即「检索到的样例题信息」)。"""
        return (
            f"知识点: {self.skill_name}\n"
            f"题库规模: {self.n_problems} 道题 / {self.n_attempts} 次作答\n"
            f"难度分布(全局正确率,越高越容易): "
            f"min={self.difficulty_min:.2f}, mean={self.difficulty_mean:.2f}, "
            f"max={self.difficulty_max:.2f}\n"
            f"样例题 ID: {', '.join(map(str, self.sample_problem_ids))}"
        )


def build_skill_contexts(data_dir: str | Path = "data") -> dict[str, SkillContext]:
    """从原始 csv 按 ``skill_name`` 汇总知识点上下文(RAG 检索层)。"""
    path = find_raw(data_dir)
    raw = pd.read_csv(
        path,
        encoding=ENCODING,
        low_memory=False,
        usecols=["problem_id", "skill_id", "skill_name", "correct"],
    ).dropna()
    raw = raw[raw["correct"].isin([0, 1])].copy()
    raw["skill_name"] = raw["skill_name"].astype(str).str.strip()
    raw = raw[raw["skill_name"] != ""]

    per_problem = raw.groupby(["skill_name", "problem_id"], as_index=False).agg(
        difficulty=("correct", "mean")
    )
    contexts: dict[str, SkillContext] = {}
    for skill_name, part in per_problem.groupby("skill_name"):
        sub = raw[raw["skill_name"] == skill_name]
        sample = tuple(
            int(x)
            for x in part.sort_values("problem_id")["problem_id"].head(_MAX_SAMPLE_PROBLEMS)
        )
        contexts[str(skill_name)] = SkillContext(
            skill_name=str(skill_name),
            n_problems=int(part["problem_id"].nunique()),
            n_attempts=int(len(sub)),
            difficulty_mean=float(part["difficulty"].mean()),
            difficulty_min=float(part["difficulty"].min()),
            difficulty_max=float(part["difficulty"].max()),
            sample_problem_ids=sample,
        )
    return contexts


def build_generation_prompt(context: SkillContext, n: int) -> str:
    """生成题的 prompt:知识点上下文 + 结构化 JSON 输出要求。"""
    schema = (
        '{"stem": "题干", "options": ["A", "B", "C", "D"], '
        '"answer": "正确选项(必须是 options 之一)", "skill_name": "'
        + context.skill_name
        + '", "difficulty_hint": 0.0}'
    )
    return (
        "你是一名出题专家。请依据下面检索到的知识点样例题信息,"
        f"命制 {n} 道同知识点的变式题,难度贴合原题库。\n\n"
        f"【知识点上下文】\n{context.to_prompt()}\n\n"
        "【输出要求】只输出一个 JSON 数组,每个元素形如:\n"
        f"{schema}\n"
        "其中 difficulty_hint 是 0~1 的难度提示(0 最难、1 最易),"
        "answer 必须精确等于 options 中的某一项。不要输出任何解释文字。"
    )


def _coerce_question(raw: dict, fallback_skill: str) -> dict:
    missing = [k for k in REQUIRED_FIELDS if k not in raw]
    if missing:
        raise GenerationError(f"生成题缺少字段 {missing}: {raw!r}")
    options = raw["options"]
    if not isinstance(options, list) or len(options) < 2:
        raise GenerationError(f"options 必须是不少于 2 项的数组: {raw!r}")
    options = [str(o) for o in options]
    answer = str(raw["answer"])
    if answer not in options:
        raise GenerationError(f"answer={answer!r} 不在 options 中: {raw!r}")
    return {
        "stem": str(raw["stem"]),
        "options": options,
        "answer": answer,
        "skill_name": str(raw.get("skill_name") or fallback_skill),
        "difficulty_hint": float(raw["difficulty_hint"]),
    }


def parse_questions(text: str, fallback_skill: str) -> list[dict]:
    """用 json.loads 解析 LLM 输出,支持数组或 ``{"questions": [...]}``。"""
    cleaned = strip_code_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"无法解析 LLM 输出为 JSON: {exc}; 原文: {cleaned[:200]!r}") from exc
    if isinstance(data, dict):
        data = data.get("questions", [data])
    if not isinstance(data, list):
        raise GenerationError(f"LLM 输出不是数组或含 questions 的对象: {data!r}")
    return [_coerce_question(q, fallback_skill) for q in data]


class QuestionGenerator:
    """生成器:独立持有 LLM 客户端(可与评审器分别注入)。"""

    def __init__(
        self,
        llm: LLMClient,
        contexts: dict[str, SkillContext] | None = None,
        max_retries: int = 1,
    ) -> None:
        self.llm = llm
        self.contexts = contexts or {}
        self.max_retries = max_retries

    def context_for(self, skill_name: str) -> SkillContext:
        ctx = self.contexts.get(skill_name)
        if ctx is None:
            # 没有检索到样例题信息时退化为仅用知识点名(仍可生成)。
            ctx = SkillContext(skill_name, 0, 0, 0.5, 0.5, 0.5, ())
        return ctx

    def generate(self, skill_name: str, n: int) -> list[dict]:
        """生成 n 道变式题(结构化 JSON)。解析失败重试,最终失败抛错。"""
        if n <= 0:
            return []
        context = self.context_for(skill_name)
        prompt = build_generation_prompt(context, n)
        last_err: Exception | None = None
        for _ in range(self.max_retries + 1):
            resp = self.llm.complete(prompt)
            try:
                questions = parse_questions(resp.text, skill_name)
            except GenerationError as exc:
                last_err = exc
                continue
            return questions
        raise GenerationError(f"生成 {skill_name!r} 变式题失败: {last_err}")
