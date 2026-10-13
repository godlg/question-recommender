"""P3 · 独立评审 agent。

评审器与生成器是**两次完全独立的 LLM 调用**:独立 prompt、独立上下文,
禁止复用生成时的上下文。两者可分别注入不同的 LLM 客户端(便于测试)。

评审输出结构化 JSON::

    {"verdict": "pass"|"fail",
     "reasons": [str],
     "checks": {"answer_correct": bool, "unambiguous": bool,
                "difficulty_plausible": bool, "skill_aligned": bool}}
"""
from __future__ import annotations

import json

from src.llm_client import LLMClient, strip_code_fence
from src.question_gen import SkillContext

CHECK_KEYS = ("answer_correct", "unambiguous", "difficulty_plausible", "skill_aligned")


class ReviewError(RuntimeError):
    """LLM 评审输出无法解析/不合规时抛出。"""


def build_review_prompt(question: dict, context: SkillContext | None = None) -> str:
    """评审 prompt —— 与生成 prompt 独立,不共享生成时上下文。"""
    ctx_line = (
        f"参考知识点分布(全局正确率 mean={context.difficulty_mean:.2f}): {context.skill_name}"
        if context is not None
        else "无额外上下文"
    )
    question_json = json.dumps(question, ensure_ascii=False, indent=2)
    return (
        "你是一名严格的独立审题人。请审查下面这道题,不要假设它一定正确。\n"
        f"知识点: {question.get('skill_name')}\n{ctx_line}\n\n"
        f"【待审题目】\n{question_json}\n\n"
        "【审查项】\n"
        "1. answer_correct: 标注答案是否真的正确;\n"
        "2. unambiguous: 题干是否无歧义、只有一个正确选项;\n"
        "3. difficulty_plausible: 难度提示是否与该知识点分布相符;\n"
        "4. skill_aligned: 题目是否确实考察该知识点。\n\n"
        "【输出要求】只输出一个 JSON 对象:\n"
        '{"verdict": "pass"|"fail", "reasons": ["..."], '
        '"checks": {"answer_correct": true, "unambiguous": true, '
        '"difficulty_plausible": true, "skill_aligned": true}}\n'
        "只要任一 checks 为 false,verdict 必须是 fail。不要输出解释文字。"
    )


def parse_review(text: str) -> dict:
    """解析评审输出;缺字段/非法 verdict 一律抛 ReviewError。"""
    cleaned = strip_code_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ReviewError(f"无法解析评审输出为 JSON: {exc}; 原文: {cleaned[:200]!r}") from exc
    if not isinstance(data, dict):
        raise ReviewError(f"评审输出不是对象: {data!r}")
    verdict = data.get("verdict")
    if verdict not in ("pass", "fail"):
        raise ReviewError(f"verdict 非法: {verdict!r}")
    checks = data.get("checks")
    if not isinstance(checks, dict) or any(k not in checks for k in CHECK_KEYS):
        raise ReviewError(f"checks 缺失或不完整: {checks!r}")
    checks = {k: bool(checks[k]) for k in CHECK_KEYS}
    reasons = data.get("reasons", [])
    if not isinstance(reasons, list):
        raise ReviewError(f"reasons 必须是数组: {reasons!r}")
    # 与 checks 保持一致:任一不过即为 fail。
    if not all(checks.values()) and verdict == "pass":
        verdict = "fail"
    return {"verdict": verdict, "reasons": [str(r) for r in reasons], "checks": checks}


class Reviewer:
    """独立评审器,持有自己的 LLM 客户端。"""

    def __init__(self, llm: LLMClient, max_retries: int = 1) -> None:
        self.llm = llm
        self.max_retries = max_retries

    def review(self, question: dict, context: SkillContext | None = None) -> dict:
        prompt = build_review_prompt(question, context)
        last_err: Exception | None = None
        for _ in range(self.max_retries + 1):
            resp = self.llm.complete(prompt)
            try:
                return parse_review(resp.text)
            except ReviewError as exc:
                last_err = exc
                continue
        raise ReviewError(f"评审 {question.get('stem')!r} 失败: {last_err}")
