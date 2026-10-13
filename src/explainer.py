"""P6 · 错题讲解生成:检索材料 → 拼 prompt → LLM 生成**有依据**的中文讲解。

链路:``question["stem"]`` + ``question["skill"]`` 构造 query → :class:`Retriever`
检索材料 chunk → 把 chunk 拼进 prompt → LLM 生成中文讲解。

降级原则(**绝不抛异常**):

  - 检索为空 → 退化为纯生成,``sources=[]``、``grounded=False``;
  - LLM 报错 / 未配置 → ``explanation=None``、``error=...``。
"""
from __future__ import annotations

from src.llm_client import LLMClient
from src.retriever import KeywordRetriever, Retriever


class Explainer:
    """讲解器:独立持有 LLM 客户端与检索器,均可注入便于测试。"""

    def __init__(
        self,
        llm: LLMClient | None,
        retriever: Retriever | None = None,
        chunks: list[dict] | None = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever or KeywordRetriever()
        self.chunks = chunks if chunks is not None else []

    def set_chunks(self, chunks: list[dict]) -> None:
        self.chunks = chunks

    def build_query(self, question: dict) -> str:
        stem = str(question.get("stem") or "")
        skill = str(question.get("skill") or "")
        return f"{stem} {skill}".strip()

    def build_prompt(self, question: dict, student_answer: str | None, hits: list[dict]) -> str:
        """把检索到的材料片段拼进 prompt。片段文本必须原样出现。"""
        skill = str(question.get("skill") or "")
        stem = str(question.get("stem") or "")
        lines = [
            "你是一名耐心的中文数学老师。请针对学生答错的这道题,"
            "**依据下面提供的讲解材料**给出针对性讲解。",
            "",
            f"【知识点】{skill}",
            f"【题目】{stem}",
        ]
        if question.get("options"):
            lines.append(f"【选项】{'、'.join(str(o) for o in question['options'])}")
        if question.get("answer") is not None:
            lines.append(f"【正确答案】{question['answer']}")
        if student_answer:
            lines.append(f"【学生答案】{student_answer}")

        if hits:
            lines.append("")
            lines.append("【讲解材料(必须依据以下内容作答,不得编造)】")
            for i, h in enumerate(hits, 1):
                lines.append(f"--- 片段 {i}(知识点:{h.get('skill', '')})---")
                lines.append(str(h.get("text", "")))
            lines.append("")
            lines.append(
                "【要求】用简体中文输出讲解:先指出学生可能的错误原因,"
                "再结合上面的材料讲清该知识点与解题步骤。不要输出与材料矛盾的内容。"
            )
        else:
            lines.append("")
            lines.append(
                "【要求】没有检索到讲解材料,请仅凭题目与知识点,用简体中文给出讲解,"
                "先分析错因再讲清解题步骤。"
            )
        return "\n".join(lines)

    def explain(
        self,
        question: dict,
        student_answer: str | None = None,
        top_k: int = 3,
    ) -> dict:
        skill = question.get("skill")
        query = self.build_query(question)
        try:
            hits = self.retriever.search(query, self.chunks, top_k=top_k, skill=skill)
        except Exception as exc:  # noqa: BLE001 - 检索异常同样降级
            hits = []
            retrieve_error = f"检索失败: {exc}"
        else:
            retrieve_error = None

        sources = [
            {
                "file": h.get("file", ""),
                "skill": h.get("skill", ""),
                "snippet": str(h.get("text", ""))[:120],
                "score": float(h.get("score", 0.0)),
            }
            for h in hits
        ]
        grounded = bool(hits)

        if self.llm is None:
            return {
                "explanation": None,
                "sources": sources,
                "grounded": grounded,
                "error": retrieve_error or "LLM 未配置",
            }

        prompt = self.build_prompt(question, student_answer, hits)
        try:
            resp = self.llm.complete(prompt)
        except Exception as exc:  # noqa: BLE001 - LLM 任何失败都降级,不抛
            return {
                "explanation": None,
                "sources": sources,
                "grounded": grounded,
                "error": retrieve_error or f"LLM 调用失败: {exc}",
            }

        text = (resp.text or "").strip()
        if not text:
            return {
                "explanation": None,
                "sources": sources,
                "grounded": grounded,
                "error": retrieve_error or "LLM 返回空内容",
            }
        return {
            "explanation": text,
            "sources": sources,
            "grounded": grounded,
            "error": retrieve_error,
        }
