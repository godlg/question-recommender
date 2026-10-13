"""P3 demo:一条命令跑完整链路 —— 生成 → 独立评审 → 统计。

用法(项目根):
    UV_SYSTEM_CERTS=1 uv run python scripts/p3_demo.py
    UV_SYSTEM_CERTS=1 uv run python scripts/p3_demo.py --skills 1 --per-skill 6 --inject-every 3

**默认走真实 LLM**:只要 ``LLM_BASE_URL`` / ``LLM_MODEL`` / ``LLM_API_KEY``
齐全,就直接调用真实 OpenAI 兼容接口(生成器与评审器各自的 prompt 独立)。
只有这三个环境变量缺失时,才回退到**离线确定性引擎**(FakeLLM,无需 API key),
并打印明确警告 —— 此时结果**不代表真实模型能力**。

注入错误与生成**解耦**:先生成,再由 demo 周期性把某题的 ``answer`` 改成
``options`` 里的另一个值(真正的错误答案),并记录注入清单。评审器**独立重算**,
报告把「注入错误检出率」与「LLM 自身质量问题」分开统计。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm_client import ENV_KEYS, FakeLLM, HttpLLM  # noqa: E402
from src.question_gen import (  # noqa: E402
    QuestionGenerator,
    SkillContext,
    build_skill_contexts,
)
from src.reviewer import Reviewer  # noqa: E402

DEFAULT_SKILLS = 1
DEFAULT_PER_SKILL = 9   # 1 × 9 -> 每 3 道注入 1 道,恰好 3 条拦截样例
DEFAULT_INJECT_EVERY = 3
REJECTED_PATH = ROOT / "p3_rejected_examples.json"

_SKILL_RE = re.compile(r"知识点:\s*(.+)")
_COUNT_RE = re.compile(r"命制\s*(\d+)\s*道")
_QUESTION_RE = re.compile(r"【待审题目】\s*(\{.*?\})\s*\n\n【审查项】", re.S)
_SUM_RE = re.compile(r"(\d+)\s*\+\s*(\d+)")


def _mock_questions(skill_name: str, n: int, counter: list[int]) -> list[dict]:
    """确定性假生成器:简单加法题(全部正确,错误由 demo 统一注入)。"""
    out = []
    for _ in range(n):
        i = counter[0]
        counter[0] += 1
        a, b = 10 + (i * 7) % 40, 5 + (i * 3) % 30
        truth = a + b
        options = [str(truth), str(truth + 1), str(truth - 1), str(truth + 2)]
        out.append(
            {
                "stem": f"{skill_name}: {a} + {b} = ?",
                "options": options,
                "answer": str(truth),
                "skill_name": skill_name,
                "difficulty_hint": 0.5,
            }
        )
    return out


def _make_mock_generator(counter: list[int]):
    def responder(prompt: str) -> str:
        skill = _SKILL_RE.search(prompt).group(1).strip()
        n = int(_COUNT_RE.search(prompt).group(1))
        return json.dumps(_mock_questions(skill, n, counter), ensure_ascii=False)

    return responder


def _make_mock_reviewer():
    """独立评审:只看题目、自己重算答案,不看生成时的上下文。"""

    def responder(prompt: str) -> str:
        question = json.loads(_QUESTION_RE.search(prompt).group(1))
        m = _SUM_RE.search(question["stem"])
        truth = int(m.group(1)) + int(m.group(2))
        checks = {
            "answer_correct": question["answer"] == str(truth),
            "unambiguous": len(set(question["options"])) == len(question["options"])
            and question["options"].count(str(truth)) == 1,
            "difficulty_plausible": 0.0 <= question["difficulty_hint"] <= 1.0,
            "skill_aligned": question["skill_name"] in question["stem"],
        }
        reasons = []
        if not checks["answer_correct"]:
            reasons.append(f"答案错误: 标注 {question['answer']}, 独立重算应为 {truth}")
        if not checks["unambiguous"]:
            reasons.append("选项存在重复或唯一解不成立")
        if not checks["difficulty_plausible"]:
            reasons.append("difficulty_hint 越界")
        if not checks["skill_aligned"]:
            reasons.append("题干未体现知识点")
        return json.dumps(
            {"verdict": "pass" if all(checks.values()) else "fail",
             "reasons": reasons, "checks": checks},
            ensure_ascii=False,
        )

    return responder


def _pick_skills(contexts: dict[str, SkillContext], k: int) -> list[str]:
    ranked = sorted(contexts.values(), key=lambda c: c.n_attempts, reverse=True)
    return [c.skill_name for c in ranked[:k]]


def _inject_errors(questions: list[dict], every: int) -> dict[int, dict]:
    """每 ``every`` 道把 ``answer`` 改成 ``options`` 中的另一个值(真实错误答案)。

    原地修改题目,返回 ``{题目下标: 注入清单}``。
    """
    injected: dict[int, dict] = {}
    if every <= 0:
        return injected
    for idx, q in enumerate(questions):
        if (idx + 1) % every != 0:
            continue
        options = q.get("options") or []
        original = q.get("answer")
        wrong = next((o for o in options if str(o) != str(original)), None)
        if wrong is None:
            continue
        q["answer"] = wrong
        injected[idx] = {
            "index": idx,
            "skill_name": q.get("skill_name"),
            "stem": q.get("stem"),
            "original_answer": original,
            "injected_answer": wrong,
        }
    return injected


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 生成 + 评审完整链路")
    parser.add_argument("--skills", type=int, default=DEFAULT_SKILLS, help="知识点个数 K(默认 1)")
    parser.add_argument("--per-skill", type=int, default=DEFAULT_PER_SKILL, help="每个知识点题数 N(默认 9)")
    parser.add_argument("--inject-every", type=int, default=DEFAULT_INJECT_EVERY, help="每 E 道注入一道错答案(默认 3,0 关闭)")
    args = parser.parse_args()

    contexts = build_skill_contexts(ROOT / "data")
    skills = _pick_skills(contexts, args.skills)

    missing = [k for k in ENV_KEYS if not os.environ.get(k)]
    if missing:
        print(
            f"[警告] 缺少环境变量 {', '.join(missing)} —— 当前**未调用真实 LLM**,\n"
            "        已回退到离线确定性引擎(FakeLLM),以下结果不代表真实模型能力。\n"
            "        配置 LLM_BASE_URL / LLM_MODEL / LLM_API_KEY 后重新运行即可走真实链路。",
            file=sys.stderr,
        )
        counter = [0]
        gen_llm = FakeLLM(_make_mock_generator(counter))
        rev_llm = FakeLLM(_make_mock_reviewer())
        mode = "离线确定性引擎(未调用真实 LLM)"
    else:
        gen_llm = HttpLLM()
        rev_llm = HttpLLM(temperature=0.0)
        mode = "真实 LLM(HTTP)"

    generator = QuestionGenerator(gen_llm, contexts)
    reviewer = Reviewer(rev_llm)

    generated: list[dict] = []
    for skill in skills:
        generated.extend(generator.generate(skill, args.per_skill))

    injected = _inject_errors(generated, args.inject_every)

    passed, rejected = [], []
    for idx, q in enumerate(generated):
        ctx = generator.context_for(q["skill_name"])
        result = reviewer.review(q, ctx)
        if result["verdict"] == "pass":
            passed.append(q)
        else:
            rejected.append(
                {
                    "index": idx,
                    "question": q,
                    "review": result,
                    "injected": idx in injected,
                    "injection": injected.get(idx),
                }
            )

    total = len(generated)
    reject_rate = len(rejected) / total if total else 0.0
    injected_total = len(injected)
    injected_rejected = sum(1 for r in rejected if r["injected"])
    detection = injected_rejected / injected_total if injected_total else 0.0
    llm_issues = len(rejected) - injected_rejected

    print(f"运行模式   : {mode}")
    print(f"知识点     : {len(skills)} 个 × {args.per_skill} 道")
    print(f"注入策略   : 每 {args.inject_every} 道注入 1 道错答案(共注入 {injected_total} 道)")
    print(f"生成 / 通过/ 拦下 : {total} / {len(passed)} / {len(rejected)}")
    print(f"拦截率     : {reject_rate:.1%}")
    print()
    print(f"{'调用':<8}{'次数':>6}{'耗时(s)':>10}{'token(估)':>12}")
    print("-" * 38)
    for label, client in (("生成器", gen_llm), ("评审器", rev_llm)):
        print(f"{label:<8}{len(client.calls):>6}{client.total_seconds:>10.2f}{client.total_tokens:>12}")
    print("-" * 38)
    grand = gen_llm.total_seconds + rev_llm.total_seconds
    grand_tok = gen_llm.total_tokens + rev_llm.total_tokens
    print(f"{'合计':<8}{len(gen_llm.calls) + len(rev_llm.calls):>6}{grand:>10.2f}{grand_tok:>12}")

    print()
    print("评审区分:")
    print(f"  注入错误检出率   : {injected_rejected}/{injected_total} = {detection:.1%}(评审器独立抓回的被注入错误)")
    print(f"  LLM 自身质量问题 : {llm_issues} 道(未被注入却被判 fail)")

    print("\n拦截样例(评审器独立重算抓出的错误):")
    for item in rejected[:3]:
        q = item["question"]
        tag = "注入" if item["injected"] else "LLM自身"
        print(f"  - [{tag}] {q['stem']} 标注={q['answer']} | {item['review']['reasons']}")

    REJECTED_PATH.write_text(
        json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n被拦下的 {len(rejected)} 条已写入 {REJECTED_PATH.name}")


if __name__ == "__main__":
    main()
