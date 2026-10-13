"""P2:对比推荐策略的学习增益。

用法（项目根）：
    UV_SYSTEM_CERTS=1 uv run python scripts/eval_recommend.py

⚠️ 这是**离线模拟**评估:模拟学习者与推荐策略共用同一套 BKT 参数,
   存在「自己评自己」偏差。它衡量策略在学习者模型下的相对优劣,
   真实效果需要线上 A/B。指标:
     - 全体掌握度:学生相关知识点(含已掌握)的平均掌握概率(奖励广撒网,仅参考)
     - 薄弱掌握度:初始最弱 5 个知识点的最终平均掌握概率 ← 个性化该看这个
     - 累计准确率:学习效率
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bkt import BKT  # noqa: E402
from src.data import load_clean  # noqa: E402
from src.recommender import QuestionPool, default_strategies  # noqa: E402
from src.simulator import simulate_student  # noqa: E402
from src.split import split_by_student_time  # noqa: E402

N_STUDENTS = 200
N_ROUNDS = 20
SEED = 42


def _width(text: str) -> int:
    return sum(2 if ord(c) > 0x2E7F else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(1, width - _width(text))


def main() -> None:
    df = load_clean(ROOT / "data")
    train, test = split_by_student_time(df, test_ratio=0.2, min_events=10)

    t0 = time.time()
    model = BKT(n_iter=20).fit(train)
    pool = QuestionPool.from_frame(train)

    students = sorted(test["user_id"].unique())
    rng = np.random.default_rng(SEED)
    sampled = rng.choice(students, size=min(N_STUDENTS, len(students)), replace=False)
    subsets = {int(u): sub for u, sub in test[test["user_id"].isin(sampled)].groupby("user_id")}

    rows = []
    for st in default_strategies():
        finals, weaks, accs = [], [], []
        for _, sub in subsets.items():
            skills = sub["skill_id"].unique().tolist()
            r = simulate_student(st, skills, pool, model.params, n_rounds=N_ROUNDS, seed=SEED)
            if not np.isnan(r.final_mastery):
                finals.append(r.final_mastery)
                weaks.append(r.weak_mastery)
                accs.append(r.accuracy)
        rows.append((st.name, float(np.mean(finals)), float(np.mean(weaks)), float(np.mean(accs))))

    print(f"模拟学生数 : {len(subsets)}(抽样自 {len(students)} 名 test 学生)")
    print(f"每人轮数   : {N_ROUNDS}")
    print(f"耗时       : {time.time() - t0:.1f}s")
    print()

    name_w = max(_width(r[0]) for r in rows)
    header = _pad("策略", name_w) + "全体掌握度   薄弱掌握度   累计准确率"
    print(header)
    print("-" * (_width(header) + 6))
    for name, fin, weak, acc in rows:
        print(_pad(name, name_w) + f"{fin:.4f}      {weak:.4f}      {acc:.4f}")


if __name__ == "__main__":
    main()
