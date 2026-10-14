"""P2:对比推荐策略的学习增益。

用法（项目根）：
    UV_SYSTEM_CERTS=1 uv run python scripts/eval_recommend.py

⚠️ 这是**离线模拟**评估:模拟学习者与推荐策略共用同一套 BKT 参数,
   存在「自己评自己」偏差。它衡量策略在学习者模型下的相对优劣,
   不是真实教学效果;真实效果需要线上 A/B。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bkt import BKT  # noqa: E402
from src.data import load_clean  # noqa: E402
from src.recommender import COLD_MIN, GainStrategy, InfoGainStrategy, QuestionPool, default_strategies  # noqa: E402
from src.simulator import simulate_student  # noqa: E402
from src.split import split_by_student_time  # noqa: E402

N_STUDENTS = 200
N_ROUNDS = 20
SEED = 42


def _width(text: str) -> int:
    return sum(2 if ord(c) > 0x2E7F else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(1, width - _width(text))


def _fmt_rows(rows: list[tuple[str, float, float, float]], name_w: int) -> None:
    header = _pad("策略", name_w) + "全体掌握度   薄弱掌握度   累计准确率"
    print(header)
    print("-" * (_width(header) + 6))
    for name, fin, weak, acc in rows:
        print(_pad(name, name_w) + f"{fin:.4f}      {weak:.4f}      {acc:.4f}")


def _eval_strategy(strategy, subsets: dict[int, pd.DataFrame], pool: QuestionPool, params: dict) -> tuple[float, float, float]:  # noqa: E501
    finals, weaks, accs = [], [], []
    for _, sub in subsets.items():
        skills = sub["skill_id"].unique().tolist()
        r = simulate_student(strategy, skills, pool, params, n_rounds=N_ROUNDS, seed=SEED)
        if not np.isnan(r.final_mastery):
            finals.append(r.final_mastery)
            weaks.append(r.weak_mastery)
            accs.append(r.accuracy)
    return float(np.mean(finals)), float(np.mean(weaks)), float(np.mean(accs))


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

    # 表 A:5 策略(增益推 = 冷启动开)
    rows_a = []
    for st in default_strategies():
        fin, weak, acc = _eval_strategy(st, subsets, pool, model.params)
        rows_a.append((st.name, fin, weak, acc))

    # 表 B:冷启动消融
    gain_cold_on = GainStrategy(cold_start=True)
    gain_cold_off = GainStrategy(cold_start=False)
    rows_b = [
        ("增益推(冷启动开)", *_eval_strategy(gain_cold_on, subsets, pool, model.params)),
        ("增益推(冷启动关)", *_eval_strategy(gain_cold_off, subsets, pool, model.params)),
    ]

    # 表 C:新知识点多的学生子集(init_counts 来自 test 段真实作答)
    # 先算每个学生的 init_counts,再按 cold_ratio 筛选
    init_counts_map: dict[int, dict[int, int]] = {}
    for uid, sub in subsets.items():
        counts: dict[int, int] = {}
        for sid in sub["skill_id"]:
            counts[sid] = counts.get(sid, 0) + 1
        init_counts_map[uid] = counts

    cold_subsets = {}
    for uid, sub in subsets.items():
        counts = init_counts_map[uid]
        skills = sub["skill_id"].unique().tolist()
        if not skills:
            continue
        cold_ratio = sum(1 for s in skills if counts.get(s, 0) < COLD_MIN) / len(skills)
        if cold_ratio >= 0.5:
            cold_subsets[uid] = sub

    rows_c = []
    for st, display_name in (
        (gain_cold_off, "增益推(冷启动关)"),
        (gain_cold_on, "增益推(冷启动开)"),
        (InfoGainStrategy(), "信息增益"),
    ):
        finals, weaks, accs = [], [], []
        for uid, sub in cold_subsets.items():
            skills = sub["skill_id"].unique().tolist()
            r = simulate_student(
                st,
                skills,
                pool,
                model.params,
                n_rounds=N_ROUNDS,
                seed=SEED,
                init_counts=init_counts_map[uid],
            )
            if not np.isnan(r.final_mastery):
                finals.append(r.final_mastery)
                weaks.append(r.weak_mastery)
                accs.append(r.accuracy)
        rows_c.append((display_name, float(np.mean(finals)), float(np.mean(weaks)), float(np.mean(accs))))

    print(f"模拟学生数 : {len(subsets)}(抽样自 {len(students)} 名 test 学生)")
    print(f"每人轮数   : {N_ROUNDS}")
    print(f"耗时       : {time.time() - t0:.1f}s")
    print()

    name_w = max(_width(r[0]) for r in (rows_a + rows_b + rows_c))

    print("表 A · 5 策略对比")
    _fmt_rows(rows_a, name_w)
    print()

    print("表 B · 冷启动消融(增益推)")
    _fmt_rows(rows_b, name_w)
    print()

    print("表 C · 新知识点多的学生子集(cold_ratio >= 0.5)")
    print(f"子集学生数 : {len(cold_subsets)}")
    _fmt_rows(rows_c, name_w)


if __name__ == "__main__":
    main()
