"""P0：一条命令打出 baseline 数字表。

用法（项目根）：
    UV_SYSTEM_CERTS=1 uv run python scripts/p0_baseline_table.py

口径：每个学生按时间切，前 80% 训练 / 后 20% 测试（见 src/split.py）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import BASELINES  # noqa: E402
from src.data import load_clean  # noqa: E402
from src.metrics import acc, auc  # noqa: E402
from src.split import split_by_student_time  # noqa: E402


def _width(text: str) -> int:
    """中文按 2 宽计算，让表格对齐。"""
    return sum(2 if ord(c) > 0x2E7F else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(1, width - _width(text))


def main() -> None:
    df = load_clean(ROOT / "data")
    train, test = split_by_student_time(df, test_ratio=0.2, min_events=10)

    print(f"清洗后总量 : {len(df):,} 行")
    print(f"学生       : {df['user_id'].nunique():,}")
    print(f"知识点     : {df['skill_id'].nunique():,}")
    print(f"题目       : {df['problem_id'].nunique():,}")
    print(f"train/test : {len(train):,} / {len(test):,} 行（每生前 80% / 后 20%）")
    print(f"test 覆盖  : {test['user_id'].nunique():,} 名学生")
    print(f"test 正确率: {test['correct'].mean():.4f}")
    print()

    name_w = max(_width(n) for n in BASELINES)
    header = _pad("baseline", name_w) + "AUC      ACC"
    print(header)
    print("-" * (_width(header) + 4))

    y = test["correct"]
    for name, fn in BASELINES.items():
        score = fn(train, test)
        print(_pad(name, name_w) + f"{auc(y, score):.4f}   {acc(y, score):.4f}")


if __name__ == "__main__":
    main()
