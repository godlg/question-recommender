"""P1:baseline 对比表 + BKT。

用法（项目根）：
    UV_SYSTEM_CERTS=1 uv run python scripts/eval_bkt.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import BASELINES  # noqa: E402
from src.bkt import BKT  # noqa: E402
from src.data import load_clean  # noqa: E402
from src.metrics import acc, auc  # noqa: E402
from src.split import split_by_student_time  # noqa: E402


def _width(text: str) -> int:
    return sum(2 if ord(c) > 0x2E7F else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(1, width - _width(text))


def main() -> None:
    df = load_clean(ROOT / "data")
    train, test = split_by_student_time(df, test_ratio=0.2, min_events=10)
    y = test["correct"]

    rows = [(name, fn(train, test)) for name, fn in BASELINES.items()]

    t0 = time.time()
    model = BKT(n_iter=20).fit(train)
    bkt_scores = model.predict(train, test)
    elapsed = time.time() - t0
    rows.append(("BKT", bkt_scores))

    n_skill = df["skill_id"].nunique()
    print(f"train/test      : {len(train):,} / {len(test):,} 行")
    print(f"知识点/已拟合   : {n_skill} / {len(model.params)}")
    print(f"BKT 拟合+预测   : {elapsed:.1f}s")
    print()

    name_w = max(_width(n) for n, _ in rows)
    header = _pad("model", name_w) + "AUC      ACC"
    print(header)
    print("-" * (_width(header) + 4))
    for name, score in rows:
        print(_pad(name, name_w) + f"{auc(y, score):.4f}   {acc(y, score):.4f}")


if __name__ == "__main__":
    main()
