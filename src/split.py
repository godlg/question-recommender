"""数据划分。

两种口径：

- ``split_by_student_time`` ← **P0 主口径**。每个学生按时间切：前 80% 进 train、
  后 20% 进 test。这本就是 KT / 推荐的任务定义（用学生历史预测其后续），
  也是「按学生历史」baseline 有意义的前提。

- ``split_by_student`` ← **冷启动对照**。整学生切走，test 学生在 train 中完全
  不出现 → 所有「用学生信息」的 baseline 退化为全局先验（实测 AUC=0.5000）。
  这是另一个评测场景（新学生冷启动），P1 之后再启用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_student_time(
    df: pd.DataFrame, test_ratio: float = 0.2, min_events: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """每个学生按 order_id 排序，后 test_ratio 的记录进 test。

    min_events 以下的短序列学生全部留 train（它们没有足够历史做预测）。
    """
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio 必须在 (0, 1) 之间")
    d = df.sort_values(["user_id", "order_id"]).reset_index(drop=True)
    n = d.groupby("user_id")["user_id"].transform("size").to_numpy()
    rank = d.groupby("user_id").cumcount().to_numpy()
    n_test = np.floor(n * test_ratio).astype(int)
    n_test = np.where(n >= min_events, np.maximum(n_test, 1), 0)
    is_test = rank >= (n - n_test)
    train = d[~is_test].reset_index(drop=True)
    test = d[is_test].reset_index(drop=True)
    return train, test


def split_by_student(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """整学生切分：test 学生与 train 学生无交集（冷启动场景）。"""
    if not 0 < test_size < 1:
        raise ValueError("test_size 必须在 (0, 1) 之间")
    students = np.array(sorted(df["user_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(students)
    n_test = max(1, int(round(len(students) * test_size)))
    test_students = set(students[:n_test].tolist())
    is_test = df["user_id"].isin(test_students)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
