"""四个预测型 baseline。

每个 baseline 都是 fn(train, test) -> 对 test 每行的「答对概率」数组。
四条都算得出 AUC，可直接互相比较；任何个性化方法都必须打赢它们。

注：spec 原文写的第 4 条是「按题库顺序」，但预测任务里顺序算不出 AUC，
故换成「按学生历史」——它才是真正要打的那个强 baseline（学生个体差异极大）。
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def predict_global(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """所有人所有题都预测为训练集全局正确率。"""
    p = float(train["correct"].mean())
    return np.full(len(test), p)


def predict_by_skill(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """用该题所属知识点在训练集上的历史正确率。"""
    prior = float(train["correct"].mean())
    rate = train.groupby("skill_id")["correct"].mean()
    return test["skill_id"].map(rate).fillna(prior).to_numpy(dtype=float)


def predict_by_student(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """用该学生在训练集上的历史正确率。"""
    prior = float(train["correct"].mean())
    rate = train.groupby("user_id")["correct"].mean()
    return test["user_id"].map(rate).fillna(prior).to_numpy(dtype=float)


def predict_random(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """随机概率。"""
    rng = np.random.default_rng(42)
    return rng.random(len(test))


BASELINES: dict[str, Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]] = {
    "全局正确率": predict_global,
    "按知识点": predict_by_skill,
    "按学生历史": predict_by_student,
    "随机": predict_random,
}
