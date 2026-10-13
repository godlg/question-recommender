"""贝叶斯知识追踪(BKT)。

每个知识点独立拟合 4 个参数(L0, T, G, S),用 EM 在 2 状态 HMM 上完成:

- L0  初始掌握概率
- T   学习率(未掌握 → 掌握)
- G   猜对率(未掌握却答对)
- S   失误率(已掌握却答错)

预测:对学生在该知识点上的**完整历史**(train 段 + test 中更早的记录)做
在线贝叶斯递推,输出每一题的「答对概率」。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-6


@dataclass(frozen=True)
class BKTParams:
    l0: float
    t: float
    g: float
    s: float


def _clip(p: float) -> float:
    return float(min(1 - EPS, max(EPS, p)))


def _emission(params: BKTParams, obs: int) -> np.ndarray:
    """[P(obs | 未掌握), P(obs | 掌握)]"""
    if obs == 1:
        return np.array([params.g, 1 - params.s])
    return np.array([1 - params.g, params.s])


def _transition(params: BKTParams) -> np.ndarray:
    """A[i, j] = P(L_{t+1}=j | L_t=i)；状态 0=未掌握, 1=掌握。"""
    return np.array([[1 - params.t, params.t], [0.0, 1.0]])


def _forward_backward(obs: np.ndarray, params: BKTParams):
    """返回 gamma[t, 状态] 和 xi[t, i, j](均为后验概率)。"""
    n = len(obs)
    a = _transition(params)
    alpha = np.zeros((n, 2))
    beta = np.zeros((n, 2))

    alpha[0] = np.array([1 - params.l0, params.l0]) * _emission(params, int(obs[0]))
    s0 = alpha[0].sum()
    alpha[0] = alpha[0] / s0 if s0 > 0 else np.array([0.5, 0.5])
    for t in range(1, n):
        alpha[t] = (alpha[t - 1] @ a) * _emission(params, int(obs[t]))
        st = alpha[t].sum()
        alpha[t] = alpha[t] / st if st > 0 else np.array([0.5, 0.5])

    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = a @ (_emission(params, int(obs[t + 1])) * beta[t + 1])
        sb = beta[t].sum()
        beta[t] = beta[t] / sb if sb > 0 else np.array([0.5, 0.5])

    gamma = alpha * beta
    gs = gamma.sum(axis=1, keepdims=True)
    gamma = np.divide(gamma, gs, out=np.full_like(gamma, 0.5), where=gs > 0)

    xi = np.zeros((max(n - 1, 0), 2, 2))
    for t in range(n - 1):
        m = np.outer(alpha[t], _emission(params, int(obs[t + 1])) * beta[t + 1]) * a
        ms = m.sum()
        xi[t] = m / ms if ms > 0 else np.full((2, 2), 0.25)
    return gamma, xi


def fit_skill(
    obs_list: list[np.ndarray], n_iter: int = 20, init: BKTParams | None = None
) -> BKTParams:
    """对一个知识点的多条学生观测序列做 EM 拟合。"""
    params = init or BKTParams(0.3, 0.2, 0.2, 0.1)
    seqs = [np.asarray(o, dtype=int) for o in obs_list if len(o) > 0]
    if not seqs:
        return params

    for _ in range(n_iter):
        sum_l0 = 0.0
        n_seq = 0
        num_t = den_t = 0.0
        num_g = den_g = 0.0
        num_s = den_s = 0.0
        for obs in seqs:
            gamma, xi = _forward_backward(obs, params)
            sum_l0 += gamma[0, 1]
            n_seq += 1
            for t in range(len(obs)):
                den_g += gamma[t, 0]
                den_s += gamma[t, 1]
                if obs[t] == 1:
                    num_g += gamma[t, 0]
                else:
                    num_s += gamma[t, 1]
            for t in range(len(obs) - 1):
                num_t += xi[t, 0, 1]
                den_t += xi[t, 0, 0] + xi[t, 0, 1]
        params = BKTParams(
            l0=_clip(sum_l0 / n_seq),
            t=_clip(num_t / den_t) if den_t > 0 else params.t,
            g=_clip(num_g / den_g) if den_g > 0 else params.g,
            s=_clip(num_s / den_s) if den_s > 0 else params.s,
        )
    return params


def update_mastery(p_m: float, obs: int, prm: BKTParams) -> float:
    """BKT 在线一步:观测 → 后验 → 学习转移。返回更新后的掌握概率。"""
    if obs == 1:
        denom = p_m * (1 - prm.s) + (1 - p_m) * prm.g
        post = (p_m * (1 - prm.s)) / denom if denom > EPS else p_m
    else:
        denom = p_m * prm.s + (1 - p_m) * (1 - prm.g)
        post = (p_m * prm.s) / denom if denom > EPS else p_m
    return post + (1 - post) * prm.t


class BKT:
    """每个知识点一套参数的 BKT。"""

    def __init__(self, n_iter: int = 20, min_sequences: int = 5):
        self.n_iter = n_iter
        self.min_sequences = min_sequences
        self.params: dict[int, BKTParams] = {}

    def fit(self, train: pd.DataFrame) -> "BKT":
        for skill, part in train.groupby("skill_id"):
            seqs = [
                sub.sort_values("order_id")["correct"].to_numpy()
                for _, sub in part.groupby("user_id")
            ]
            if len(seqs) >= self.min_sequences:
                self.params[int(skill)] = fit_skill(seqs, n_iter=self.n_iter)
        return self

    def predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """对 test 每条记录给「答对概率」,用到该 (user, skill) 的全部已知历史。"""
        cols = ["user_id", "skill_id", "order_id", "correct"]
        tr = train[cols].copy()
        tr["_idx"] = -1
        te = test.reset_index(drop=True)
        te_idx = te.copy()
        te_idx["_idx"] = np.arange(len(te))
        merged = pd.concat([tr, te_idx], ignore_index=True).sort_values(
            ["user_id", "skill_id", "order_id"], kind="mergesort"
        )

        preds = np.full(len(te), np.nan)
        for (_, skill), grp in merged.groupby(["user_id", "skill_id"], sort=False):
            prm = self.params.get(int(skill))
            if prm is None:
                continue
            p_mastery = prm.l0
            for obs, idx in zip(grp["correct"].to_numpy(), grp["_idx"].to_numpy()):
                p_correct = p_mastery * (1 - prm.s) + (1 - p_mastery) * prm.g
                if idx >= 0:
                    preds[int(idx)] = p_correct
                if obs == 1:
                    denom = p_mastery * (1 - prm.s) + (1 - p_mastery) * prm.g
                    post = (p_mastery * (1 - prm.s)) / denom if denom > EPS else p_mastery
                else:
                    denom = p_mastery * prm.s + (1 - p_mastery) * (1 - prm.g)
                    post = (p_mastery * prm.s) / denom if denom > EPS else p_mastery
                p_mastery = post + (1 - post) * prm.t

        prior = float(train["correct"].mean())
        return np.where(np.isnan(preds), prior, preds)
