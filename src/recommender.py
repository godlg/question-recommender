"""推荐策略:候选生成 + 打分排序。

⚠️ 设计取舍:候选筛选是**结构化条件过滤**(知识点 + 难度),不是向量检索。
按知识点 ID / 难度是精确条件,用向量相似度是降级;这份数据集也没有题干文本可嵌入。
RAG 的正当位置在 P3(LLM 生成变式题时检索教材/样例题),不在候选筛选。

策略:
  - RandomStrategy      随机 (对照)
  - SequentialStrategy  按题库顺序 (对照)
  - ZPDStrategy         难度匹配:挑 p 最接近 target 的题 ← 朴素 ZPD,会被"已经中等掌握"的题吸走
  - GainStrategy        期望掌握度增益最大 ← 把「该学什么」形式化,推荐用这个
  - InfoGainStrategy    闭式期望熵减最大 ← 严格信息增益(不确定性下降)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from src.bkt import BKTParams, update_mastery

ZPD_TARGET = 0.7  # 最近发展区:目标正确率
COLD_MIN = 3      # 冷启动门控:某知识点作答次数低于该值时不信任 BKT 先验
PidList = list[int]
ProbFn = Callable[[np.ndarray], np.ndarray]  # 给定 problem_id 数组 → 该生答对概率
State = dict[int, float]                      # skill → 当前掌握概率
Params = dict[int, BKTParams]
Counts = dict[int, int]                       # skill → 该生在该知识点的作答次数


@dataclass
class QuestionPool:
    """题目池。difficulty = 全局正确率(越高越容易)。"""

    problem_id: np.ndarray
    skill: np.ndarray
    difficulty: np.ndarray
    skill_lookup: dict[int, int] = field(default_factory=dict)
    by_skill: dict[int, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "QuestionPool":
        g = df.groupby("problem_id").agg(
            skill=("skill_id", lambda s: int(s.iloc[0])),
            difficulty=("correct", "mean"),
        )
        pool = cls(
            problem_id=g.index.to_numpy(dtype=int),
            skill=g["skill"].to_numpy(dtype=int),
            difficulty=g["difficulty"].to_numpy(dtype=float),
        )
        pool.skill_lookup = dict(zip(pool.problem_id.tolist(), pool.skill.tolist()))
        for sid, sub in g.groupby("skill"):
            pool.by_skill[int(sid)] = sub.index.to_numpy(dtype=int)
        return pool

    def problems_of(self, skills) -> np.ndarray:
        parts = [self.by_skill[s] for s in skills if s in self.by_skill]
        return np.concatenate(parts) if parts else np.array([], dtype=int)


class Strategy(Protocol):
    name: str

    def recommend(
        self,
        pool: QuestionPool,
        prob_fn: ProbFn,
        candidates: np.ndarray,
        n: int,
        state: State | None = None,
        params: Params | None = None,
        counts: Counts | None = None,
    ) -> PidList: ...


def _dedup_by_skill(pool: QuestionPool, ordered_pids, n: int) -> PidList:
    """按给点顺序取题,一轮内同一知识点只取一道。"""
    chosen: PidList = []
    used: set[int] = set()
    for pid in ordered_pids:
        pid = int(pid)
        sid = pool.skill_lookup.get(pid)
        if sid in used:
            continue
        chosen.append(pid)
        used.add(sid)
        if len(chosen) >= n:
            break
    return chosen


@dataclass
class RandomStrategy:
    """随机推(对照)。随机源是实例级的,连续调用不会每轮同一题。"""

    name: str = "随机推"
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def recommend(self, pool, prob_fn, candidates, n, state=None, params=None, counts=None):
        if len(candidates) == 0:
            return []
        k = min(n, len(candidates))
        return [int(x) for x in self._rng.choice(candidates, size=k, replace=False)]


@dataclass
class SequentialStrategy:
    """顺序推(对照):按 problem_id 从小到大。"""

    name: str = "顺序推"

    def recommend(self, pool, prob_fn, candidates, n, state=None, params=None, counts=None):
        if len(candidates) == 0:
            return []
        return [int(x) for x in np.sort(candidates)[:n]]


@dataclass
class ZPDStrategy:
    """朴素 ZPD:挑预测正确率最接近 target 的题(一轮内不重复知识点)。

    ⚠️ 缺陷:会让「已经中等掌握」的题反复吸引选择,绕开真正薄弱的知识点。
    """

    name: str = "ZPD推"
    target: float = ZPD_TARGET

    def recommend(self, pool, prob_fn, candidates, n, state=None, params=None, counts=None):
        if len(candidates) == 0:
            return []
        candidates = np.asarray(candidates)
        gap = np.abs(prob_fn(candidates) - self.target)
        order = candidates[np.argsort(gap, kind="stable")]
        return _dedup_by_skill(pool, order, n)


@dataclass
class GainStrategy:
    """期望掌握度增益最大:选「做完这道题,学生掌握度期望提升最大」的题。

    这是把「该练什么」形式化的做法 —— 用 BKT 算每道题的期望增益
        E[Δm] = p(对)·Δm(对) + p(错)·Δm(错)
    增益天然偏向「尚未掌握、但难度不至于全错」的知识点,即最近发展区。

    冷启动门控(cold_start=True 且调用方给出 counts 时):
        某知识点作答次数 < cold_min 时,该生的掌握度 `m = l0` 只是 BKT 的**默认参数**,
        不是这个学生的证据。在错误先验上算出的期望增益会挑错题(比如把「没测过」
        误当「已会」或「不会」)。所以此时不信先验,优先去**测**这些作答次数不足的
        知识点,排序键为 `(作答次数升序, problem_id 升序)`。
        这只是「先验不可信 → 先补测」的**门控**,不是严格信息增益;严格熵减版本
        由 `InfoGainStrategy` 承担。
        兼容性:counts is None 时行为与改动前**逐位一致**(忽略门控)。
        **默认关闭**(P8 实测:200 人样本 -1.2pt / 50 人样本 +0.2pt,方向随样本翻转;
        见 README P8 与 scripts/eval_recommend.py 表 B/C),需要主动探索时显式开启。
    """

    name: str = "增益推"
    cold_start: bool = False
    cold_min: int = COLD_MIN

    def recommend(self, pool, prob_fn, candidates, n, state=None, params=None, counts=None):
        if len(candidates) == 0 or state is None or params is None:
            return []
        candidates = np.asarray(candidates)

        if self.cold_start and counts is not None:
            cold = [
                int(pid)
                for pid in candidates
                if counts.get(pool.skill_lookup[int(pid)], 0) < self.cold_min
            ]
            if cold:
                order = sorted(
                    cold, key=lambda pid: (counts.get(pool.skill_lookup[pid], 0), pid)
                )
                return _dedup_by_skill(pool, order, n)

        gains = np.full(len(candidates), -np.inf)
        for i, pid in enumerate(candidates):
            sid = pool.skill_lookup[int(pid)]
            prm = params.get(sid)
            if prm is None:
                continue
            m = state.get(sid, prm.l0)
            p_correct = m * (1 - prm.s) + (1 - m) * prm.g
            d_ok = update_mastery(m, 1, prm) - m
            d_bad = update_mastery(m, 0, prm) - m
            gains[i] = p_correct * d_ok + (1 - p_correct) * d_bad
        order = candidates[np.argsort(-gains, kind="stable")]
        return _dedup_by_skill(pool, order, n)


def _entropy(p: float) -> float:
    """伯努利熵 H(p),约定 0·log0 = 0(p<=0 或 p>=1 时为 0)。"""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


@dataclass
class InfoGainStrategy:
    """闭式期望熵减最大:选「做完这道题,掌握度不确定性下降最多」的题。

    与「增益推」的区别:增益推问「掌握度涨多少」,信息增益问「不确定性降多少」。
    对每个候选知识点:
        m       = state.get(sid, prm.l0)
        p_ok    = m·(1-s) + (1-m)·g
        m_ok    = update_mastery(m, 1, prm)
        m_bad   = update_mastery(m, 0, prm)
        score   = H(m) - (p_ok·H(m_ok) + (1-p_ok)·H(m_bad))
    m≈0.5(最不确定)的题分数最高;m≈0.95(几乎确定)的题分数接近 0。
    """

    name: str = "信息增益"

    def recommend(self, pool, prob_fn, candidates, n, state=None, params=None, counts=None):
        if len(candidates) == 0 or state is None or params is None:
            return []
        candidates = np.asarray(candidates)
        scores = np.full(len(candidates), -np.inf)
        for i, pid in enumerate(candidates):
            sid = pool.skill_lookup[int(pid)]
            prm = params.get(sid)
            if prm is None:
                continue
            m = state.get(sid, prm.l0)
            p_ok = m * (1 - prm.s) + (1 - m) * prm.g
            m_ok = update_mastery(m, 1, prm)
            m_bad = update_mastery(m, 0, prm)
            scores[i] = _entropy(m) - (p_ok * _entropy(m_ok) + (1 - p_ok) * _entropy(m_bad))
        order = candidates[np.argsort(-scores, kind="stable")]
        return _dedup_by_skill(pool, order, n)


def default_strategies() -> list[Strategy]:
    return [
        RandomStrategy(),
        SequentialStrategy(),
        ZPDStrategy(),
        GainStrategy(),   # 冷启动门控默认开启
        InfoGainStrategy(),
    ]
