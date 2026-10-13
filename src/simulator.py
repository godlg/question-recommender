"""离线模拟:用 BKT 当「模拟学习者」,对比不同推荐策略的学习增益。

⚠️ 局限(必须诚实声明,面试会被问):
    模拟器与推荐策略共用同一套 BKT → 存在「自己评自己」的偏差。
    它衡量的是**在给定学习者模型下各策略的相对优劣**(近似最优控制),
    不是真实教学效果;真实效果需要线上 A/B 实验。

指标:
    - final_mastery:模拟结束后对**全部**相关知识点的平均掌握概率
      ⚠️ 平均值奖励「广撒网」,单看它会低估聚焦型策略
    - weak_mastery :学生**初始最弱 5 个**知识点在模拟结束时的平均掌握概率
    - accuracy     :累计答对率(学习效率)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bkt import BKTParams, update_mastery
from src.recommender import Params, ProbFn, QuestionPool, State, Strategy

N_WEAK = 5


@dataclass
class SimResult:
    final_mastery: float
    weak_mastery: float
    accuracy: float
    rounds: int


def _prob_fn(state: State, params: Params, pool: QuestionPool) -> ProbFn:
    """给定学生当前掌握度,返回「这些题答对的概率」。"""

    def fn(pids: np.ndarray) -> np.ndarray:
        skills = np.array([pool.skill_lookup[int(p)] for p in pids])
        p_mastery = np.array([state.get(int(s), 0.0) for s in skills])
        g = np.array([params[int(s)].g for s in skills])
        s_slip = np.array([params[int(s)].s for s in skills])
        return p_mastery * (1 - s_slip) + (1 - p_mastery) * g

    return fn


def simulate_student(
    strategy: Strategy,
    student_skills: list[int],
    pool: QuestionPool,
    params: Params,
    n_rounds: int = 20,
    seed: int = 0,
) -> SimResult:
    """一个学生:每轮推荐 1 道**没做过的**题 → 按 BKT 采样作答 → 在线更新掌握度。"""
    skills = [s for s in student_skills if s in params]
    if not skills:
        return SimResult(float("nan"), float("nan"), float("nan"), 0)

    rng = np.random.default_rng(seed)
    init = {s: params[s].l0 for s in skills}
    weak = sorted(skills, key=lambda s: init[s])[:N_WEAK]
    state: State = dict(init)
    candidates = pool.problems_of(skills)
    n_correct = 0
    done = 0
    used: set[int] = set()

    for _ in range(n_rounds):
        avail = candidates if not used else candidates[~np.isin(candidates, list(used))]
        if len(avail) == 0:
            break
        pids = strategy.recommend(
            pool, _prob_fn(state, params, pool), avail, n=1, state=state, params=params
        )
        if not pids:
            break
        pid = pids[0]
        used.add(pid)
        sid = pool.skill_lookup[pid]
        if sid not in state:
            continue
        prm = params[sid]
        p_m = state[sid]
        p_correct = p_m * (1 - prm.s) + (1 - p_m) * prm.g
        obs = int(rng.random() < p_correct)
        n_correct += obs
        done += 1
        state[sid] = update_mastery(p_m, obs, prm)

    final = float(np.mean(list(state.values())))
    weak_final = float(np.mean([state[s] for s in weak]))
    accuracy = n_correct / done if done else float("nan")
    return SimResult(final, weak_final, accuracy, done)
