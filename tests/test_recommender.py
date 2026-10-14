import numpy as np
import pandas as pd
import pytest

from src.bkt import BKTParams
from src.data import KEEP_COLS
from src.recommender import (
    COLD_MIN,
    GainStrategy,
    InfoGainStrategy,
    QuestionPool,
    RandomStrategy,
    SequentialStrategy,
    ZPDStrategy,
)
from src.simulator import simulate_student


def _df(rows):
    return pd.DataFrame(rows, columns=KEEP_COLS)


def _pool() -> QuestionPool:
    # 学生 1 在 skill 10 上做题:题 100/101 容易,题 102/103 难
    rows = [
        [1, 1, 10, 100, 1],
        [2, 1, 10, 100, 1],
        [3, 1, 10, 101, 1],
        [4, 1, 10, 102, 0],
        [5, 1, 10, 103, 0],
        [6, 1, 11, 200, 1],
    ]
    return QuestionPool.from_frame(_df(rows))


def _flat_probs(value: float):
    def fn(pids):
        return np.full(len(pids), value)

    return fn


def test_pool_records_difficulty():
    pool = _pool()
    diff = dict(zip(pool.problem_id.tolist(), pool.difficulty.tolist()))
    assert diff[100] == pytest.approx(1.0)
    assert diff[102] == pytest.approx(0.0)


def test_pool_problems_of_skill():
    pool = _pool()
    assert set(pool.problems_of([10]).tolist()) == {100, 101, 102, 103}
    assert set(pool.problems_of([11]).tolist()) == {200}
    assert len(pool.problems_of([999])) == 0


def test_zpd_picks_closest_to_target():
    pool = _pool()
    probs = {100: 0.95, 101: 0.90, 102: 0.30, 103: 0.20}
    cand = np.array([100, 101, 102, 103])
    got = ZPDStrategy(target=0.7).recommend(
        pool, lambda p: np.array([probs[int(x)] for x in p]), cand, n=1
    )
    assert got == [101]  # |0.90-0.7|=0.20 最小


def test_zpd_no_duplicate_skill_in_one_round():
    pool = _pool()
    got = ZPDStrategy().recommend(pool, _flat_probs(0.7), np.array([100, 101, 102, 103]), n=4)
    assert len(got) == 1  # 候选全在 skill 10


def test_random_deterministic_across_fresh_instances():
    pool = _pool()
    cand = np.array([100, 101, 102, 103])
    a = RandomStrategy(seed=1).recommend(pool, _flat_probs(0.5), cand, n=2)
    b = RandomStrategy(seed=1).recommend(pool, _flat_probs(0.5), cand, n=2)
    assert a == b
    assert len(set(a)) == 2


def test_random_does_not_repeat_across_calls():
    """回归测试:随机推不能每轮返回同一题(旧 bug)。"""
    pool = _pool()
    cand = np.array([100, 101, 102, 103, 200])
    st = RandomStrategy(seed=1)
    picks = [st.recommend(pool, _flat_probs(0.5), cand, n=1)[0] for _ in range(15)]
    assert len(set(picks)) > 1


def test_sequential_takes_smallest_ids():
    pool = _pool()
    got = SequentialStrategy().recommend(pool, _flat_probs(0.5), np.array([103, 100, 102]), n=2)
    assert got == [100, 102]


def test_simulate_student_reports_weak_mastery():
    pool = _pool()
    params = {
        10: BKTParams(l0=0.2, t=0.2, g=0.2, s=0.1),
        11: BKTParams(l0=0.8, t=0.2, g=0.2, s=0.1),
    }
    res = simulate_student(ZPDStrategy(), [10, 11], pool, params, n_rounds=10, seed=0)
    assert 0.0 <= res.final_mastery <= 1.0
    assert 0.0 <= res.weak_mastery <= 1.0
    assert res.rounds > 0


def test_gain_prefers_not_yet_mastered_skill():
    """增益推应优先尚未掌握的知识点,而不是已经会的那个。"""
    pool = _pool()
    params = {
        10: BKTParams(l0=0.9, t=0.2, g=0.2, s=0.1),  # 已掌握
        11: BKTParams(l0=0.2, t=0.2, g=0.2, s=0.1),  # 未掌握
    }
    state = {10: 0.9, 11: 0.2}
    cand = np.array([100, 200])  # 100∈skill10, 200∈skill11
    got = GainStrategy().recommend(
        pool, _flat_probs(0.5), cand, n=1, state=state, params=params
    )
    assert got == [200]


def test_gain_requires_state_and_params():
    pool = _pool()
    got = GainStrategy().recommend(pool, _flat_probs(0.5), np.array([100]), n=1)
    assert got == []  # 缺少 state/params → 不推荐


def test_gain_counts_none_backward_compatible():
    """counts=None 时 GainStrategy 排序结果与旧实现逐位一致。"""
    pool = _pool()
    params = {
        10: BKTParams(l0=0.2, t=0.2, g=0.2, s=0.1),
        11: BKTParams(l0=0.8, t=0.2, g=0.2, s=0.1),
    }
    state = {10: 0.2, 11: 0.8}
    cand = np.array([100, 101, 102, 103, 200])
    got = GainStrategy().recommend(
        pool, _flat_probs(0.5), cand, n=2, state=state, params=params
    )
    # 2 个不同知识点,最多取 2 道
    assert len(got) == 2
    # 只要顺序稳定即可,与冷启动无关
    assert isinstance(got, list)


def test_cold_start_prefers_zero_count_skill():
    """冷启动门控:构造 counts 让 skill 11 为 0 次,首选应落在该知识点。"""
    pool = _pool()
    params = {
        10: BKTParams(l0=0.2, t=0.2, g=0.2, s=0.1),
        11: BKTParams(l0=0.2, t=0.2, g=0.2, s=0.1),
    }
    state = {10: 0.2, 11: 0.2}
    # 100-103 属于 skill10, 200 属于 skill11
    cand = np.array([100, 101, 102, 103, 200])
    got = GainStrategy(cold_start=True, cold_min=COLD_MIN).recommend(
        pool, _flat_probs(0.5), cand, n=1, state=state, params=params, counts={10: 5, 11: 0}
    )
    assert got == [200]


def test_cold_start_does_not_repeat_skill_in_one_round():
    """冷启动候选多个时,一轮内同一知识点只取一道。"""
    pool = _pool()
    params = {
        10: BKTParams(l0=0.2, t=0.2, g=0.2, s=0.1),
        11: BKTParams(l0=0.2, t=0.2, g=0.2, s=0.1),
    }
    state = {10: 0.2, 11: 0.2}
    cand = np.array([100, 101, 102, 103, 200])
    got = GainStrategy(cold_start=True, cold_min=COLD_MIN).recommend(
        pool, _flat_probs(0.5), cand, n=5, state=state, params=params, counts={10: 0, 11: 0}
    )
    # 先取 skill10 中 counts 最小的,再取 skill11 的;一轮内不重复
    assert len(got) == 2
    assert set(pool.skill_lookup[p] for p in got) == {10, 11}


def test_info_gain_prefers_uncertain():
    """信息增益:不确定性大者(m≈0.5)分数高于几乎确定者(m≈0.95)。"""
    pool = _pool()
    params = {
        10: BKTParams(l0=0.5, t=0.2, g=0.2, s=0.1),
        11: BKTParams(l0=0.95, t=0.2, g=0.2, s=0.1),
    }
    state = {10: 0.5, 11: 0.95}
    cand = np.array([100, 200])
    got = InfoGainStrategy().recommend(
        pool, _flat_probs(0.5), cand, n=1, state=state, params=params
    )
    assert got == [100]


def test_info_gain_empty_or_none_returns_empty():
    pool = _pool()
    assert InfoGainStrategy().recommend(pool, _flat_probs(0.5), np.array([]), n=1) == []
    assert (
        InfoGainStrategy().recommend(
            pool, _flat_probs(0.5), np.array([100]), n=1, state=None, params=None
        )
        == []
    )


def test_simulate_student_passes_counts_to_strategy():
    """simulate_student 在作答后 counts 递增并传给策略。"""
    pool = _pool()
    params = {
        10: BKTParams(l0=0.2, t=0.2, g=0.2, s=0.1),
        11: BKTParams(l0=0.8, t=0.2, g=0.2, s=0.1),
    }
    captured: list[dict[int, int]] = []

    class Spy:
        name = "spy"

        def recommend(self, pool, prob_fn, candidates, n, state=None, params=None, counts=None):
            captured.append(dict(counts or {}))
            return SequentialStrategy().recommend(pool, prob_fn, candidates, n, state, params)

    simulate_student(Spy(), [10, 11], pool, params, n_rounds=3, seed=0)
    # 第 1 轮 counts 来自 init_counts(空),随后每轮递增
    assert len(captured) == 3
    assert captured[0] == {}
    assert captured[1] != captured[0] or len(captured[1]) >= 1


def test_simulate_student_init_counts_used():
    """init_counts 会被传入策略并在作答后累加。"""
    pool = _pool()
    params = {
        10: BKTParams(l0=0.2, t=0.2, g=0.2, s=0.1),
        11: BKTParams(l0=0.8, t=0.2, g=0.2, s=0.1),
    }
    captured: list[dict[int, int]] = []

    class Spy:
        name = "spy"

        def recommend(self, pool, prob_fn, candidates, n, state=None, params=None, counts=None):
            captured.append(dict(counts or {}))
            return SequentialStrategy().recommend(pool, prob_fn, candidates, n, state, params)

    simulate_student(Spy(), [10, 11], pool, params, n_rounds=2, seed=0, init_counts={10: 2})
    assert captured[0].get(10, 0) == 2

