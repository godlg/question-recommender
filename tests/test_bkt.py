import numpy as np
import pandas as pd
import pytest

from src.bkt import BKT, fit_skill
from src.data import KEEP_COLS


def _df(rows):
    return pd.DataFrame(rows, columns=KEEP_COLS)


def test_fit_returns_valid_probabilities():
    obs = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    p = fit_skill([obs], n_iter=20)
    for v in (p.l0, p.t, p.g, p.s):
        assert 0.0 < v < 1.0


def test_fit_detects_learning_after_failures():
    # 前 5 次全错、后 15 次全对 → 学习率应明显大于 0
    obs = np.array([0] * 5 + [1] * 15)
    p = fit_skill([obs], n_iter=30)
    assert p.t > 0.05


def test_predict_length_and_range():
    train = _df([[i, 1, 10, 100, 1] for i in range(1, 21)])
    test = _df([[i, 1, 10, 100, 1] for i in range(21, 26)])
    model = BKT(n_iter=10).fit(train)
    preds = model.predict(train, test)
    assert len(preds) == len(test)
    assert np.all((preds >= 0) & (preds <= 1))


def test_predict_high_after_correct_streak():
    train = _df([[i, 1, 10, 100, 1] for i in range(1, 21)])
    test = _df([[i, 1, 10, 100, 1] for i in range(21, 26)])
    preds = BKT(n_iter=10).fit(train).predict(train, test)
    assert preds[-1] > 0.5


def test_unseen_skill_falls_back_to_prior():
    train = _df([[1, 1, 10, 100, 1], [2, 1, 10, 100, 0]])
    test = _df([[3, 1, 99, 200, 1]])
    preds = BKT(n_iter=5).fit(train).predict(train, test)
    assert preds[0] == pytest.approx(train["correct"].mean())


def test_predict_uses_train_history():
    """test 记录应能用上 train 里的历史 → 学生已连续答对后,预测应偏高。"""
    train = _df([[i, 7, 10, 100, 1] for i in range(1, 11)])  # 学生 7 全对 10 次
    test = _df([[11, 7, 10, 100, 1]])                          # 学生 7 的下一条
    preds = BKT(n_iter=10).fit(train).predict(train, test)
    assert preds[0] > 0.5
