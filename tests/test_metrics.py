import math

import pytest

from src.metrics import acc, auc, recall_at_k


def test_auc_perfect_ranking():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_auc_reversed_ranking():
    assert auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auc_constant_scores_is_half():
    assert auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auc_single_class_is_nan():
    assert math.isnan(auc([1, 1, 1], [0.1, 0.5, 0.9]))


def test_acc_threshold():
    assert acc([1, 0, 1, 0], [0.9, 0.1, 0.6, 0.4]) == 1.0
    assert acc([1, 0], [0.4, 0.6]) == 0.0


def test_recall_at_k():
    # 3 个正例，按分数取 top-2 命中 2 个 → 2/3
    y = [1, 0, 1, 1, 0]
    s = [0.9, 0.1, 0.8, 0.7, 0.2]
    assert recall_at_k(y, s, k=2) == pytest.approx(2 / 3)


def test_recall_at_k_no_positive_is_nan():
    assert math.isnan(recall_at_k([0, 0], [0.5, 0.1], k=1))
