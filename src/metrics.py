"""评测指标：AUC / ACC / Recall@k。"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y_true: Sequence, y_score: Sequence) -> float:
    """ROC-AUC。标签只有一类时返回 nan。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def acc(y_true: Sequence, y_score: Sequence, threshold: float = 0.5) -> float:
    """把概率按阈值二值化后算准确率。"""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    return float((y_pred == y_true).mean())


def recall_at_k(y_true: Sequence, y_score: Sequence, k: int) -> float:
    """Recall@k：按分数取 top-k，命中正例的比例。（P0 只留接口）"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")
    k = min(k, len(y_true))
    top = np.argsort(-y_score)[:k]
    return float((y_true[top] == 1).sum() / n_pos)
