import pandas as pd
import pytest

from src.data import KEEP_COLS
from src.split import split_by_student, split_by_student_time


def make_seq_df(n_students: int = 6, per_student: int = 20) -> pd.DataFrame:
    rows = []
    oid = 1
    for uid in range(1, n_students + 1):
        for _ in range(per_student):
            rows.append([oid, uid, uid % 3, 900 + oid, oid % 2])
            oid += 1
    return pd.DataFrame(rows, columns=KEEP_COLS)


# ---------- 时间切分（P0 主口径） ----------


def test_time_split_keeps_history_in_train():
    train, test = split_by_student_time(make_seq_df(), test_ratio=0.2)
    assert set(test["user_id"]) <= set(train["user_id"])
    assert len(test) > 0


def test_time_split_test_is_the_tail():
    train, test = split_by_student_time(make_seq_df(n_students=1, per_student=10), test_ratio=0.2)
    assert test["order_id"].tolist() == [9, 10]
    assert len(train) == 8


def test_time_split_short_sequences_all_train():
    train, test = split_by_student_time(
        make_seq_df(n_students=1, per_student=5), test_ratio=0.2, min_events=10
    )
    assert len(test) == 0
    assert len(train) == 5


def test_time_split_no_row_loss():
    df = make_seq_df()
    train, test = split_by_student_time(df)
    assert len(train) + len(test) == len(df)


def test_time_split_rejects_bad_ratio():
    with pytest.raises(ValueError):
        split_by_student_time(make_seq_df(), test_ratio=1.5)


# ---------- 整学生切分（冷启动对照） ----------


def test_student_split_no_leak():
    train, test = split_by_student(make_seq_df(), test_size=0.2)
    assert set(train["user_id"]) & set(test["user_id"]) == set()


def test_student_split_no_row_loss():
    df = make_seq_df()
    train, test = split_by_student(df, test_size=0.2)
    assert len(train) + len(test) == len(df)


def test_student_split_deterministic():
    df = make_seq_df()
    a = split_by_student(df, seed=1)[1]["user_id"].tolist()
    b = split_by_student(df, seed=1)[1]["user_id"].tolist()
    assert a == b


def test_student_split_rejects_bad_ratio():
    with pytest.raises(ValueError):
        split_by_student(make_seq_df(), test_size=0.0)
