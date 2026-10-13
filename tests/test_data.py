import pandas as pd

from src.data import KEEP_COLS, clean


def make_df(rows):
    return pd.DataFrame(rows, columns=KEEP_COLS)


def test_clean_drops_missing_skill():
    df = make_df([[1, 100, 5, 900, 1], [2, 100, None, 901, 0]])
    out = clean(df)
    assert len(out) == 1
    assert out.iloc[0]["skill_id"] == 5


def test_clean_drops_non_binary_label():
    df = make_df([[1, 100, 5, 900, 1], [2, 100, 5, 901, 2]])
    out = clean(df)
    assert set(out["correct"]) <= {0, 1}
    assert len(out) == 1


def test_keeps_same_user_order_different_skill():
    """同一题关联多个知识点会复制成多行 —— 不能按 (user, order) 去重。"""
    df = make_df([[1, 100, 5, 900, 1], [1, 100, 7, 900, 1]])
    out = clean(df)
    assert len(out) == 2


def test_drops_exact_duplicate_row():
    df = make_df([[1, 100, 5, 900, 1], [1, 100, 5, 900, 1]])
    out = clean(df)
    assert len(out) == 1


def test_sorted_by_user_then_order():
    df = make_df([[3, 200, 5, 901, 0], [2, 100, 5, 902, 1], [1, 100, 7, 903, 1]])
    out = clean(df)
    assert out["user_id"].tolist() == [100, 100, 200]
    assert out["order_id"].tolist() == [1, 2, 3]


def test_output_dtypes_are_int():
    out = clean(make_df([[1, 100, 5, 900, 1]]))
    for col in KEEP_COLS:
        assert out[col].dtype == "int64"
