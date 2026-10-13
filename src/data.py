"""加载 ASSISTments 2009-2010 并统一 schema。

口径 v1：
  1. 只取 order_id / user_id / skill_id / problem_id / correct 五列
  2. 丢掉 skill_id 缺失的行（63,755 行，约 15.9%）
  3. 只保留 correct ∈ {0, 1}
  4. 去掉「完全重复」的行
     ⚠️ 不能按 (user_id, order_id) 去重：ASSISTments skill builder 会把一道题
        按关联知识点复制成多行，按这两列去重会误删 54,896 行（实测 283,105 vs 338,001）
  5. 按 (user_id, order_id) 排序

实测结果：338,001 行（与 EduKTM 的 assistment-2009-2010-skill 口径一致）
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_FILENAME = "skill_builder_data_corrected.csv"
KEEP_COLS = ["order_id", "user_id", "skill_id", "problem_id", "correct"]
ENCODING = "ISO-8859-15"


def find_raw(data_dir: str | Path = "data") -> Path:
    """在 data/ 里递归找原始 csv。"""
    data_dir = Path(data_dir)
    hits = sorted(data_dir.glob(f"**/{RAW_FILENAME}"))
    if not hits:
        raise FileNotFoundError(f"{data_dir} 下找不到 {RAW_FILENAME}；先跑下载命令")
    return hits[0]


def load_raw(data_dir: str | Path = "data") -> pd.DataFrame:
    """读原始 csv，只取需要的五列。"""
    path = find_raw(data_dir)
    return pd.read_csv(path, encoding=ENCODING, low_memory=False, usecols=KEEP_COLS)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """统一下来的干净表（口径 v1）。"""
    out = df[KEEP_COLS].copy()
    out = out.dropna(subset=KEEP_COLS)
    out = out[out["correct"].isin([0, 1])]
    for col in KEEP_COLS:
        out[col] = out[col].astype("int64")
    out = out.drop_duplicates()
    out = out.sort_values(["user_id", "order_id"]).reset_index(drop=True)
    return out


def load_clean(data_dir: str | Path = "data") -> pd.DataFrame:
    """一步到位：找文件 → 读 → 清洗。"""
    return clean(load_raw(data_dir))
