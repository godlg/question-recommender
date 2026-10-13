# question-recommender

个性化推荐题目的 Agent(实习面试项目):用**知识追踪**估计学生能力 → 卡在**最近发展区**选题 → 推荐。

> P0 已完成:数据链路 + 评测骨架打通,baseline 数字表就位。

## 冷启动(第一次跑)

```bash
# 1. 建环境
uv sync

# 2. 下载数据(ASSISTments 2009-2010,中科大镜像,约 8.7MB)
UV_SYSTEM_CERTS=1 uv run --python 3.13 --with EduData \
  python -c "from EduData import get_data; get_data('assistment-2009-2010-skill','data/')"

# 3. 跑 baseline 数字表
UV_SYSTEM_CERTS=1 uv run python scripts/p0_baseline_table.py

# 4. 测试
uv run pytest -q
```

> `UV_SYSTEM_CERTS=1` 在本机必需:否则 uv 下载报 `invalid peer certificate: UnknownIssuer`。

## 数据与预处理口径(v1)

数据集:ASSISTments 2009-2010 skill builder(corrected 版)。

| 步骤 | 规则 |
|---|---|
| 取列 | `order_id / user_id / skill_id / problem_id / correct`(其余 25 列丢弃) |
| 丢缺失 | 丢掉 `skill_id` 缺失的行(63,755 行,15.9%);其余四列无缺失 |
| 标签 | 只保留 `correct ∈ {0,1}`(实测无需过滤) |
| 去重 | 只去**完全重复**的行。⚠️ **不能按 `(user_id, order_id)` 去重**——skill builder 把一道题按关联知识点复制成多行,按这两列去重会误删 54,896 行 |
| 排序 | 按 `(user_id, order_id)` |
| 读编码 | `ISO-8859-15` |

实测:**338,001 行 / 4,163 学生 / 123 知识点 / 17,751 题**(与 EduKTM 的 `assistment-2009-2010-skill` 口径一致)。

## 划分

**每个学生按时间切**(`split_by_student_time`):前 80% 进 train、后 20% 进 test;交互数 < 10 的学生全留 train。
这是 KT / 推荐的任务定义(用学生历史预测其后续),也是「按学生历史」baseline 有意义的前提。

另提供 `split_by_student`(整学生切走)作**冷启动对照**:此时 test 学生在 train 中不出现,所有「用学生信息」的 baseline 退化为全局先验(实测 AUC=0.5000)。这是另一个评测场景,P1 之后再启用。

## P0 实测结果(2026-10-12)

```
清洗后总量 : 338,001 行
学生       : 4,163
知识点     : 123
题目       : 17,751
train/test : 272,590 / 65,411 行(每生前 80% / 后 20%)
test 覆盖  : 3,114 名学生
test 正确率: 0.6439

baseline      AUC      ACC
----------------------------
全局正确率    0.5000   0.6439
按知识点      0.6281   0.6561
按学生历史    0.6690   0.6788   ← P1 的 KT 必须超过它
随机          0.4985   0.5030
```

## 目录

```
src/data.py       加载 + 清洗 + 统一 schema
src/split.py      按学生时间切分(主) / 整学生切分(冷启动对照)
src/metrics.py    AUC / ACC / Recall@k
src/baselines.py  四个预测型 baseline
scripts/          可执行脚本
tests/            pytest
```

## 测试

`uv run pytest -q` → **22 passed**
