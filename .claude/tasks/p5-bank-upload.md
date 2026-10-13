# P5 · 题库上传 + 按学生推荐(核心链路)

## 目标

用户上传**自己的题库**(带题干/知识点/难度)→ 学生做题 → 系统按学生掌握度推荐合适的题。

## 现状(动手前必读)

- 仓库根 `D:\question-recommender`。已有 `app.py`(FastAPI + 热力图前端,P4)、
  `src/{data,split,bkt,recommender,simulator,metrics,baselines}.py`、`src/{llm_client,question_gen,reviewer}.py`。
  **当前 92 passed**;命令 `uv run pytest -q`。
- 现有题源是 ASSISTments(`data/2009_skill_builder_data_corrected/*.csv`,**只有 problem_id、没有题干**)。
- 推荐用 `src/recommender.py` 的 `GainStrategy`(期望掌握度增益最大),BKT 在 `src/bkt.py`。
- 环境:uv;联网命令带 `UV_SYSTEM_CERTS=1`。

## 需求

### P0-1 标准题库格式(CSV / Excel)

必需列(**列名固定,大小写不敏感**):

| 列 | 说明 |
|---|---|
| `stem` | 题干 |
| `answer` | 正确答案(有 options 时必须是其中之一) |
| `skill` | 知识点名 |
| `options` | 选项,用 `|` 分隔(如 `x=3|x=5|x=4`)。**可为空**(填空题) |
| `difficulty` | 0~1,越大越容易。可空 → 默认 0.5 |

`question_id` 由系统生成(从 1 递增)。解析时:缺必需列 → 报错并说清缺哪列;单行坏(如 answer 不在 options 里)
→ **跳过并计数**,不整体失败。Excel 读取用 `openpyxl`(未装则加入 `pyproject.toml`)。

### P0-2 题库上传与存储

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/bank/upload` | multipart 传 CSV/Excel → 解析校验 → **替换当前题库**(内存 + 落盘)→ 返回 `{n_questions, n_skills, n_skipped, source_name}` |
| GET | `/api/bank` | 当前题库信息:`{n_questions, n_skills, source_name}` |
| GET | `/api/bank/sample?n=5` | 样例题(带题干),供前端展示 |

- 落盘到 `data/bank.jsonl`(**每行一道题的标准 JSON**);服务启动时若存在则自动加载。
- 上传解析失败(格式错/缺列)要返回 **400 + 明确原因**,不要 500。

### P0-3 学生与推荐(适配上传的题库)

- 新题库没有历史作答 → **BKT 用默认参数**:`L0=0.3, T=0.2, G=0.2, S=0.1`(定义为模块常量,README 里说明)。
- 新增 `POST /api/students`(body 可选 `{"name": str}`)→ 创建一个**空画像学生**(所有知识点掌握度 = L0),返回 `{user_id}`。
- 现有 `GET /api/students`、`GET /api/students/{uid}/state`、`GET /api/students/{uid}/recommend?k=`、
  `POST /api/students/{uid}/answer` **逻辑不变,但题源改成当前题库**:
  - `recommend` 用 `GainStrategy` 从**当前题库**选 k 道,返回 `{question_id, stem, options, answer, skill, difficulty}`
    (题干题库自带,**不再调 LLM**;`with_text` 参数可保留但默认不再需要)
  - `answer` 按 `question_id` 更新该学生在该 `skill` 上的掌握度
- **未上传题库时**,保持原有 ASSISTments 行为(向后兼容,现有测试不得变红)。

### P0-4 前端 `web/index.html`

- 新增"题库"区:文件选择 + 上传按钮 + 显示 `n_questions / n_skills / source_name`;
  上传中/失败要有提示。
- 推荐卡片直接展示题库自带的题干 + 选项(不再显示"正在生成…")。
- 保留掌握度热力图与学生切换;新增"新建学生"按钮。

### P0-5 样例题库(必须附)

生成 `data/sample_bank.csv`:约 **20 道中文数学题**,覆盖 **4~5 个知识点**
(如"一元一次方程 / 分数运算 / 几何周长 / 概率初步"),每题带 `stem,options,answer,skill,difficulty`,
**答案必须正确**(我会抽查),难度分布 0.3~0.9 混开。

### P0-6 测试(`tests/test_bank.py` 新建)

≥ 10 用例:CSV 解析正确 / 缺列报 400 / 坏行跳过计数 / Excel 解析 /
上传后 `/api/bank` 数字对 / 落盘与重载 / 上传后推荐来自新题库且带题干 /
新建学生掌握度为默认值 / answer 更新掌握度 / 坏上传不破坏已有题库。

## DoD

- `uv run pytest -q` 全绿,**现有 92 个测试一个都不能红**。
- 真起服务:上传 `data/sample_bank.csv` → `/api/bank` 显示 20 题 →
  `POST /api/students` 建学生 → `recommend?k=3` 返回**该题库的题(带题干)** →
  `answer` 后该学生掌握度变化。**(我会亲自复现)**
- README 增加 P5 章节:题库格式表 + 上传方式 + "为什么这条链路不用 LLM"。

## 约束

- **本任务不引入任何 LLM 调用**(纯本地:BKT 默认参数 + GainStrategy)。
- 不引入重依赖;Excel 用 `openpyxl`。
- 不破坏现有 ASSISTments 路径与全部既有测试。
