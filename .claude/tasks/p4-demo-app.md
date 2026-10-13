# P4 · 闭环演示(做题 → 掌握度 → 推荐)

## 现状(动手前必读)

- 仓库根 `D:\question-recommender`。已完成 P0–P3,**当前 72 passed**。
- 已有模块:`src/data.py`(`load_clean`)、`src/split.py`(`split_by_student_time`)、
  `src/baselines.py`、`src/bkt.py`(`BKT` 类 + `BKTParams` + `update_mastery`)、
  `src/recommender.py`(`QuestionPool.from_frame` / `GainStrategy` / `default_strategies`)、
  `src/simulator.py`、`src/llm_client.py`、`src/question_gen.py`、`src/reviewer.py`
- 模型:BKT 每个知识点 4 参数(L0/T/G/S)用 EM 拟合;推荐:`GainStrategy`(期望掌握度增益最大)。
- 数据:`data/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv`(读编码 `ISO-8859-15`)
- 环境:uv;联网命令带 `UV_SYSTEM_CERTS=1`。**BKT 全量拟合约 80 秒,只应在服务启动时做一次。**
- ⚠️ 已知坑:`api.commandcode.ai` 会拦 `Python-urllib` 的 UA(403);本项目 `HttpLLM` 已修。
  本任务不涉及 LLM 调用。

## 需求

### P0-1 `app.py`(FastAPI 服务)

启动时执行一次:加载清洗数据 → 拟合 BKT → 建 `QuestionPool`,全部缓存到进程内(别每次请求重算)。

端点:

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 返回 `web/index.html` |
| GET | `/api/students?limit=20` | 抽样学生(取 test 中历史足够的学生) |
| GET | `/api/students/{uid}/state` | 该学生每个知识点的掌握度 `[{skill_id, skill_name, mastery}]` + 概览(`mean_mastery`, `n_weak`) |
| GET | `/api/students/{uid}/recommend?k=5` | 用 `GainStrategy` 推荐 k 道题,返回 `[{problem_id, skill_id, skill_name, difficulty}]` |
| POST | `/api/students/{uid}/answer` | body `{"problem_id": int, "correct": bool}`;更新该学生掌握度 → 返回新 `state` + 新推荐 |

- 学生状态存在进程内 dict(以 uid 为键),**初始化方式**:用该学生在数据中的真实历史记录
  逐条 `update_mastery` 递推,得到"当前掌握度"。
- 环境变量 `P4_FAST=1` 时用小子集数据(如只取 50 个学生、跳过全量拟合)快速启动,便于测试与演示。

### P0-2 `web/index.html`(单文件,零构建、零外部 CDN,离线可用)

- 左侧:**掌握度热力图**(知识点 × 掌握度;颜色深浅表示掌握程度),可滚动
- 右侧:当前推荐题列表(默认 5 道),每道配「答对 / 答错」两个按钮
- 点击按钮 → `POST answer` → **热力图与推荐列表实时更新**
- 顶部:切换学生的下拉框

### P0-3 `tests/test_app.py`

用 FastAPI `TestClient` 测每个端点(用 `P4_FAST=1` 或注入最小模型,**不要跑 80 秒全量拟合**)。**≥ 8 个用例。**

## DoD

- `uv run pytest -q` 全绿,**现有 72 个测试一个都不能红**。
- `P4_FAST=1 uv run uvicorn app:app --port 8010` 起服务后:
  - `GET /` → 200 且返回 HTML
  - `GET /api/students` → 200
  - `GET /api/students/{uid}/state` → 200 且含掌握度数据
  - `GET /api/students/{uid}/recommend?k=5` → 返回 5 道题
  - `POST /api/students/{uid}/answer` → 该学生的掌握度**确实发生变化**
- README 增加 P4 章节(能力 + 启动命令)。

## 约束

- 不引入重型前端依赖(无 npm、无 CDN 外链)。
- 模型只加载一次(进程级缓存),请求内不得重新拟合。
- 新增依赖只能用已有的(`fastapi`/`uvicorn` 若未装可加入 `pyproject.toml`,但不要引入其它重依赖)。
