# P4-FIX2 · 推荐结果必须带中文题面

## 问题

ASSISTments 数据集**只有 `problem_id`,没有题干文本**。所以 P4 页面上"推荐题"只显示
`problem_id` + 英文知识点名(如 `Perimeter of a Polygon`),**看不到题目内容**。

而 P3 已经能生成中文题(`src/question_gen.py` 的 `QuestionGenerator` + `SkillContext`),
但没有接进 P4 —— 两条线没打通。

## 目标闭环

**BKT 估计能力 → 增益选出知识点与难度 → LLM 现场生成该题的中文题面 → 学生作答 → 更新画像**

## 需求

### 1. `app.py`

- 新增**进程内缓存** `problem_id -> {"stem","options","answer"}`(服务生命周期内有效)。
- `GET /api/students/{uid}/recommend` 增加查询参数 `with_text`(默认 `1`):
  对每道推荐题,调 `QuestionGenerator` 生成一道**中文**题面。
  - 输入上下文:该题的知识点名 + 该题的 `difficulty`(作为难度要求,例如"难度提示 0.4,偏难")。
  - prompt 要求:**中文题干 + 4 个选项 + 正确答案**,输出结构化 JSON(复用 P3 的解析链)。
  - 把结果按 `problem_id` 缓存 —— 同一个 problem_id **只生成一次**。
- 响应里每道题增加 `text` 字段:`{"stem": str, "options": [str], "answer": str}` 或 `null`。
- **LLM 不可用(env 缺失)或调用失败时不得报错** —— `text` 置 `null`,接口仍返回 200。
- `with_text=0` 时不调用 LLM(直接返回 `text: null`)。

### 2. `web/index.html`

- 推荐卡片展示中文题干 + 选项;`text` 为 `null` 时显示占位文案(如"题面不可用")。
- 首次生成可能较慢 → 要有加载中提示。

### 3. 测试(`tests/test_app.py` 增补)

- 注入 `FakeLLM`:断言 `with_text=1` 时每道题带 `text.stem` 且为中文;
- 断言 `with_text=0` 时 `text` 全为 `null` 且**未调用** LLM;
- 断言**缓存命中**:同一 `problem_id` 第二次请求不再增加 LLM 调用次数;
- 断言 LLM 抛错时接口仍返回 200 且 `text: null`。

## DoD

- `uv run pytest -q` 全绿,**现有 87 个测试一个都不能红**。
- 真 LLM 起服务:`GET /api/students/14/recommend?k=3&with_text=1` 每道题都有**中文** `text.stem`(我会亲自复现)。
- 第二次请求同一批推荐明显更快(缓存生效)。

## 约束

- 不引入新依赖。
- LLM 未配置/失败必须优雅降级,不得 500。
- 只改 `app.py` / `web/index.html` / `tests/test_app.py`;不要动 `src/` 下的算法模块
  (可以 import 用)。
