# P8:冷启动探索 + 信息增益策略(与增益推对照)

## 目标(两件事,不要扩大范围)

1. **冷启动门控**:该生在某知识点作答次数 < 3 时,不信 BKT 先验(先验只是默认参数,不是证据),优先去测这些知识点。
2. **新增第 5 个策略 `InfoGainStrategy`(信息增益)**,与增益推对照,重点看它能否在「新知识点多的学生」子集上赢。

## 现状事实(已核对,行号为当前 main)

- `src/recommender.py`
  - `Strategy` 协议:`recommend(self, pool, prob_fn, candidates, n, state=None, params=None)`(第 61-72 行)
  - 已有 4 策略:RandomStrategy / SequentialStrategy / ZPDStrategy / GainStrategy(第 91-166 行)
  - `GainStrategy`:`gains[i] = p_correct*d_ok + (1-p_correct)*d_bad`(第 150-166 行)
  - `_dedup_by_skill(pool, ordered_pids, n)`:一轮内同一知识点只取一道(第 75-88 行)
  - `default_strategies()` 返回 4 个(第 169-170 行)
- `src/simulator.py`
  - `simulate_student(strategy, student_skills, pool, params, n_rounds=20, seed=0)`(第 47-94 行)
  - 每轮:`strategy.recommend(pool, _prob_fn(state, params, pool), avail, n=1, state=state, params=params)`(第 73-75 行)
  - 掌握度初值 `init = {s: params[s].l0 for s in skills}`(第 61 行)
- `app.py`
  - `AppState.history: dict[int, list[tuple[int, int]]]  # uid → [(skill_id, correct)]`(第 63 行)
  - `POST /api/students/{uid}/answer`:只更新 `mastery[sid]`(第 508-509 行),**没有记录作答次数**
  - `_recommend_payload`:用 `GainStrategy().recommend(pool, prob_fn, candidates, n=k, state=mastery, params=params)`(第 406-408 行)
  - `_state_payload`:每 skill 只返 `skill_id / skill_name / mastery`(第 319-330 行)
- `scripts/eval_recommend.py`:4 策略表(第 57-78 行);N_STUDENTS=200,N_ROUNDS=20,SEED=42
- 基线数字(seed 42,已实测):随机推 0.6362 / 顺序推 0.6625 / ZPD推 0.6260 / 增益推 0.6918(薄弱掌握度 0.6182/0.6432/0.6077/0.6843)

## 改动 1:`src/recommender.py`

### 1.1 常量与协议
- 新增 `COLD_MIN = 3`。
- `Strategy` 协议与 4 个现有策略的 `recommend` 签名**追加可选参数** `counts: dict[int, int] | None = None`(不改位置、不给默认以外的新行为)。Random/Sequential/ZPD 接受但忽略即可。

### 1.2 `GainStrategy` 增加冷启动门控
- dataclass 追加字段:`cold_start: bool = True`、`cold_min: int = COLD_MIN`。
- **兼容性是硬要求**:`counts is None` 时,排序结果必须与改动前**逐位一致**(旧测试与旧对照数字不能变)。
- `counts` 给出且 `cold_start=True` 时:
  - 冷启动候选 = 候选题中 `counts.get(skill, 0) < cold_min` 的题;
  - 冷启动候选非空 → **只在冷启动候选中选**,排序键 `(counts 升序, problem_id 升序)`,再走 `_dedup_by_skill(pool, order, n)`;
  - 冷启动候选为空(所有相关知识点都测过 ≥ cold_min 次)→ 走原有期望增益排序。
- 注释写清语义:这是「不信先验」的**门控**,不是严格信息增益(严格版由 InfoGainStrategy 承担)。为什么门控是必要的:新知识点 `m = L0` 是默认参数,不是该生的证据,期望增益在错误先验上算出来会挑错题。

### 1.3 新增 `InfoGainStrategy`
- `name = "信息增益"`;与 GainStrategy 同构(都用 `state` / `params` / `prob_fn` 语义)。
- 打分 = **闭式期望熵减**:
  ```
  H(p) = 0 if p<=0 or p>=1 else -(p*log(p) + (1-p)*log(1-p))
  m      = state.get(sid, prm.l0)
  p_ok   = m*(1-prm.s) + (1-m)*prm.g
  m_ok   = update_mastery(m, 1, prm)      # 复用 src.bkt.update_mastery
  m_bad  = update_mastery(m, 0, prm)
  score  = H(m) - (p_ok*H(m_ok) + (1-p_ok)*H(m_bad))
  ```
- 按 score 降序,`_dedup_by_skill` 取 n;候选空 / `state is None` / `params is None` → 返回 `[]`。
- 熵函数写成模块级 `_entropy(p: float) -> float`,注释标明 `0·log0 = 0` 约定。

### 1.4 `default_strategies()`
返回 5 个:随机推 / 顺序推 / ZPD推 / 增益推(默认 `cold_start=True`)/ 信息增益。`GainStrategy.name` 保持 `"增益推"` 不变(消融时由脚本构造 `cold_start=False` 的实例并自行显示名字)。

## 改动 2:`src/simulator.py`

- `simulate_student` 追加参数 `init_counts: dict[int, int] | None = None`。
- 函数内部 `counts = dict(init_counts or {})`;每轮成功作答后 `counts[sid] = counts.get(sid, 0) + 1`。
- 调用策略时传 `counts=counts`。
- **不要改**掌握度初值(`init = l0`)、返回值结构与已有三个指标口径 —— 保证与 P2 数字可比。

## 改动 3:`app.py`

- `AppState` 追加字段 `n_answers: dict[int, dict[int, int]] = field(default_factory=dict)`(uid → skill_id → 次数)。
- `POST /api/students/{uid}/answer`:更新 mastery 的同时把该生该知识点计数 +1。
- 新增小工具函数(如 `_counts_of(st, uid) -> dict[int, int]`):把 `st.history[uid]`(数据集学生)按 skill 计数,再叠加 `st.n_answers[uid]`(新建学生的作答)。
- `_recommend_payload`:把 `counts` 传给 `GainStrategy().recommend(..., counts=counts)`,默认策略即「冷启动 + 增益推」。
- `GET /state` 的每个 skill 增加字段 `n_answers: int`(该生在该知识点的作答次数)—— 用于区分「未测过(先验)」与「测得不会(后验)」,也是冷启动是否生效的可观测证据。
- 前端(`web/index.html`)**只在有现成展示位时顺带显示 n_answers;不要为此重构页面**。

## 改动 4:`scripts/eval_recommend.py`

保持 seed 42 / 200 学生 / 20 轮 / 抽样口径不变,输出**三张表**:

- **表 A · 5 策略**:`default_strategies()`(增益推 = 冷启动开),列:全体掌握度 / 薄弱掌握度 / 累计准确率。
- **表 B · 冷启动消融**:`增益推(冷启动关)` vs `增益推(冷启动开)` 同样三列 → 回答「冷启动到底是帮忙还是添乱」。
- **表 C · 子集:新知识点多的学生**:
  - 该生 `init_counts` = 其 **test 段**真实作答按 skill 计数;
  - `cold_ratio` = `init_counts` 中 `< COLD_MIN` 的知识点数 / 该生相关知识点总数;
  - 子集 = `cold_ratio >= 0.5` 的学生;打印**该子集学生数**;
  - 在该子集上比 `增益推(冷启动开)` vs `信息增益` 的薄弱掌握度(**这正是本阶段要回答的问题**)。
  - `init_counts` 只用于冷启动判定,**不改变**掌握度初值(仍 `l0`),保证与旧口径可比。
- 打印学生数 / 轮数 / 耗时,总耗时 < 300s。

## 改动 5:`README.md`

- 新增 P8 小节:冷启动门控 + 信息增益策略 + 三张表数字。
- 旧的「增益推 0.6918 vs 随机 0.6362」结论**保留**,但注明口径(冷启动关闭时的 P2 数字);若冷启动开启后数字变化,写出新数字。
- 诚实保留既有局限声明(离线模拟与策略共用同一套 BKT = 自己评自己,真效果需线上 A/B)。

## 明确不做(DON'Ts)

- 不改 BKT 的参数拟合 / EM / 切分口径 / 数据加载;**不新增任何依赖**(numpy/pandas 已够)。
- 不改检索、讲解、题库、材料相关代码。
- 不重构 `Strategy` 协议(只追加可选 `counts` kwarg)。
- 不加 DKT/IRT/强化学习;不做多轮对话;不接 LLM。
- 不重排 / 不格式化与任务无关的现有代码;不留临时脚本、调试打印、`$null` 之类脏文件。
- 不允许 mock / 假数据 / hardcode 结果绕过实现:新策略必须是真算法,测试必须真跑策略代码。

## DoD(逐条给真实证据,不许只写"已完成")

1. **测试**:`cd /d/question-recommender && UV_SYSTEM_CERTS=1 uv run pytest -q` 全绿,并报告改动前后的 passed 数。新增测试 ≥6 条,至少覆盖:
   - `counts=None` 时 `GainStrategy` 与旧实现返回**相同 pid 序列**;
   - 冷启动门控:构造 `counts` 让某知识点 0 次 → 首选落在该知识点;
   - 冷启动一轮内不重复知识点;
   - `InfoGainStrategy`:不确定性大者优先(`m≈0.5` 的题分数 > `m≈0.95` 的题);
   - `InfoGainStrategy` 候选空 / `state=None` → `[]`;
   - `simulate_student` 在作答后 `counts` 递增并传给策略(可用自定义策略捕获断言)。
2. **对照实验**:`UV_SYSTEM_CERTS=1 uv run python scripts/eval_recommend.py`,贴出完整三张表 + 耗时。
3. **服务侧实测**:起服务(`P4_FAST=1 ... uvicorn app:app --port 8010`),新建学生 → `GET /api/students/{uid}/recommend?k=5` 应覆盖不同知识点;对同一知识点 `POST /answer` 3 次后,`GET /api/students/{uid}/state` 中该知识点 `n_answers >= 3` 且它退出冷启动候选。贴真实 curl/urllib 输出。
4. **改动面**:`git status` / `git diff --stat` 只落在 `src/recommender.py`、`src/simulator.py`、`app.py`、`scripts/eval_recommend.py`、`tests/`、`README.md`(前端仅在有现成展示位时允许)。
5. **注意**:评测脚本必须从项目根运行(用相对路径找 `data/`);BKT 全量拟合约 80s,属正常。

## 交付报告格式

按顺序给:① 改动文件清单(diff 摘要)② pytest 输出尾部 ③ 三张表原文 ④ 服务侧实测输出 ⑤ 与旧数字的差异及原因一句话。**所有数字必须来自真实跑出的终端输出。**
