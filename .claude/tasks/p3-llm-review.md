# P3 · LLM 生成变式题 + 独立评审 agent

## 背景

项目 `question-recommender` 已完成 P0(评测骨架)/ P1(BKT 能力估计,AUC 0.7298)/ P2(推荐策略离线对比)。
现在补最后一块:**用 LLM 生成新题 + 用一个独立 agent 校验题质量**。
这也是全项目**唯一真正需要 RAG 的位置** —— 检索该知识点的样例题信息作为生成上下文。

## 现状(动手前必须先读)

- 仓库根:`D:\question-recommender`
- 已有模块:`src/data.py`(加载/清洗)、`src/split.py`、`src/metrics.py`、`src/baselines.py`、
  `src/bkt.py`(BKT,含 `update_mastery`)、`src/recommender.py`(题目池 + 4 策略)、`src/simulator.py`
- 测试:`tests/`,**当前 38 passed**;命令 `uv run pytest -q`
- 数据:`data/2009_skill_builder_data_corrected/skill_builder_data_corrected.csv`(读编码 `ISO-8859-15`)
  ⚠️ **该数据集没有题干文本**,只有 `problem_id / skill_id / skill_name / correct`。
  所以生成变式题的输入是 **`skill_name`**(而不是原题题干)。
- 运行环境:uv;任何联网命令要带 `UV_SYSTEM_CERTS=1`(本机 TLS 链问题)

## 需求

### P0(必须完成)

1. **`src/llm_client.py`** —— LLM 调用的薄封装。
   - 配置从环境变量读:`LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY`;缺失时抛清晰异常。
   - 提供 `FakeLLM`(可注入),**让测试完全不依赖真实 API**。

2. **`src/question_gen.py`**
   - `SkillContext`:从数据集按 `skill_name` 汇总该知识点的信息(知识点名 + 难度分布:该知识点题目全局正确率)。
   - `generate(skill_name, n)` → 调 LLM 生成 n 道变式题,输出**结构化 JSON**:
     `{"stem": str, "options": [str], "answer": str, "skill_name": str, "difficulty_hint": float}`
   - 用 `json.loads` 解析(不要正则抠文本);解析失败要抛错或重试,**不得静默吞掉**。

3. **`src/reviewer.py`** —— **独立**评审 agent
   - 输入:一道生成的题;输出结构化:
     `{"verdict": "pass"|"fail", "reasons": [str], "checks": {"answer_correct": bool, "unambiguous": bool, "difficulty_plausible": bool, "skill_aligned": bool}}`
   - **必须与生成器是独立的调用**(独立 prompt / 独立上下文),禁止复用生成时的上下文。
   - 生成器与评审器要能分别注入不同的 LLM 客户端(便于测试)。

4. **`scripts/p3_demo.py`** —— 一条命令跑完整链路
   - 取 K 个知识点 × n 道 = 共 100 道 → 生成 → 评审 → 打印统计(生成数 / 通过 / 拦下 / 拦截率)
   - 把被拦下的例子写入 `p3_rejected_examples.json`

### P1(尽量完成)

5. 记录每次调用的耗时与 token 估算并汇总打印。
6. README 增加 P3 章节(能力 + 快速开始 + 实测数字)。

## 验收 DoD

- `uv run pytest -q` **全绿**,新增测试 ≥ 10 用例;且**测试在无 API key 的情况下也能全过**(用 fake LLM)。
- `UV_SYSTEM_CERTS=1 uv run python scripts/p3_demo.py` 跑通,输出统计表,
  且 `p3_rejected_examples.json` 里**至少有 3 条真实的「评审拦下错误答案」例子**。
- 现有 38 个测试一个都不能红。

## 约束

- 不引入新的重依赖(优先用已装依赖或标准库 `urllib`)。
- 所有 LLM 输出走**结构化 JSON**。
- 生成器 / 评审器的 LLM 调用必须解耦、可分别注入 fake。
