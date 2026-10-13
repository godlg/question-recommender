# question-recommender

个性化推荐题目的 Agent(实习面试项目):用**知识追踪**估计学生能力 → 卡在**最近发展区**选题 → 推荐。

> P0(评测骨架)✅ · P1(能力估计 BKT,AUC 0.7298)✅ · P2(推荐链路 + 离线对照)✅ · P3(LLM 生成题 + 独立评审)✅ · P4(闭环演示服务)✅ · P5(题库上传 + 按学生推荐)✅ · P6(错题讲解 RAG)✅
> **增益推把平均掌握度从 0.6362(随机)提到 0.6918。**

## 冷启动(第一次跑)

```bash
# 1. 建环境
uv sync

# 2. 下载数据(ASSISTments 2009-2010,中科大镜像,约 8.7MB)
UV_SYSTEM_CERTS=1 uv run --python 3.13 --with EduData \
  python -c "from EduData import get_data; get_data('assistment-2009-2010-skill','data/')"

# 3. 跑对比表
UV_SYSTEM_CERTS=1 uv run python scripts/eval_bkt.py          # 能力估计:4 baseline + BKT
UV_SYSTEM_CERTS=1 uv run python scripts/eval_recommend.py    # 推荐:4 策略离线对比

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

**每个学生按时间切**(`split_by_student_time`):前 80% train / 后 20% test;交互数 < 10 的学生全留 train。
另提供 `split_by_student`(整学生切走)作**冷启动对照**——此时 test 学生在 train 中不出现,所有「用学生信息」的 baseline 退化为全局先验(实测 AUC=0.5000)。

## P1:能力估计实测(2026-10-12)

```
train/test : 272,590 / 65,411 行;知识点 123(已拟合 117)

model        AUC      ACC
全局正确率   0.5000   0.6439
按知识点     0.6281   0.6561
按学生历史   0.6690   0.6788
随机         0.4985   0.5030
BKT          0.7298   0.7132   ← 打赢最强 baseline +0.061
```

## P2:推荐策略离线对比(2026-10-12)

⚠️ **这是离线模拟评估**:模拟学习者与推荐策略共用同一套 BKT 参数,存在「自己评自己」的偏差。
它衡量策略在学习者模型下的**相对优劣**,不是真实教学效果(真实效果需线上 A/B)。

```
模拟 200 名学生 × 20 轮(每轮推 1 道没做过的题)

策略              全体掌握度  薄弱掌握度  累计准确率
随机推            0.6362     0.6182     0.4730
顺序推            0.6625     0.6432     0.5107
ZPD推(难度匹配)   0.6260     0.6077     0.5502
增益推(期望增益)  0.6918     0.6843     0.4367   ← 最优
```

**关键洞察**:增益推答对率最低(0.437)但掌握度最高——它不挑简单题讨好学生,而是在**最该学的地方**投入。只看答对率会误判策略。

策略定义:
- `随机推` 从候选中随机取(对照)
- `顺序推` 按 problem_id 升序(对照)
- `ZPD推` 挑预测正确率最接近 0.7 的题 —— **朴素实现会跑偏**:它被"已经中等掌握"的题吸引,绕开真正薄弱的知识点
- `增益推` **最大化期望掌握度增益** `E[Δm] = p(对)·Δm(对) + p(错)·Δm(错)` —— 把「该练什么」形式化

## P3:LLM 生成变式题 + 独立评审(2026-10-13)

全项目**唯一真正需要 RAG 的位置**。数据集没有题干文本,所以检索的是该知识点的
**样例题统计信息**(题库规模 / 难度分布 / 代表题 ID)作为生成上下文,
生成输入是 `skill_name`。

链路:`知识点上下文(RAG)` → `QuestionGenerator` 生成结构化 JSON 题
→ `Reviewer` **独立**评审(独立 prompt / 独立 LLM 调用,禁止复用生成上下文)。

```
$ UV_SYSTEM_CERTS=1 uv run python scripts/p3_demo.py --skills 1 --per-skill 9
运行模式   : 真实 LLM(HTTP)
知识点     : 1 个 × 9 道
注入策略   : 每 3 道注入 1 道错答案(共注入 3 道)
生成 / 通过/ 拦下 : 9 / 5 / 4
拦截率     : 44.4%

调用          次数     耗时(s)    token(估)
生成器          1     17.71         607
评审器          9     87.34        3195
合计           10    105.05        3802

评审区分:
  注入错误检出率   : 3/3 = 100.0%(评审器独立抓回的被注入错误)
  LLM 自身质量问题 : 1 道(未被注入却被判 fail)
```

**默认走真实 LLM**:`LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` 齐全时直接调
真实 OpenAI 兼容接口(统计里 token > 0 且耗时 > 0)。三者缺失才回退到**离线确定性
引擎**(FakeLLM),并打印明确警告「当前未调用真实 LLM」。

> ⚠️ 真实链路必须带浏览器 `User-Agent`:`HttpLLM` 默认发送
> `Mozilla/5.0 (compatible; question-recommender/0.1)` —— urllib 默认 UA
> (`Python-urllib/3.x`)会被 `api.commandcode.ai` 的 WAF 拦成 **HTTP 403**。

**注入错误与生成解耦**:先生成,再由 demo 按 `--inject-every E`(默认 3)周期性把
某题的 `answer` 改成 `options` 里的另一个值(真正的错误答案),并记录注入清单。评审器
独立重算,报告把 **注入错误检出率**(检测召回)与 **LLM 自身质量问题**(未被注入却被判
fail)分开统计,拦截样例 ≥ 3 条,写入 `p3_rejected_examples.json`(**真实 LLM 生成**)。

- CLI 规模:`--skills K --per-skill N`(默认 `1 × 9`)。
- 结构化输出:所有 LLM 返回一律 `json.loads` 解析,失败重试、最终抛错,不静默吞掉。
- 生成器 / 评审器各自持有独立 LLM 客户端,可分别注入 fake 便于测试。

## P4:闭环演示(做题 → 掌握度 → 推荐)(2026-10-13)

`app.py`(FastAPI)+ `web/index.html`(单文件,零构建、零 CDN,离线可用)把 P1–P3
的链路包成一个**可交互的闭环**:学生做题 → BKT 在线更新掌握度 → 立即重推下一批题。

```bash
# 快速演示(小子集,秒级启动,适合测试/演示)
P4_FAST=1 uv run uvicorn app:app --port 8010

# 完整模型(启动时全量拟合 BKT 一次,约 80 秒;之后请求内不再重算)
uv run uvicorn app:app --port 8010
# 浏览器打开 http://127.0.0.1:8010
```

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 返回 `web/index.html` |
| GET | `/api/students?limit=20` | 抽样学生(优先 test 中历史足够者) |
| GET | `/api/students/{uid}/state` | 每个知识点掌握度 + 概览(`mean_mastery` / `n_weak`) |
| GET | `/api/students/{uid}/recommend?k=5` | `GainStrategy` 推荐 k 道题(含难度) |
| POST | `/api/students/{uid}/answer` | body `{problem_id, correct}` → 更新掌握度 → 返回新 state + 新推荐 |
| POST | `/api/materials/upload` | multipart 上传 md/txt 讲解材料 → 入库 → `{n_chunks, skills, files}` |
| GET | `/api/materials` | 材料库概况 `{n_chunks, skills, files}` |
| POST | `/api/students/{uid}/explain` | body `{problem_id, student_answer}` → `{explanation, sources, grounded, error}` |

**关键设计**:
- 模型只加载一次 —— 启动时加载清洗数据 → 拟合 BKT → 建 `QuestionPool`,全部缓存到进程内;
  请求内**绝不重新拟合**。
- 学生「当前掌握度」= 用其在数据中的**真实历史**逐条 `update_mastery` 递推得到。
- 前端左侧是**掌握度热力图**(颜色深浅=掌握程度,可滚动),右侧是推荐题 + 「答对/答错」按钮;
  点击 → `POST answer` → 热力图与推荐列表**实时刷新**;顶部下拉框切换学生。

## P6:错题讲解(RAG)(2026-10-13)

学生答错题 → **检索该知识点的讲解材料** → LLM 生成**有依据的**中文讲解。
链路:`题目 stem + skill` 构造 query → `Retriever` 检索材料 chunk → 拼进 prompt
→ LLM 生成讲解,并回传 `sources`(知识点 / 文件 / 片段 / 分数)与 `grounded` 标记。

### 材料格式与上传

- 支持 `.md` / `.txt`;
- **按 Markdown 二级标题 `## 标题` 切 chunk,标题即知识点名**(与题库 `skill` 字段对齐);
- 没有 `##` 的纯文本 → 整个文件作为 1 个 chunk,知识点名取**文件名(去扩展名)**;
- 落盘:`data/materials/<原文件名>`(保留原文)+ `data/materials_index.json`(chunk 索引)。

```bash
# 前端:顶部「讲解材料」栏选择 data/sample_materials/数学知识点讲解.md → 上传材料
# 或直接调接口:
curl -F "file=@data/sample_materials/数学知识点讲解.md" \
  http://127.0.0.1:8010/api/materials/upload
curl http://127.0.0.1:8010/api/materials     # → 5 个 chunk / 5 个知识点
```

内置样例材料 `data/sample_materials/数学知识点讲解.md` 含 5 节
(一元一次方程 / 分数运算 / 几何周长 / 概率初步 / 百分数应用),与
`data/sample_bank.csv` 的知识点**完全对齐**。

### 检索为什么默认用词法而不是向量

- **精确条件优先,词法/语义只用于排序**:推荐链路与本任务同一思路 —— 先用
  `skill` 做**知识点精确过滤**(`一元一次方程` 的题绝不命中 `分数运算` 的材料),
  再在**同一知识点内**用 `jieba` 分词 + `rank_bm25` 做词法排序。知识点是结构化精确
  条件,用向量相似度反而会串知识点。
- **零向量库、零模型下载**:`jieba` + `rank_bm25` 是纯本地轻量依赖;材料是中文短节,
  词法(尤其术语如「通分」「移项」)已足够区分。向量检索的价值在语义改写/同义表达,
  当前材料与题库术语高度一致,收益有限。
- **可插拔**:`Retriever` 是协议,`KeywordRetriever` 只是默认实现;换成
  `VectorRetriever`(先 `skill` 过滤再用 embedding 余弦排序)即可,返回结构一致,
  `Explainer` 与接口无需改动。

```bash
uv run pytest tests/test_materials.py tests/test_explainer.py -q   # P6 共 35 用例
```

## P7:向量语义检索(embedding 可插拔)(2026-10-13)

P6 的检索 = **知识点精确过滤 + BM25 词法排序** —— 材料**按知识点组织**时才有效。
材料没组织(长文 / 照片 OCR 出来的文本)时过滤失效,只能靠**语义相似度** → 这时才需要向量。

模式由 `RETRIEVAL_MODE` 决定:`keyword`(**默认**)/ `vector` / `hybrid`(RRF 融合两路)。
**embedder 未配置时 `vector` / `hybrid` 自动降级回 `keyword`** —— 不报错、不 500。

### 两种 embedding 配法(二选一)

```bash
# A. OpenAI 兼容端点(最省事)
export EMBED_PROVIDER=openai
export EMBED_BASE_URL=https://your-endpoint/v1
export EMBED_MODEL=your-embedding-model
export EMBED_API_KEY=sk-...

# B. 本地 ONNX 模型(离线;目录内需有 model.onnx + tokenizer.json)
export EMBED_PROVIDER=onnx
export EMBED_ONNX_DIR=/path/to/model-dir
```

- B 只用 `onnxruntime` + `tokenizers`,**不依赖 torch / transformers / sentence-transformers**。
- **不配任何模型**(默认)→ 照旧走词法检索,系统完全正常。
- 想只验证链路:`EMBED_PROVIDER=fake`(词袋假向量,零外部依赖)。

### 检索策略

| 材料形态 | 行为 |
|---|---|
| chunk 有知识点标签 | 先按知识点**精确过滤**,再在候选内排序 |
| chunk **无**标签(没按知识点组织) | **全库检索** ← 向量模式的价值所在 |

## 目录

```
app.py              P4 FastAPI 演示服务(启动时拟合一次,进程级缓存)
web/index.html      P4 单文件前端(零构建、零外部 CDN)
src/data.py         加载 + 清洗 + 统一 schema
src/split.py        按学生时间切分(主) / 整学生切分(冷启动对照)
src/metrics.py      AUC / ACC / Recall@k
src/baselines.py    四个预测型 baseline
src/bkt.py          贝叶斯知识追踪(每知识点 EM 拟合 L0/T/G/S)
src/recommender.py  题目池 + 4 个推荐策略
src/simulator.py    离线模拟学习者(带局限声明)
src/llm_client.py   LLM 薄封装(HttpLLM / FakeLLM,耗时 + token 估算)
src/question_gen.py 知识点上下文(RAG)+ 变式题生成
src/reviewer.py     独立评审 agent
src/materials.py    P6 讲解材料库(上传 / 按 ## 切分 / 落盘)
src/retriever.py    P6 材料检索(可插拔;jieba + rank_bm25 默认实现)
src/explainer.py    P6 检索式错题讲解(命中材料 → grounded 生成)
data/sample_materials/数学知识点讲解.md  P6 样例材料(5 节,与样例题库对齐)
scripts/            可执行脚本
tests/              pytest
```

## 测试

`uv run pytest -q` → **147 passed**(原 112 + P6 新增 35)
