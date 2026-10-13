# P6 · RAG:错题讲解

## 目标

学生答错题 → **检索该知识点的讲解材料** → LLM 生成**有依据的**针对性讲解。
(真实资料暂不上传;代码与验真用内置样例材料完成)

## 现状(动手前必读)

- 仓库根 `D:\question-recommender`。已有 `app.py`(P5:题库上传 + 推荐)、
  `src/{bank,bkt,recommender,llm_client,question_gen,reviewer}.py`。**当前 112 passed**。
- `src/llm_client.py` 已有 `HttpLLM`(读 `LLM_BASE_URL/LLM_MODEL/LLM_API_KEY`,**已带 User-Agent**)和 `FakeLLM`(可注入)。
- 题库里的知识点名是 `skill` 字段(如 `一元一次方程`);讲解材料的标题要与它对齐。
- 环境:uv;联网命令带 `UV_SYSTEM_CERTS=1`。
- ⚠️ 本机 `api.commandcode.ai` 会拦 `Python-urllib` 的 UA → `HttpLLM` 已修,不要动这段。

## 需求

### P0-1 材料库 + 切分 —— `src/materials.py`

- 支持上传 `.md` / `.txt`。
- **切分规则**:按 Markdown 二级标题 `## 标题` 切 chunk,**标题即知识点名**(与题库 `skill` 对齐);
  没有 `##` 的纯文本 → 整个文件作为一个 chunk,知识点名取**文件名(去扩展名)**。
- 落盘:`data/materials/<原文件名>`(保留原文)+ `data/materials_index.json`(chunk 索引)。
- `MaterialStore`:`add_from_bytes(name, raw)` / `load_all()` / `chunks` / `skills()` / `save()`。

### P0-2 检索 —— `src/retriever.py`(**可插拔**)

- 定义 `Retriever` 协议:`search(query: str, chunks: list, top_k: int, skill: str | None) -> list[dict]`
- **默认实现 `KeywordRetriever`**:`jieba` 分词 + `rank_bm25`
  - `skill` 非空时**先按知识点精确过滤**,再在候选内做词法排序(与推荐链路同一思路:精确条件优先,词法/语义只用于同知识点内的排序)
  - 返回 `[{file, skill, text, score}]`
- 接口要能直接换成向量实现(留注释说明扩展点即可,**本任务不实现向量**)。

### P0-3 讲解生成 —— `src/explainer.py`

- `Explainer(llm, retriever)`:`explain(question: dict, student_answer: str | None, top_k: int = 3) -> dict`
  - 用 `question["stem"]` + `question["skill"]` 构造 query → 检索 → 拼 prompt → LLM 生成中文讲解
  - 返回 `{"explanation": str|None, "sources": [{file, skill, snippet, score}], "grounded": bool, "error": str|None}`
  - **检索为空**:降级为纯生成,`sources=[]`、`grounded=False`
  - **LLM 报错/未配置**:`explanation=None`、`error=...`,**不抛异常**

### P0-4 接口 + 前端

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/materials/upload` | multipart 上传 md/txt → 入库 → `{n_chunks, skills, files}` |
| GET | `/api/materials` | `{n_chunks, skills, files}` |
| POST | `/api/students/{uid}/explain` | body `{"problem_id": int, "student_answer": str\|None}` → `{explanation, sources, grounded, error}` |

- 前端 `web/index.html`:每道推荐题加**「看讲解」**按钮 → 展示讲解文本 + **引用来源**(知识点 / 文件名 / 片段)。
- 上传区增加"讲解材料"上传入口。

### P0-5 样例材料(必须附)

生成 `data/sample_materials/数学知识点讲解.md`:含 `## 一元一次方程`、`## 分数运算`、`## 几何周长`、
`## 概率初步`、`## 百分数应用` 五节(与 `data/sample_bank.csv` 的知识点**完全对齐**),
每节 150~300 字中文讲解,**内容必须数学正确**。

### P0-6 测试

`tests/test_materials.py` + `tests/test_explainer.py`,**≥ 12 用例**,全部用 `FakeLLM` + 内置样例材料:

- 按 `##` 切分正确 / 无 `##` 时按文件名 / 落盘后重载一致
- 检索:同知识点优先;`skill` 过滤生效;空材料返回空
- explain:命中材料时 `grounded=True` 且 **prompt 里确实含检索到的片段**;无材料时 `grounded=False`
- LLM 抛错 → 优雅降级(不抛异常)
- `/api/materials/upload` 传坏文件 → 400

## DoD

- `uv run pytest -q` 全绿,**现有 112 个测试一个都不能红**。
- 真起服务 + **真 LLM**:上传 `data/sample_materials/数学知识点讲解.md` →
  `GET /api/materials` 显示 5 个 chunk → `POST /explain`(拿题库里一道分数运算的题)
  → 返回**中文讲解**且 `sources` 非空、`grounded=True`。**(我会亲自复现)**
- README 增加 P6 章节:材料格式 + 上传方式 + "检索为什么默认用词法而不是向量"。

## 约束

- **材料为空也要能跑**(降级路径必须有测试)。
- 依赖只加 `jieba`、`rank_bm25`(轻量);不引入向量库。
- 不破坏 P5 链路与全部既有测试。
