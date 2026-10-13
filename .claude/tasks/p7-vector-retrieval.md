# P7 · 向量语义检索(embedding 可插拔,先不选模型)

## 背景

- P6 的检索 = BM25 词法 + 知识点精确过滤。**材料没按知识点组织**时过滤失效 → 需要**语义**检索。
- 已确认:commandcode 通道**没有 embedding 端点**(`/embeddings` → 404)。
- **模型先不选** —— 用户之后自己配。本任务只写**可插拔**的代码,并用**假 embedder** 验证逻辑。

## 需求

### P0-1 `src/embedder.py`(可插拔,**不绑定任何具体模型**)

- `Embedder` 协议:`encode(texts: list[str]) -> np.ndarray`(返回**已 L2 归一化**的向量)+ `dim: int` + `name: str`
- 两个真实实现(都从环境变量读配置,**默认不启用**):
  1. **`OpenAICompatEmbedder`** —— 任何 OpenAI 兼容 embedding 端点:
     `EMBED_BASE_URL` / `EMBED_MODEL` / `EMBED_API_KEY`
     ⚠️ 请求头**必须带 `User-Agent`**(原因见 `src/llm_client.py` 的注释:本机 WAF 拦 `Python-urllib`)
  2. **`LocalONNXEmbedder`** —— 本地 ONNX 模型目录 `EMBED_ONNX_DIR`(目录内需有 `model.onnx` + `tokenizer.json`)
     用 **`onnxruntime`**(本机已装 1.20.1)+ **`tokenizers`** 库;批推理 + mean pooling + L2 归一化。
     **不引 torch / transformers / fastembed / sentence-transformers。**
- **`FakeEmbedder`**(测试用,生产也可显式启用):
  - **必须是词袋语义向量**(按 token 计数构造);**绝不要用 sha256/哈希** —— 哈希向量语义恒为噪声、测不出召回(以前踩过)
  - 维度可配(默认 64)
- **`get_embedder() -> Embedder | None`**:按 `EMBED_PROVIDER` 选择(`openai` / `onnx` / `fake` / 未设或 `none` → `None`)
  - 任何初始化失败(**模型目录不存在 / 端点不通**)**不得抛异常** —— 记 warning 后返回 `None`

### P0-2 `src/retriever.py`:向量 + 混合

- `VectorRetriever(embedder)`:余弦相似度
- 保留 `KeywordRetriever`(BM25)
- `RETRIEVAL_MODE`:`keyword`(**默认**)| `vector` | `hybrid`
  - **embedder 为 None 时,`vector`/`hybrid` 自动降级为 `keyword`**(不报错,返回值里标明**实际生效模式**)
- **检索策略(关键)**:
  1. chunk 有 `skill` 标签 → **先按知识点精确过滤,再在候选内排序**
  2. chunk **无 `skill` 标签**(材料没组织)→ **全库检索** ← 这就是"没结构材料"的场景
- `hybrid`:RRF 融合两路(`rrf_k=60`)

### P0-3 向量缓存

- 索引 `data/materials_index.json` 增加 `embedding`(list[float])与 `embedder_name` 字段
- `embedder_name` 变化 → 重算;否则复用(**重载不重算**)

### P0-4 接口 / 前端

- `GET /api/materials` 增加 `retrieval_mode`(实际生效)+ `embedder`(名字或 null)
- 前端材料区显示当前检索模式

### P0-5 测试(≥ 12 用例,**全部用 FakeEmbedder**)

`tests/test_embedder.py` + 扩充 `tests/test_retriever.py`:

- 向量模式:相似文本召回优先(用词袋可控构造)
- skill 过滤优先:有标签时只在同知识点内排序
- **无标签 chunk 走全库检索**
- hybrid RRF 融合正确
- `embedder=None` → 自动 keyword,不报错
- 向量缓存:重载后一致;`embedder_name` 变化触发重算
- `OpenAICompatEmbedder`:monkeypatch 假响应,断言请求体/头(**含 User-Agent**)
- `LocalONNXEmbedder`:**模型目录不存在 → 返回 None 且不抛异常**(不需要真模型)

## DoD

- `uv run pytest -q` 全绿,**现有 147 个测试一个都不能红**
- 用假 embedder 起服务:`EMBED_PROVIDER=fake RETRIEVAL_MODE=vector` → `/api/materials` 显示 `retrieval_mode: vector`,
  对一份**没有 `##` 结构的纯文本材料**也能 explain 出 `grounded=True` 的结果。**(我会亲自复现)**
- **默认配置(不配任何模型)下系统完全照旧走 keyword**,不发生任何报错
- README 增加 P7:embedding 两种配法(OpenAI 兼容 / 本地 ONNX)、模式选择、回退策略

## 约束

- **不下载、不绑定任何具体模型**(禁 fastembed / sentence-transformers / transformers / torch)
- 新增依赖只允许 `tokenizers`(ONNX 路径用)
- embedder 不可用必须优雅回退,不得 500
- 不破坏 P5/P6 与全部既有测试
