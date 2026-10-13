# P7 续 · 只剩三件事(embedder/retriever 已写完,**别重写**)

## 现状(先读,别重复劳动)

上一轮被强杀,但下列内容**已经写完,且现有 147 个测试全绿**:

- `src/embedder.py`(263 行):`Embedder` 协议 + `l2_normalize` + `FakeEmbedder`(**词袋**,非哈希) +
  `OpenAICompatEmbedder`(带 `User-Agent`)+ `LocalONNXEmbedder` + `get_embedder()`(失败返回 `None`)
- `src/retriever.py`:`KeywordRetriever` / `VectorRetriever` / `HybridRetriever` +
  `rrf_fuse` / `filter_by_skill` / `resolve_mode` / `MODES`
- `pyproject.toml` 已加 `tokenizers`

**不要重写这两个模块的实现。** 若确实发现 bug,可以修,但必须在报告里单独说明。

## 剩余任务

### 1. 新建 `tests/test_embedder.py`(≥ 8 用例)

- `FakeEmbedder`:共享 token 多的文本余弦更高;**维度过小**不崩;`dim <= 0` 抛 `EmbedderError`
- `l2_normalize`:零向量不产生 NaN
- `get_embedder()`:`EMBED_PROVIDER` 未设 / `none` → `None`;`fake` → `FakeEmbedder`;
  `onnx` 且 `EMBED_ONNX_DIR` 不存在 → **返回 None 且不抛异常**;`openai` 缺变量 → `None`
- `OpenAICompatEmbedder`:monkeypatch `urlopen`,断言请求体含 `input` / `model`,
  且**请求头里是 `User-Agent: Mozilla...` 而非 urllib 默认**(回归防护)

### 2. 新建 `tests/test_retriever.py`(≥ 10 用例)

- `filter_by_skill`:指定 skill 时只留同知识点 chunk;无标签 chunk 会被滤掉
- `VectorRetriever` + `FakeEmbedder`:相似文本排前
- `resolve_mode`:`RETRIEVAL_MODE=vector` 但 embedder 为 `None` → **降级为 `keyword`**
- `HybridRetriever`:RRF 融合后,两路都命中的候选排最前
- **无 `skill` 标签的 chunk 能通过全库检索被召回**(这是"材料没结构"的核心场景)

### 3. 向量缓存 + 接线

- **`src/materials.py`**:落盘时把 `embedding`(list[float])与 `embedder_name` 写进 `data/materials_index.json`;
  `embedder_name` 变化或字段缺失 → 重算;否则复用(**重载不重算**);无 embedder 时不写这两个字段
- **`app.py`**:
  - 启动时 `get_embedder()` + `resolve_mode()`,存入 `AppState`
  - `_ensure_explainer` 使用选定的 retriever(默认 keyword 行为不变)
  - `GET /api/materials` 增加 `retrieval_mode`(实际生效)与 `embedder`(名字或 null)
- **`web/index.html`**:材料区显示当前检索模式 + embedder 名(未配置时显示"词法模式(未配向量)")
- **README** 增加 P7 段:两种 embedding 配法(OpenAI 兼容 / 本地 ONNX 目录)、模式怎么选、回退策略

## DoD

- `uv run pytest -q` 全绿,**现有 147 个测试一个都不能红**
- **默认配置**(`EMBED_PROVIDER` 未设)系统行为完全照旧
- `EMBED_PROVIDER=fake RETRIEVAL_MODE=vector` 起服务 → `GET /api/materials` 显示 `retrieval_mode: "vector"`;
  对一份**没有 `##` 结构的纯文本材料**能 explain 出 `grounded=True`。**(我会亲自复现)**

## 约束

- **绝不下载/安装任何 embedding 模型**(禁 fastembed / sentence-transformers / transformers / torch)
- 新增依赖只允许 `tokenizers`
- 不得破坏 P5/P6 与全部既有测试
