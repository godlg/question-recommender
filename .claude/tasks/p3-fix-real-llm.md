# P3-FIX · 打通真实 LLM 路径 + 改造 demo

## 根因(已实测确认,别重复排查)

`HttpLLM._invoke` 用 urllib 发请求,默认 `User-Agent: Python-urllib/3.x`,
被 `api.commandcode.ai` 的 WAF 拦掉 → **HTTP 403 Forbidden**。

实测(同一 key / 同一 endpoint / 同一 body):
```
urllib 默认 UA      → HTTP 403
User-Agent: Mozilla → HTTP 200
无 proxy 时同样如此(与 proxy 无关)
```
**这是上一轮 P3 默认退回「离线确定性引擎」的唯一原因。** 加上 UA 后真实链路已验证可用
(实测生成 2 道中文数学题 + 评审 pass,tokens=308,7.3s)。

## 需求

### P0-1 修 `src/llm_client.py`

- 请求头加 `User-Agent`,默认 `Mozilla/5.0 (compatible; question-recommender/0.1)`,
  并支持构造参数 `user_agent=` 覆盖。
- 新增测试:断言发出的请求头里含该 UA(可 monkeypatch `urllib.request.urlopen` 捕获 Request)。

### P0-2 改造 `scripts/p3_demo.py`

1. **默认走真实 LLM**(当 `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` 齐全时);
   只有 env 缺失时才回退离线引擎,且必须**明确打印警告**,说明"当前未调用真实 LLM"。
2. 规模由 CLI 参数控制:`--skills K --per-skill N`(默认取小值,如 `1 × 6`,避免跑太久)。
3. 生成之后**注入已知错误**:按 `--inject-every E`(默认 3)周期性把某题的 `answer`
   改成 `options` 中的另一个值(真正的错误答案),并记录注入清单。
4. 用评审器独立检测,报告必须**区分**三件事:
   - **注入检出率**:被注入的题里有多少被判 fail(检测召回)
   - **LLM 自身质量问题数**:没被注入却被判 fail 的题(LLM 自己出的问题)
   - 拦截样例(至少 3 条)
5. `p3_rejected_examples.json` 里写的是**真实 LLM 生成**的题。

## DoD

- `uv run pytest -q` 全绿,现有 **65 个测试一个都不能红**。
- `LLM_*` 配好时,`uv run python scripts/p3_demo.py` 的统计里 **token > 0 且耗时 > 0**
  (证明真的调用了 LLM,而不是离线引擎)。
- 报告里明确区分「注入错误检出率」与「LLM 自身质量问题」。
- env 齐全时不再出现"离线确定性引擎"标签。

## 约束

- 不引入新重依赖。
- 现有测试与既有模块接口尽量不动(只加 UA + 改 demo)。
