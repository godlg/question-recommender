"""LLM 调用的薄封装(不引入重依赖,只用标准库 urllib)。

- ``HttpLLM`` : OpenAI 兼容的 ``/chat/completions``。配置来自环境变量
  ``LLM_BASE_URL`` / ``LLM_MODEL`` / ``LLM_API_KEY``,任一缺失即抛清晰异常。
- ``FakeLLM`` : 可注入的假客户端,让测试与离线 demo 完全不依赖真实 API。

每次调用都会记录耗时与 token 估算(供 P1 汇总打印)。
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

ENV_KEYS = ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY")

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; question-recommender/0.1)"


class LLMConfigError(RuntimeError):
    """缺少 LLM 配置(环境变量)时抛出。"""


class LLMError(RuntimeError):
    """LLM 调用失败(网络 / 协议 / 返回格式)时抛出。"""


def estimate_tokens(text: str) -> int:
    """粗略 token 估算:CJK 字符按 1 个 token,其余按 4 字符 1 个 token。"""
    if not text:
        return 0
    cjk = sum(
        1
        for ch in text
        if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" or "\uac00" <= ch <= "\ud7a3"
    )
    other = len(text) - cjk
    return cjk + max(1, round(other / 4)) if other else cjk


def strip_code_fence(text: str) -> str:
    """去掉模型可能多写的 Markdown 代码块围栏。真正的解析仍交给 json.loads。"""
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


@dataclass
class LLMResponse:
    text: str
    elapsed_s: float
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    def complete(self, prompt: str) -> LLMResponse: ...


class _TrackedClient:
    """公共基类:把每次调用的耗时 / token 估算记到 ``calls`` 里。"""

    def __init__(self) -> None:
        self.calls: list[LLMResponse] = []

    def _invoke(self, prompt: str) -> str:  # pragma: no cover - 抽象
        raise NotImplementedError

    def complete(self, prompt: str) -> LLMResponse:
        t0 = time.perf_counter()
        text = self._invoke(prompt)
        resp = LLMResponse(
            text=text,
            elapsed_s=time.perf_counter() - t0,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(text),
        )
        self.calls.append(resp)
        return resp

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def total_seconds(self) -> float:
        return sum(c.elapsed_s for c in self.calls)

    def reset_stats(self) -> None:
        self.calls.clear()


class FakeLLM(_TrackedClient):
    """可注入的假客户端。

    ``responder`` 可以是:
      - 字符串:每次返回同一段文本
      - 字符串序列:按调用顺序返回,用尽后停在最后一个
      - 可调用对象 ``f(prompt) -> str``:按输入动态生成(离线 demo 常用)
    """

    def __init__(self, responder: str | Sequence[str] | Callable[[str], str] = "{}") -> None:
        super().__init__()
        self._responder = responder
        self._i = 0

    def _invoke(self, prompt: str) -> str:
        r = self._responder
        if callable(r):
            return r(prompt)
        if isinstance(r, str):
            return r
        idx = min(self._i, len(r) - 1)
        self._i += 1
        return r[idx]


class HttpLLM(_TrackedClient):
    """OpenAI 兼容 ``/chat/completions`` 的极简客户端(标准库 urllib)。"""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.7,
        user_agent: str | None = None,
    ) -> None:
        super().__init__()
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        missing = [
            name
            for name, value in (
                ("LLM_BASE_URL", self.base_url),
                ("LLM_MODEL", self.model),
                ("LLM_API_KEY", self.api_key),
            )
            if not value
        ]
        if missing:
            raise LLMConfigError(
                "缺少 LLM 配置: " + ", ".join(missing) + " —— 请设置对应环境变量后再试"
            )
        self.timeout = timeout
        self.temperature = temperature
        if not self.base_url.endswith("/chat/completions"):
            self.base_url = f"{self.base_url}/chat/completions"

    def _invoke(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": self.user_agent,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - 统一包成 LLMError
            raise LLMError(f"LLM 调用失败: {exc}") from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"LLM 返回格式异常: {body!r}") from exc
