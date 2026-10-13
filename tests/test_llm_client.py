import json

import pytest

from src.llm_client import (
    DEFAULT_USER_AGENT,
    ENV_KEYS,
    FakeLLM,
    HttpLLM,
    LLMConfigError,
    estimate_tokens,
    strip_code_fence,
)


class _FakeHTTPResponse:
    """最小可用的假响应:支持上下文管理器与 ``read()``。"""

    def __init__(self, body: dict) -> None:
        self._body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def _capture_request(monkeypatch) -> dict:
    """monkeypatch ``urlopen``,返回一个会填入最后一次 Request 的容器。"""
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["request"] = req
        return _FakeHTTPResponse(
            {"choices": [{"message": {"content": "ok"}}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return captured


def test_http_llm_raises_on_missing_config(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(LLMConfigError):
        HttpLLM()


def test_http_llm_config_from_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LLM_MODEL", "demo-model")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    client = HttpLLM()
    assert client.model == "demo-model"
    assert client.base_url == "https://api.example.com/v1/chat/completions"


def test_http_llm_appends_completions_path_only_once(monkeypatch):
    client = HttpLLM(
        base_url="https://api.example.com/v1/chat/completions",
        model="m",
        api_key="k",
    )
    assert client.base_url == "https://api.example.com/v1/chat/completions"


def test_http_llm_sends_default_user_agent(monkeypatch):
    captured = _capture_request(monkeypatch)
    client = HttpLLM(base_url="https://api.example.com/v1", model="m", api_key="k")
    client.complete("hi")
    req = captured["request"]
    assert req.get_header("User-agent") == DEFAULT_USER_AGENT


def test_http_llm_user_agent_override(monkeypatch):
    captured = _capture_request(monkeypatch)
    client = HttpLLM(
        base_url="https://api.example.com/v1",
        model="m",
        api_key="k",
        user_agent="CustomAgent/9.9",
    )
    client.complete("hi")
    assert captured["request"].get_header("User-agent") == "CustomAgent/9.9"


def test_fake_llm_records_stats():
    fake = FakeLLM("hello world, this is a response")
    resp = fake.complete("some prompt text")
    assert resp.text.startswith("hello")
    assert len(fake.calls) == 1
    assert fake.total_tokens > 0
    assert fake.total_seconds >= 0.0


def test_fake_llm_sequence_stops_at_last():
    fake = FakeLLM(["a", "b"])
    assert [fake.complete("p").text for _ in range(4)] == ["a", "b", "b", "b"]


def test_fake_llm_callable_uses_prompt():
    fake = FakeLLM(lambda prompt: prompt.upper())
    assert fake.complete("abc").text == "ABC"


def test_estimate_tokens_handles_cjk_and_ascii():
    assert estimate_tokens("") == 0
    assert estimate_tokens("你好世界") == 4
    assert estimate_tokens("a" * 40) == 10


def test_strip_code_fence():
    fenced = "```json\n{\"a\": 1}\n```"
    assert strip_code_fence(fenced) == '{"a": 1}'
    assert strip_code_fence('{"a": 1}') == '{"a": 1}'
