"""P6:讲解生成 + 接口测试(FakeLLM + 内置样例材料,全离线)。"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as appmod
from app import AppState, app, set_state
from src.bkt import BKTParams
from src.explainer import Explainer
from src.llm_client import FakeLLM
from src.recommender import QuestionPool

MATERIAL = (
    "## 分数运算\n"
    "异分母分数相加减要先通分。例:1/2 + 1/3 = 3/6 + 2/6 = 5/6。\n"
    "\n"
    "## 一元一次方程\n"
    "移项要变号,例:2x + 3 = 7 解得 x = 2。\n"
)

FRACTION_Q = {
    "stem": "计算:1/2 + 1/3 = ?",
    "options": ["5/6", "2/5", "1/6"],
    "answer": "5/6",
    "skill": "分数运算",
}


def _chunks() -> list[dict]:
    from src.materials import split_markdown

    return [{"file": "数学.md", **c} for c in split_markdown(MATERIAL, "数学")]


# --- Explainer ------------------------------------------------------------


def test_explain_grounded_when_material_hit():
    e = Explainer(FakeLLM("你的错误在于没有先通分。"), chunks=_chunks())
    r = e.explain(FRACTION_Q, student_answer="2/5")
    assert r["grounded"] is True
    assert r["sources"]
    assert r["explanation"] == "你的错误在于没有先通分。"
    assert r["error"] is None


def test_prompt_contains_retrieved_snippet():
    """命中材料时,prompt 里必须真的含有检索到的片段文本。"""
    seen: dict[str, str] = {}

    def responder(prompt: str) -> str:
        seen["p"] = prompt
        return "讲解"

    e = Explainer(FakeLLM(responder), chunks=_chunks())
    e.explain(FRACTION_Q, student_answer="2/5")
    prompt = seen["p"]
    assert "3/6 + 2/6 = 5/6" in prompt
    assert "分数运算" in prompt
    assert "1/2 + 1/3" in prompt


def test_explain_sources_shape():
    e = Explainer(FakeLLM("x"), chunks=_chunks())
    r = e.explain(FRACTION_Q)
    src = r["sources"][0]
    assert set(src) == {"file", "skill", "snippet", "score"}
    assert src["skill"] == "分数运算"
    assert src["file"] == "数学.md"


def test_explain_ungrounded_without_material():
    e = Explainer(FakeLLM("纯生成讲解"), chunks=[])
    r = e.explain(FRACTION_Q, student_answer="2/5")
    assert r["grounded"] is False
    assert r["sources"] == []
    assert r["explanation"] == "纯生成讲解"


def test_explain_skill_filter_prevents_cross_skill():
    """分数运算的题不应命中一元一次方程的材料。"""
    e = Explainer(FakeLLM("x"), chunks=_chunks())
    r = e.explain(FRACTION_Q)
    assert all(s["skill"] == "分数运算" for s in r["sources"])


def test_explain_llm_error_degrades():
    def boom(_prompt: str) -> str:
        raise RuntimeError("LLM 挂了")

    e = Explainer(FakeLLM(boom), chunks=_chunks())
    r = e.explain(FRACTION_Q)
    assert r["explanation"] is None
    assert r["error"] is not None
    assert r["grounded"] is True  # 材料命中仍记录来源


def test_explain_no_llm_returns_error():
    e = Explainer(None, chunks=_chunks())
    r = e.explain(FRACTION_Q)
    assert r["explanation"] is None
    assert r["error"] is not None


def test_explain_empty_llm_text_returns_error():
    e = Explainer(FakeLLM("   "), chunks=_chunks())
    r = e.explain(FRACTION_Q)
    assert r["explanation"] is None
    assert r["error"] is not None


# --- 接口 -----------------------------------------------------------------


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [[1, 1, 10, 100, 1], [2, 1, 10, 101, 0]],
        columns=["order_id", "user_id", "skill_id", "problem_id", "correct"],
    )


def _state() -> AppState:
    pool = QuestionPool.from_frame(_frame())
    return AppState(
        pool=pool,
        params={10: BKTParams(l0=0.3, t=0.2, g=0.2, s=0.1)},
        skill_names={10: "分数运算"},
        history={1: [(10, 1)]},
        student_ids=[1],
    )


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "MATERIALS_DIR", tmp_path / "materials")
    monkeypatch.setattr(appmod, "MATERIALS_INDEX", tmp_path / "materials_index.json")
    st = _state()
    set_state(st)
    yield TestClient(app)
    set_state(None)  # type: ignore[arg-type]


def _upload_material(client: TestClient, text: str = MATERIAL, name: str = "数学.md"):
    return client.post(
        "/api/materials/upload",
        files={"file": (name, text.encode("utf-8"), "text/markdown")},
    )


def test_upload_material_returns_summary(client):
    r = _upload_material(client)
    assert r.status_code == 200
    body = r.json()
    assert body["n_chunks"] == 2
    assert set(body["skills"]) == {"分数运算", "一元一次方程"}
    assert body["files"] == ["数学.md"]


def test_materials_info_after_upload(client):
    _upload_material(client)
    info = client.get("/api/materials").json()
    assert info["n_chunks"] == 2


def test_materials_info_empty(client):
    assert client.get("/api/materials").json()["n_chunks"] == 0


def test_upload_bad_file_returns_400(client):
    r = client.post(
        "/api/materials/upload",
        files={"file": ("x.pdf", b"%PDF", "application/pdf")},
    )
    assert r.status_code == 400


def test_explain_endpoint_returns_sources(client):
    _upload_material(client)
    st = appmod.get_state()
    st.explainer = Explainer(FakeLLM("先通分再相加。"), chunks=st.material_store().chunks)
    r = client.post("/api/students/1/explain", json={"problem_id": 100})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True
    assert body["sources"]
    assert body["explanation"] == "先通分再相加。"
    assert body["skill"] == "分数运算"


def test_explain_endpoint_ungrounded_when_no_material(client):
    st = appmod.get_state()
    st.explainer = Explainer(FakeLLM("纯生成"), chunks=[])
    r = client.post("/api/students/1/explain", json={"problem_id": 100})
    assert r.status_code == 200
    assert r.json()["grounded"] is False
    assert r.json()["sources"] == []


def test_explain_endpoint_unknown_problem_404(client):
    _upload_material(client)
    r = client.post("/api/students/1/explain", json={"problem_id": 99999})
    assert r.status_code == 404


def test_explain_endpoint_unknown_student_404(client):
    _upload_material(client)
    r = client.post("/api/students/999/explain", json={"problem_id": 100})
    assert r.status_code == 404
