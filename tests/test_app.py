"""P4:闭环演示服务测试。

用 FastAPI TestClient + 注入最小模型,不跑 80 秒全量拟合。
"""
from __future__ import annotations

import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as appmod
from app import AppState, app, set_state
from src.bkt import BKTParams
from src.data import KEEP_COLS
from src.recommender import QuestionPool


def _frame() -> pd.DataFrame:
    # 学生 1/2 在 skill 10、11 上都有历史
    rows = [
        [1, 1, 10, 100, 1],
        [2, 1, 10, 100, 1],
        [3, 1, 10, 101, 1],
        [4, 1, 11, 200, 0],
        [5, 1, 11, 201, 0],
        [6, 2, 10, 100, 1],
        [7, 2, 10, 101, 0],
        [8, 2, 11, 200, 1],
    ]
    return pd.DataFrame(rows, columns=KEEP_COLS)


def _state() -> AppState:
    pool = QuestionPool.from_frame(_frame())
    params = {
        10: BKTParams(l0=0.3, t=0.2, g=0.2, s=0.1),
        11: BKTParams(l0=0.5, t=0.2, g=0.2, s=0.1),
    }
    history = {
        1: [(10, 1), (10, 1), (11, 0)],
        2: [(10, 1), (11, 1)],
    }
    return AppState(
        pool=pool,
        params=params,
        skill_names={10: "Box and Whisker", 11: "Circle Graph"},
        history=history,
        student_ids=[1, 2],
    )


@pytest.fixture()
def client() -> TestClient:
    set_state(_state())
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    set_state(None)  # type: ignore[arg-type]


def test_index_returns_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "掌握度" in r.text


def test_list_students(client):
    r = client.get("/api/students?limit=20")
    assert r.status_code == 200
    data = r.json()
    assert [s["user_id"] for s in data["students"]] == [1, 2]
    assert data["total"] == 2


def test_student_state_has_mastery(client):
    r = client.get("/api/students/1/state")
    assert r.status_code == 200
    data = r.json()
    assert {s["skill_id"] for s in data["skills"]} == {10, 11}
    assert all(0.0 <= s["mastery"] <= 1.0 for s in data["skills"])
    assert "mean_mastery" in data["overview"]
    assert "n_weak" in data["overview"]


def test_student_state_show_real_skill_names(client):
    """热力图知识点名必须是真实名,不能是 skill_<id> 占位。"""
    data = client.get("/api/students/1/state").json()
    names = {s["skill_id"]: s["skill_name"] for s in data["skills"]}
    assert names == {10: "Box and Whisker", 11: "Circle Graph"}
    assert all(not s["skill_name"].startswith("skill_") for s in data["skills"])


def test_student_state_unknown_uid_404(client):
    assert client.get("/api/students/999/state").status_code == 404


def test_recommend_show_real_skill_names(client):
    recs = client.get("/api/students/1/recommend?k=5").json()["recommendations"]
    assert recs
    for rec in recs:
        assert not rec["skill_name"].startswith("skill_")


def test_skill_names_fallback_when_missing():
    """映射里查不到时才回退 f"skill_{id}"。"""
    st = _state()
    st.skill_names = {10: "Box and Whisker"}
    set_state(st)
    data = TestClient(app).get("/api/students/1/state").json()
    names = {s["skill_id"]: s["skill_name"] for s in data["skills"]}
    assert names[10] == "Box and Whisker"
    assert names[11] == "skill_11"


def test_load_skill_names_picks_most_common(tmp_path):
    """同一 skill_id 多名字时取出现次数最多的那个。"""
    csv = tmp_path / "skill_builder_data_corrected.csv"
    csv.write_text(
        "skill_id,skill_name\n"
        "1,Alpha\n"
        "1,Alpha\n"
        "1,Beta\n"
        "2,Only\n",
        encoding="ISO-8859-15",
    )
    names = appmod.load_skill_names(tmp_path)
    assert names == {1: "Alpha", 2: "Only"}


def test_recommend_returns_k(client):
    r = client.get("/api/students/1/recommend?k=5")
    assert r.status_code == 200
    recs = r.json()["recommendations"]
    assert len(recs) >= 1
    for rec in recs:
        assert set(rec) >= {"problem_id", "skill_id", "skill_name", "difficulty"}


def test_recommend_k_one(client):
    recs = client.get("/api/students/1/recommend?k=1").json()["recommendations"]
    assert len(recs) == 1


def test_answer_updates_mastery(client):
    before = {
        s["skill_id"]: s["mastery"]
        for s in client.get("/api/students/1/state").json()["skills"]
    }
    # 题 200 属于 skill 11;答错应改变该知识点掌握度
    r = client.post("/api/students/1/answer", json={"problem_id": 200, "correct": False})
    assert r.status_code == 200
    after = {
        s["skill_id"]: s["mastery"] for s in r.json()["state"]["skills"]
    }
    assert after[11] != before[11]


def test_answer_returns_new_recommendations(client):
    r = client.post("/api/students/1/answer", json={"problem_id": 100, "correct": True})
    assert r.status_code == 200
    body = r.json()
    assert "state" in body
    assert "recommendations" in body
    assert len(body["recommendations"]) >= 1


def test_answer_unknown_problem_404(client):
    r = client.post("/api/students/1/answer", json={"problem_id": 99999, "correct": True})
    assert r.status_code == 404


def test_overview_n_weak_counts_below_half(client):
    st = _state()
    # 学生 1 的 skill 11 初始 l0=0.5,答错后应 < 0.5
    st.mastery[1] = {10: 0.9, 11: 0.1}
    set_state(st)
    data = TestClient(app).get("/api/students/1/state").json()
    assert data["overview"]["n_weak"] == 1


def test_build_state_fast_mode(monkeypatch):
    """P4_FAST=1 时能在小子集上完成一次真实启动(不跑全量拟合)。"""
    monkeypatch.setenv("P4_FAST", "1")
    st = appmod.build_state()
    assert len(st.params) > 0
    assert len(st.student_ids) > 0
    assert len(st.pool.problem_id) > 0
