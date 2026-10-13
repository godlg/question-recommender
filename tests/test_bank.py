"""P5:题库上传 + 按学生推荐测试。"""
from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as appmod
from app import AppState, app, set_state
from src.bank import (
    DEFAULT_PARAMS,
    Bank,
    BankError,
    load_bank,
    parse_frame,
    parse_upload,
    save_bank,
)
from src.bkt import BKTParams
from src.recommender import QuestionPool

CSV = (
    "stem,options,answer,skill,difficulty\n"
    "1+1=?,1|2|3,2,加法,0.9\n"
    "2+2=?,2|3|4,4,加法,0.8\n"
    "3×3=?,6|9|12,9,乘法,0.6\n"
)


def _csv_bytes(text: str, encoding: str = "utf-8") -> bytes:
    return text.encode(encoding)


def _state() -> AppState:
    """最小 AppState(ASSISTments 路径),用于上传测试。"""
    frame = pd.DataFrame(
        [[1, 1, 10, 100, 1], [2, 1, 10, 101, 0]],
        columns=["order_id", "user_id", "skill_id", "problem_id", "correct"],
    )
    pool = QuestionPool.from_frame(frame)
    return AppState(
        pool=pool,
        params={10: BKTParams(l0=0.3, t=0.2, g=0.2, s=0.1)},
        skill_names={10: "原始技能"},
        history={1: [(10, 1)]},
        student_ids=[1],
    )


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch, tmp_path):
    monkeypatch.setattr(appmod, "BANK_PATH", tmp_path / "bank.jsonl")
    set_state(_state())
    yield
    set_state(None)  # type: ignore[arg-type]


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _upload(client: TestClient, content: bytes, filename: str = "bank.csv"):
    return client.post(
        "/api/bank/upload",
        files={"file": (filename, content, "text/csv")},
    )


# --- 解析层 ---------------------------------------------------------------


def test_parse_csv_generates_sequential_ids():
    bank = parse_frame(pd.read_csv(io.StringIO(CSV)))
    assert bank.n_questions == 3
    assert [q["question_id"] for q in bank.questions] == [1, 2, 3]
    assert bank.n_skills == 2
    assert bank.skill_names == {1: "加法", 2: "乘法"}
    assert bank.questions[0]["options"] == ["1", "2", "3"]


def test_parse_missing_column_raises():
    with pytest.raises(BankError) as exc:
        parse_frame(pd.DataFrame({"stem": ["a"], "answer": ["a"], "skill": ["s"]}))
    assert "options" in str(exc.value)
    assert "difficulty" in str(exc.value)


def test_parse_counts_bad_rows_and_keeps_good():
    bad = (
        "stem,options,answer,skill,difficulty\n"
        "1+1=?,1|2,2,加法,0.9\n"          # good
        "缺答案,1|2|3,,加法,0.5\n"        # bad: answer 空
        "答案不在选项,1|2|3,9,加法,0.5\n"  # bad: answer 不属于 options
    )
    bank = parse_frame(pd.read_csv(io.StringIO(bad)))
    assert bank.n_questions == 1
    assert bank.n_skipped == 2
    assert bank.questions[0]["question_id"] == 1


def test_parse_options_empty_for_fill_blank():
    bank = parse_frame(
        pd.DataFrame(
            {
                "stem": ["填空"],
                "options": [""],
                "answer": ["42"],
                "skill": ["算术"],
                "difficulty": [None],
            }
        )
    )
    assert bank.questions[0]["options"] == []
    assert bank.questions[0]["difficulty"] == 0.5


def test_parse_column_names_case_insensitive():
    bank = parse_frame(
        pd.DataFrame(
            {
                "STEM": ["s"],
                "Answer": ["a"],
                "SKILL": ["k"],
                "Options": ["a|b"],
                "Difficulty": [0.4],
            }
        )
    )
    assert bank.n_questions == 1


def test_parse_upload_excel():
    df = pd.DataFrame(
        {
            "stem": ["x+1=2"],
            "options": ["x=1|x=2"],
            "answer": ["x=1"],
            "skill": ["方程"],
            "difficulty": [0.7],
        }
    )
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    bank = parse_upload("q.xlsx", buf.getvalue())
    assert bank.n_questions == 1
    assert bank.source_name == "q.xlsx"


def test_upload_rejects_empty_file():
    with pytest.raises(BankError):
        parse_upload("q.csv", b"")


def test_save_and_load_roundtrip(tmp_path):
    bank = parse_frame(pd.read_csv(io.StringIO(CSV)), source_name="my.csv")
    path = save_bank(bank, tmp_path / "bank.jsonl")
    reloaded = load_bank(path)
    assert reloaded is not None
    assert reloaded.n_questions == bank.n_questions
    assert reloaded.source_name == "my.csv"
    assert reloaded.skill_names == bank.skill_names


def test_load_bank_missing_returns_none(tmp_path):
    assert load_bank(tmp_path / "nope.jsonl") is None


# --- 服务层 ---------------------------------------------------------------


def test_upload_then_bank_info(client):
    r = _upload(client, _csv_bytes(CSV))
    assert r.status_code == 200
    body = r.json()
    assert body["n_questions"] == 3
    assert body["n_skills"] == 2
    assert body["n_skipped"] == 0
    assert body["source_name"] == "bank.csv"

    info = client.get("/api/bank").json()
    assert info == {"n_questions": 3, "n_skills": 2, "source_name": "bank.csv"}


def test_bank_info_empty_when_not_uploaded(client):
    assert client.get("/api/bank").json()["n_questions"] == 0


def test_upload_missing_column_returns_400(client):
    bad = "stem,answer,skill\n1+1,2,加法\n"
    r = _upload(client, _csv_bytes(bad))
    assert r.status_code == 400
    assert "options" in r.json()["detail"]


def test_bad_upload_does_not_destroy_existing_bank(client):
    assert _upload(client, _csv_bytes(CSV)).status_code == 200
    assert client.get("/api/bank").json()["n_questions"] == 3

    r = _upload(client, b"stem,answer,skill\nx,y,z\n")
    assert r.status_code == 400
    # 旧题库仍在
    assert client.get("/api/bank").json()["n_questions"] == 3


def test_bank_sample_returns_stems(client):
    _upload(client, _csv_bytes(CSV))
    data = client.get("/api/bank/sample?n=2").json()
    assert len(data["questions"]) == 2
    q = data["questions"][0]
    assert set(q) == {"question_id", "stem", "options", "answer", "skill", "difficulty"}


def test_upload_persists_to_disk(client, tmp_path):
    _upload(client, _csv_bytes(CSV))
    path = appmod.BANK_PATH
    assert path.exists()
    reloaded = load_bank(path)
    assert reloaded is not None and reloaded.n_questions == 3


def test_create_student_default_mastery(client):
    _upload(client, _csv_bytes(CSV))
    r = client.post("/api/students", json={"name": "小明"})
    assert r.status_code == 200
    uid = r.json()["user_id"]

    state = client.get(f"/api/students/{uid}/state").json()
    # 所有知识点掌握度 = L0 = 0.3
    assert state["skills"]
    for s in state["skills"]:
        assert s["mastery"] == pytest.approx(DEFAULT_PARAMS.l0)
    assert state["overview"]["n_weak"] == len(state["skills"])


def test_create_student_appears_in_list(client):
    _upload(client, _csv_bytes(CSV))
    uid = client.post("/api/students", json={}).json()["user_id"]
    ids = [s["user_id"] for s in client.get("/api/students").json()["students"]]
    assert uid in ids


def test_recommend_after_upload_uses_bank_and_has_stem(client):
    _upload(client, _csv_bytes(CSV))
    uid = client.post("/api/students", json={}).json()["user_id"]
    recs = client.get(f"/api/students/{uid}/recommend?k=2").json()["recommendations"]
    assert recs
    for rec in recs:
        assert rec["stem"]  # 题干来自题库
        assert rec["options"]
        assert rec["answer"] in rec["options"]
        assert rec["skill_name"] in {"加法", "乘法"}
        assert 1 <= rec["question_id"] <= 3


def test_answer_updates_mastery_in_bank_mode(client):
    _upload(client, _csv_bytes(CSV))
    uid = client.post("/api/students", json={}).json()["user_id"]
    recs = client.get(f"/api/students/{uid}/recommend?k=1").json()["recommendations"]
    qid = recs[0]["question_id"]

    before = {
        s["skill_id"]: s["mastery"]
        for s in client.get(f"/api/students/{uid}/state").json()["skills"]
    }
    r = client.post(
        f"/api/students/{uid}/answer", json={"problem_id": qid, "correct": True}
    )
    assert r.status_code == 200
    after = {s["skill_id"]: s["mastery"] for s in r.json()["state"]["skills"]}
    assert after != before


def test_upload_bank_does_not_touch_assistments_recommend(client):
    """未上传题库时,推荐仍走 ASSISTments 路径(向后兼容)。"""
    recs = client.get("/api/students/1/recommend?k=1").json()["recommendations"]
    assert recs
    assert recs[0]["stem"] is None  # ASSISTments 无题干
    assert recs[0]["problem_id"] in {100, 101}
