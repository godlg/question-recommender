"""P4:闭环演示服务(做题 → 掌握度 → 推荐)。

启动时**执行一次**重活:加载清洗数据 → 拟合 BKT → 建 QuestionPool,
全部缓存到进程内;请求内绝不重新拟合。

端点:
    GET  /                          → web/index.html
    GET  /api/students?limit=20     → 抽样学生
    GET  /api/students/{uid}/state  → 每个知识点掌握度 + 概览
    GET  /api/students/{uid}/recommend?k=5&with_text=1 → 用 GainStrategy 推荐 k 道题,
        并用 LLM 现场生成中文题面(text 字段,进程内按 problem_id 缓存)
    POST /api/students/{uid}/answer → 作答 → 更新掌握度 → 返回新 state + 新推荐

学生「当前掌握度」= 用其历史记录逐条 update_mastery 递推得到。

环境变量 P4_FAST=1 时用小子集(前 50 名学生、跳过全量拟合)快速启动。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.bkt import BKT, BKTParams, update_mastery
from src.data import ENCODING, find_raw, load_clean
from src.llm_client import HttpLLM, LLMConfigError
from src.question_gen import QuestionGenerator, parse_questions
from src.recommender import GainStrategy, Params, QuestionPool, State
from src.split import split_by_student_time

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"

FAST_STUDENTS = 50
FAST_SKILLS = 30


def _is_fast() -> bool:
    return os.environ.get("P4_FAST") == "1"


@dataclass
class AppState:
    """进程级缓存:模型只拟合一次。"""

    pool: QuestionPool
    params: Params
    skill_names: dict[int, str]
    history: dict[int, list[tuple[int, int]]]  # uid → [(skill_id, correct)]
    student_ids: list[int]
    mastery: dict[int, State] = field(default_factory=dict)
    # 进程内缓存:problem_id → {"stem","options","answer"},同一题只生成一次
    text_cache: dict[int, dict] = field(default_factory=dict)
    generator: QuestionGenerator | None = None
    generator_unavailable: bool = False

    def text_of(self, pid: int) -> dict | None:
        return self.text_cache.get(int(pid))

    def state_of(self, uid: int) -> State:
        if uid not in self.mastery:
            raise KeyError(uid)
        return self.mastery[uid]

    def ensure_state(self, uid: int) -> State:
        if uid not in self.mastery:
            self.mastery[uid] = self._replay(uid)
        return self.mastery[uid]

    def _replay(self, uid: int) -> State:
        """用该生真实历史逐条 update_mastery,得到当前掌握度。"""
        state: State = {sid: prm.l0 for sid, prm in self.params.items()}
        for sid, obs in self.history.get(uid, []):
            prm = self.params.get(sid)
            if prm is None:
                continue
            state[sid] = update_mastery(state.get(sid, prm.l0), obs, prm)
        return state


def load_skill_names(data_dir: Path = ROOT / "data") -> dict[int, str]:
    """从原始 csv 建 skill_id → skill_name 映射。

    同一 skill_id 可能有多个名字,取**出现次数最多**的那个。
    """
    path = find_raw(data_dir)
    raw = pd.read_csv(
        path, encoding=ENCODING, low_memory=False, usecols=["skill_id", "skill_name"]
    ).dropna(subset=["skill_id", "skill_name"])
    raw["skill_id"] = raw["skill_id"].astype("int64")
    counts = (
        raw.groupby(["skill_id", "skill_name"]).size().reset_index(name="n")
    )
    counts = counts.sort_values(
        ["skill_id", "n"], ascending=[True, False]
    ).drop_duplicates(subset="skill_id", keep="first")
    return {int(r.skill_id): str(r.skill_name) for r in counts.itertuples()}


def build_state() -> AppState:
    """加载数据 → 拟合 → 建池,全部一次完成。"""
    df = load_clean(ROOT / "data")
    fast = _is_fast()

    if fast:
        keep = sorted(df["user_id"].unique())[:FAST_STUDENTS]
        df = df[df["user_id"].isin(keep)].reset_index(drop=True)
        keep_skills = sorted(df["skill_id"].unique())[:FAST_SKILLS]
        df = df[df["skill_id"].isin(keep_skills)].reset_index(drop=True)

    train, test = split_by_student_time(df, test_ratio=0.2, min_events=10)

    model = BKT(n_iter=2 if fast else 20, min_sequences=2 if fast else 5).fit(train)
    params: Params = model.params
    if not params:
        raise RuntimeError("没有拟合出任何知识点参数;检查数据或 P4_FAST 设置")

    pool = QuestionPool.from_frame(df)

    real_names = load_skill_names(ROOT / "data")
    skill_names = {
        int(s): real_names.get(int(s), f"skill_{int(s)}") for s in params
    }

    # 学生在数据中的真实历史(全量,按 order_id 排序)用于递推当前掌握度
    hist = (
        df.sort_values(["user_id", "order_id"])
        .groupby("user_id")[["skill_id", "correct"]]
        .apply(lambda sub: [(int(s), int(c)) for s, c in sub.to_numpy()])
    )
    all_history = {int(u): v for u, v in hist.items()}

    # 供下拉框选择:优先 test 中历史足够的学生,不足则回退全部
    test_ids = sorted(int(u) for u in test["user_id"].unique())
    if not test_ids:
        test_ids = sorted(int(u) for u in df["user_id"].unique())

    state = AppState(
        pool=pool,
        params=params,
        skill_names=skill_names,
        history=all_history,
        student_ids=test_ids,
    )
    for uid in test_ids[:FAST_STUDENTS if fast else len(test_ids)]:
        state.ensure_state(uid)
    return state


_STATE: AppState | None = None


def get_state() -> AppState:
    global _STATE
    if _STATE is None:
        _STATE = build_state()
    return _STATE


def set_state(state: AppState) -> None:
    """测试注入用:覆盖进程级状态。"""
    global _STATE
    _STATE = state


def _difficulty_label(difficulty: float) -> str:
    if difficulty >= 0.7:
        return "偏易"
    if difficulty >= 0.45:
        return "中等"
    return "偏难"


def build_text_prompt(skill_name: str, difficulty: float) -> str:
    """让 LLM 现场生成一道中文题面(4 选项 + 答案),输出结构化 JSON。"""
    schema = (
        '{"stem": "题干", "options": ["A", "B", "C", "D"], '
        '"answer": "正确选项(必须是 options 之一)", "skill_name": "'
        + skill_name
        + '", "difficulty_hint": 0.0}'
    )
    return (
        "你是一名中文出题专家。请依据下面的知识点与难度要求,"
        "命制 1 道同知识点的中文变式题。\n\n"
        f"【知识点】{skill_name}\n"
        f"【难度提示】{difficulty:.2f}(0 最难、1 最易,{_difficulty_label(difficulty)})\n\n"
        "【输出要求】只输出一个 JSON 数组,每个元素形如:\n"
        f"{schema}\n"
        "题干与选项必须使用简体中文;difficulty_hint 是 0~1 的难度提示;"
        "answer 必须精确等于 options 中的某一项。不要输出任何解释文字。"
    )


def _ensure_generator(st: AppState) -> QuestionGenerator | None:
    """惰性建 LLM 生成器;env 缺失时标记不可用并返回 None(不抛错)。"""
    if st.generator is not None:
        return st.generator
    if st.generator_unavailable:
        return None
    try:
        st.generator = QuestionGenerator(HttpLLM())
    except LLMConfigError:
        st.generator_unavailable = True
        return None
    return st.generator


def _text_for(
    st: AppState, pid: int, sid: int, difficulty: float, generate: bool
) -> dict | None:
    """取(必要时生成)某题的中文题面。任何失败都返回 None,绝不抛错。"""
    cached = st.text_of(pid)
    if cached is not None:
        return cached
    if not generate:
        return None
    gen = _ensure_generator(st)
    if gen is None:
        return None
    skill_name = st.skill_names.get(int(sid), f"skill_{int(sid)}")
    try:
        questions = parse_questions(
            gen.llm.complete(build_text_prompt(skill_name, difficulty)).text,
            skill_name,
        )
    except Exception:  # noqa: BLE001 - LLM/解析任何失败都降级为无题面
        return None
    if not questions:
        return None
    q = questions[0]
    text = {"stem": q["stem"], "options": list(q["options"]), "answer": q["answer"]}
    st.text_cache[int(pid)] = text
    return text


class AnswerIn(BaseModel):
    problem_id: int
    correct: bool


app = FastAPI(title="question-recommender P4 demo")


def _overview(mastery: State) -> dict:
    vals = np.array(list(mastery.values()), dtype=float)
    return {
        "mean_mastery": float(vals.mean()) if len(vals) else 0.0,
        "n_weak": int((vals < 0.5).sum()),
    }


def _state_payload(st: AppState, uid: int) -> dict:
    mastery = st.ensure_state(uid)
    rows = [
        {
            "skill_id": int(sid),
            "skill_name": st.skill_names.get(int(sid), f"skill_{int(sid)}"),
            "mastery": float(m),
        }
        for sid, m in sorted(mastery.items())
    ]
    return {"user_id": uid, "skills": rows, "overview": _overview(mastery)}


def _recommend_payload(
    st: AppState, uid: int, k: int, with_text: bool = True
) -> list[dict]:
    mastery = st.ensure_state(uid)
    known_skills = [s for s in mastery if s in st.params]
    candidates = st.pool.problems_of(known_skills)

    def prob_fn(pids: np.ndarray) -> np.ndarray:
        skills = np.array([st.pool.skill_lookup[int(p)] for p in pids])
        g = np.array([st.params[int(s)].g for s in skills])
        s_slip = np.array([st.params[int(s)].s for s in skills])
        m = np.array([mastery.get(int(s), st.params[int(s)].l0) for s in skills])
        return m * (1 - s_slip) + (1 - m) * g

    pids = GainStrategy().recommend(
        st.pool, prob_fn, candidates, n=k, state=mastery, params=st.params
    )
    out = []
    for pid in pids:
        sid = st.pool.skill_lookup[int(pid)]
        difficulty = float(st.pool.difficulty[
            np.where(st.pool.problem_id == int(pid))[0][0]
        ])
        out.append(
            {
                "problem_id": int(pid),
                "skill_id": int(sid),
                "skill_name": st.skill_names.get(int(sid), f"skill_{int(sid)}"),
                "difficulty": difficulty,
                "text": _text_for(st, int(pid), int(sid), difficulty, with_text),
            }
        )
    return out


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    path = WEB_DIR / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="web/index.html 不存在")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/api/students")
def students(limit: int = Query(20, ge=1, le=500)) -> dict:
    st = get_state()
    ids = st.student_ids[:limit]
    return {"students": [{"user_id": uid} for uid in ids], "total": len(st.student_ids)}


@app.get("/api/students/{uid}/state")
def student_state(uid: int) -> dict:
    st = get_state()
    if uid not in st.student_ids and uid not in st.history:
        raise HTTPException(status_code=404, detail=f"学生 {uid} 不存在")
    return _state_payload(st, uid)


@app.get("/api/students/{uid}/recommend")
def recommend(
    uid: int,
    k: int = Query(5, ge=1, le=50),
    with_text: int = Query(1),
) -> dict:
    st = get_state()
    if uid not in st.student_ids and uid not in st.history:
        raise HTTPException(status_code=404, detail=f"学生 {uid} 不存在")
    return {
        "user_id": uid,
        "recommendations": _recommend_payload(st, uid, k, with_text=bool(with_text)),
    }


@app.post("/api/students/{uid}/answer")
def answer(uid: int, body: AnswerIn) -> dict:
    st = get_state()
    if uid not in st.student_ids and uid not in st.history:
        raise HTTPException(status_code=404, detail=f"学生 {uid} 不存在")
    sid = st.pool.skill_lookup.get(int(body.problem_id))
    if sid is None:
        raise HTTPException(status_code=404, detail=f"题目 {body.problem_id} 不存在")
    prm: BKTParams | None = st.params.get(sid)
    if prm is None:
        raise HTTPException(status_code=404, detail=f"知识点 {sid} 无 BKT 参数")

    mastery = st.ensure_state(uid)
    mastery[sid] = update_mastery(mastery.get(sid, prm.l0), int(body.correct), prm)

    return {
        "user_id": uid,
        "state": _state_payload(st, uid),
        "recommendations": _recommend_payload(st, uid, 5),
    }
