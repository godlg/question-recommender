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
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.bank import DEFAULT_PARAMS, Bank, BankError, load_bank, parse_upload, save_bank
from src.bkt import BKT, BKTParams, update_mastery
from src.data import ENCODING, find_raw, load_clean
from src.embedder import Embedder, get_embedder
from src.explainer import Explainer
from src.llm_client import HttpLLM, LLMConfigError
from src.materials import MaterialError, MaterialStore
from src.question_gen import QuestionGenerator, parse_questions
from src.recommender import GainStrategy, Params, QuestionPool, State
from src.retriever import build_retriever, resolve_mode
from src.split import split_by_student_time

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
BANK_PATH = ROOT / "data" / "bank.jsonl"
MATERIALS_DIR = ROOT / "data" / "materials"
MATERIALS_INDEX = ROOT / "data" / "materials_index.json"

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
    # P5:上传的自定义题库。非 None 时推荐/作答改用该题库(向后兼容 ASSISTments)
    bank: Bank | None = None
    extra_students: list[int] = field(default_factory=list)
    # P6:讲解材料库 + 检索式讲解器(惰性初始化)
    materials: MaterialStore | None = None
    explainer: Explainer | None = None
    explainer_unavailable: bool = False
    # P7:embedding 检索(默认 None → 全程 keyword,行为与 P6 一致)
    embedder: Embedder | None = None
    retrieval_mode: str = "keyword"  # 实际生效模式(embedder 缺失时已降级)
    # P8:该生在各知识点的作答次数(uid → skill_id → count),与 history 解耦
    n_answers: dict[int, dict[int, int]] = field(default_factory=dict)

    def material_store(self) -> MaterialStore:
        if self.materials is None:
            self.materials = MaterialStore(
                MATERIALS_DIR, MATERIALS_INDEX, embedder=self.embedder
            )
            self.materials.load_all()
        return self.materials

    def retriever(self):
        """按实际生效模式构造检索器(embedder 为 None 时固定 keyword)。"""
        retriever, _ = build_retriever(self.retrieval_mode, self.embedder)
        return retriever

    def text_of(self, pid: int) -> dict | None:
        return self.text_cache.get(int(pid))

    def state_of(self, uid: int) -> State:
        if uid not in self.mastery:
            raise KeyError(uid)
        return self.mastery[uid]

    def active_pool(self) -> QuestionPool:
        return self.bank.pool() if self.bank is not None else self.pool

    def active_params(self) -> Params:
        return self.bank.params() if self.bank is not None else self.params

    def active_skill_names(self) -> dict[int, str]:
        return self.bank.skill_names if self.bank is not None else self.skill_names

    def has_student(self, uid: int) -> bool:
        return uid in self.student_ids or uid in self.history or uid in self.mastery

    def create_student(self, name: str | None = None) -> int:
        """建一个空画像学生(所有知识点掌握度 = 默认 L0),返回新 uid。"""
        existing = set(self.student_ids) | set(self.history)
        uid = max(existing, default=0) + 1
        while uid in self.mastery:
            uid += 1
        params = self.active_params()
        self.mastery[uid] = {sid: prm.l0 for sid, prm in params.items()}
        self.extra_students.append(uid)
        return uid

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
    # P7:embedding 可选启用;未配置 / 初始化失败 → None(全程 keyword)
    state.embedder = get_embedder()
    state.retrieval_mode = resolve_mode(None, state.embedder)
    for uid in test_ids[:FAST_STUDENTS if fast else len(test_ids)]:
        state.ensure_state(uid)
    # P5:启动时若已落盘题库,自动加载(不覆盖 ASSISTments 路径)
    bank = load_bank(BANK_PATH)
    if bank is not None:
        state.bank = bank
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


def _counts_of(st: AppState, uid: int) -> dict[int, int]:
    """合并数据集中学生的真实历史计数 + 新建学生的在线作答计数。"""
    counts: dict[int, int] = {}
    for sid, _ in st.history.get(uid, []):
        counts[sid] = counts.get(sid, 0) + 1
    for sid, c in st.n_answers.get(uid, {}).items():
        counts[sid] = counts.get(sid, 0) + c
    return counts


def _state_payload(st: AppState, uid: int) -> dict:
    mastery = st.ensure_state(uid)
    names = st.active_skill_names()
    counts = _counts_of(st, uid)
    rows = [
        {
            "skill_id": int(sid),
            "skill_name": names.get(int(sid), f"skill_{int(sid)}"),
            "mastery": float(m),
            "n_answers": counts.get(int(sid), 0),
        }
        for sid, m in sorted(mastery.items())
    ]
    return {"user_id": uid, "skills": rows, "overview": _overview(mastery)}


def _bank_text(bank: Bank, pid: int) -> dict | None:
    """从上传题库取题干(不调 LLM)。"""
    q = bank.question(pid)
    if q is None:
        return None
    return {
        "stem": q["stem"],
        "options": list(q["options"]),
        "answer": q["answer"],
    }


def _question_for(st: AppState, pid: int) -> dict:
    """构造讲解用题目 dict(stem / options / answer / skill)。"""
    pool = st.active_pool()
    sid = pool.skill_lookup.get(int(pid))
    names = st.active_skill_names()
    skill = names.get(int(sid), f"skill_{int(sid)}") if sid is not None else ""
    text = None
    if st.bank is not None:
        text = _bank_text(st.bank, int(pid))
    if text is None:
        text = st.text_of(pid) or {}
    return {
        "problem_id": int(pid),
        "stem": text.get("stem"),
        "options": text.get("options"),
        "answer": text.get("answer"),
        "skill": skill,
    }


def _ensure_explainer(st: AppState) -> Explainer:
    """惰性建讲解器。无 LLM 配置时仍建(用于检索),llm=None 会降级为无讲解文本。"""
    if st.explainer is not None:
        return st.explainer
    if st.explainer_unavailable:
        llm = None
    else:
        try:
            llm = HttpLLM()
        except LLMConfigError:
            st.explainer_unavailable = True
            llm = None
    store = st.material_store()
    st.explainer = Explainer(llm, chunks=store.chunks)
    return st.explainer


def _materials_info(store: MaterialStore) -> dict:
    return store.summary()



def _recommend_payload(
    st: AppState, uid: int, k: int, with_text: bool = True
) -> list[dict]:
    mastery = st.ensure_state(uid)
    pool = st.active_pool()
    params = st.active_params()
    names = st.active_skill_names()
    bank = st.bank

    known_skills = [s for s in mastery if s in params]
    candidates = pool.problems_of(known_skills)

    def prob_fn(pids: np.ndarray) -> np.ndarray:
        skills = np.array([pool.skill_lookup[int(p)] for p in pids])
        g = np.array([params[int(s)].g for s in skills])
        s_slip = np.array([params[int(s)].s for s in skills])
        m = np.array([mastery.get(int(s), params[int(s)].l0) for s in skills])
        return m * (1 - s_slip) + (1 - m) * g

    counts = _counts_of(st, uid)
    pids = GainStrategy().recommend(
        pool, prob_fn, candidates, n=k, state=mastery, params=params, counts=counts
    )
    out = []
    for pid in pids:
        sid = pool.skill_lookup[int(pid)]
        difficulty = float(pool.difficulty[
            np.where(pool.problem_id == int(pid))[0][0]
        ])
        if bank is not None:
            # 题库自带题干 → 不再调 LLM;with_text 参数保留但默认不起作用
            text = _bank_text(bank, int(pid))
        else:
            text = _text_for(st, int(pid), int(sid), difficulty, with_text)
        out.append(
            {
                "question_id": int(pid),
                "problem_id": int(pid),
                "skill_id": int(sid),
                "skill_name": names.get(int(sid), f"skill_{int(sid)}"),
                "difficulty": difficulty,
                "stem": text["stem"] if text else None,
                "options": text["options"] if text else None,
                "answer": text["answer"] if text else None,
                "text": text,
            }
        )
    return out


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    path = WEB_DIR / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="web/index.html 不存在")
    return HTMLResponse(path.read_text(encoding="utf-8"))


class StudentIn(BaseModel):
    name: str | None = None


def _bank_info(bank: Bank) -> dict:
    return {
        "n_questions": bank.n_questions,
        "n_skills": bank.n_skills,
        "source_name": bank.source_name,
    }


@app.get("/api/students")
def students(limit: int = Query(20, ge=1, le=500)) -> dict:
    st = get_state()
    ids = list(st.extra_students) + st.student_ids
    return {"students": [{"user_id": uid} for uid in ids[:limit]], "total": len(ids)}


@app.post("/api/students")
def create_student(body: StudentIn | None = None) -> dict:
    st = get_state()
    name = body.name if body is not None else None
    uid = st.create_student(name)
    return {"user_id": uid, "name": name}


@app.get("/api/students/{uid}/state")
def student_state(uid: int) -> dict:
    st = get_state()
    if not st.has_student(uid):
        raise HTTPException(status_code=404, detail=f"学生 {uid} 不存在")
    return _state_payload(st, uid)


@app.get("/api/students/{uid}/recommend")
def recommend(
    uid: int,
    k: int = Query(5, ge=1, le=50),
    with_text: int = Query(1),
) -> dict:
    st = get_state()
    if not st.has_student(uid):
        raise HTTPException(status_code=404, detail=f"学生 {uid} 不存在")
    return {
        "user_id": uid,
        "recommendations": _recommend_payload(st, uid, k, with_text=bool(with_text)),
    }


@app.post("/api/students/{uid}/answer")
def answer(uid: int, body: AnswerIn) -> dict:
    st = get_state()
    if not st.has_student(uid):
        raise HTTPException(status_code=404, detail=f"学生 {uid} 不存在")
    pool = st.active_pool()
    params = st.active_params()
    sid = pool.skill_lookup.get(int(body.problem_id))
    if sid is None:
        raise HTTPException(status_code=404, detail=f"题目 {body.problem_id} 不存在")
    prm: BKTParams | None = params.get(sid)
    if prm is None:
        raise HTTPException(status_code=404, detail=f"知识点 {sid} 无 BKT 参数")

    mastery = st.ensure_state(uid)
    mastery[sid] = update_mastery(mastery.get(sid, prm.l0), int(body.correct), prm)

    # P8:记录该生在该知识点的作答次数,供冷启动门控使用
    if uid not in st.n_answers:
        st.n_answers[uid] = {}
    st.n_answers[uid][sid] = st.n_answers[uid].get(sid, 0) + 1

    return {
        "user_id": uid,
        "state": _state_payload(st, uid),
        "recommendations": _recommend_payload(st, uid, 5),
    }


@app.post("/api/bank/upload")
async def upload_bank(file: UploadFile = File(...)) -> dict:
    st = get_state()
    content = await file.read()
    try:
        bank = parse_upload(file.filename, content)
        if bank.n_questions == 0:
            raise BankError("没有解析到任何有效题目(检查必需列与坏行)")
    except BankError as exc:
        # 解析失败不得破坏已有题库
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    save_bank(bank, BANK_PATH)
    st.bank = bank
    return _bank_info(bank) | {"n_skipped": bank.n_skipped}


@app.get("/api/bank")
def bank_info() -> dict:
    st = get_state()
    if st.bank is None:
        return {"n_questions": 0, "n_skills": 0, "source_name": None}
    return _bank_info(st.bank)


@app.get("/api/bank/sample")
def bank_sample(n: int = Query(5, ge=1, le=100)) -> dict:
    st = get_state()
    if st.bank is None:
        return {"questions": []}
    out = []
    for q in st.bank.questions[:n]:
        out.append(
            {
                "question_id": q["question_id"],
                "stem": q["stem"],
                "options": q["options"],
                "answer": q["answer"],
                "skill": q["skill"],
                "difficulty": q["difficulty"],
            }
        )
    return {"questions": out}


@app.post("/api/materials/upload")
async def upload_materials(file: UploadFile = File(...)) -> dict:
    st = get_state()
    store = st.material_store()
    content = await file.read()
    try:
        store.add_from_bytes(file.filename or "", content)
    except MaterialError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store.save()
    # 材料更新后,让讲解器看到最新 chunk
    if st.explainer is not None:
        st.explainer.set_chunks(store.chunks)
    return store.summary()


@app.get("/api/materials")
def materials_info() -> dict:
    st = get_state()
    return {
        **st.material_store().summary(),
        "retrieval_mode": st.retrieval_mode,
        "embedder": st.embedder.name if st.embedder else None,
    }


class ExplainIn(BaseModel):
    problem_id: int
    student_answer: str | None = None


@app.post("/api/students/{uid}/explain")
def explain(uid: int, body: ExplainIn) -> dict:
    st = get_state()
    if not st.has_student(uid):
        raise HTTPException(status_code=404, detail=f"学生 {uid} 不存在")
    pool = st.active_pool()
    if int(body.problem_id) not in pool.skill_lookup:
        raise HTTPException(status_code=404, detail=f"题目 {body.problem_id} 不存在")
    question = _question_for(st, int(body.problem_id))
    explainer = _ensure_explainer(st)
    result = explainer.explain(question, body.student_answer)
    return {
        "problem_id": int(body.problem_id),
        "skill": question["skill"],
        "explanation": result["explanation"],
        "sources": result["sources"],
        "grounded": result["grounded"],
        "error": result["error"],
    }
