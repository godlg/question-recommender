import json

import pytest

from src.llm_client import FakeLLM
from src.question_gen import (
    GenerationError,
    QuestionGenerator,
    SkillContext,
    build_generation_prompt,
    parse_questions,
)


def _skill_context() -> SkillContext:
    return SkillContext(
        skill_name="Addition",
        n_problems=3,
        n_attempts=10,
        difficulty_mean=0.6,
        difficulty_min=0.2,
        difficulty_max=0.9,
        sample_problem_ids=(101, 102, 103),
    )


def _question(**over) -> dict:
    base = {
        "stem": "1 + 1 = ?",
        "options": ["1", "2", "3", "4"],
        "answer": "2",
        "skill_name": "Addition",
        "difficulty_hint": 0.5,
    }
    base.update(over)
    return base


def test_context_prompt_includes_stats():
    text = _skill_context().to_prompt()
    assert "Addition" in text
    assert "3 道题" in text
    assert "0.60" in text


def test_build_generation_prompt_mentions_n_and_schema():
    prompt = build_generation_prompt(_skill_context(), n=4)
    assert "命制 4 道" in prompt
    assert "difficulty_hint" in prompt
    assert "Addition" in prompt


def test_parse_questions_from_array():
    text = json.dumps([_question()])
    got = parse_questions(text, "Addition")
    assert len(got) == 1
    assert got[0]["answer"] == "2"
    assert got[0]["options"] == ["1", "2", "3", "4"]


def test_parse_questions_from_questions_object_and_fence():
    text = "```json\n" + json.dumps({"questions": [_question()]}) + "\n```"
    got = parse_questions(text, "Addition")
    assert got[0]["skill_name"] == "Addition"


def test_parse_questions_raises_on_bad_json():
    with pytest.raises(GenerationError):
        parse_questions("not json at all", "Addition")


def test_parse_questions_raises_on_missing_field():
    bad = _question()
    del bad["difficulty_hint"]
    with pytest.raises(GenerationError):
        parse_questions(json.dumps([bad]), "Addition")


def test_parse_questions_raises_when_answer_not_in_options():
    with pytest.raises(GenerationError):
        parse_questions(json.dumps([_question(answer="999")]), "Addition")


def test_generate_retries_then_succeeds():
    good = json.dumps([_question()])
    fake = FakeLLM(["garbage", good])
    gen = QuestionGenerator(fake, {"Addition": _skill_context()}, max_retries=1)
    got = gen.generate("Addition", 1)
    assert got[0]["stem"] == "1 + 1 = ?"
    assert len(fake.calls) == 2


def test_generate_raises_after_retries_exhausted():
    fake = FakeLLM("still not json")
    gen = QuestionGenerator(fake, {"Addition": _skill_context()}, max_retries=1)
    with pytest.raises(GenerationError):
        gen.generate("Addition", 1)
    assert len(fake.calls) == 2


def test_generate_zero_returns_empty_without_calls():
    fake = FakeLLM("{}")
    gen = QuestionGenerator(fake)
    assert gen.generate("Addition", 0) == []
    assert fake.calls == []


def test_context_for_unknown_skill_falls_back():
    ctx = QuestionGenerator(FakeLLM("[]")).context_for("unknown")
    assert ctx.skill_name == "unknown"
    assert ctx.n_problems == 0
