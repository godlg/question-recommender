import json

import pytest

from src.llm_client import FakeLLM
from src.question_gen import SkillContext
from src.reviewer import ReviewError, Reviewer, build_review_prompt, parse_review


def _question() -> dict:
    return {
        "stem": "3 + 4 = ?",
        "options": ["6", "7", "8", "9"],
        "answer": "7",
        "skill_name": "Addition",
        "difficulty_hint": 0.5,
    }


def _review(**over) -> dict:
    base = {
        "verdict": "pass",
        "reasons": [],
        "checks": {
            "answer_correct": True,
            "unambiguous": True,
            "difficulty_plausible": True,
            "skill_aligned": True,
        },
    }
    base.update(over)
    return base


def test_review_prompt_is_independent_of_generation():
    text = build_review_prompt(_question())
    assert "独立审题人" in text
    assert "answer_correct" in text
    assert "3 + 4 = ?" in text


def test_parse_review_ok():
    got = parse_review(json.dumps(_review()))
    assert got["verdict"] == "pass"
    assert set(got["checks"]) == {
        "answer_correct",
        "unambiguous",
        "difficulty_plausible",
        "skill_aligned",
    }


def test_parse_review_raises_on_bad_json():
    with pytest.raises(ReviewError):
        parse_review("nope")


def test_parse_review_raises_on_bad_verdict():
    with pytest.raises(ReviewError):
        parse_review(json.dumps(_review(verdict="maybe")))


def test_parse_review_raises_on_incomplete_checks():
    with pytest.raises(ReviewError):
        parse_review(json.dumps({"verdict": "pass", "reasons": [], "checks": {"answer_correct": True}}))


def test_parse_review_forces_fail_when_a_check_fails():
    payload = _review(verdict="pass")
    payload["checks"]["answer_correct"] = False
    got = parse_review(json.dumps(payload))
    assert got["verdict"] == "fail"


def test_reviewer_flags_wrong_answer_via_independent_client():
    checks = _review()["checks"] | {"answer_correct": False}
    payload = json.dumps({"verdict": "fail", "reasons": ["答案错误"], "checks": checks})
    reviewer = Reviewer(FakeLLM(payload))
    result = reviewer.review(_question(), SkillContext("Addition", 1, 1, 0.5, 0.5, 0.5, ()))
    assert result["verdict"] == "fail"
    assert result["checks"]["answer_correct"] is False
    assert result["reasons"] == ["答案错误"]


def test_reviewer_retries_then_raises():
    fake = FakeLLM("garbage")
    reviewer = Reviewer(fake, max_retries=1)
    with pytest.raises(ReviewError):
        reviewer.review(_question())
    assert len(fake.calls) == 2
