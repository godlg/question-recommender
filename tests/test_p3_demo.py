import json

from scripts.p3_demo import (
    _inject_errors,
    _make_mock_generator,
    _make_mock_reviewer,
    _mock_questions,
)


def _questions(n: int) -> list[dict]:
    return [
        {
            "stem": f"Addition: {10 + i} + {5 + i} = ?",
            "options": [str(15 + 2 * i), str(16 + 2 * i), str(14 + 2 * i)],
            "answer": str(15 + 2 * i),
            "skill_name": "Addition",
            "difficulty_hint": 0.5,
        }
        for i in range(n)
    ]


def test_inject_errors_periodic_and_logged():
    qs = _questions(6)
    injected = _inject_errors(qs, every=3)
    assert set(injected) == {2, 5}
    for idx, item in injected.items():
        assert qs[idx]["answer"] == item["injected_answer"]
        assert item["injected_answer"] != item["original_answer"]
        assert item["injected_answer"] in qs[idx]["options"]


def test_inject_errors_disabled_when_every_zero():
    qs = _questions(4)
    assert _inject_errors(qs, every=0) == {}
    assert qs[0]["answer"] == "15"


def test_inject_errors_skips_when_no_alternative_option():
    qs = [{"stem": "s", "options": ["7"], "answer": "7", "skill_name": "k", "difficulty_hint": 0.5}]
    assert _inject_errors(qs, every=1) == {}


def test_mock_generator_emits_all_correct_questions():
    counter = [0]
    responder = _make_mock_generator(counter)
    text = responder("知识点: Addition\n请命制 3 道同知识点的变式题")
    questions = json.loads(text)
    assert len(questions) == 3
    for q in questions:
        a, b = q["stem"].split(": ")[1].split(" = ?")[0].split(" + ")
        assert q["answer"] == str(int(a) + int(b))


def test_mock_reviewer_flags_injected_answer():
    qs = _mock_questions("Addition", 1, [0])
    q = qs[0]
    reviewer = _make_mock_reviewer()
    from src.reviewer import build_review_prompt

    good = json.loads(reviewer(build_review_prompt(q)))
    assert good["verdict"] == "pass"

    q["answer"] = next(o for o in q["options"] if o != q["answer"])
    bad = json.loads(reviewer(build_review_prompt(q)))
    assert bad["verdict"] == "fail"
    assert bad["checks"]["answer_correct"] is False
