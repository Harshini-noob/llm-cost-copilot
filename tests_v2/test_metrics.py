import asyncio
from decimal import Decimal

import eval.metrics as metrics
import pytest
from eval.dataset import load_dataset


def _record(method, **fields):
    record = {
        "prompt_id": "test_001",
        "task_type": "math",
        "prompt": "Evaluate this answer.",
        "evaluation_method": method,
    }
    record.update(fields)
    return record


def test_numeric_evaluation_normalizes_formatting_with_exact_value():
    result = asyncio.run(
        metrics.evaluate_output(
            _record("numeric_exact", reference_answer="120"),
            "**Answer:** `120.00`.\n",
        )
    )

    assert result.evaluation_status == "scored"
    assert result.correct is True
    assert result.score == 1.0


def test_numeric_evaluation_rejects_wrong_or_ambiguous_values():
    record = _record("numeric_exact", reference_answer="120")

    wrong = asyncio.run(metrics.evaluate_output(record, "121"))
    ambiguous = asyncio.run(
        metrics.evaluate_output(record, "121, not 120")
    )
    multiple_answers = asyncio.run(
        metrics.evaluate_output(record, "Final answer: 120 or 121")
    )
    negated = asyncio.run(
        metrics.evaluate_output(record, "120 is not the correct answer")
    )

    assert wrong.correct is False
    assert wrong.score == 0.0
    assert ambiguous.correct is False
    assert multiple_answers.correct is False
    assert negated.correct is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("42", Decimal("42")),
        ("1,234.50", Decimal("1234.50")),
        ("-2.5", Decimal("-2.5")),
        ("1.2e3", Decimal("1.2e3")),
        ("$120", Decimal("120")),
        ("€120", Decimal("120")),
        ("£120", Decimal("120")),
        ("₹120", Decimal("120")),
        ("Rs. 120", Decimal("120")),
        ("INR 120", Decimal("120")),
    ],
)
def test_strict_decimal_parser_accepts_supported_numeric_forms(text, expected):
    assert metrics._parse_number(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "The notebook cost was 120 in the story",
        "1,23",
        "12..0",
        "NaN",
        "Infinity",
        "-Infinity",
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_strict_decimal_parser_rejects_prose_malformed_and_nonfinite(text):
    assert metrics._parse_number(text) is None


@pytest.mark.parametrize(
    "output",
    [
        "120\nHope this helps!",
        "Final answer: 120. Hope this helps!",
        "Final answer: 120\nThank you!",
    ],
)
def test_numeric_evaluation_ignores_common_polite_closings(output):
    result = asyncio.run(
        metrics.evaluate_output(
            _record("numeric_exact", reference_answer="120"),
            output,
        )
    )

    assert result.correct is True
    assert result.score == 1.0


def test_explicit_final_answer_marker_wins_over_intermediate_values():
    result = asyncio.run(
        metrics.evaluate_output(
            _record("numeric_exact", reference_answer="120"),
            "Intermediate value: 15\nFinal answer: 120",
        )
    )

    assert result.correct is True


def test_polite_closing_does_not_promote_an_intermediate_value():
    result = asyncio.run(
        metrics.evaluate_output(
            _record("numeric_exact", reference_answer="120"),
            "Intermediate value: 120\nHope this helps!",
        )
    )

    assert result.correct is False


def test_qwen_explanations_extract_the_final_numeric_answer():
    records = {
        record["prompt_id"]: record
        for record in load_dataset()
        if record["task_type"] == "math"
    }
    outputs = {
        "math_001": (
            "First, find the cost of one notebook:\n"
            "3 notebooks = Rs. 45\n"
            "So, 1 notebook = 45 ÷ 3 = Rs. 15\n"
            "Now, cost of 8 notebooks = 8 × 15 = Rs. 120\n"
            "120"
        ),
        "math_002": (
            "3x + 7 = 22\n"
            "Subtract 7 from both sides:\n"
            "3x = 15\n"
            "Divide both sides by 3:\n"
            "x = 5"
        ),
    }

    results = {
        prompt_id: asyncio.run(
            metrics.evaluate_output(records[prompt_id], output)
        )
        for prompt_id, output in outputs.items()
    }

    assert records["math_001"]["reference_answer"] == "120"
    assert records["math_002"]["reference_answer"] == "5"
    assert records["math_002"]["answer_variable"] == "x"
    assert all(result.correct is True for result in results.values())


def test_numeric_evaluation_supports_currency_and_uses_only_final_answer():
    record = _record("numeric_exact", reference_answer="120")

    currency = asyncio.run(
        metrics.evaluate_output(
            record,
            "3 notebooks cost Rs. 45.\nFinal answer: Rs. 120.",
        )
    )
    wrong_final = asyncio.run(
        metrics.evaluate_output(
            record,
            "An intermediate result is 120.\nFinal answer: 121.",
        )
    )

    assert currency.correct is True
    assert wrong_final.correct is False


def test_numeric_variable_assignment_requires_configured_variable():
    record = _record(
        "numeric_exact",
        reference_answer="5",
        answer_variable="x",
    )

    matching = asyncio.run(
        metrics.evaluate_output(record, "Work: 3x = 15\nx = 5")
    )
    wrong_variable = asyncio.run(
        metrics.evaluate_output(record, "y = 5")
    )
    wrong_case = asyncio.run(
        metrics.evaluate_output(record, "X = 5")
    )

    assert matching.correct is True
    assert wrong_variable.correct is False
    assert wrong_case.correct is False


def test_numeric_evaluation_leaves_missing_reference_unscored():
    result = asyncio.run(
        metrics.evaluate_output(_record("numeric_exact"), "120")
    )

    assert result.evaluation_status == "unscored"
    assert result.correct is None
    assert result.score is None


def test_code_evaluation_is_unsupported_without_sandbox():
    result = asyncio.run(
        metrics.evaluate_output(
            _record(
                "unit_test",
                task_type="code",
                test_cases=[{"input": 2, "expected": True}],
            ),
            "def is_even(n): return n % 2 == 0",
        )
    )

    assert result.evaluation_status == "unsupported"
    assert result.correct is None
    assert "sandbox" in result.evaluation_error


def test_knowledge_evaluation_accepts_configured_alternative():
    result = asyncio.run(
        metrics.evaluate_output(
            _record(
                "normalized_exact",
                task_type="knowledge",
                reference_answer="Jupiter",
                acceptable_answers=["The planet Jupiter"],
            ),
            "  JUPITER!  ",
        )
    )

    assert result.evaluation_status == "scored"
    assert result.correct is True


def test_knowledge_evaluation_requires_a_reference_answer():
    result = asyncio.run(
        metrics.evaluate_output(
            _record("normalized_exact", task_type="knowledge"),
            "Jupiter",
        )
    )

    assert result.evaluation_status == "unscored"
    assert result.correct is None


def test_extraction_evaluation_checks_expected_fields_and_values():
    result = asyncio.run(
        metrics.evaluate_output(
            _record(
                "json_exact",
                task_type="extraction",
                reference_answer={"name": "Ananya", "age": 21},
            ),
            '```json\n{"name":"ANANYA","age":21,"extra":"ok"}\n```',
        )
    )

    assert result.evaluation_status == "scored"
    assert result.correct is True


def test_extraction_evaluation_rejects_wrong_values_and_invalid_json():
    record = _record(
        "json_exact",
        task_type="extraction",
        reference_answer={"age": 21},
    )

    wrong_value = asyncio.run(
        metrics.evaluate_output(record, '{"age":22}')
    )
    malformed = asyncio.run(metrics.evaluate_output(record, '{"age":'))

    assert wrong_value.correct is False
    assert malformed.evaluation_status == "scored"
    assert malformed.correct is False
    assert malformed.score == 0.0


def test_extraction_missing_reference_is_unscored():
    result = asyncio.run(
        metrics.evaluate_output(
            _record("json_exact", task_type="extraction"),
            '{"name":"Ananya"}',
        )
    )

    assert result.evaluation_status == "unscored"
    assert result.correct is None


@pytest.mark.parametrize(
    ("output", "expected", "correct"),
    [
        ("C", "C", True),
        ("Answer: B.", "A", False),
        ("Reasoning...\nFinal answer: D", "A", False),
        ("```text\nA\n```", "D", False),
    ],
)
def test_multiple_choice_evaluation_matches_unambiguous_labels(
    output,
    expected,
    correct,
):
    result = asyncio.run(
        metrics.evaluate_output(
            _record(
                "multiple_choice_exact",
                task_type="knowledge",
                reference_answer=expected,
                choice_labels=["A", "B", "C", "D"],
            ),
            output,
        )
    )

    assert result.evaluation_status == "scored"
    assert result.correct is correct
    assert result.score == float(correct)


@pytest.mark.parametrize(
    "output",
    [
        "A or C",
        "The answer may be B or D.",
        "Reasoning mentions A and B.",
        "",
    ],
)
def test_multiple_choice_ambiguous_outputs_remain_unscored(output):
    result = asyncio.run(
        metrics.evaluate_output(
            _record(
                "multiple_choice_exact",
                task_type="knowledge",
                reference_answer="A",
                choice_labels=["A", "B", "C", "D"],
            ),
            output,
        )
    )

    assert result.evaluation_status == "unscored"
    assert result.correct is None


def test_multiple_choice_invalid_reference_is_unscored():
    result = asyncio.run(
        metrics.evaluate_output(
            _record(
                "multiple_choice_exact",
                task_type="knowledge",
                reference_answer="E",
                choice_labels=["A", "B", "C", "D"],
            ),
            "A",
        )
    )

    assert result.evaluation_status == "unscored"
    assert result.correct is None


def test_multiple_choice_conflicting_final_labels_are_unscored():
    result = asyncio.run(
        metrics.evaluate_output(
            _record(
                "multiple_choice_exact",
                task_type="knowledge",
                reference_answer="A",
                choice_labels=["A", "B", "C", "D"],
            ),
            "Final answer: A\nB",
        )
    )

    assert result.evaluation_status == "unscored"
    assert result.correct is None


@pytest.mark.parametrize(
    "output",
    [
        "Answer: A\nFinal answer: B",
        "A\nB",
    ],
)
def test_multiple_choice_conflicting_labels_anywhere_are_unscored(output):
    result = asyncio.run(
        metrics.evaluate_output(
            _record(
                "multiple_choice_exact",
                task_type="knowledge",
                reference_answer="A",
                choice_labels=["A", "B", "C", "D"],
            ),
            output,
        )
    )

    assert result.evaluation_status == "unscored"
    assert result.correct is None


def test_judge_requires_configuration_and_reference(monkeypatch):
    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)

    async def unexpected_call(**kwargs):
        raise AssertionError("judge must not be called without labels/config")

    monkeypatch.setattr(metrics.litellm, "acompletion", unexpected_call)
    no_reference = asyncio.run(
        metrics.evaluate_output(
            _record("llm_judge", task_type="summarization"),
            "A summary.",
            model_id="gpt_oss_20b",
        )
    )
    no_configuration = asyncio.run(
        metrics.evaluate_output(
            _record(
                "llm_judge",
                task_type="summarization",
                reference_answer="Reference summary.",
            ),
            "A summary.",
            model_id="gpt_oss_20b",
        )
    )

    assert no_reference.evaluation_status == "unscored"
    assert no_reference.correct is None
    assert no_configuration.evaluation_status == "unscored"
    assert no_configuration.correct is None


def test_judge_uses_separate_model_and_keeps_score_for_review(monkeypatch):
    calls = []
    monkeypatch.setenv("GEMINI_API_KEY", "judge-test-key")

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [{
                "message": {
                    "content": '{"score":0.75,"rationale":"Mostly complete."}'
                }
            }]
        }

    monkeypatch.setattr(metrics.litellm, "acompletion", fake_acompletion)
    result = asyncio.run(
        metrics.evaluate_output(
            _record(
                "llm_judge",
                task_type="summarization",
                reference_answer="Reference summary.",
            ),
            "Candidate summary.",
            model_id="gpt_oss_20b",
            judge_model="gemini/gemini-3.5-flash-lite",
        )
    )

    assert result.evaluation_status == "needs_review"
    assert result.score == 0.75
    assert result.correct is None
    assert result.evaluation_details == "Mostly complete."
    assert calls[0]["model"] == "gemini/gemini-3.5-flash-lite"
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert calls[0]["num_retries"] == 0


def test_judge_cannot_be_the_evaluated_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    async def unexpected_call(**kwargs):
        raise AssertionError("the evaluated model cannot judge itself")

    monkeypatch.setattr(metrics.litellm, "acompletion", unexpected_call)
    result = asyncio.run(
        metrics.evaluate_output(
            _record(
                "llm_judge",
                task_type="summarization",
                reference_answer="Reference summary.",
            ),
            "Candidate summary.",
            model_id="gpt_oss_20b",
            judge_model="groq/openai/gpt-oss-20b",
        )
    )

    assert result.evaluation_status == "unscored"
    assert result.correct is None


def test_judge_rejects_invalid_structured_score(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "judge-test-key")

    async def fake_acompletion(**kwargs):
        return {
            "choices": [{
                "message": {"content": '{"score":1.5,"rationale":"Invalid."}'}
            }]
        }

    monkeypatch.setattr(metrics.litellm, "acompletion", fake_acompletion)
    result = asyncio.run(
        metrics.evaluate_output(
            _record(
                "llm_judge",
                task_type="chit_chat",
                reference_answer="Reference answer.",
            ),
            "Candidate answer.",
            model_id="gpt_oss_20b",
            judge_model="gemini/gemini-3.5-flash-lite",
        )
    )

    assert result.evaluation_status == "unscored"
    assert result.score is None
    assert result.correct is None
