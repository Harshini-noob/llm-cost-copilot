import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import litellm

from model_registry_v2 import get_model_by_id


_NUMERIC_ANSWER_PATTERN = re.compile(
    r"\s*(?:(?:therefore|thus|hence|so)\s*[,;:]?\s*)?"
    r"(?:(?:the\s+)?(?:final\s+)?(?:numeric(?:al)?\s+)?"
    r"answer\s*(?::|is)\s*)?"
    r"(?:[$€£₹]\s*|(?:Rs\.?|INR)\s*)?"
    r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*[.!]?\s*",
    flags=re.IGNORECASE,
)
_VARIABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_]\w*$")
_POLITE_CLOSING = (
    r"(?:i\s+hope\s+(?:that\s+)?this\s+helps|"
    r"hope\s+(?:that\s+)?this\s+helps|"
    r"let\s+me\s+know\s+if\s+you\s+(?:need|have)\b.*|"
    r"feel\s+free\s+to\s+(?:ask|reach\s+out)\b.*|"
    r"thank\s+you(?:\s+for\s+(?:asking|reading))?|"
    r"thanks(?:\s+for\s+(?:asking|reading))?|"
    r"happy\s+to\s+help)"
)
_POLITE_CLOSING_LINE_PATTERN = re.compile(
    rf"^\s*{_POLITE_CLOSING}\s*[.!?]*\s*$",
    flags=re.IGNORECASE,
)
_POLITE_CLOSING_SUFFIX_PATTERN = re.compile(
    rf"(?:[.!?;,]\s+|\s+){_POLITE_CLOSING}\s*[.!?]*\s*$",
    flags=re.IGNORECASE,
)

JUDGE_RUBRIC = (
    "Evaluate how well the candidate answer addresses the prompt and matches "
    "the reference answer. Score 1.0 only when the answer is fully correct, "
    "relevant, and sufficiently complete; score 0.0 when it is incorrect or "
    "irrelevant, and use intermediate values for partial coverage. Do not "
    "reward unsupported claims. Return one JSON object with exactly these "
    "fields: score (number from 0 to 1) and rationale (brief string)."
)


@dataclass(frozen=True)
class EvaluationResult:
    evaluation_status: str
    score: float | None
    correct: bool | None
    evaluation_method: str
    evaluation_error: str | None = None
    evaluation_details: str | None = None


def _result(
    method: str,
    status: str,
    score: float | None = None,
    correct: bool | None = None,
    error: str | None = None,
    details: str | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        evaluation_status=status,
        score=score,
        correct=correct,
        evaluation_method=method,
        evaluation_error=error,
        evaluation_details=details,
    )


def _has_reference(record: dict[str, Any]) -> bool:
    value = record.get("reference_answer")
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list)):
        return bool(value)
    return True


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = normalized.strip("`*_\"'“”‘’")
    while normalized and unicodedata.category(normalized[0]).startswith("P"):
        normalized = normalized[1:].lstrip()
    while normalized and unicodedata.category(normalized[-1]).startswith("P"):
        normalized = normalized[:-1].rstrip()
    return re.sub(r"\s+", " ", normalized)


def _parse_number(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            number = Decimal(str(value))
        except InvalidOperation:
            return None
        return number if number.is_finite() else None
    if not isinstance(value, str):
        return None

    text = value.replace("```", "").replace("`", "")
    text = text.replace("**", "").replace("__", "").strip()
    match = _NUMERIC_ANSWER_PATTERN.fullmatch(text)
    if match is None:
        return None
    try:
        number = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _configured_answer_variable(record: dict[str, Any]) -> str | None:
    variable = record.get("answer_variable")
    if variable is None:
        reference = record.get("reference_answer")
        if isinstance(reference, str):
            assignment = re.fullmatch(
                r"\s*([A-Za-z_]\w*)\s*=\s*.+?\s*",
                reference,
            )
            if assignment:
                variable = assignment.group(1)
    if variable is None:
        return None
    return variable if isinstance(variable, str) and _VARIABLE_NAME_PATTERN.fullmatch(variable) else ""


def _assignment_number(value: Any, variable: str) -> Decimal | None:
    if not isinstance(value, str) or not variable:
        return None
    assignment = re.fullmatch(
        rf"\s*(?:(?:therefore|thus|hence|so)\s*[,;:]?\s*)?"
        r"([A-Za-z_]\w*)\s*=\s*(.*?)\s*[.!]?\s*",
        value,
        flags=re.IGNORECASE,
    )
    if not assignment or assignment.group(1) != variable:
        return None
    return _parse_number(assignment.group(2))


def _final_answer_line(output: Any) -> Any:
    if not isinstance(output, str):
        return output
    lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip()
        and not re.fullmatch(r"\s*```(?:[A-Za-z0-9_-]+)?\s*", line)
    ]
    while lines and _POLITE_CLOSING_LINE_PATTERN.fullmatch(lines[-1]):
        lines.pop()
    if lines:
        closing = _POLITE_CLOSING_SUFFIX_PATTERN.search(lines[-1])
        if closing:
            lines[-1] = lines[-1][: closing.start()].rstrip(" \t,;:.!?")
    return lines[-1] if lines else ""


def _evaluate_numeric_exact(record: dict[str, Any], output: Any) -> EvaluationResult:
    method = record["evaluation_method"]
    variable = _configured_answer_variable(record)
    if variable == "":
        return _result(
            method,
            "unscored",
            error="answer_variable must be a valid variable name.",
        )

    reference = record["reference_answer"]
    expected = (
        _assignment_number(reference, variable)
        if variable
        else None
    )
    if expected is None:
        expected = _parse_number(reference)
    if expected is None:
        return _result(
            method,
            "unscored",
            error="Reference answer is not a single valid number.",
        )

    final_line = _final_answer_line(output)
    actual = (
        _assignment_number(final_line, variable)
        if variable
        else None
    )
    if actual is None:
        actual = _parse_number(final_line)
    correct = actual is not None and actual == expected
    return _result(method, "scored", float(correct), correct)


def _evaluate_normalized_exact(
    record: dict[str, Any],
    output: Any,
) -> EvaluationResult:
    method = record["evaluation_method"]
    reference = record["reference_answer"]
    alternatives = record.get("acceptable_answers", [])
    if not isinstance(alternatives, list):
        return _result(
            method,
            "unscored",
            error="acceptable_answers must be a list when provided.",
        )
    expected = [reference, *alternatives]
    normalized_expected = {
        _normalize_text(value)
        for value in expected
        if isinstance(value, str)
    }
    if not normalized_expected:
        return _result(
            method,
            "unscored",
            error="No valid reference or acceptable answer is configured.",
        )

    correct = (
        isinstance(output, str)
        and _normalize_text(output) in normalized_expected
    )
    return _result(method, "scored", float(correct), correct)


def _extract_choice_label(output: Any, labels: set[str]) -> str | None:
    if not isinstance(output, str):
        return None

    text = output.strip()
    fenced = re.fullmatch(
        r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL,
    )
    if fenced:
        text = fenced.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    answer_markers = re.compile(
        r"^(?:final\s+answer|answer|choice)\s*(?::|is)\s*"
        r"([A-Za-z0-9]+)\s*[.!)]?$",
        flags=re.IGNORECASE,
    )
    marked = [
        match.group(1).upper()
        for line in lines
        if (match := answer_markers.fullmatch(line))
    ]
    standalone_pattern = re.compile(r"[\(\[]?([A-Za-z0-9]+)[\)\].]?")
    standalone = [
        match.group(1).upper()
        for line in lines
        if (match := standalone_pattern.fullmatch(line))
        and match.group(1).upper() in labels
    ]
    recognized = set(marked + standalone)
    if len(recognized) > 1:
        return None
    if marked:
        return marked[0] if recognized else None
    if standalone:
        return standalone[-1]
    return None


def _evaluate_multiple_choice_exact(
    record: dict[str, Any],
    output: Any,
) -> EvaluationResult:
    method = record["evaluation_method"]
    labels = record.get("choice_labels")
    if (
        not isinstance(labels, list)
        or not labels
        or any(not isinstance(label, str) or not label for label in labels)
    ):
        return _result(
            method,
            "unscored",
            error="choice_labels must be a non-empty list of strings.",
        )

    normalized_labels = {label.upper() for label in labels}
    if len(normalized_labels) != len(labels):
        return _result(
            method,
            "unscored",
            error="choice_labels must contain unique labels.",
        )

    reference = record.get("reference_answer")
    if not isinstance(reference, str) or reference.upper() not in normalized_labels:
        return _result(
            method,
            "unscored",
            error="Reference answer is not one of the configured choice labels.",
        )

    actual = _extract_choice_label(output, normalized_labels)
    if actual is None:
        return _result(
            method,
            "unscored",
            error="Output does not contain one unambiguous answer label.",
        )

    correct = actual == reference.upper()
    return _result(method, "scored", float(correct), correct)


def _decode_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    text = value.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        text = fenced.group(1)
    return json.loads(text)


def _json_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _json_value_matches(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _json_value_matches(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected)
            )
        )
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, (int, float)):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
        try:
            return Decimal(str(actual)) == Decimal(str(expected))
        except InvalidOperation:
            return False
    if isinstance(expected, str):
        return (
            isinstance(actual, str)
            and _normalize_text(actual) == _normalize_text(expected)
        )
    return type(actual) is type(expected) and actual == expected


def _evaluate_json_exact(record: dict[str, Any], output: Any) -> EvaluationResult:
    method = record["evaluation_method"]
    try:
        expected = _decode_json(record["reference_answer"])
    except (json.JSONDecodeError, TypeError, ValueError):
        return _result(
            method,
            "unscored",
            error="Reference answer is not valid JSON.",
        )

    if not isinstance(expected, dict) or not expected:
        return _result(
            method,
            "unscored",
            error="Reference answer must be a non-empty JSON object.",
        )

    try:
        actual = _decode_json(output)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _result(method, "scored", 0.0, False)

    correct = isinstance(actual, dict) and _json_value_matches(actual, expected)
    return _result(method, "scored", float(correct), correct)


def _lite_llm_provider(model: str) -> tuple[str, tuple[str, ...]] | None:
    if model.startswith("groq/"):
        return "groq", ("GROQ_API_KEY",)
    if model.startswith("gemini/"):
        return "google", ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    if model.startswith("openrouter/"):
        return "openrouter", ("OPENROUTER_API_KEY",)
    return None


def _judge_is_evaluated_model(judge_model: str, model_id: str | None) -> bool:
    if not model_id:
        return False
    try:
        model_config = get_model_by_id(model_id)
    except ValueError:
        return judge_model == model_id

    equivalents = {model_id, model_config["model_name"]}
    provider_prefix = {
        "groq": "groq/",
        "google": "gemini/",
        "openrouter": "openrouter/",
    }.get(model_config["provider"])
    if provider_prefix:
        equivalents.add(provider_prefix + model_config["model_name"])

    provider_config = _lite_llm_provider(judge_model)
    if provider_config:
        prefix = {
            "groq": "groq/",
            "google": "gemini/",
            "openrouter": "openrouter/",
        }[provider_config[0]]
        equivalents.add(judge_model[len(prefix) :])
    return judge_model in equivalents or any(
        judge_model == prefix + model_config["model_name"]
        for prefix in ("groq/", "gemini/", "openrouter/")
    )


def _response_value(value: Any, key: str, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


async def _evaluate_with_judge(
    record: dict[str, Any],
    output: Any,
    model_id: str | None,
    judge_model: str | None,
) -> EvaluationResult:
    method = record["evaluation_method"]
    if not _has_reference(record):
        return _result(
            method,
            "unscored",
            error="A reference label is required for judge evaluation.",
        )

    judge_model = judge_model or os.getenv("EVAL_JUDGE_MODEL")
    if not judge_model:
        return _result(
            method,
            "unscored",
            error="EVAL_JUDGE_MODEL is not configured.",
        )
    if _judge_is_evaluated_model(judge_model, model_id):
        return _result(
            method,
            "unscored",
            error="Judge model must differ from the evaluated model.",
        )

    provider_config = _lite_llm_provider(judge_model)
    if provider_config is None:
        return _result(
            method,
            "unscored",
            error="Judge model must use a supported LiteLLM provider prefix.",
        )
    provider, key_names = provider_config
    api_key = next((os.getenv(name) for name in key_names if os.getenv(name)), None)
    if not api_key:
        return _result(
            method,
            "unscored",
            error=f"Credentials for the configured {provider} judge are missing.",
        )

    prompt = (
        f"Prompt:\n{record['prompt']}\n\n"
        f"Reference answer:\n{json.dumps(record['reference_answer'], ensure_ascii=False)}"
        f"\n\nCandidate answer:\n{output}"
    )
    try:
        response = await litellm.acompletion(
            model=judge_model,
            api_key=api_key,
            messages=[
                {"role": "system", "content": JUDGE_RUBRIC},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            num_retries=0,
        )
        choices = _response_value(response, "choices", [])
        message = _response_value(choices[0], "message") if choices else None
        content = _response_value(message, "content")
        judge_result = json.loads(content)
        score = judge_result.get("score")
        rationale = judge_result.get("rationale")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
            or not isinstance(rationale, str)
            or not rationale.strip()
        ):
            raise ValueError("Invalid judge score schema.")
    except Exception as exc:
        return _result(
            method,
            "unscored",
            error=f"Judge evaluation failed ({type(exc).__name__}).",
        )

    return _result(
        method,
        "needs_review",
        score=float(score),
        details=rationale.strip()[:2000],
    )


async def evaluate_output(
    record: dict[str, Any],
    output: Any,
    *,
    model_id: str | None = None,
    judge_model: str | None = None,
) -> EvaluationResult:
    """Evaluate one successful model output without conflating execution status."""
    method = record.get("evaluation_method", "")
    if method == "unit_test":
        return _result(
            method,
            "unsupported",
            error="Generated code is not executed without an isolation sandbox.",
        )
    if method in {
        "numeric_exact",
        "normalized_exact",
        "json_exact",
        "multiple_choice_exact",
    }:
        if not _has_reference(record):
            return _result(
                method,
                "unscored",
                error="A reference answer is required for deterministic evaluation.",
            )
        if method == "numeric_exact":
            return _evaluate_numeric_exact(record, output)
        if method == "normalized_exact":
            return _evaluate_normalized_exact(record, output)
        if method == "multiple_choice_exact":
            return _evaluate_multiple_choice_exact(record, output)
        return _evaluate_json_exact(record, output)
    if method == "llm_judge":
        return await _evaluate_with_judge(record, output, model_id, judge_model)
    return _result(
        method,
        "unsupported",
        error="Evaluation method is not supported.",
    )