import re
from decimal import Decimal

from fastapi import Depends, FastAPI, HTTPException
from google.genai.errors import APIError as GeminiAPIError
from groq import APIError as GroqAPIError

from auth import verify_api_key
from budget import BudgetExceededError, BudgetTracker, InvalidTokenUsageError
from logger import log_full_request
from models import call_model
from routing_policy import (
    InvalidConstraintError,
    NoEligibleModelError,
    select_model,
)
from router import max_tokens_for_model

app = FastAPI()

QUALITY_PASS_THRESHOLD = 3
QUALITY_JUDGE_MODEL = "openai/gpt-oss-120b"
QUALITY_JUDGE_MAX_TOKENS = 60


async def verify_quality(prompt: str, answer: str,
                         budget_tracker: BudgetTracker) -> dict:
    judge_prompt = f"""Rate this answer's quality for the given question, 1-5.
1 = wrong or unhelpful, 5 = correct and complete.

Question: "{prompt}"
Answer: "{answer}"

Respond with ONLY a number 1-5."""
    try:
        result = await call_model(
            judge_prompt,
            model=QUALITY_JUDGE_MODEL,
            max_tokens=QUALITY_JUDGE_MAX_TOKENS,
            budget_tracker=budget_tracker,
            call_type="quality_judge",
        )
    except BudgetExceededError:
        return {"score": None, "passed": None, "status": "unavailable"}
    except InvalidTokenUsageError:
        return {"score": None, "passed": None, "status": "unavailable"}
    except (GroqAPIError, GeminiAPIError):
        return {"score": None, "passed": None, "status": "unavailable"}

    match = re.search(r"\b([1-5])\b", result["answer"].strip())
    if match is None:
        return {"score": None, "passed": None, "status": "unavailable"}

    score = int(match.group(1))
    passed = score >= QUALITY_PASS_THRESHOLD
    budget_tracker.attach_quality(score, passed)
    return {"score": score, "passed": passed, "status": "verified"}


def _persist_request(prompt: str, tier: str, routing_mode: str,
                    max_cost_usd, min_quality, tracker: BudgetTracker,
                    quality_status: str, escalation_attempted: bool,
                    escalation_reason):
    return log_full_request(
        prompt=prompt,
        tier=tier,
        routing_mode=routing_mode,
        max_cost_usd=max_cost_usd,
        min_quality=min_quality,
        model_calls_data=tracker.model_calls,
        estimated_cost_usd=tracker.estimated_cost,
        actual_total_cost_usd=tracker.actual_cost,
        budget_accounted_cost_usd=tracker.accounted_cost,
        remaining_budget_usd=tracker.remaining_budget,
        quality_verification_status=quality_status,
        escalation_attempted=escalation_attempted,
        escalation_reason=escalation_reason,
        budget_exceeded=(
            tracker.budget_limit is not None
            and tracker.accounted_cost > tracker.budget_limit
        ),
    )


@app.post("/query")
async def query(prompt: str, routing_mode: str = "balanced",
                max_cost_usd: Decimal = None, min_quality: float = None,
                api_key_id: str = Depends(verify_api_key)):
    try:
        decision = await select_model(
            prompt,
            routing_mode=routing_mode,
            max_cost_usd=max_cost_usd,
            min_quality=min_quality,
        )
    except InvalidConstraintError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except NoEligibleModelError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    tier = decision["tier"]
    max_tokens = max_tokens_for_model(decision["model"], tier)
    tracker = BudgetTracker(max_cost_usd)
    try:
        primary_reservation = await tracker.reserve(
            prompt, decision["model"], max_tokens
        )
    except BudgetExceededError as e:
        raise HTTPException(status_code=422, detail=f"No eligible model is affordable: {e}") from e

    try:
        result = await call_model(
            prompt,
            model=decision["model"],
            max_tokens=max_tokens,
            budget_tracker=tracker,
            _reservation=primary_reservation,
            tier=tier,
        )
    except BudgetExceededError as e:
        if tracker.model_calls:
            _persist_request(
                prompt, tier, routing_mode, max_cost_usd, min_quality,
                tracker, "unavailable", False, str(e),
            )
        raise HTTPException(
            status_code=422,
            detail=f"No eligible fallback or retry is affordable: {e}",
        ) from e
    except (GroqAPIError, GeminiAPIError) as e:
        if tracker.model_calls:
            _persist_request(
                prompt, tier, routing_mode, max_cost_usd, min_quality,
                tracker, "unavailable", False, "Provider call failed",
            )
        raise HTTPException(
            status_code=502,
            detail=f"Provider call failed: {type(e).__name__}",
        ) from e
    except InvalidTokenUsageError as e:
        if tracker.model_calls:
            _persist_request(
                prompt, tier, routing_mode, max_cost_usd, min_quality,
                tracker, "unavailable", False, "Provider returned invalid token usage",
            )
        raise HTTPException(status_code=502, detail=str(e)) from e
    verification = await verify_quality(prompt, result["answer"], tracker)

    escalation_attempted = False
    escalation_reason = None
    if verification["status"] == "verified" and not verification["passed"]:
        stronger = [
            (candidate, max_tokens_for_model(candidate["model"], tier))
            for candidate in decision["candidates_considered"]
            if candidate["expected_quality"] > decision["expected_quality"]
        ]
        stronger.sort(key=lambda item: item[0]["expected_quality"])
        next_choice = None
        for candidate, candidate_max_tokens in stronger:
            try:
                reservation = await tracker.reserve(
                    prompt, candidate["model"], candidate_max_tokens
                )
            except BudgetExceededError:
                continue
            else:
                next_choice = candidate, candidate_max_tokens, reservation
                break

        if next_choice is not None:
            candidate, candidate_max_tokens, reservation = next_choice
            escalation_attempted = True
            escalation_reason = (
                f"Quality check failed (score {verification['score']}/5); escalated"
            )
            try:
                result = await call_model(
                    prompt,
                    model=candidate["model"],
                    max_tokens=candidate_max_tokens,
                    budget_tracker=tracker,
                    is_escalation=True,
                    _reservation=reservation,
                    tier=tier,
                )
            except BudgetExceededError as e:
                escalation_reason = f"Escalation call was blocked by the remaining budget: {e}"
            except (GroqAPIError, GeminiAPIError) as e:
                escalation_reason = (
                    f"Escalation provider call failed: {type(e).__name__}"
                )
            else:
                verification = await verify_quality(prompt, result["answer"], tracker)
        elif stronger:
            escalation_reason = (
                "Quality check failed but no stronger model fits the remaining budget"
            )
        else:
            escalation_reason = (
                "Quality check failed but no stronger candidate was available"
            )
    elif verification["status"] == "unavailable":
        escalation_reason = "Quality verification unavailable; escalation was skipped"

    request_id = _persist_request(
        prompt=prompt,
        tier=tier,
        routing_mode=routing_mode,
        max_cost_usd=max_cost_usd,
        min_quality=min_quality,
        tracker=tracker,
        quality_status=verification["status"],
        escalation_attempted=escalation_attempted,
        escalation_reason=escalation_reason,
    )

    result.update(tracker.as_response_fields())
    result["request_id"] = request_id
    result["quality_score"] = verification["score"]
    result["quality_passed"] = verification["passed"]
    result["quality_verification_status"] = verification["status"]
    result["escalated"] = escalation_attempted
    result["escalation_attempted"] = escalation_attempted
    result["escalation_reason"] = escalation_reason
    result["total_cost_usd"] = float(tracker.accounted_cost)
    result["model_calls"] = tracker.model_calls
    result["routing_mode"] = routing_mode
    result["routing_reason"] = decision["reason"]
    result["candidates_considered"] = decision.get("candidates_considered", [])
    result["estimated_primary_cost_usd"] = decision.get("estimated_cost")
    result["rejected_candidates"] = decision.get("rejected", [])

    return result


def _get_runtime_router():
    router = getattr(app.state, "runtime_router", None)
    if router is not None:
        return router
    from routers.registry import RouterVersionManager

    mgr = RouterVersionManager()
    if mgr.get_active_version() is not None:
        router = mgr.load_active()
        app.state.runtime_router = router
        return router
    return None


def _get_decision_logger():
    dlog = getattr(app.state, "decision_logger", None)
    if dlog is not None:
        return dlog
    from telemetry.decision_log import DecisionLogger

    dlog = DecisionLogger()
    app.state.decision_logger = dlog
    return dlog


@app.post("/v1/chat/completions")
async def chat_completions(
    payload: dict,
    api_key_id: str = Depends(verify_api_key),
):
    from pydantic import ValidationError
    from serving.chat_completions import ChatCompletionRequest, execute_chat_completion

    try:
        req = ChatCompletionRequest.model_validate(payload)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e

    caller = getattr(app.state, "model_caller", call_model)
    return await execute_chat_completion(
        req,
        runtime_router=_get_runtime_router(),
        decision_logger=_get_decision_logger(),
        model_caller=caller,
    )


@app.post("/v1/feedback")
async def record_outcome_feedback(
    payload: dict,
    api_key_id: str = Depends(verify_api_key),
):
    from pydantic import ValidationError
    from serving.chat_completions import FeedbackRequest

    try:
        req = FeedbackRequest.model_validate(payload)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e

    dlog = _get_decision_logger()
    try:
        return dlog.record_feedback(
            req.request_id,
            feedback_type=req.feedback_type,
            raw_value=req.raw_value,
            notes=req.notes,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
