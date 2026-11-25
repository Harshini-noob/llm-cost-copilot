"""
serving/chat_completions.py

OpenAI-compatible /v1/chat/completions handler with:
- Unified M5 RuntimeRouter integration & lambda presets
- Live model_registry validation
- Ranked fallback across eligible candidates with BudgetTracker enforcement
- Separation of expected_cost_usd, actual_cost_usd, and budget_accounted_cost_usd
- Auditable SQLite DecisionLogger recording
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
import time
from typing import Any, Awaitable, Callable
import uuid

from fastapi import HTTPException
from google.genai.errors import APIError as GeminiAPIError
from groq import APIStatusError, RateLimitError
from pydantic import BaseModel, Field, field_validator

from budget import BudgetExceededError, BudgetTracker, InvalidTokenUsageError
from model_registry import MODELS, get_model_by_id
from models import call_model
from routers.base import LAMBDA_PRESETS, RouteDecision, RuntimeRouter
from telemetry.decision_log import DecisionLogger

PROVIDER_TRANSIENT_ERRORS = (
    asyncio.TimeoutError,
    TimeoutError,
    RateLimitError,
    APIStatusError,
    GeminiAPIError,
    InvalidTokenUsageError,
    ConnectionError,
    OSError,
)


class ChatMessage(BaseModel):
    role: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="auto", min_length=1)
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_cost_usd: Decimal | None = Field(default=None, gt=0)
    task_category: str = Field(default="unknown")

    @field_validator("messages")
    @classmethod
    def _non_empty_messages(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if not v:
            raise ValueError("messages must contain at least one message")
        return v


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=1)
    feedback_type: str = Field(..., min_length=1)
    raw_value: Any
    notes: str | None = None


def messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Convert OpenAI chat messages deterministically into a single router prompt string."""
    if len(messages) == 1:
        return messages[0].content.strip()
    return "\n".join(f"{m.role.strip()}: {m.content.strip()}" for m in messages).strip()


def live_model_ids() -> list[str]:
    """Return available live model IDs from model_registry.MODELS."""
    return [
        entry["model"]
        for entry in MODELS.values()
        if entry.get("available", True)
    ]


async def execute_chat_completion(
    req: ChatCompletionRequest,
    *,
    runtime_router: RuntimeRouter | None,
    decision_logger: DecisionLogger | None = None,
    model_caller: Callable[..., Awaitable[dict[str, Any]]] = call_model,
) -> dict[str, Any]:
    """Execute an OpenAI-compatible chat completion with ranked fallback and budget tracking."""
    t_start = time.perf_counter()
    prompt = messages_to_prompt(req.messages)
    if not prompt:
        raise HTTPException(status_code=400, detail="Converted prompt is empty")

    live_models = live_model_ids()
    model_key = req.model.strip()
    is_auto = model_key.lower() in LAMBDA_PRESETS or model_key.lower().startswith("auto:")

    route_decision: RouteDecision | None = None
    if is_auto:
        if runtime_router is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No active runtime router bundle is loaded for auto routing. "
                    "Train and promote a router bundle or specify an explicit live model ID."
                ),
            )
        try:
            route_decision = runtime_router.route(
                prompt,
                lam=model_key,
                eligible_models=live_models,
                task_category=req.task_category,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

        ranked_candidates = [m for m in route_decision.ranked_candidates if m in live_models]
        if not ranked_candidates:
            raise HTTPException(
                status_code=422,
                detail="No routed candidates are registered in the live model registry",
            )
        predicted_scores = route_decision.predicted_scores
        router_version = route_decision.router_version
        lambda_value: float | None = route_decision.lambda_value
        router_latency_ms = route_decision.router_latency_ms
        expected_costs = route_decision.expected_costs_usd
    else:
        try:
            get_model_by_id(model_key)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        if model_key not in live_models:
            raise HTTPException(status_code=422, detail=f"Model '{model_key}' is currently unavailable")
        ranked_candidates = [model_key]
        predicted_scores = {model_key: None}
        router_version = "explicit"
        lambda_value = None
        router_latency_ms = 0.0
        expected_costs = {}

    tracker = BudgetTracker(req.max_cost_usd)
    fallback_attempts: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    chosen_result: dict[str, Any] | None = None
    chosen_model: str | None = None
    attempted_models: set[str] = set()
    saw_budget_error = False

    for candidate in ranked_candidates:
        if candidate in attempted_models:
            continue
        attempted_models.add(candidate)

        try:
            reservation = await tracker.reserve(prompt, candidate, req.max_tokens)
        except BudgetExceededError as e:
            saw_budget_error = True
            err_rec = {
                "model": candidate,
                "error_type": "BudgetExceededError",
                "error": str(e),
                "cost_usd": 0.0,
            }
            fallback_attempts.append(err_rec)
            errors.append(err_rec)
            continue

        try:
            res = await model_caller(
                prompt,
                model=candidate,
                max_tokens=req.max_tokens,
                allow_fallback=False,
                budget_tracker=tracker,
                _reservation=reservation,
            )
            chosen_result = res
            chosen_model = candidate
            break
        except asyncio.CancelledError:
            raise
        except BudgetExceededError as e:
            saw_budget_error = True
            err_rec = {
                "model": candidate,
                "error_type": "BudgetExceededError",
                "error": str(e),
                "cost_usd": 0.0,
            }
            fallback_attempts.append(err_rec)
            errors.append(err_rec)
        except PROVIDER_TRANSIENT_ERRORS as e:
            last_call_cost = (
                float(tracker.model_calls[-1].get("cost_usd", 0.0))
                if tracker.model_calls
                else 0.0
            )
            err_rec = {
                "model": candidate,
                "error_type": type(e).__name__,
                "error": str(e),
                "cost_usd": last_call_cost,
            }
            fallback_attempts.append(err_rec)
            errors.append(err_rec)

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0

    if chosen_result is None or chosen_model is None:
        if decision_logger is not None:
            decision_logger.log_decision(
                prompt=prompt,
                chosen_model=ranked_candidates[0],
                predicted_scores=predicted_scores,
                ranked_candidates=ranked_candidates,
                lambda_value=lambda_value if lambda_value is not None else 0.0,
                router_version=router_version,
                expected_cost_usd=float(expected_costs.get(ranked_candidates[0], 0.0)),
                latency_ms=total_latency_ms,
                router_latency_ms=router_latency_ms,
                actual_cost_usd=None,
                budget_accounted_cost_usd=float(tracker.accounted_cost),
                fallback_attempts=fallback_attempts,
                errors=errors,
                quality_status="failed",
                task_category=req.task_category,
            )
        status_code = 422 if saw_budget_error and len(fallback_attempts) == len(attempted_models) else 502
        raise HTTPException(
            status_code=status_code,
            detail={
                "message": "All eligible candidate models failed or exceeded budget",
                "attempts": fallback_attempts,
            },
        )

    input_tokens = chosen_result.get("input_tokens")
    output_tokens = chosen_result.get("output_tokens")
    cost_is_estimated = bool(chosen_result.get("cost_is_estimated", False))
    actual_cost_usd = None if cost_is_estimated else float(tracker.actual_cost)

    if chosen_model in expected_costs:
        expected_cost_usd = float(expected_costs[chosen_model])
    elif tracker.model_calls:
        expected_cost_usd = float(tracker.model_calls[-1].get("estimated_cost_usd", 0.0))
    else:
        expected_cost_usd = 0.0

    request_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
    quality_status = "unverified"

    if decision_logger is not None:
        prompt_emb = None
        if runtime_router is not None and is_auto:
            try:
                if hasattr(runtime_router, "_knn") and runtime_router._knn.encoder is not None:
                    prompt_emb = runtime_router._knn.encoder.encode_one(prompt)
                elif hasattr(runtime_router, "_lr") and runtime_router._lr.encoder is not None:
                    prompt_emb = runtime_router._lr.encoder.encode_one(prompt)
            except Exception:
                prompt_emb = None

        decision_logger.log_decision(
            request_id=request_id,
            prompt=prompt,
            chosen_model=chosen_model,
            predicted_scores=predicted_scores,
            ranked_candidates=ranked_candidates,
            lambda_value=lambda_value if lambda_value is not None else 0.0,
            router_version=router_version,
            expected_cost_usd=expected_cost_usd,
            latency_ms=total_latency_ms,
            router_latency_ms=router_latency_ms,
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            actual_cost_usd=actual_cost_usd,
            budget_accounted_cost_usd=float(tracker.accounted_cost),
            fallback_attempts=fallback_attempts,
            errors=errors,
            quality_status=quality_status,
            task_category=req.task_category,
            prompt_embedding=prompt_emb,
        )

    prompt_tok_val = input_tokens if isinstance(input_tokens, int) else 0
    comp_tok_val = output_tokens if isinstance(output_tokens, int) else 0

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": chosen_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": chosen_result.get("answer", ""),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tok_val,
            "completion_tokens": comp_tok_val,
            "total_tokens": prompt_tok_val + comp_tok_val,
        },
        "routing": {
            "chosen_model": chosen_model,
            "predicted_scores": predicted_scores,
            "ranked_candidates": ranked_candidates,
            "router_version": router_version,
            "lambda": lambda_value,
            "expected_cost_usd": expected_cost_usd,
            "actual_cost_usd": actual_cost_usd,
            "budget_accounted_cost_usd": float(tracker.accounted_cost),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(total_latency_ms, 3),
            "router_latency_ms": round(router_latency_ms, 3),
            "fallback_attempts": fallback_attempts,
            "quality_status": quality_status,
        },
    }
