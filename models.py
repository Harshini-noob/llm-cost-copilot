from providers.groq_provider import GroqProvider
from providers.gemini_provider import GeminiProvider
from model_registry import get_model_by_id
from groq import RateLimitError, APIStatusError
from google.genai.errors import APIError as GeminiAPIError
import asyncio
from budget import BudgetTracker, token_cost, worst_case_cost

_providers = {
    "groq": GroqProvider(),
    "gemini": GeminiProvider(),
}

FALLBACK_MODEL = {
    "openai/gpt-oss-120b": "openai/gpt-oss-20b",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",
    "gemini-2.5-flash": "openai/gpt-oss-120b",
}


async def _raw_call(prompt: str, model: str, max_tokens: int) -> dict:
    model_info = get_model_by_id(model)
    provider = _providers[model_info["provider"]]

    result = await provider.generate(prompt, model=model, max_tokens=max_tokens)

    input_tokens = result.get("input_tokens")
    output_tokens = result.get("output_tokens")
    if (
        isinstance(input_tokens, int) and not isinstance(input_tokens, bool)
        and input_tokens >= 0
        and isinstance(output_tokens, int) and not isinstance(output_tokens, bool)
        and output_tokens >= 0
    ):
        result["cost_usd"] = float(token_cost(model, input_tokens, output_tokens))
        result["cost_is_estimated"] = False
    else:
        result["cost_usd"] = float(worst_case_cost(prompt, model, max_tokens))
        result["cost_is_estimated"] = True
    result["fell_back"] = False
    return result


async def call_model(prompt: str, model: str = "openai/gpt-oss-20b", max_tokens: int = 500,
                      allow_fallback: bool = True, _is_retry: bool = False,
                      budget_tracker: BudgetTracker = None,
                      call_type: str = "generation",
                      is_escalation: bool = False,
                      _was_fallback: bool = False) -> dict:
    if budget_tracker is not None:
        budget_tracker.authorize(prompt, model, max_tokens)

    try:
        result = await _raw_call(prompt, model, max_tokens)
    except Exception as e:
        if budget_tracker is not None:
            budget_tracker.record_failure(
                prompt, model, max_tokens, call_type, is_escalation,
                _was_fallback, e
            )

        if isinstance(e, (RateLimitError, GeminiAPIError)):
            if not allow_fallback:
                raise

            fallback = FALLBACK_MODEL.get(model)
            if _is_retry or fallback is None or fallback == model:
                raise
            print(f"Warning: rate limit or quota hit on '{model}', falling back to '{fallback}'")
            result = await call_model(
                prompt, model=fallback, max_tokens=max_tokens,
                allow_fallback=allow_fallback, _is_retry=True,
                budget_tracker=budget_tracker, call_type=call_type,
                is_escalation=is_escalation, _was_fallback=True
            )
            result["fell_back"] = True
            return result

        if isinstance(e, APIStatusError):
            if _is_retry:
                raise
            print(f"Warning: API error on '{model}': {e}. Retrying once in 3s...")
            await asyncio.sleep(3)
            return await call_model(
                prompt, model=model, max_tokens=max_tokens,
                allow_fallback=allow_fallback, _is_retry=True,
                budget_tracker=budget_tracker, call_type=call_type,
                is_escalation=is_escalation, _was_fallback=_was_fallback
            )

        raise

    if budget_tracker is not None:
        record = budget_tracker.record_response(
            prompt, model, max_tokens, result, call_type, is_escalation,
            _was_fallback
        )
        result["cost_usd"] = record["cost_usd"]
        result["cost_is_estimated"] = record["cost_is_estimated"]

    return result