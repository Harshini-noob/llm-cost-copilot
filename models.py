from providers.groq_provider import GroqProvider
from providers.gemini_provider import GeminiProvider
from model_registry import get_model_by_id
from groq import RateLimitError, APIStatusError
from google.genai.errors import APIError as GeminiAPIError
import asyncio

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

    input_cost = (result["input_tokens"] / 1_000_000) * model_info["input_cost_per_million"]
    output_cost = (result["output_tokens"] / 1_000_000) * model_info["output_cost_per_million"]

    result["cost_usd"] = round(input_cost + output_cost, 8)
    result["fell_back"] = False
    return result


async def call_model(prompt: str, model: str = "openai/gpt-oss-20b", max_tokens: int = 500,
                      allow_fallback: bool = True, _is_retry: bool = False) -> dict:
    try:
        return await _raw_call(prompt, model, max_tokens)

    except (RateLimitError, GeminiAPIError) as e:
        if not allow_fallback:
            raise

        fallback = FALLBACK_MODEL.get(model)
        if _is_retry or fallback is None or fallback == model:
            raise
        print(f"Warning: rate limit or quota hit on '{model}', falling back to '{fallback}'")
        result = await call_model(prompt, model=fallback, max_tokens=max_tokens,
                                   allow_fallback=allow_fallback, _is_retry=True)
        result["fell_back"] = True
        return result

    except APIStatusError as e:
        if _is_retry:
            raise
        print(f"Warning: API error on '{model}': {e}. Retrying once in 3s...")
        await asyncio.sleep(3)  # non-blocking sleep — doesn't freeze the event loop
        return await call_model(prompt, model=model, max_tokens=max_tokens,
                                 allow_fallback=allow_fallback, _is_retry=True)