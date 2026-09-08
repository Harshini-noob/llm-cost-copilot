import time
from providers.groq_provider import GroqProvider
from providers.gemini_provider import GeminiProvider
from model_registry import get_model_by_id
from groq import RateLimitError, APIStatusError

_providers = {
    "groq": GroqProvider(),
    "gemini": GeminiProvider(),
}

# Fallback only applies within Groq for now — Gemini's free tier isn't
# rate-limit-fragile in the same way, and cross-provider fallback would
# need its own careful design (different quality/latency tradeoffs).
FALLBACK_MODEL = {
    "openai/gpt-oss-120b": "openai/gpt-oss-20b",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b",  # nothing cheaper to fall back to on Groq
}


def _raw_call(prompt: str, model: str, max_tokens: int) -> dict:
    """Dispatches to the correct provider based on the model registry,
    then attaches cost using registry pricing (providers only report
    raw token counts — cost calculation is a registry concern, not a
    provider concern, so pricing logic lives here, not duplicated
    per-provider)."""
    model_info = get_model_by_id(model)
    provider = _providers[model_info["provider"]]

    result = provider.generate(prompt, model=model, max_tokens=max_tokens)

    input_cost = (result["input_tokens"] / 1_000_000) * model_info["input_cost_per_million"]
    output_cost = (result["output_tokens"] / 1_000_000) * model_info["output_cost_per_million"]

    result["cost_usd"] = round(input_cost + output_cost, 8)
    result["fell_back"] = False
    return result


def call_model(prompt: str, model: str = "openai/gpt-oss-20b", max_tokens: int = 500,
               allow_fallback: bool = True, _is_retry: bool = False) -> dict:
    try:
        return _raw_call(prompt, model, max_tokens)

    except RateLimitError as e:
        if not allow_fallback:
            raise

        fallback = FALLBACK_MODEL.get(model)
        if _is_retry or fallback is None or fallback == model:
            raise
        print(f"⚠️  Rate limit hit on '{model}', falling back to '{fallback}'")
        result = call_model(prompt, model=fallback, max_tokens=max_tokens,
                             allow_fallback=allow_fallback, _is_retry=True)
        result["fell_back"] = True
        return result

    except APIStatusError as e:
        if _is_retry:
            raise
        print(f"⚠️  API error on '{model}': {e}. Retrying once in 3s...")
        time.sleep(3)
        return call_model(prompt, model=model, max_tokens=max_tokens,
                           allow_fallback=allow_fallback, _is_retry=True)