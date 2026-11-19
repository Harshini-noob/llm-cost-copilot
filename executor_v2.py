import os
import time
from collections.abc import Mapping

import litellm
from dotenv import load_dotenv

from model_registry_v2 import get_model_by_id


load_dotenv()


class _ExecutionError(Exception):
    pass


_API_KEY_ENV_VARS = {
    "groq": ("GROQ_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def _read(value, key, default=None):
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _litellm_model_name(model_config: dict) -> str:
    provider = model_config["provider"].lower()
    model_name = model_config["model_name"]
    prefixes = {
        "groq": "groq/",
        "google": "gemini/",
        "openrouter": "openrouter/",
    }

    if provider not in prefixes:
        raise _ExecutionError("Unsupported provider configuration.")

    prefix = prefixes[provider]
    return model_name if model_name.startswith(prefix) else f"{prefix}{model_name}"


def _provider_api_key(provider: str) -> str:
    env_vars = _API_KEY_ENV_VARS.get(provider)
    if env_vars is None:
        raise _ExecutionError("Unsupported provider configuration.")

    for env_var in env_vars:
        api_key = os.getenv(env_var)
        if api_key:
            return api_key

    expected = " or ".join(env_vars)
    raise _ExecutionError(
        f"Missing required environment variable: {expected}."
    )


def _extract_response(response) -> tuple[str, int, int]:
    choices = _read(response, "choices", [])
    if not choices:
        raise _ExecutionError("Provider response did not contain generated text.")

    message = _read(choices[0], "message")
    output = _read(message, "content")
    if not isinstance(output, str):
        raise _ExecutionError("Provider response did not contain generated text.")

    usage = _read(response, "usage")
    input_tokens = _read(usage, "prompt_tokens")
    if input_tokens is None:
        input_tokens = _read(usage, "input_tokens", 0)
    output_tokens = _read(usage, "completion_tokens")
    if output_tokens is None:
        output_tokens = _read(usage, "output_tokens", 0)

    return output, int(input_tokens or 0), int(output_tokens or 0)


def _calculate_cost(
    model_config: dict,
    input_tokens: int,
    output_tokens: int,
) -> float:
    input_cost = input_tokens * model_config["input_price_per_1m"] / 1_000_000
    output_cost = (
        output_tokens * model_config["output_price_per_1m"] / 1_000_000
    )
    return round(input_cost + output_cost, 10)


def _failure_response(
    model_id: str,
    provider: str | None,
    latency_ms: float,
    error: str,
) -> dict:
    return {
        "output": None,
        "model_id": model_id,
        "provider": provider,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0.0,
        "latency_ms": round(latency_ms, 2),
        "success": False,
        "error": error,
    }


async def execute_model(
    model_id: str,
    prompt: str,
    adapters: dict | None = None,
) -> dict:
    start_time = time.perf_counter()
    provider = None

    try:
        model_config = get_model_by_id(model_id)
    except ValueError:
        return _failure_response(
            model_id=model_id,
            provider=None,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            error="Model ID is not in the enabled registry.",
        )

    provider = model_config["provider"]
    try:
        if adapters is not None:
            adapter = adapters.get(provider)
            if adapter is None:
                raise _ExecutionError(
                    f"No adapter configured for provider: {provider}"
                )
            response = await adapter(model_config=model_config, prompt=prompt)
            output = response["output"]
            input_tokens = int(response.get("input_tokens") or 0)
            output_tokens = int(response.get("output_tokens") or 0)
        else:
            api_key = _provider_api_key(provider)
            response = await litellm.acompletion(
                model=_litellm_model_name(model_config),
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key,
                num_retries=0,
            )
            output, input_tokens, output_tokens = _extract_response(response)

        return {
            "output": output,
            "model_id": model_config["id"],
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": _calculate_cost(model_config, input_tokens, output_tokens),
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            "success": True,
            "error": None,
        }
    except _ExecutionError as exc:
        error = str(exc)
    except Exception as exc:
        error = f"Execution failed ({type(exc).__name__})."

    return _failure_response(
        model_id=model_id,
        provider=provider,
        latency_ms=(time.perf_counter() - start_time) * 1000,
        error=error,
    )