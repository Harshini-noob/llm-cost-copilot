import asyncio
import pytest

import executor_v2
from executor_v2 import execute_model
from model_registry_v2 import get_model_by_id


# -------------------------
# Mock Provider
# -------------------------

async def fake_provider(model_config, prompt):
    return {
        "output": f"Mock response to: {prompt}",
        "input_tokens": 100,
        "output_tokens": 50,
    }


# -------------------------
# Test 1: Successful execution
# -------------------------

def test_executor_success():

    adapters = {
        model_config["provider"]: fake_provider
        for model_config in [
            get_model_by_id("gpt_oss_20b")
        ]
    }

    result = asyncio.run(
        execute_model(
            model_id="gpt_oss_20b",
            prompt="Explain binary search",
            adapters=adapters,
        )
    )

    assert result["success"] is True
    assert result["output"] is not None
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 50
    assert result["cost"] > 0
    assert result["latency_ms"] >= 0

    print("\nSUCCESS RESULT:")
    print(result)


# -------------------------
# Test 2: Cost calculation
# -------------------------

def test_executor_cost_calculation():

    model_config = get_model_by_id("gpt_oss_20b")

    adapters = {
        model_config["provider"]: fake_provider
    }

    result = asyncio.run(
        execute_model(
            model_id="gpt_oss_20b",
            prompt="Test prompt",
            adapters=adapters,
        )
    )

    expected_cost = (
        100 * model_config["input_price_per_1m"] / 1_000_000
        +
        50 * model_config["output_price_per_1m"] / 1_000_000
    )

    assert result["cost"] == round(expected_cost, 10)


# -------------------------
# Test 3: Provider failure
# -------------------------

async def failing_provider(model_config, prompt):
    raise RuntimeError("Simulated provider failure")


def test_executor_failure():

    model_config = get_model_by_id("gpt_oss_20b")

    adapters = {
        model_config["provider"]: failing_provider
    }

    result = asyncio.run(
        execute_model(
            model_id="gpt_oss_20b",
            prompt="Test failure",
            adapters=adapters,
        )
    )

    assert result["success"] is False
    assert result["output"] is None
    assert result["cost"] == 0.0
    assert "RuntimeError" in result["error"]

    print("\nFAILURE RESULT:")
    print(result)


# -------------------------
# Test 4: Missing adapter
# -------------------------

def test_missing_adapter():

    result = asyncio.run(
        execute_model(
            model_id="gpt_oss_20b",
            prompt="Test missing adapter",
            adapters={},
        )
    )

    assert result["success"] is False
    assert "No adapter configured" in result["error"]


def test_litellm_execution_extracts_usage_and_calculates_registry_cost(
    monkeypatch,
):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-a-real-credential")
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [{"message": {"content": "generated response"}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 40},
        }

    monkeypatch.setattr(executor_v2.litellm, "acompletion", fake_acompletion)
    result = asyncio.run(execute_model("gpt_oss_20b", "test prompt"))

    config = get_model_by_id("gpt_oss_20b")
    expected_cost = (
        120 * config["input_price_per_1m"] / 1_000_000
        + 40 * config["output_price_per_1m"] / 1_000_000
    )
    assert result["success"] is True
    assert result["output"] == "generated response"
    assert result["input_tokens"] == 120
    assert result["output_tokens"] == 40
    assert result["cost"] == round(expected_cost, 10)
    assert calls[0]["model"] == "groq/openai/gpt-oss-20b"
    assert calls[0]["api_key"] == "test-key-not-a-real-credential"
    assert calls[0]["num_retries"] == 0


def test_google_api_key_alias_is_passed_to_litellm(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [{"message": {"content": "generated response"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(executor_v2.litellm, "acompletion", fake_acompletion)
    result = asyncio.run(
        execute_model("gemini_3_5_flash_lite", "test prompt")
    )

    assert result["success"] is True
    assert calls[0]["api_key"] == "google-test-key"
    assert calls[0]["model"] == "gemini/gemini-3.5-flash-lite"


def test_litellm_model_names_resolve_for_each_provider():
    assert executor_v2._litellm_model_name(
        get_model_by_id("gemini_3_5_flash_lite")
    ) == "gemini/gemini-3.5-flash-lite"
    assert executor_v2._litellm_model_name(
        get_model_by_id("qwen3_30b_a3b_openrouter")
    ) == "openrouter/qwen/qwen3-30b-a3b-instruct-2507"


def test_latency_uses_perf_counter(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    ticks = iter((12.0, 12.125))
    monkeypatch.setattr(executor_v2.time, "perf_counter", lambda: next(ticks))

    async def fake_acompletion(**kwargs):
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    monkeypatch.setattr(executor_v2.litellm, "acompletion", fake_acompletion)
    result = asyncio.run(execute_model("gpt_oss_20b", "prompt"))

    assert result["latency_ms"] == 125.0


def test_missing_api_key_fails_without_calling_litellm(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    async def unexpected_call(**kwargs):
        pytest.fail("LiteLLM must not be called without provider credentials")

    monkeypatch.setattr(executor_v2.litellm, "acompletion", unexpected_call)
    result = asyncio.run(execute_model("gpt_oss_20b", "prompt"))

    assert result["success"] is False
    assert "GROQ_API_KEY" in result["error"]
    assert "test-key" not in result["error"]


def test_provider_exception_is_standardized_without_leaking_secret(monkeypatch):
    secret = "sample-secret-value"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret)

    async def failing_acompletion(**kwargs):
        raise RuntimeError(f"request failed with {secret}")

    monkeypatch.setattr(executor_v2.litellm, "acompletion", failing_acompletion)
    result = asyncio.run(
        execute_model("qwen3_30b_a3b_openrouter", "prompt")
    )

    assert result["success"] is False
    assert "RuntimeError" in result["error"]
    assert secret not in result["error"]
    assert result["input_tokens"] == result["output_tokens"] == 0


def test_invalid_model_id_returns_standard_failure():
    result = asyncio.run(execute_model("not-registered", "prompt"))

    assert result["success"] is False
    assert result["model_id"] == "not-registered"
    assert result["provider"] is None
    assert result["error"] == "Model ID is not in the enabled registry."