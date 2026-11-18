import asyncio
import pytest

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
    assert "Simulated provider failure" in result["error"]

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