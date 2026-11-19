from model_registry_v2 import (
    MODELS,
    get_model_by_id,
    get_pricing_dict,
    load_config,
)


def test_models_loaded():
    assert len(MODELS) == 4
    assert all(model["enabled"] for model in MODELS.values())
    historical = next(
        model
        for model in load_config()["models"]
        if model["id"] == "gemini_3_8_flash"
    )
    assert historical["enabled"] is False
    assert historical["id"] not in MODELS


def test_lookup_by_id():
    model = get_model_by_id("gpt_oss_20b")
    assert model["model_name"] == "openai/gpt-oss-20b"


def test_lookup_by_provider_model_name():
    model = get_model_by_id("openai/gpt-oss-20b")
    assert model["id"] == "gpt_oss_20b"


def test_pricing():
    pricing = get_pricing_dict()

    assert "openai/gpt-oss-20b" in pricing
    assert pricing["openai/gpt-oss-20b"]["input"] == 0.075
    assert pricing["openai/gpt-oss-20b"]["output"] == 0.30


def test_added_models_have_verified_registry_pricing():
    gemini = get_model_by_id("gemini_3_5_flash_lite")
    qwen = get_model_by_id("qwen3_30b_a3b_openrouter")

    assert (gemini["input_price_per_1m"], gemini["output_price_per_1m"]) == (
        0.30,
        2.50,
    )
    assert (qwen["input_price_per_1m"], qwen["output_price_per_1m"]) == (
        0.10,
        0.30,
    )