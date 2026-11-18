from model_registry_v2 import MODELS, get_model_by_id, get_pricing_dict


def test_models_loaded():
    assert len(MODELS) >= 1


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