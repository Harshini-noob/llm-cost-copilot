import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config" / "models.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_models():
    config = load_config()

    return {
        model["id"]: model
        for model in config["models"]
        if model.get("enabled", True)
    }


MODELS = load_models()


def get_model_by_id(model_id: str) -> dict:
    if model_id in MODELS:
        return MODELS[model_id]

    for entry in MODELS.values():
        if entry["model_name"] == model_id:
            return entry

    raise ValueError(f"Model '{model_id}' not found in V2 registry")


def get_pricing_dict() -> dict:
    return {
        entry["model_name"]: {
            "input": entry["input_price_per_1m"],
            "output": entry["output_price_per_1m"],
        }
        for entry in MODELS.values()
    }