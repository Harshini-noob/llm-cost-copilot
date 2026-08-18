MODELS = {
    "small": {
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "quality_tier": 0.75,
        "input_cost_per_million": 0.075,
        "output_cost_per_million": 0.30,
        "context_window": 131072,
        "avg_latency_sec": 0.5,
        "available": True,
    },
    "medium": {
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "quality_tier": 0.85,
        "input_cost_per_million": 0.15,
        "output_cost_per_million": 0.60,
        "context_window": 131072,
        "avg_latency_sec": 1.1,
        "available": True,
    },
    "gemini_flash": {
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "quality_tier": 0.90,
        "input_cost_per_million": 0.0,
        "output_cost_per_million": 0.0,
        "context_window": 1000000,
        "avg_latency_sec": 2.5,
        "available": True,
    },
}


def get_model_by_id(model_id: str) -> dict:
    for entry in MODELS.values():
        if entry["model"] == model_id:
            return entry
    raise ValueError(f"Model '{model_id}' not found in registry")


def get_pricing_dict() -> dict:
    return {
        entry["model"]: {
            "input": entry["input_cost_per_million"],
            "output": entry["output_cost_per_million"]
        }
        for entry in MODELS.values()
    }