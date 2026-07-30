# Central registry — the single source of truth for every model's identity,
# pricing, and quality metadata.

MODELS = {
    "small": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "quality_tier": 0.75,
        "input_cost_per_million": 0.05,
        "output_cost_per_million": 0.08,
        "context_window": 131072,
        "avg_latency_sec": 0.6,
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
    "large": {
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "quality_tier": 1.0,
        "input_cost_per_million": 0.59,
        "output_cost_per_million": 0.79,
        "context_window": 131072,
        "avg_latency_sec": 1.4,
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