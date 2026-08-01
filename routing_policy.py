from model_registry import MODELS
from router import classify_llm, MAX_TOKENS_BY_TIER
from cost_estimator import estimate_request_cost, estimate_tokens
from dynamic_quality import get_live_quality_score

COMPLEXITY_MIN_QUALITY = {"simple": 0.70, "medium": 0.75, "complex": 0.90}
QUALITY_SAFETY_MARGIN = 0.03


class InvalidConstraintError(Exception):
    pass


def build_model_metadata() -> dict:
    """Builds model metadata fresh each time it's called, checking live quality
    data per model instead of relying on a fixed dict computed once at import time.
    Falls back to the registry's hand-set quality_tier if not enough real
    verification data exists yet for a given model."""
    metadata = {}
    for entry in MODELS.values():
        model_id = entry["model"]
        quality_info = get_live_quality_score(model_id, fallback_score=entry["quality_tier"])

        metadata[model_id] = {
            "input_price": entry["input_cost_per_million"],
            "output_price": entry["output_cost_per_million"],
            "avg_latency_sec": entry["avg_latency_sec"],
            "expected_quality": quality_info["quality"],
            "quality_source": quality_info["source"],
            "quality_sample_count": quality_info["sample_count"],
            "context_window": entry["context_window"],
            "available": entry["available"],
        }
    return metadata


def validate_constraints(routing_mode: str, max_cost_usd: float = None, min_quality: float = None):
    valid_modes = {"economy", "balanced", "quality", "latency"}
    if routing_mode not in valid_modes:
        raise InvalidConstraintError(f"routing_mode must be one of {valid_modes}, got '{routing_mode}'")
    if max_cost_usd is not None and max_cost_usd <= 0:
        raise InvalidConstraintError("max_cost_usd must be positive")
    if min_quality is not None and not (0.0 <= min_quality <= 1.0):
        raise InvalidConstraintError("min_quality must be between 0.0 and 1.0")


def estimate_cost(model: str, prompt: str, tier: str) -> float:
    expected_output = MAX_TOKENS_BY_TIER.get(tier, 300)
    return estimate_request_cost(prompt, model, expected_output_tokens=expected_output)


def select_model(prompt: str, routing_mode: str = "balanced",
                  max_cost_usd: float = None, min_quality: float = None) -> dict:

    validate_constraints(routing_mode, max_cost_usd, min_quality)

    model_metadata = build_model_metadata()

    tier = classify_llm(prompt)
    required_quality = min_quality if min_quality is not None else COMPLEXITY_MIN_QUALITY[tier]
    prompt_tokens = estimate_tokens(prompt)

    candidates = []
    rejected = []

    for model, meta in model_metadata.items():
        if not meta["available"]:
            rejected.append({"model": model, "reason": "unavailable"})
            continue

        if prompt_tokens > meta["context_window"]:
            rejected.append({"model": model, "reason": "exceeds context window"})
            continue

        if meta["expected_quality"] < required_quality + QUALITY_SAFETY_MARGIN:
            rejected.append({
                "model": model,
                "reason": f"quality {meta['expected_quality']} below required {required_quality}"
            })
            continue

        cost = estimate_cost(model, prompt, tier)
        if max_cost_usd is not None and cost > max_cost_usd:
            rejected.append({
                "model": model,
                "reason": f"estimated cost {cost} exceeds max_cost_usd {max_cost_usd}"
            })
            continue

        candidates.append({
            "model": model,
            "estimated_cost": cost,
            "expected_quality": meta["expected_quality"],
            "avg_latency_sec": meta["avg_latency_sec"],
            "quality_source": meta["quality_source"],
            "quality_sample_count": meta["quality_sample_count"],
        })

    if not candidates:
        fallback = "llama-3.3-70b-versatile"
        return {
            "model": fallback,
            "tier": tier,
            "reason": "No candidate met constraints; defaulted to strongest model.",
            "candidates_considered": [],
            "rejected": rejected
        }

    if routing_mode == "economy":
        best = min(candidates, key=lambda c: c["estimated_cost"])
        reason = f"Cheapest model meeting quality ≥ {required_quality}"

    elif routing_mode == "quality":
        best = max(candidates, key=lambda c: c["expected_quality"])
        reason = "Highest expected quality among eligible models"

    elif routing_mode == "latency":
        best = min(candidates, key=lambda c: c["avg_latency_sec"])
        reason = f"Fastest model meeting quality ≥ {required_quality}"

    else:  # balanced — normalized weighting across candidates
        costs = [c["estimated_cost"] for c in candidates]
        latencies = [c["avg_latency_sec"] for c in candidates]
        min_cost, max_cost = min(costs), max(costs)
        min_lat, max_lat = min(latencies), max(latencies)

        for c in candidates:
            cost_norm = 0 if max_cost == min_cost else (c["estimated_cost"] - min_cost) / (max_cost - min_cost)
            lat_norm = 0 if max_lat == min_lat else (c["avg_latency_sec"] - min_lat) / (max_lat - min_lat)
            c["score"] = (c["expected_quality"] * 0.5) + ((1 - cost_norm) * 0.35) + ((1 - lat_norm) * 0.15)

        best = max(candidates, key=lambda c: c["score"])
        reason = "Best balance of cost, quality, and latency"

    return {
        "model": best["model"],
        "tier": tier,
        "reason": reason,
        "estimated_cost": best["estimated_cost"],
        "expected_quality": best["expected_quality"],
        "candidates_considered": candidates,
        "rejected": rejected
    }