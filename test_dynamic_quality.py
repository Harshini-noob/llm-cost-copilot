from dynamic_quality import get_live_quality_score
from model_registry import MODELS

for entry in MODELS.values():
    model = entry["model"]
    correct_fallback = entry["quality_tier"]
    result = get_live_quality_score(model, fallback_score=correct_fallback)
    print(model, "→", result)