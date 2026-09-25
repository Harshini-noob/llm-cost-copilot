import asyncio
from dynamic_quality import get_live_quality_score
from model_registry import MODELS

async def main():
    for entry in MODELS.values():
        model = entry["model"]
        correct_fallback = entry["quality_tier"]
        result = await get_live_quality_score(model, fallback_score=correct_fallback)
        print(model, "->", result)

asyncio.run(main())