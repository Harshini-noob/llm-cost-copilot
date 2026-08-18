import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

MIN_SAMPLES_REQUIRED = 5


def get_live_quality_score(model: str, fallback_score: float, sample_size: int = 50) -> dict:
    try:
        result = (
            supabase.table("model_calls")
            .select("id, quality_scores(score)")
            .eq("model", model)
            .order("created_at", desc=True)
            .limit(sample_size)
            .execute()
        )
    except Exception as e:
        # Network/connection issue with Supabase — degrade gracefully
        # instead of crashing the entire routing decision over a
        # "nice to have" live quality lookup.
        print(f"⚠️  Could not reach Supabase for live quality data on '{model}': {e}")
        return {
            "quality": fallback_score,
            "source": "static_fallback_connection_error",
            "sample_count": 0
        }

    scores = []
    for row in result.data:
        for qs in row.get("quality_scores", []):
            if qs.get("score") is not None:
                scores.append(qs["score"])

    if len(scores) < MIN_SAMPLES_REQUIRED:
        return {
            "quality": fallback_score,
            "source": "static_fallback",
            "sample_count": len(scores)
        }

    avg_raw_score = sum(scores) / len(scores)
    normalized = avg_raw_score / 5.0

    return {
        "quality": round(normalized, 3),
        "source": "live_data",
        "sample_count": len(scores)
    }