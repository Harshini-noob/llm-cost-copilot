import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

_supabase_client = None

def get_supabase():
    global _supabase_client
    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_KEY must be set in .env — "
                "Supabase client cannot be initialized."
            )
        _supabase_client = create_client(url, key)
    return _supabase_client


MIN_SAMPLES_REQUIRED = 5


def get_live_quality_score(model: str, fallback_score: float, sample_size: int = 50) -> dict:
    try:
        result = (
            get_supabase().table("model_calls")
            .select("id, quality_scores(score)")
            .eq("model", model)
            .order("created_at", desc=True)
            .limit(sample_size)
            .execute()
        )
    except Exception as e:
        # Covers both connection issues AND a missing/invalid .env
        # (RuntimeError from get_supabase() lands here too) — either way,
        # degrade gracefully instead of crashing the routing decision
        # over a "nice to have" live quality lookup.
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