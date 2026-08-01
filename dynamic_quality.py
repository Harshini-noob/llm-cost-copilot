import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

MIN_SAMPLES_REQUIRED = 5  # don't trust a live score until we have enough real data


def get_live_quality_score(model: str, fallback_score: float, sample_size: int = 50) -> dict:
    """
    Query the last `sample_size` quality_scores for this model (joined through
    model_calls) and return a live average, normalized to 0-1 to match the
    registry's scale (raw scores are 1-5).

    If there isn't enough real data yet, honestly falls back to the
    hand-set registry score instead of pretending we have live data.
    """
    result = (
        supabase.table("model_calls")
        .select("id, quality_scores(score)")
        .eq("model", model)
        .order("created_at", desc=True)
        .limit(sample_size)
        .execute()
    )

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

    avg_raw_score = sum(scores) / len(scores)  # this is on a 1-5 scale
    normalized = avg_raw_score / 5.0            # convert to 0-1 to match registry scale

    return {
        "quality": round(normalized, 3),
        "source": "live_data",
        "sample_count": len(scores)
    }
    