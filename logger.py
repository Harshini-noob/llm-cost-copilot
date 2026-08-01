import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def log_full_request(prompt: str, tier: str, routing_mode: str,
                      max_cost_usd: float, min_quality: float,
                      model_calls_data: list, quality_data: dict = None) -> str:
    """
    Logs a complete request lifecycle to Supabase:
    - one requests row
    - one or more model_calls rows (more than one only if escalation happened)
    - one quality_scores row per model_calls row that was verified

    model_calls_data: list of dicts, each shaped like:
        {"model": ..., "input_tokens": ..., "output_tokens": ..., "cost_usd": ...,
         "latency_sec": ..., "is_escalation": bool, "fell_back": bool,
         "quality_score": int, "quality_passed": bool}
    """
    req = supabase.table("requests").insert({
        "prompt": prompt,
        "tier": tier,
        "routing_mode": routing_mode,
        "max_cost_usd": max_cost_usd,
        "min_quality": min_quality
    }).execute()
    request_id = req.data[0]["id"]

    for call in model_calls_data:
        call_result = supabase.table("model_calls").insert({
            "request_id": request_id,
            "model": call["model"],
            "input_tokens": call.get("input_tokens"),
            "output_tokens": call.get("output_tokens"),
            "cost_usd": call.get("cost_usd"),
            "latency_sec": call.get("latency_sec"),
            "is_escalation": call.get("is_escalation", False),
            "fell_back": call.get("fell_back", False)
        }).execute()
        call_id = call_result.data[0]["id"]

        if "quality_score" in call:
            supabase.table("quality_scores").insert({
                "model_call_id": call_id,
                "score": call["quality_score"],
                "passed": call.get("quality_passed", False)
            }).execute()

    return request_id