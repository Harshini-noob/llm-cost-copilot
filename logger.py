from decimal import Decimal

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


def log_full_request(prompt: str, tier: str, routing_mode: str,
                      max_cost_usd: float, min_quality: float,
                      model_calls_data: list, quality_data: dict = None,
                      estimated_cost_usd=None, actual_total_cost_usd=None,
                      budget_accounted_cost_usd=None, remaining_budget_usd=None,
                      quality_verification_status=None,
                      escalation_attempted=None, escalation_reason=None,
                      budget_exceeded=None) -> str:
    """
    Logs a complete request lifecycle to Supabase:
    - one requests row
    - one model_calls row per provider attempt, including judges, retries, and fallbacks
    - one quality_scores row per model_calls row that was verified

    model_calls_data: list of dicts, each shaped like:
        {"model": ..., "input_tokens": ..., "output_tokens": ..., "cost_usd": ...,
         "latency_sec": ..., "is_escalation": bool, "fell_back": bool,
         "quality_score": int, "quality_passed": bool}
    """
    supabase = get_supabase()

    def numeric_value(value):
        return str(value) if isinstance(value, Decimal) else value

    req = supabase.table("requests").insert({
        "prompt": prompt,
        "tier": tier,
        "routing_mode": routing_mode,
        "max_cost_usd": numeric_value(max_cost_usd),
        "min_quality": min_quality,
        "estimated_cost_usd": numeric_value(estimated_cost_usd),
        "actual_total_cost_usd": numeric_value(actual_total_cost_usd),
        "budget_accounted_cost_usd": numeric_value(budget_accounted_cost_usd),
        "remaining_budget_usd": numeric_value(remaining_budget_usd),
        "quality_verification_status": quality_verification_status,
        "escalation_attempted": escalation_attempted,
        "escalation_reason": escalation_reason,
        "budget_exceeded": budget_exceeded,
    }).execute()
    request_id = req.data[0]["id"]

    for call in model_calls_data:
        call_result = supabase.table("model_calls").insert({
            "request_id": request_id,
            "model": call["model"],
            "input_tokens": call.get("input_tokens"),
            "output_tokens": call.get("output_tokens"),
            "cost_usd": numeric_value(call.get("cost_usd")),
            "actual_cost_usd": numeric_value(call.get("actual_cost_usd")),
            "estimated_cost_usd": numeric_value(call.get("estimated_cost_usd")),
            "cost_is_estimated": call.get("cost_is_estimated", False),
            "latency_sec": call.get("latency_sec"),
            "is_escalation": call.get("is_escalation", False),
            "fell_back": call.get("fell_back", False),
            "call_type": call.get("call_type", "generation"),
            "call_succeeded": call.get("call_succeeded", True),
            "error_type": call.get("error_type"),
        }).execute()
        call_id = call_result.data[0]["id"]

        if "quality_score" in call:
            supabase.table("quality_scores").insert({
                "model_call_id": call_id,
                "score": call["quality_score"],
                "passed": call.get("quality_passed", False)
            }).execute()

    return request_id