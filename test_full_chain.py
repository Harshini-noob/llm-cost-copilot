import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# 1. Insert a request
req = supabase.table("requests").insert({
    "prompt": "Explain how vaccines work",
    "tier": "medium",
    "routing_mode": "balanced"
}).execute()

request_id = req.data[0]["id"]
print("Created request:", request_id)

# 2. Insert a model call linked to that request
call = supabase.table("model_calls").insert({
    "request_id": request_id,
    "model": "openai/gpt-oss-120b",
    "input_tokens": 39,
    "output_tokens": 400,
    "cost_usd": 0.00026311,
    "latency_sec": 1.1,
    "is_escalation": False,
    "fell_back": False
}).execute()

call_id = call.data[0]["id"]
print("Created model_call:", call_id)

# 3. Insert a quality score linked to that model call
score = supabase.table("quality_scores").insert({
    "model_call_id": call_id,
    "score": 4,
    "passed": True
}).execute()

print("Created quality_score:", score.data[0]["id"])