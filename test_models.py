from models import call_model
from logger import log_full_request

result = call_model("What is 2+2?", model="llama-3.1-8b-instant")

request_id = log_full_request(
    prompt="What is 2+2?",
    tier="simple",
    routing_mode="manual-test",
    max_cost_usd=None,
    min_quality=None,
    model_calls_data=[{
        "model": result["model"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "cost_usd": result["cost_usd"],
        "latency_sec": result["latency_sec"],
        "is_escalation": False,
        "fell_back": result["fell_back"],
    }]
)

print(result)
print("Logged as request_id:", request_id)