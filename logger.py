import json
import time

def log_request(prompt: str, result: dict, tier: str = "manual"):
    entry = {
        "timestamp": time.time(),
        "prompt": prompt,
        "tier": tier,
        **result
    }
    with open("data/logs.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")