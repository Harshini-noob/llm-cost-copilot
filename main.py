from fastapi import FastAPI
from models import call_model
from logger import log_request
from router import classify_llm, MODEL_MAP, MAX_TOKENS_BY_TIER, NEXT_TIER, get_confidence

app = FastAPI()

CONFIDENCE_THRESHOLD = 6

@app.post("/query")
async def query(prompt: str):
    tier = classify_llm(prompt)
    model = MODEL_MAP[tier]
    max_tokens = MAX_TOKENS_BY_TIER[tier]

    result = call_model(prompt, model=model, max_tokens=max_tokens)

    confidence = get_confidence(prompt, result["answer"])
    escalated = False

    if confidence < CONFIDENCE_THRESHOLD and NEXT_TIER[tier] != tier:
        escalated = True
        new_tier = NEXT_TIER[tier]
        new_model = MODEL_MAP[new_tier]
        new_max_tokens = MAX_TOKENS_BY_TIER[new_tier]
        result = call_model(prompt, model=new_model, max_tokens=new_max_tokens)
        tier = new_tier  # update tier label to reflect what actually answered

    result["confidence"] = confidence
    result["escalated"] = escalated

    log_request(prompt, result, tier=tier)
    return result