from fastapi import FastAPI
from models import call_model
from logger import log_request
from router import classify_llm, MODEL_MAP, MAX_TOKENS_BY_TIER

app = FastAPI()

@app.post("/query")
async def query(prompt: str):
    tier = classify_llm(prompt)
    model = MODEL_MAP[tier]
    max_tokens = MAX_TOKENS_BY_TIER[tier]

    result = call_model(prompt, model=model, max_tokens=max_tokens)
    log_request(prompt, result, tier=tier)
    return result