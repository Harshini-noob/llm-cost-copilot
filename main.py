from unittest import result

from fastapi import FastAPI, HTTPException, Depends
from models import call_model, client
from router import MAX_TOKENS_BY_TIER
from logger import log_full_request
from routing_policy import select_model, InvalidConstraintError, build_model_metadata
from auth import verify_api_key

app = FastAPI()

QUALITY_PASS_THRESHOLD = 3

def verify_quality(prompt: str, answer: str) -> dict:
    judge_prompt = f"""Rate this answer's quality for the given question, 1-5.
1 = wrong or unhelpful, 5 = correct and complete.

Question: "{prompt}"
Answer: "{answer}"

Respond with ONLY a number 1-5."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=5, temperature=0
    )
    try:
        score = int("".join(filter(str.isdigit, response.choices[0].message.content.strip())))
    except:
        score = 5
    return {"score": score, "passed": score >= QUALITY_PASS_THRESHOLD}


@app.post("/query")
async def query(prompt: str, routing_mode: str = "balanced",
                max_cost_usd: float = None, min_quality: float = None,
                api_key_id: str = Depends(verify_api_key)):

    try:
        decision = select_model(prompt, routing_mode=routing_mode,
                                 max_cost_usd=max_cost_usd, min_quality=min_quality)
    except InvalidConstraintError as e:
        raise HTTPException(status_code=400, detail=str(e))

    tier = decision["tier"]
    model = decision["model"]
    max_tokens = MAX_TOKENS_BY_TIER[tier]
    spent_so_far = 0.0
    model_calls_log = []

    result = call_model(prompt, model=model, max_tokens=max_tokens)
    spent_so_far += result["cost_usd"]

    verification = verify_quality(prompt, result["answer"])
    model_calls_log.append({
        "model": result["model"], "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"], "cost_usd": result["cost_usd"],
        "latency_sec": result["latency_sec"], "is_escalation": False,
        "fell_back": result["fell_back"],
        "quality_score": verification["score"], "quality_passed": verification["passed"]
    })

    escalated = False
    escalation_reason = None
    model_metadata = build_model_metadata()

    if not verification["passed"]:
        remaining_budget = (max_cost_usd - spent_so_far) if max_cost_usd is not None else None
        stronger = [c for c in decision["candidates_considered"]
                    if model_metadata[c["model"]]["expected_quality"] > decision["expected_quality"]]

        if stronger:
            stronger.sort(key=lambda c: c["expected_quality"])
            next_choice = stronger[0]
            fits_budget = remaining_budget is None or next_choice["estimated_cost"] <= remaining_budget

            if fits_budget:
                escalated = True
                escalation_reason = f"Quality check failed (score {verification['score']}/5); escalated"
                result = call_model(prompt, model=next_choice["model"], max_tokens=max_tokens)
                spent_so_far += result["cost_usd"]
                verification = verify_quality(prompt, result["answer"])

                model_calls_log.append({
                    "model": result["model"], "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"], "cost_usd": result["cost_usd"],
                    "latency_sec": result["latency_sec"], "is_escalation": True,
                    "fell_back": result["fell_back"],
                    "quality_score": verification["score"], "quality_passed": verification["passed"]
                })
            else:
                escalation_reason = "Quality check failed but no budget remained to escalate"
        else:
            escalation_reason = "Quality check failed but no stronger candidate was available"

    request_id = log_full_request(
        prompt=prompt, tier=tier, routing_mode=routing_mode,
        max_cost_usd=max_cost_usd, min_quality=min_quality,
        model_calls_data=model_calls_log
    )

    result["request_id"] = request_id
    result["quality_score"] = verification["score"]
    result["quality_passed"] = verification["passed"]
    result["escalated"] = escalated
    result["escalation_reason"] = escalation_reason
    result["total_cost_usd"] = round(spent_so_far, 8)
    result["routing_mode"] = routing_mode
    result["routing_reason"] = decision["reason"]
    result["rejected_candidates"] = decision.get("rejected", [])
    result["candidates_considered"] = decision.get("candidates_considered", [])

    return result
    

