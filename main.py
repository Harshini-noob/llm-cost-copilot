from fastapi import FastAPI, HTTPException
from models import call_model, client
from logger import log_request
from routing_policy import select_model, InvalidConstraintError, MODEL_METADATA
from router import MAX_TOKENS_BY_TIER
from cost_estimator import estimate_request_cost

app = FastAPI()

QUALITY_PASS_THRESHOLD = 3  # out of 5, from judge

def verify_quality(prompt: str, answer: str) -> dict:
    """Real verification: judge rates the answer on its own merits (not vs a reference,
    to avoid an extra expensive call on every single request)."""
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
        score = 5  # fail open — don't block a response over a parsing hiccup
    return {"score": score, "passed": score >= QUALITY_PASS_THRESHOLD}


@app.post("/query")
async def query(prompt: str, routing_mode: str = "balanced",
                max_cost_usd: float = None, min_quality: float = None):

    try:
        decision = select_model(prompt, routing_mode=routing_mode,
                                 max_cost_usd=max_cost_usd, min_quality=min_quality)
    except InvalidConstraintError as e:
        raise HTTPException(status_code=400, detail=str(e))

    tier = decision["tier"]
    model = decision["model"]
    max_tokens = MAX_TOKENS_BY_TIER[tier]
    spent_so_far = 0.0

    result = call_model(prompt, model=model, max_tokens=max_tokens)
    spent_so_far += result["cost_usd"]

    verification = verify_quality(prompt, result["answer"])
    escalated = False
    escalation_reason = None

    if not verification["passed"]:
        # Only escalate to candidates STRONGER than what we already tried,
        # and only if remaining budget allows it
        remaining_budget = (max_cost_usd - spent_so_far) if max_cost_usd is not None else None
        stronger = [c for c in decision["candidates_considered"]
                    if MODEL_METADATA[c["model"]]["expected_quality"] > decision["expected_quality"]]

        if stronger:
            stronger.sort(key=lambda c: c["expected_quality"])
            next_choice = stronger[0]
            fits_budget = remaining_budget is None or next_choice["estimated_cost"] <= remaining_budget

            if fits_budget:
                escalated = True
                escalation_reason = f"Quality check failed (score {verification['score']}/5); escalated to stronger model"
                result = call_model(prompt, model=next_choice["model"], max_tokens=max_tokens)
                spent_so_far += result["cost_usd"]
                verification = verify_quality(prompt, result["answer"])  # re-verify the NEW response
                tier = next_choice["model"]
            else:
                escalation_reason = "Quality check failed but no budget remained to escalate"
        else:
            escalation_reason = "Quality check failed but no stronger candidate was available"

    result["quality_score"] = verification["score"]
    result["quality_passed"] = verification["passed"]
    result["escalated"] = escalated
    result["escalation_reason"] = escalation_reason
    result["total_cost_usd"] = round(spent_so_far, 8)
    result["routing_mode"] = routing_mode
    result["routing_reason"] = decision["reason"]
    result["rejected_candidates"] = decision.get("rejected", [])

    log_request(prompt, result, tier=tier)
    return result