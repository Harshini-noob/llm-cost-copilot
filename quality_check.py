from models import call_model, client
from router import classify_llm, MODEL_MAP, MAX_TOKENS_BY_TIER
from test_prompts import TEST_PROMPTS
from groq import RateLimitError
import time

REFERENCE_MODEL = "llama-3.3-70b-versatile"
REFERENCE_MAX_TOKENS = 500

def judge_quality(prompt: str, routed_answer: str, reference_answer: str) -> dict:
    judge_prompt = f"""You are evaluating two answers to the same question for QUALITY only.
Do not favor the longer answer. Do not favor either answer for style.
Judge only on: correctness, completeness of the core point, and usefulness to the user.

Question: "{prompt}"

Answer A: "{routed_answer}"

Answer B: "{reference_answer}"

Is Answer A as good as Answer B for actually answering this question?
Respond in exactly this format, nothing else:
SCORE: <number 1-5, where 5 = fully equivalent quality, 1 = significantly worse>
REASON: <one short sentence>"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=60,
        temperature=0
    )

    text = response.choices[0].message.content.strip()
    score = None
    reason = ""
    for line in text.split("\n"):
        if line.upper().startswith("SCORE"):
            try:
                score = int("".join(filter(str.isdigit, line.split(":")[1])))
            except:
                score = None
        if line.upper().startswith("REASON"):
            reason = line.split(":", 1)[1].strip()

    return {"score": score, "reason": reason, "raw": text}


print(f"{'PROMPT':<45} {'TIER':<10} {'SCORE':<7} {'REASON'}")
print("-" * 100)

scores = []
skipped_due_to_rate_limit = 0

for item in TEST_PROMPTS:
    prompt = item["prompt"]
    tier = classify_llm(prompt)
    routed_model = MODEL_MAP[tier]
    max_tokens = MAX_TOKENS_BY_TIER[tier]

    routed_result = call_model(prompt, model=routed_model, max_tokens=max_tokens)  # live-app style, fallback OK

    if routed_model == REFERENCE_MODEL:
        print(f"{prompt[:42]:<45} {tier:<10} {'N/A':<7} (same as reference model)")
        continue

    try:
        reference_result = call_model(prompt, model=REFERENCE_MODEL, max_tokens=REFERENCE_MAX_TOKENS,
                                       allow_fallback=False)  # MUST stay fixed for a valid comparison
    except RateLimitError:
        print(f"{prompt[:42]:<45} {tier:<10} {'SKIP':<7} reference model rate-limited, skipping to preserve integrity")
        skipped_due_to_rate_limit += 1
        time.sleep(5)
        continue

    judgment = judge_quality(prompt, routed_result["answer"], reference_result["answer"])
    if judgment["score"] is not None:
        scores.append(judgment["score"])

    print(f"{prompt[:42]:<45} {tier:<10} {str(judgment['score']):<7} {judgment['reason']}")
    time.sleep(1)

print("-" * 100)
if scores:
    avg = sum(scores) / len(scores)
    print(f"Average quality score (routed vs reference): {avg:.2f} / 5")
    print(f"Queries judged: {len(scores)}")
if skipped_due_to_rate_limit:
    print(f"Queries skipped due to rate limit (reference model unavailable): {skipped_due_to_rate_limit}")