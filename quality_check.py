import asyncio
import sys
from models import call_model
from providers.groq_provider import GroqProvider
from router import classify, MODEL_MAP, MAX_TOKENS_BY_TIER
from test_prompts import TEST_PROMPTS
from groq import RateLimitError

_groq = GroqProvider()

REFERENCE_MODEL = "openai/gpt-oss-120b"
REFERENCE_MAX_TOKENS = 500


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or "ascii", errors="replace").decode(sys.stdout.encoding or "ascii"))


async def judge_quality(prompt: str, routed_answer: str, reference_answer: str) -> dict:
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

    result = await _groq.generate(judge_prompt, model="openai/gpt-oss-120b", max_tokens=350, temperature=0)
    text = result["answer"].strip()

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


async def main():
    safe_print(f"{'PROMPT':<45} {'TIER':<10} {'SCORE':<7} {'REASON'}")
    safe_print("-" * 100)

    scores = []
    skipped_due_to_rate_limit = 0

    for item in TEST_PROMPTS:
        prompt = item["prompt"]
        tier = classify(prompt)
        routed_model = MODEL_MAP[tier]
        max_tokens = MAX_TOKENS_BY_TIER[tier]

        if routed_model == "gemini-2.5-flash":
            await asyncio.sleep(13)

        routed_result = await call_model(prompt, model=routed_model, max_tokens=max_tokens)

        if routed_model == REFERENCE_MODEL:
            safe_print(f"{prompt[:42]:<45} {tier:<10} {'N/A':<7} (same as reference model)")
            continue

        try:
            reference_result = await call_model(prompt, model=REFERENCE_MODEL, max_tokens=REFERENCE_MAX_TOKENS,
                                                 allow_fallback=False)
        except RateLimitError:
            safe_print(f"{prompt[:42]:<45} {tier:<10} {'SKIP':<7} reference model rate-limited, skipping to preserve integrity")
            skipped_due_to_rate_limit += 1
            await asyncio.sleep(5)
            continue

        judgment = await judge_quality(prompt, routed_result["answer"], reference_result["answer"])
        if judgment["score"] is None:
            safe_print(f"   [PARSE FAILURE] raw judge output: {judgment['raw']!r}")
        else:
            scores.append(judgment["score"])

        safe_print(f"{prompt[:42]:<45} {tier:<10} {str(judgment['score']):<7} {judgment['reason']}")
        await asyncio.sleep(1)

    safe_print("-" * 100)
    if scores:
        avg = sum(scores) / len(scores)
        safe_print(f"Average quality score (routed vs reference): {avg:.2f} / 5")
        safe_print(f"Queries judged: {len(scores)}")
    if skipped_due_to_rate_limit:
        safe_print(f"Queries skipped due to rate limit (reference model unavailable): {skipped_due_to_rate_limit}")

asyncio.run(main())