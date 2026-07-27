import os, time
from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIStatusError

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PRICING = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
}

FALLBACK_MODEL = {
    "llama-3.3-70b-versatile": "openai/gpt-oss-120b",
    "openai/gpt-oss-120b": "llama-3.1-8b-instant",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant",
}

def _raw_call(prompt: str, model: str, max_tokens: int):
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    latency = time.time() - start
    usage = response.usage

    input_cost = (usage.prompt_tokens / 1_000_000) * PRICING[model]["input"]
    output_cost = (usage.completion_tokens / 1_000_000) * PRICING[model]["output"]

    return {
        "answer": response.choices[0].message.content,
        "model": model,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "cost_usd": round(input_cost + output_cost, 8),
        "latency_sec": round(latency, 3),
        "fell_back": False
    }


def call_model(prompt: str, model: str = "llama-3.1-8b-instant", max_tokens: int = 500,
               allow_fallback: bool = True, _is_retry: bool = False):
    try:
        return _raw_call(prompt, model, max_tokens)

    except RateLimitError as e:
        if not allow_fallback:
            # measurement scripts need a FIXED model — surface the error instead of swapping silently
            raise

        fallback = FALLBACK_MODEL.get(model)
        if _is_retry or fallback == model:
            raise
        print(f"⚠️  Rate limit hit on '{model}', falling back to '{fallback}'")
        result = call_model(prompt, model=fallback, max_tokens=max_tokens,
                             allow_fallback=allow_fallback, _is_retry=True)
        result["fell_back"] = True
        return result

    except APIStatusError as e:
        if _is_retry:
            raise
        print(f"⚠️  API error on '{model}': {e}. Retrying once in 3s...")
        time.sleep(3)
        return call_model(prompt, model=model, max_tokens=max_tokens,
                           allow_fallback=allow_fallback, _is_retry=True)