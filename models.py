import os, time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PRICING = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
}

def call_model(prompt: str, model: str = "llama-3.1-8b-instant", max_tokens: int = 500):
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
        "latency_sec": round(latency, 3)
    }