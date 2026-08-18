import os, time
from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIStatusError
from .base import ModelProvider

load_dotenv()


class GroqProvider(ModelProvider):
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    
    def generate(self, prompt: str, model: str, max_tokens: int, temperature: float = None) -> dict:
        start = time.time()
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = self.client.chat.completions.create(**kwargs)
        latency = time.time() - start
        usage = response.usage

        return {
            "answer": response.choices[0].message.content,
            "model": model,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "latency_sec": round(latency, 3),
        }