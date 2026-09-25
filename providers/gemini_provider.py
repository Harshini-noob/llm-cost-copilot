import time
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from .base import ModelProvider

load_dotenv()


class GeminiProvider(ModelProvider):
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    async def generate(self, prompt: str, model: str, max_tokens: int,
                        temperature: float = None, reasoning_effort: str = None) -> dict:
        start = time.time()
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # disable internal reasoning
        )
        if temperature is not None:
            config.temperature = temperature

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config
        )
        latency = time.time() - start
        usage = response.usage_metadata

        return {
            "answer": response.text,
            "model": model,
            "input_tokens": usage.prompt_token_count,
            "output_tokens": usage.candidates_token_count,
            "latency_sec": round(latency, 3),
        }