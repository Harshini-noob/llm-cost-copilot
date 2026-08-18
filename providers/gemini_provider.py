import os, time
from dotenv import load_dotenv
from google import genai
from .base import ModelProvider

load_dotenv()


class GeminiProvider(ModelProvider):
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate(self, prompt: str, model: str, max_tokens: int, temperature: float = None) -> dict:
        start = time.time()
        config = {"max_output_tokens": max_tokens}
        if temperature is not None:
            config["temperature"] = temperature

        response = self.client.models.generate_content(
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