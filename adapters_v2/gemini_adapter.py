import os
from dotenv import load_dotenv
from google import genai

load_dotenv()


async def gemini_adapter(model_config: dict, prompt: str) -> dict:

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError("Gemini API key not found in environment")

    client = genai.Client(api_key=api_key)

    response = await client.aio.models.generate_content(
        model=model_config["model_name"],
        contents=prompt,
    )

    usage = response.usage_metadata

    return {
        "output": response.text,
        "input_tokens": usage.prompt_token_count or 0,
        "output_tokens": usage.candidates_token_count or 0,
    }