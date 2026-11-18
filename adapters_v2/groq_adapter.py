import os
from dotenv import load_dotenv
load_dotenv()
from groq import AsyncGroq


async def groq_adapter(model_config: dict, prompt: str) -> dict:

    client = AsyncGroq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    response = await client.chat.completions.create(
        model=model_config["model_name"],
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return {
        "output": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }