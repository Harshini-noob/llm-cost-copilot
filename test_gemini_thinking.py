# test_gemini_thinking_fix.py
import asyncio
from providers.gemini_provider import GeminiProvider

async def main():
    provider = GeminiProvider()
    result = await provider.generate(
        "Design a database schema for an e-commerce platform.",
        model="gemini-2.5-flash", max_tokens=600
    )
    print(result["answer"])
    print("Output tokens used:", result["output_tokens"])

asyncio.run(main())