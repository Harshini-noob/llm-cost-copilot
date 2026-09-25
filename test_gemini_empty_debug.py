# test_gemini_empty_debug.py
import asyncio
from providers.gemini_provider import GeminiProvider

async def main():
    provider = GeminiProvider()
    result = await provider.generate(
        "Design a database schema for an e-commerce platform.",
        model="gemini-2.5-flash", max_tokens=600
    )
    print(repr(result))

asyncio.run(main())