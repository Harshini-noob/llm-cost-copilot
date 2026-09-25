# test_async_gemini.py
import asyncio
from providers.gemini_provider import GeminiProvider

async def main():
    provider = GeminiProvider()
    result = await provider.generate("What is 2+2?", model="gemini-2.5-flash", max_tokens=50)
    print(result)

asyncio.run(main())