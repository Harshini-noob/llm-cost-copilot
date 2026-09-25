# test_async_groq.py
import asyncio
from providers.groq_provider import GroqProvider

async def main():
    provider = GroqProvider()
    result = await provider.generate("What is 2+2?", model="openai/gpt-oss-20b", max_tokens=50)
    print(result)

asyncio.run(main())