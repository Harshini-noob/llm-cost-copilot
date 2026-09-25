# test_async_models.py
import asyncio
from models import call_model

async def main():
    result = await call_model("What is 2+2?", model="openai/gpt-oss-20b")
    print(result)

asyncio.run(main())