# test_async_router.py
import asyncio
from router import classify, classify_llm, get_confidence

async def main():
    print("Rule-based:", classify("What is 2+2?"))
    print("LLM-based:", await classify_llm("What is 2+2?"))
    print("Confidence:", await get_confidence("What is 2+2?", "The answer is 4."))

asyncio.run(main())