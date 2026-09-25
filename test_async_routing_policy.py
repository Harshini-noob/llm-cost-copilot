# test_async_routing_policy.py
import asyncio
from routing_policy import select_model

async def main():
    decision = await select_model("What is 2+2?", routing_mode="economy")
    print(decision["model"])
    print(decision["candidates_considered"])

asyncio.run(main())