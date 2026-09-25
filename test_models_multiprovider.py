import asyncio
from models import call_model

async def main():
    groq_result = await call_model("What is 2+2?", model="openai/gpt-oss-20b")
    print("Groq:", groq_result)

    gemini_result = await call_model("What is 2+2?", model="gemini-2.5-flash")
    print("Gemini:", gemini_result)

asyncio.run(main())