# test_gemini_fallback.py
import asyncio
from models import call_model

async def main():
    # Force many rapid Gemini calls to trigger the real 429 quota error
    for i in range(8):
        try:
            result = await call_model(f"What is {i}+{i}?", model="gemini-2.5-flash", max_tokens=50)
            print(f"Call {i}: model={result['model']}, fell_back={result['fell_back']}")
        except Exception as e:
            print(f"Call {i} FAILED with unhandled: {type(e).__name__}: {e}")

asyncio.run(main())