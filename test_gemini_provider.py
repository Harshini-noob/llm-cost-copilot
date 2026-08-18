# test_gemini_provider.py
from providers.gemini_provider import GeminiProvider

provider = GeminiProvider()
result = provider.generate("What is 2+2?", model="gemini-2.5-flash", max_tokens=50)
print(result)