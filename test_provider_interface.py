from providers.groq_provider import GroqProvider

provider = GroqProvider()
result = provider.generate("What is 2+2?", model="llama-3.1-8b-instant", max_tokens=50)
print(result)