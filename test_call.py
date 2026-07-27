import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)

print("Answer:", response.choices[0].message.content)
print("Tokens used:", response.usage.total_tokens)