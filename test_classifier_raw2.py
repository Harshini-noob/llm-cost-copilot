# test_classifier_raw2.py
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Respond with only one word: simple, medium, or complex. Query: What is 2+2?"}],
    max_tokens=50,
    temperature=0,
    reasoning_effort="low"
)
print(repr(response.choices[0].message.content))