# test_models_multiprovider.py
from models import call_model

groq_result = call_model("What is 2+2?", model="llama-3.1-8b-instant")
print("Groq:", groq_result)

gemini_result = call_model("What is 2+2?", model="gemini-2.5-flash")
print("Gemini:", gemini_result)