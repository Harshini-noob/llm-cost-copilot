# test_classifier_debug.py
from router import classify_llm

test_prompts = [
    "What is 2+2?",
    "Design a recommendation system architecture for an e-commerce platform.",
    "Explain what a REST API is."
]

for p in test_prompts:
    print(p, "→", classify_llm(p))