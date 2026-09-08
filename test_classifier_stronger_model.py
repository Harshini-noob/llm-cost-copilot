# test_classifier_stronger_model.py
from providers.groq_provider import GroqProvider

_groq = GroqProvider()

judge_prompt = """Rate this query's complexity as exactly one word: simple, medium, or complex.

simple = factual lookup, definitions, short direct answers
medium = explanations, comparisons, writing code, moderate reasoning
complex = multi-step reasoning, system/architecture design, deep multi-part analysis

Query: "Design a recommendation system architecture for an e-commerce platform, including data pipeline."

Respond with only one word, nothing else."""

result = _groq.generate(judge_prompt, model="openai/gpt-oss-120b", max_tokens=20,
                         temperature=0, reasoning_effort="low")
print(repr(result["answer"]))