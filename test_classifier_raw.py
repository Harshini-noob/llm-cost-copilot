# test_classifier_raw.py
from providers.groq_provider import GroqProvider

_groq = GroqProvider()

judge_prompt = """Rate this query's complexity as exactly one word: simple, medium, or complex.

simple = factual lookup, definitions, short direct answers
medium = explanations, comparisons, writing code, moderate reasoning
complex = multi-step reasoning, system/architecture design, deep multi-part analysis

Query: "What is 2+2?"

Respond with only one word, nothing else."""

result = _groq.generate(judge_prompt, model="openai/gpt-oss-20b", max_tokens=20, temperature=0)
print(repr(result["answer"]))