# test_classifier_final_check.py
from providers.groq_provider import GroqProvider

_groq = GroqProvider()

judge_prompt = """Rate this query's complexity as exactly one word: simple, medium, or complex.
simple = factual lookup, definitions, short direct answers
medium = explanations, comparisons, writing code, moderate reasoning
complex = multi-step reasoning, system/architecture design, deep multi-part analysis
Query: "Analyze the causes of the 2008 financial crisis."
Respond with only one word, nothing else."""

result = _groq.generate(judge_prompt, model="openai/gpt-oss-120b", max_tokens=150, temperature=0)
print(repr(result["answer"]))