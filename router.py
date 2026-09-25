from providers.groq_provider import GroqProvider
from models import call_model

_groq = GroqProvider()

MODEL_MAP = {
    "simple": "openai/gpt-oss-20b",
    "medium": "openai/gpt-oss-120b",
    "complex": "gemini-2.5-flash"
}

MAX_TOKENS_BY_TIER = {
    "simple": 250,
    "medium": 600,
    "complex":1200
}

NEXT_TIER = {
    "simple": "medium",
    "medium": "complex",
    "complex": "complex"
}


def classify(prompt: str) -> str:
    # unchanged — pure logic, no I/O, stays synchronous
    prompt_lower = prompt.lower()
    word_count = len(prompt.split())
    hard_keywords = ["design a", "architecture", "build a system", "implement a full"]
    medium_keywords = ["explain", "analyze", "compare", "write code", "why", "how does"]
    if any(kw in prompt_lower for kw in hard_keywords):
        return "complex"
    if any(kw in prompt_lower for kw in medium_keywords):
        return "medium"
    if word_count < 15:
        return "simple"
    return "medium"


async def classify_llm(prompt: str) -> str:
    judge_prompt = f"""Rate this query's complexity as exactly one word: simple, medium, or complex.
simple = factual lookup, definitions, short direct answers
medium = explanations, comparisons, writing code, moderate reasoning
complex = multi-step reasoning, system/architecture design, deep multi-part analysis
Query: "{prompt}"
Respond with only one word, nothing else."""
    result = await _groq.generate(judge_prompt, model="openai/gpt-oss-120b", max_tokens=150,
                                   temperature=0)
    label = result["answer"].strip().lower()
    if label not in ["simple", "medium", "complex"]:
        return "medium"
    return label


async def get_confidence(prompt: str, answer: str) -> int:
    confidence_prompt = f"""You just answered this question. Rate your OWN confidence
in the correctness and completeness of your answer, from 1-10.
1 = you're likely missing something or unsure
10 = you're certain this answer is correct and complete
Question: "{prompt}"
Your answer: "{answer}"
Respond with ONLY a number from 1-10, nothing else."""
    result = await _groq.generate(confidence_prompt, model="openai/gpt-oss-120b", max_tokens=150,
                                   temperature=0)
    try:
        return int("".join(filter(str.isdigit, result["answer"].strip())))
    except:
        return 10