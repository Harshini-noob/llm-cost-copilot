from providers.groq_provider import GroqProvider
from models import call_model

_groq = GroqProvider()

MODEL_MAP = {
    "simple": "openai/gpt-oss-20b",
    "medium": "openai/gpt-oss-120b",
    "complex": "gemini-2.5-flash"
}

MAX_TOKENS_BY_TIER = {
    "simple": 60,
    "medium": 400,
    "complex": 600
}

NEXT_TIER = {
    "simple": "medium",
    "medium": "complex",
    "complex": "complex"
}


def classify(prompt: str) -> str:
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


def classify_llm(prompt: str) -> str:
    judge_prompt = f"""..."""  # unchanged
    result = _groq.generate(judge_prompt, model="openai/gpt-oss-20b", max_tokens=5, temperature=0)
    label = result["answer"].strip().lower()
    if label not in ["simple", "medium", "complex"]:
        return "medium"
    return label


def get_confidence(prompt: str, answer: str) -> int:
    confidence_prompt = f"""..."""  # unchanged
    result = _groq.generate(confidence_prompt, model="openai/gpt-oss-20b", max_tokens=5, temperature=0)
    try:
        return int("".join(filter(str.isdigit, result["answer"].strip())))
    except:
        return 10