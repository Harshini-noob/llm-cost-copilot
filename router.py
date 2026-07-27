from models import client, call_model

MODEL_MAP = {
    "simple": "llama-3.1-8b-instant",
    "medium": "openai/gpt-oss-120b",
    "complex": "llama-3.3-70b-versatile"
}

MAX_TOKENS_BY_TIER = {
    "simple": 60,
    "medium": 400,
    "complex": 500  # capped to match BASELINE_MAX_TOKENS — complex uses the
                     # same model as baseline, so this tier can only ever win
                     # via a tighter token budget, never a cheaper model.
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
    judge_prompt = f"""Rate this query's complexity as exactly one word: simple, medium, or complex.

simple = factual lookup, definitions, short direct answers
medium = explanations, comparisons, writing/debugging a single function or small code snippet, moderate reasoning
complex = designing or architecting a system/schema/algorithm from scratch, multi-step reasoning, deep multi-part analysis

Examples:
"Write code for a binary search tree in Python." -> medium
"Write a function to reverse a linked list." -> medium
"Design a database schema for an e-commerce platform." -> complex
"Design an algorithm to detect fraud in real-time transactions." -> complex
"What is the capital of France?" -> simple
"Explain how vaccines work in the immune system." -> medium
"Design a system architecture for a real-time chat app." -> complex

Query: "{prompt}"

Respond with only one word, nothing else."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=5,
        temperature=0,
      # best-effort determinism only — Groq does not guarantee
                 # identical output even with temperature=0 + seed set
    )

    label = response.choices[0].message.content.strip().lower()

    if label not in ["simple", "medium", "complex"]:
        return "medium"
    return label


def classify_llm_with_cost(prompt: str) -> tuple[str, float]:
    """Same as classify_llm, but also returns the $ cost of the classification
    call itself, via call_model (which already computes cost from usage).
    Use this wherever you need an honest end-to-end cost total, since the
    classifier call is a real Groq API call with a real price tag."""
    judge_prompt = f"""Rate this query's complexity as exactly one word: simple, medium, or complex.

simple = factual lookup, definitions, short direct answers
medium = explanations, comparisons, writing/debugging a single function or small code snippet, moderate reasoning
complex = designing or architecting a system/schema/algorithm from scratch, multi-step reasoning, deep multi-part analysis

Examples:
"Write code for a binary search tree in Python." -> medium
"Write a function to reverse a linked list." -> medium
"Design a database schema for an e-commerce platform." -> complex
"Design an algorithm to detect fraud in real-time transactions." -> complex
"What is the capital of France?" -> simple
"Explain how vaccines work in the immune system." -> medium
"Design a system architecture for a real-time chat app." -> complex

Query: "{prompt}"

Respond with only one word, nothing else."""

    result = call_model(judge_prompt, model="llama-3.1-8b-instant", max_tokens=5)
    label = result["answer"].strip().lower()

    if label not in ["simple", "medium", "complex"]:
        label = "medium"
    return label, result["cost_usd"]