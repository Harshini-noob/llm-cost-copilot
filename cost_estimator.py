from model_registry import get_model_by_id

TOKENS_PER_WORD = 1.3


def estimate_tokens(text: str) -> int:
    word_count = len(text.split())
    return max(1, round(word_count * TOKENS_PER_WORD))


def estimate_request_cost(prompt: str, model: str, expected_output_tokens: int = 300) -> float:
    model_info = get_model_by_id(model)

    estimated_input_tokens = estimate_tokens(prompt)

    input_cost = (estimated_input_tokens / 1_000_000) * model_info["input_cost_per_million"]
    output_cost = (expected_output_tokens / 1_000_000) * model_info["output_cost_per_million"]

    return round(input_cost + output_cost, 8)

