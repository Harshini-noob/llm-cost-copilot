from decimal import Decimal

from budget import token_cost

TOKENS_PER_WORD = 1.3


def estimate_tokens(text: str) -> int:
    word_count = len(text.split())
    return max(1, round(word_count * TOKENS_PER_WORD))


def estimate_request_cost(prompt: str, model: str, expected_output_tokens: int = 300) -> float:
    return float(estimate_request_cost_decimal(prompt, model, expected_output_tokens))


def estimate_request_cost_decimal(
        prompt: str, model: str, expected_output_tokens: int = 300) -> Decimal:
    estimated_input_tokens = estimate_tokens(prompt)
    return token_cost(model, estimated_input_tokens, expected_output_tokens)

