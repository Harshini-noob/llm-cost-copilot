from decimal import Decimal
from typing import Optional

from model_registry import get_model_by_id

USD_PER_MILLION_TOKENS = Decimal("1000000")


def decimal_usd(value) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def token_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    model_info = get_model_by_id(model)
    input_price = decimal_usd(model_info["input_cost_per_million"])
    output_price = decimal_usd(model_info["output_cost_per_million"])
    return (
        Decimal(input_tokens) * input_price
        + Decimal(output_tokens) * output_price
    ) / USD_PER_MILLION_TOKENS


def worst_case_cost(prompt: str, model: str, max_tokens: int) -> Decimal:
    from cost_estimator import estimate_tokens

    return token_cost(model, estimate_tokens(prompt), max_tokens)


def _valid_token_count(value) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


class BudgetExceededError(Exception):
    def __init__(self, model: str, estimated_cost: Decimal, remaining_budget: Decimal):
        self.model = model
        self.estimated_cost = estimated_cost
        self.remaining_budget = remaining_budget
        super().__init__(
            f"Model '{model}' requires up to ${estimated_cost} but only "
            f"${remaining_budget} remains in the request budget"
        )


class BudgetTracker:
    def __init__(self, budget_limit=None):
        self.budget_limit = None if budget_limit is None else decimal_usd(budget_limit)
        self.actual_cost = Decimal("0")
        self.estimated_cost = Decimal("0")
        self.accounted_cost = Decimal("0")
        self.model_calls = []
        self._last_generation_index = None

    @property
    def remaining_budget(self):
        if self.budget_limit is None:
            return None
        return self.budget_limit - self.accounted_cost

    def can_afford(self, prompt: str, model: str, max_tokens: int) -> bool:
        if self.budget_limit is None:
            return True
        return worst_case_cost(prompt, model, max_tokens) <= self.remaining_budget

    def authorize(self, prompt: str, model: str, max_tokens: int):
        estimate = worst_case_cost(prompt, model, max_tokens)
        if self.budget_limit is not None and estimate > self.remaining_budget:
            raise BudgetExceededError(model, estimate, self.remaining_budget)
        return estimate

    def record_response(self, prompt: str, model: str, max_tokens: int,
                        result: dict, call_type: str, is_escalation: bool,
                        fell_back: bool):
        estimate = worst_case_cost(prompt, model, max_tokens)
        input_tokens = _valid_token_count(result.get("input_tokens"))
        output_tokens = _valid_token_count(result.get("output_tokens"))
        usage_known = input_tokens is not None and output_tokens is not None
        actual = token_cost(model, input_tokens, output_tokens) if usage_known else None
        accounted = actual if actual is not None else estimate

        self.estimated_cost += estimate
        self.accounted_cost += accounted
        if actual is not None:
            self.actual_cost += actual

        record = self._record(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_sec=result.get("latency_sec"),
            call_type=call_type,
            is_escalation=is_escalation,
            fell_back=fell_back,
            succeeded=True,
            estimated=estimate,
            actual=actual,
            accounted=accounted,
            error_type=None,
        )
        if call_type == "generation":
            self._last_generation_index = len(self.model_calls) - 1
        return record

    def record_failure(self, prompt: str, model: str, max_tokens: int,
                       call_type: str, is_escalation: bool, fell_back: bool,
                       error: Exception):
        estimate = worst_case_cost(prompt, model, max_tokens)
        self.estimated_cost += estimate
        self.accounted_cost += estimate
        return self._record(
            model=model,
            input_tokens=None,
            output_tokens=None,
            latency_sec=None,
            call_type=call_type,
            is_escalation=is_escalation,
            fell_back=fell_back,
            succeeded=False,
            estimated=estimate,
            actual=None,
            accounted=estimate,
            error_type=type(error).__name__,
        )

    def attach_quality(self, score, passed):
        if self._last_generation_index is not None:
            record = self.model_calls[self._last_generation_index]
            record["quality_score"] = score
            record["quality_passed"] = passed

    def _record(self, model: str, input_tokens, output_tokens, latency_sec,
                call_type: str, is_escalation: bool, fell_back: bool,
                succeeded: bool, estimated: Decimal, actual,
                accounted: Decimal, error_type):
        record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": accounted,
            "actual_cost_usd": actual,
            "estimated_cost_usd": estimated,
            "cost_is_estimated": actual is None,
            "latency_sec": latency_sec,
            "is_escalation": is_escalation,
            "fell_back": fell_back,
            "call_type": call_type,
            "call_succeeded": succeeded,
            "error_type": error_type,
        }
        self.model_calls.append(record)
        return record

    def as_response_fields(self):
        return {
            "estimated_cost_usd": float(self.estimated_cost),
            "actual_total_cost_usd": float(self.actual_cost),
            "budget_accounted_cost_usd": float(self.accounted_cost),
            "remaining_budget_usd": (
                None if self.remaining_budget is None else float(self.remaining_budget)
            ),
            "budget_limit_usd": (
                None if self.budget_limit is None else float(self.budget_limit)
            ),
            "budget_exceeded": (
                self.budget_limit is not None
                and self.accounted_cost > self.budget_limit
            ),
        }
