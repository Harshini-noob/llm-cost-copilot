import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from model_registry import get_model_by_id

USD_PER_MILLION_TOKENS = Decimal("1000000")


def decimal_usd(value) -> Decimal:
    amount = value if isinstance(value, Decimal) else Decimal(str(value))
    if not amount.is_finite():
        raise ValueError("Monetary values must be finite")
    return amount


def token_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    input_tokens = validate_token_count(input_tokens)
    output_tokens = validate_token_count(output_tokens)
    if input_tokens is None or output_tokens is None:
        raise InvalidTokenUsageError("Token counts are required to calculate actual cost")
    model_info = get_model_by_id(model)
    input_price = decimal_usd(model_info["input_cost_per_million"])
    output_price = decimal_usd(model_info["output_cost_per_million"])
    return (
        Decimal(input_tokens) * input_price
        + Decimal(output_tokens) * output_price
    ) / USD_PER_MILLION_TOKENS


def worst_case_cost(prompt: str, model: str, max_tokens: int) -> Decimal:
    max_tokens = validate_token_count(max_tokens)
    if max_tokens is None:
        raise InvalidTokenUsageError("A maximum output-token limit is required")
    from cost_estimator import estimate_tokens

    return token_cost(model, estimate_tokens(prompt), max_tokens)


class InvalidTokenUsageError(ValueError):
    pass


def validate_token_count(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidTokenUsageError(f"Invalid token count: {value!r}")
    return value


@dataclass(frozen=True)
class BudgetReservation:
    tracker_id: int
    reservation_id: int
    model: str
    estimated_input_tokens: int
    max_tokens: int
    estimated_cost: Decimal


class BudgetExceededError(Exception):
    def __init__(self, model: str, estimated_cost: Decimal, remaining_budget: Decimal):
        self.model = model
        self.estimated_cost = estimated_cost
        self.remaining_budget = remaining_budget
        super().__init__(
            f"Model '{model}' requires up to ${estimated_cost} but only "
            f"${remaining_budget} remains in the request budget"
        )


class ReservationError(RuntimeError):
    """Base class for invalid budget-reservation lifecycle operations."""


class ReservationAlreadyClaimedError(ReservationError):
    pass


class ReservationAlreadySettledError(ReservationError):
    pass


class InvalidReservationError(ReservationError):
    pass


class BudgetTracker:
    def __init__(self, budget_limit=None):
        self.budget_limit = None if budget_limit is None else decimal_usd(budget_limit)
        if self.budget_limit is not None and self.budget_limit < 0:
            raise ValueError("budget_limit must not be negative")

        self.actual_cost = Decimal("0")
        self.estimated_cost = Decimal("0")
        self.accounted_cost = Decimal("0")
        self._reserved_cost = Decimal("0")
        self._reservations = {}
        self._claimed_reservations = set()
        self._settled_reservations = set()
        self._next_reservation_id = 0
        self._lock = asyncio.Lock()
        self.model_calls = []
        self._last_generation_index = None

    @property
    def remaining_budget(self):
        if self.budget_limit is None:
            return None
        return self.budget_limit - self.accounted_cost - self._reserved_cost

    @property
    def reserved_cost(self):
        return self._reserved_cost

    async def can_afford(self, prompt: str, model: str, max_tokens: int) -> bool:
        estimate = worst_case_cost(prompt, model, max_tokens)
        async with self._lock:
            return (
                self.budget_limit is None
                or estimate <= self.budget_limit - self.accounted_cost - self._reserved_cost
            )

    async def reserve(
        self, prompt: str, model: str, max_tokens: int
    ) -> BudgetReservation:
        from cost_estimator import estimate_tokens

        input_tokens = estimate_tokens(prompt)
        estimate = token_cost(model, input_tokens, max_tokens)
        async with self._lock:
            remaining = (
                None if self.budget_limit is None
                else self.budget_limit - self.accounted_cost - self._reserved_cost
            )
            if remaining is not None and estimate > remaining:
                raise BudgetExceededError(model, estimate, remaining)

            reservation = BudgetReservation(
                tracker_id=id(self),
                reservation_id=self._next_reservation_id,
                model=model,
                estimated_input_tokens=input_tokens,
                max_tokens=max_tokens,
                estimated_cost=estimate,
            )
            self._next_reservation_id += 1
            self._reservations[reservation.reservation_id] = reservation
            self._reserved_cost += estimate
            return reservation

    async def claim(self, reservation: BudgetReservation):
        async with self._lock:
            active = self._reservations.get(reservation.reservation_id)
            if active is not reservation:
                if reservation.reservation_id in self._settled_reservations:
                    raise ReservationAlreadySettledError(
                        "Budget reservation was already settled"
                    )
                raise InvalidReservationError("Budget reservation is invalid")
            if reservation.reservation_id in self._claimed_reservations:
                raise ReservationAlreadyClaimedError(
                    "Budget reservation has already been claimed"
                )
            self._claimed_reservations.add(reservation.reservation_id)

    async def release(self, reservation: BudgetReservation):
        async with self._lock:
            active = self._reservations.get(reservation.reservation_id)
            if active is not reservation:
                if reservation.reservation_id in self._settled_reservations:
                    raise ReservationAlreadySettledError(
                        "Budget reservation was already settled"
                    )
                raise InvalidReservationError("Budget reservation is invalid")
            if reservation.reservation_id in self._claimed_reservations:
                raise ReservationAlreadyClaimedError(
                    "A claimed budget reservation cannot be released"
                )
            del self._reservations[reservation.reservation_id]
            self._reserved_cost -= reservation.estimated_cost
            self._settled_reservations.add(reservation.reservation_id)

    async def record_response(
        self, reservation: BudgetReservation, result: dict, call_type: str,
        is_escalation: bool, fell_back: bool,
    ):
        input_tokens = validate_token_count(result.get("input_tokens"))
        output_tokens = validate_token_count(result.get("output_tokens"))
        usage_known = input_tokens is not None and output_tokens is not None
        actual = (
            token_cost(reservation.model, input_tokens, output_tokens)
            if usage_known else None
        )
        return await self._settle(
            reservation=reservation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_sec=result.get("latency_sec"),
            call_type=call_type,
            is_escalation=is_escalation,
            fell_back=fell_back,
            succeeded=True,
            actual=actual,
            error_type=None,
        )

    async def record_failure(
        self, reservation: BudgetReservation, call_type: str,
        is_escalation: bool, fell_back: bool, error: BaseException,
    ):
        return await self._settle(
            reservation=reservation,
            input_tokens=None,
            output_tokens=None,
            latency_sec=None,
            call_type=call_type,
            is_escalation=is_escalation,
            fell_back=fell_back,
            succeeded=False,
            actual=None,
            error_type=type(error).__name__,
        )

    async def _settle(
        self, reservation: BudgetReservation, input_tokens, output_tokens,
        latency_sec, call_type: str, is_escalation: bool, fell_back: bool,
        succeeded: bool, actual, error_type,
    ):
        async with self._lock:
            active = self._reservations.get(reservation.reservation_id)
            if active is not reservation:
                if reservation.reservation_id in self._settled_reservations:
                    raise ReservationAlreadySettledError(
                        "Budget reservation was already settled"
                    )
                raise InvalidReservationError("Budget reservation is invalid")
            if reservation.reservation_id not in self._claimed_reservations:
                raise InvalidReservationError(
                    "Budget reservation must be claimed before it is settled"
                )

            del self._reservations[reservation.reservation_id]
            self._claimed_reservations.remove(reservation.reservation_id)
            self._settled_reservations.add(reservation.reservation_id)
            self._reserved_cost -= reservation.estimated_cost
            self.estimated_cost += reservation.estimated_cost
            accounted = (
                actual if actual is not None else reservation.estimated_cost
            )
            self.accounted_cost += accounted
            if actual is not None:
                self.actual_cost += actual

            record = {
                "model": reservation.model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": accounted,
                "actual_cost_usd": actual,
                "estimated_cost_usd": reservation.estimated_cost,
                "cost_is_estimated": actual is None,
                "latency_sec": latency_sec,
                "is_escalation": is_escalation,
                "fell_back": fell_back,
                "call_type": call_type,
                "call_succeeded": succeeded,
                "error_type": error_type,
            }
            self.model_calls.append(record)
            if succeeded and call_type == "generation":
                self._last_generation_index = len(self.model_calls) - 1
            return record

    def attach_quality(self, score, passed):
        if self._last_generation_index is not None:
            record = self.model_calls[self._last_generation_index]
            record["quality_score"] = score
            record["quality_passed"] = passed

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
