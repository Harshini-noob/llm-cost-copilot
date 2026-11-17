import asyncio
import importlib
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient
from groq import APIStatusError, RateLimitError

import main
import models
from auth import verify_api_key
from budget import (
    BudgetExceededError,
    BudgetTracker,
    InvalidTokenUsageError,
    ReservationAlreadyClaimedError,
    ReservationAlreadySettledError,
    token_cost,
    worst_case_cost,
)
from cost_estimator import estimate_tokens
from logger import log_full_request
from model_registry import MODELS
from routing_policy import (
    InvalidConstraintError,
    NoEligibleModelError,
    select_model,
    validate_constraints,
)
from router import MAX_TOKENS_BY_TIER


PROMPT = "What is two plus two?"
PRIMARY_MODEL = "openai/gpt-oss-20b"
STRONGER_MODEL = "openai/gpt-oss-120b"


class FakeProvider:
    def __init__(self, respond=None):
        self.respond = respond or self.default_response
        self.calls = []

    @staticmethod
    def default_response(prompt, model, max_tokens):
        return {
            "answer": "5" if "Rate this answer's quality" in prompt else "answer",
            "model": model,
            "input_tokens": 1,
            "output_tokens": 1,
            "latency_sec": 0.01,
        }

    async def generate(self, prompt, model, max_tokens, **kwargs):
        self.calls.append((prompt, model, max_tokens))
        response = self.respond(prompt, model, max_tokens)
        if isinstance(response, Exception):
            raise response
        return response


class BlockingProvider(FakeProvider):
    def __init__(self, start_count):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.start_count = start_count

    async def generate(self, prompt, model, max_tokens, **kwargs):
        self.calls.append((prompt, model, max_tokens))
        if len(self.calls) >= self.start_count:
            self.started.set()
        await self.release.wait()
        return {
            "answer": "answer",
            "model": model,
            "input_tokens": estimate_tokens(PROMPT),
            "output_tokens": max_tokens,
            "latency_sec": 0.01,
        }


class FakeSupabase:
    def __init__(self):
        self.inserts = []

    def table(self, name):
        return FakeTable(self, name)


class FakeTable:
    def __init__(self, client, name):
        self.client = client
        self.name = name
        self.payload = None

    def insert(self, payload):
        self.payload = payload
        self.client.inserts.append((self.name, payload))
        return self

    def execute(self):
        row_id = "request-test" if self.name == "requests" else f"{self.name}-test"
        return type("Response", (), {"data": [{"id": row_id}]})()


def routing_decision(model=PRIMARY_MODEL, stronger=False):
    candidates = [{
        "model": model,
        "estimated_cost": 0.001,
        "expected_quality": 0.75,
        "avg_latency_sec": 1.0,
    }]
    if stronger:
        candidates.append({
            "model": STRONGER_MODEL,
            "estimated_cost": 0.002,
            "expected_quality": 0.85,
            "avg_latency_sec": 1.2,
        })
    return {
        "tier": "simple",
        "model": model,
        "reason": "test decision",
        "estimated_cost": candidates[0]["estimated_cost"],
        "expected_quality": candidates[0]["expected_quality"],
        "candidates_considered": candidates,
        "rejected": [],
    }


class BudgetEnforcementTests(unittest.IsolatedAsyncioTestCase):
    def query_mocks(self, decision):
        return (
            patch("main.select_model", new=AsyncMock(return_value=decision)),
            patch("main.log_full_request", return_value="request-test"),
        )

    async def test_very_small_budget_rejects_before_provider_call(self):
        provider = FakeProvider()
        decision = routing_decision()
        selector, logger = self.query_mocks(decision)
        with patch.dict(models._providers, {"groq": provider}), selector, logger:
            with self.assertRaises(HTTPException) as raised:
                await main.query(
                    PROMPT, max_cost_usd=Decimal("0.0000000001"),
                    api_key_id="test",
                )
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("affordable", raised.exception.detail)
        self.assertEqual(provider.calls, [])

    async def test_exact_worst_case_budget_is_eligible(self):
        prompt = "What is two plus two?"
        expected = worst_case_cost(
            prompt, PRIMARY_MODEL, 25
        )
        metadata = {
            entry["model"]: {
                "expected_quality": 0.99,
                "available": True,
                "context_window": entry["context_window"],
                "avg_latency_sec": 1.0,
                "quality_source": "test",
                "quality_sample_count": 0,
            }
            for entry in MODELS.values()
        }
        with patch.dict(
            MODELS["small"],
            {"max_output_tokens_by_tier": {"simple": 25}},
        ), patch("routing_policy.build_model_metadata", new=AsyncMock(return_value=metadata)):
            decision = await select_model(
                prompt, routing_mode="economy", max_cost_usd=expected
            )
        eligible = {candidate["model"] for candidate in decision["candidates_considered"]}
        self.assertIn(PRIMARY_MODEL, eligible)
        candidate = next(
            candidate for candidate in decision["candidates_considered"]
            if candidate["model"] == PRIMARY_MODEL
        )
        self.assertEqual(candidate["max_tokens"], 25)
        self.assertEqual(candidate["estimated_cost"], float(expected))

    async def test_output_limit_is_resolved_for_escalation_candidate(self):
        prompt = PROMPT
        judge_count = 0

        def respond(text, model, max_tokens):
            nonlocal judge_count
            if "Rate this answer's quality" in text:
                judge_count += 1
                answer = "1" if judge_count == 1 else "5"
            else:
                answer = "answer"
            return {
                "answer": answer,
                "model": model,
                "input_tokens": 1,
                "output_tokens": 1,
                "latency_sec": 0.01,
            }

        provider = FakeProvider(respond)
        selector, logger = self.query_mocks(routing_decision(stronger=True))
        with patch.dict(
            MODELS["medium"],
            {"max_output_tokens_by_tier": {"simple": 25}},
        ):
            with patch.dict(models._providers, {"groq": provider}), selector, logger:
                response = await main.query(
                    prompt, max_cost_usd=Decimal("0.01"), api_key_id="test"
                )

        self.assertTrue(response["escalation_attempted"])
        self.assertEqual(
            [call[2] for call in provider.calls],
            [250, 60, 25, 60],
        )
        escalation_cost = response["model_calls"][2]["estimated_cost_usd"]
        self.assertEqual(
            escalation_cost,
            worst_case_cost(prompt, STRONGER_MODEL, 25),
        )

    async def test_routing_returns_422_when_no_model_is_affordable(self):
        selector = patch(
            "main.select_model",
            new=AsyncMock(side_effect=NoEligibleModelError(
                "No eligible model is affordable within max_cost_usd"
            )),
        )
        with selector:
            with self.assertRaises(HTTPException) as raised:
                await main.query(
                    PROMPT, max_cost_usd=Decimal("0.0000000001"),
                    api_key_id="test",
                )
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("affordable", raised.exception.detail)

    async def test_actual_cost_over_initial_estimate_is_reported(self):
        provider = FakeProvider(lambda prompt, model, max_tokens: {
            "answer": "answer",
            "model": model,
            "input_tokens": 1,
            "output_tokens": 100000,
            "latency_sec": 0.01,
        })
        limit = worst_case_cost(PROMPT, PRIMARY_MODEL, MAX_TOKENS_BY_TIER["simple"])
        selector, logger = self.query_mocks(routing_decision())
        with patch.dict(models._providers, {"groq": provider}), selector, logger:
            response = await main.query(
                PROMPT, max_cost_usd=limit, api_key_id="test"
            )
        self.assertTrue(response["budget_exceeded"])
        self.assertGreater(
            Decimal(str(response["actual_total_cost_usd"])), limit
        )
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(response["quality_verification_status"], "unavailable")

    async def test_escalation_is_skipped_when_remaining_budget_is_insufficient(self):
        provider = FakeProvider(lambda prompt, model, max_tokens: {
            "answer": "1" if "Rate this answer's quality" in prompt else "answer",
            "model": model,
            "input_tokens": 1,
            "output_tokens": 1,
            "latency_sec": 0.01,
        })
        limit = worst_case_cost(
            PROMPT, STRONGER_MODEL, MAX_TOKENS_BY_TIER["simple"]
        )
        selector, logger = self.query_mocks(routing_decision(stronger=True))
        with patch.dict(models._providers, {"groq": provider}), selector, logger:
            response = await main.query(
                PROMPT, max_cost_usd=limit, api_key_id="test"
            )
        self.assertFalse(response["escalation_attempted"])
        self.assertIn("remaining budget", response["escalation_reason"])
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(
            [call["call_type"] for call in response["model_calls"]],
            ["generation", "quality_judge"],
        )

    async def test_judge_cost_is_included_in_request_total(self):
        provider = FakeProvider()
        selector, logger = self.query_mocks(routing_decision())
        with patch.dict(models._providers, {"groq": provider}), selector, logger:
            response = await main.query(
                PROMPT, max_cost_usd=Decimal("0.01"), api_key_id="test"
            )
        self.assertEqual(len(response["model_calls"]), 2)
        self.assertEqual(response["model_calls"][1]["call_type"], "quality_judge")
        call_costs = sum(
            (call["actual_cost_usd"] for call in response["model_calls"]),
            Decimal("0"),
        )
        self.assertAlmostEqual(
            response["actual_total_cost_usd"], float(call_costs)
        )
        self.assertGreater(response["actual_total_cost_usd"], 0)

    async def test_failed_provider_attempt_is_reserved_before_fallback(self):
        def respond(prompt, model, max_tokens):
            if model == STRONGER_MODEL:
                return RateLimitError(
                    "rate limited",
                    response=httpx.Response(
                        429, request=httpx.Request("POST", "https://provider.test")
                    ),
                    body={},
                )
            return {
                "answer": "answer",
                "model": model,
                "input_tokens": 2,
                "output_tokens": 3,
                "latency_sec": 0.02,
            }

        provider = FakeProvider(respond)
        tracker = BudgetTracker(Decimal("0.01"))
        with patch.dict(models._providers, {"groq": provider}):
            result = await models.call_model(
                PROMPT, model=STRONGER_MODEL, max_tokens=50,
                budget_tracker=tracker,
            )
        self.assertTrue(result["fell_back"])
        self.assertEqual(len(tracker.model_calls), 2)
        self.assertFalse(tracker.model_calls[0]["call_succeeded"])
        self.assertGreater(tracker.model_calls[0]["cost_usd"], 0)
        self.assertTrue(tracker.model_calls[1]["fell_back"])
        fallback_actual = token_cost(PRIMARY_MODEL, 2, 3)
        failed_reservation = worst_case_cost(PROMPT, STRONGER_MODEL, 50)
        self.assertEqual(
            tracker.accounted_cost, failed_reservation + fallback_actual
        )

    async def test_fallback_is_blocked_when_failed_attempt_reserves_budget(self):
        def respond(prompt, model, max_tokens):
            return RateLimitError(
                "rate limited",
                response=httpx.Response(
                    429, request=httpx.Request("POST", "https://provider.test")
                ),
                body={},
            )

        provider = FakeProvider(respond)
        estimate = worst_case_cost(PROMPT, STRONGER_MODEL, 50)
        tracker = BudgetTracker(estimate)
        with patch.dict(models._providers, {"groq": provider}):
            with self.assertRaises(BudgetExceededError):
                await models.call_model(
                    PROMPT, model=STRONGER_MODEL, max_tokens=50,
                    budget_tracker=tracker,
                )
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(tracker.accounted_cost, estimate)
        self.assertEqual(tracker.remaining_budget, Decimal("0"))

    async def test_retry_reserves_and_accounts_each_attempt(self):
        replies = iter([
            APIStatusError(
                "server error",
                response=httpx.Response(
                    500, request=httpx.Request("POST", "https://provider.test")
                ),
                body={},
            ),
            {
                "answer": "answer",
                "model": STRONGER_MODEL,
                "input_tokens": 1,
                "output_tokens": 2,
                "latency_sec": 0.01,
            },
        ])
        provider = FakeProvider(lambda prompt, model, max_tokens: next(replies))
        tracker = BudgetTracker(Decimal("0.01"))
        with patch.dict(models._providers, {"groq": provider}):
            with patch(
                "models.asyncio.sleep", new=AsyncMock(return_value=None)
            ):
                await models.call_model(
                    PROMPT, model=STRONGER_MODEL, max_tokens=50,
                    budget_tracker=tracker,
                )
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(len(tracker.model_calls), 2)
        self.assertFalse(tracker.model_calls[0]["call_succeeded"])
        self.assertTrue(tracker.model_calls[1]["call_succeeded"])
        self.assertEqual(
            tracker.accounted_cost,
            worst_case_cost(PROMPT, STRONGER_MODEL, 50)
            + token_cost(STRONGER_MODEL, 1, 2),
        )

    async def test_multiple_calls_share_one_cumulative_budget(self):
        max_tokens = MAX_TOKENS_BY_TIER["simple"]
        per_call = worst_case_cost(PROMPT, PRIMARY_MODEL, max_tokens)
        input_tokens = estimate_tokens(PROMPT)
        provider = FakeProvider(lambda prompt, model, output_limit: {
            "answer": "answer",
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_limit,
            "latency_sec": 0.01,
        })
        tracker = BudgetTracker(per_call * 2)
        with patch.dict(models._providers, {"groq": provider}):
            await models.call_model(
                PROMPT, model=PRIMARY_MODEL, max_tokens=max_tokens,
                budget_tracker=tracker,
            )
            await models.call_model(
                PROMPT, model=PRIMARY_MODEL, max_tokens=max_tokens,
                budget_tracker=tracker,
            )
            with self.assertRaises(BudgetExceededError):
                await models.call_model(
                    PROMPT, model=PRIMARY_MODEL, max_tokens=max_tokens,
                    budget_tracker=tracker,
                )
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(tracker.actual_cost, per_call * 2)
        self.assertEqual(tracker.remaining_budget, Decimal("0"))

    async def test_concurrent_calls_reserve_and_reconcile_without_double_counting(self):
        max_tokens = MAX_TOKENS_BY_TIER["simple"]
        per_call = worst_case_cost(PROMPT, PRIMARY_MODEL, max_tokens)
        provider = BlockingProvider(start_count=2)
        tracker = BudgetTracker(per_call * 2)
        with patch.dict(models._providers, {"groq": provider}):
            first = asyncio.create_task(models.call_model(
                PROMPT, model=PRIMARY_MODEL, max_tokens=max_tokens,
                budget_tracker=tracker,
            ))
            second = asyncio.create_task(models.call_model(
                PROMPT, model=PRIMARY_MODEL, max_tokens=max_tokens,
                budget_tracker=tracker,
            ))
            await asyncio.wait_for(provider.started.wait(), timeout=1)
            self.assertEqual(tracker.reserved_cost, per_call * 2)
            self.assertEqual(tracker.remaining_budget, Decimal("0"))
            provider.release.set()
            await asyncio.gather(first, second)
        self.assertEqual(tracker.reserved_cost, Decimal("0"))
        self.assertEqual(tracker.actual_cost, per_call * 2)
        self.assertEqual(tracker.accounted_cost, per_call * 2)

    async def test_concurrent_calls_deny_second_when_only_one_reservation_fits(self):
        max_tokens = MAX_TOKENS_BY_TIER["simple"]
        per_call = worst_case_cost(PROMPT, PRIMARY_MODEL, max_tokens)
        provider = BlockingProvider(start_count=1)
        tracker = BudgetTracker(per_call)
        with patch.dict(models._providers, {"groq": provider}):
            first = asyncio.create_task(models.call_model(
                PROMPT, model=PRIMARY_MODEL, max_tokens=max_tokens,
                budget_tracker=tracker,
            ))
            await asyncio.wait_for(provider.started.wait(), timeout=1)
            with self.assertRaises(BudgetExceededError):
                await models.call_model(
                    PROMPT, model=PRIMARY_MODEL, max_tokens=max_tokens,
                    budget_tracker=tracker,
                )
            self.assertEqual(len(provider.calls), 1)
            provider.release.set()
            await first
        self.assertEqual(tracker.accounted_cost, per_call)
        self.assertEqual(tracker.reserved_cost, Decimal("0"))

    async def test_same_reservation_cannot_authorize_concurrent_provider_calls(self):
        max_tokens = MAX_TOKENS_BY_TIER["simple"]
        tracker = BudgetTracker()
        reservation = await tracker.reserve(PROMPT, PRIMARY_MODEL, max_tokens)
        provider = BlockingProvider(start_count=1)

        with patch.dict(models._providers, {"groq": provider}):
            first = asyncio.create_task(models.call_model(
                PROMPT, model=PRIMARY_MODEL, max_tokens=max_tokens,
                budget_tracker=tracker, _reservation=reservation,
            ))
            await asyncio.wait_for(provider.started.wait(), timeout=1)
            with self.assertRaises(ReservationAlreadyClaimedError):
                await models.call_model(
                    PROMPT, model=PRIMARY_MODEL, max_tokens=max_tokens,
                    budget_tracker=tracker, _reservation=reservation,
                )
            self.assertEqual(len(provider.calls), 1)
            provider.release.set()
            await first

        self.assertEqual(len(tracker.model_calls), 1)
        self.assertEqual(tracker.reserved_cost, Decimal("0"))

    async def test_duplicate_reservation_settlement_is_rejected(self):
        tracker = BudgetTracker()
        reservation = await tracker.reserve(PROMPT, PRIMARY_MODEL, 50)
        await tracker.claim(reservation)
        response = {"input_tokens": 1, "output_tokens": 1, "latency_sec": 0.01}
        await tracker.record_response(
            reservation, response, "generation", False, False
        )

        with self.assertRaises(ReservationAlreadySettledError):
            await tracker.record_response(
                reservation, response, "generation", False, False
            )

        self.assertEqual(len(tracker.model_calls), 1)
        self.assertEqual(tracker.reserved_cost, Decimal("0"))

    async def test_provider_failure_after_claim_keeps_conservative_cost(self):
        provider = FakeProvider(lambda prompt, model, max_tokens: RuntimeError("failed"))
        tracker = BudgetTracker()
        estimate = worst_case_cost(PROMPT, PRIMARY_MODEL, 50)
        with patch.dict(models._providers, {"groq": provider}):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                await models.call_model(
                    PROMPT, model=PRIMARY_MODEL, max_tokens=50,
                    budget_tracker=tracker, allow_fallback=False,
                )

        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(tracker.accounted_cost, estimate)
        self.assertEqual(tracker.reserved_cost, Decimal("0"))
        self.assertFalse(tracker.model_calls[0]["call_succeeded"])

    async def test_cancellation_during_provider_call_settles_reservation(self):
        class CancellableProvider(FakeProvider):
            def __init__(self):
                super().__init__()
                self.started = asyncio.Event()

            async def generate(self, prompt, model, max_tokens, **kwargs):
                self.calls.append((prompt, model, max_tokens))
                self.started.set()
                await asyncio.Event().wait()

        provider = CancellableProvider()
        estimate = worst_case_cost(PROMPT, PRIMARY_MODEL, 50)
        tracker = BudgetTracker(estimate)
        with patch.dict(models._providers, {"groq": provider}):
            task = asyncio.create_task(models.call_model(
                PROMPT, model=PRIMARY_MODEL, max_tokens=50,
                budget_tracker=tracker,
            ))
            await asyncio.wait_for(provider.started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertEqual(tracker.accounted_cost, estimate)
        self.assertEqual(tracker.remaining_budget, Decimal("0"))
        self.assertEqual(tracker.reserved_cost, Decimal("0"))
        self.assertEqual(tracker.model_calls[0]["error_type"], "CancelledError")

    async def test_cancellation_while_claim_waits_releases_unclaimed_reservation(self):
        tracker = BudgetTracker()
        reservation = await tracker.reserve(PROMPT, PRIMARY_MODEL, 50)
        await tracker._lock.acquire()
        task = asyncio.create_task(models.call_model(
            PROMPT, model=PRIMARY_MODEL, max_tokens=50,
            budget_tracker=tracker, _reservation=reservation,
        ))
        await asyncio.sleep(0)
        task.cancel()
        tracker._lock.release()

        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(tracker.reserved_cost, Decimal("0"))
        self.assertEqual(tracker.accounted_cost, Decimal("0"))
        self.assertEqual(tracker.model_calls, [])

    async def test_no_budget_constraint_keeps_remaining_budget_unbounded(self):
        provider = FakeProvider()
        selector, logger = self.query_mocks(routing_decision())
        with patch.dict(models._providers, {"groq": provider}), selector, logger:
            response = await main.query(PROMPT, api_key_id="test")
        self.assertIsNone(response["budget_limit_usd"])
        self.assertIsNone(response["remaining_budget_usd"])
        self.assertFalse(response["budget_exceeded"])
        self.assertEqual(len(provider.calls), 2)

    def test_http_response_exposes_budget_and_call_fields(self):
        provider = FakeProvider()
        selector, logger = self.query_mocks(routing_decision())
        main.app.dependency_overrides[verify_api_key] = lambda: "test"
        try:
            with patch.dict(models._providers, {"groq": provider}), selector, logger:
                response = TestClient(main.app).post(
                    "/query",
                    params={"prompt": PROMPT, "max_cost_usd": "0.01"},
                    headers={"X-API-Key": "test"},
                )
        finally:
            main.app.dependency_overrides.pop(verify_api_key, None)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["budget_limit_usd"], 0.01)
        self.assertIn("actual_total_cost_usd", body)
        self.assertIn("remaining_budget_usd", body)
        self.assertIn("budget_exceeded", body)
        self.assertEqual(body["quality_verification_status"], "verified")
        self.assertEqual(len(body["model_calls"]), 2)
        self.assertEqual(
            [candidate["model"] for candidate in body["candidates_considered"]],
            [PRIMARY_MODEL],
        )

    async def test_zero_and_missing_usage_are_not_confused(self):
        replies = iter([
            {
                "answer": "zero",
                "model": PRIMARY_MODEL,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_sec": 0.01,
            },
            {
                "answer": "unknown",
                "model": PRIMARY_MODEL,
                "input_tokens": None,
                "output_tokens": None,
                "latency_sec": 0.01,
            },
        ])
        provider = FakeProvider(lambda prompt, model, max_tokens: next(replies))
        tracker = BudgetTracker()
        with patch.dict(models._providers, {"groq": provider}):
            await models.call_model(
                PROMPT, model=PRIMARY_MODEL, max_tokens=50,
                budget_tracker=tracker,
            )
            await models.call_model(
                PROMPT, model=PRIMARY_MODEL, max_tokens=50,
                budget_tracker=tracker,
            )
        self.assertEqual(tracker.model_calls[0]["actual_cost_usd"], 0)
        self.assertFalse(tracker.model_calls[0]["cost_is_estimated"])
        self.assertIsNone(tracker.model_calls[1]["actual_cost_usd"])
        self.assertTrue(tracker.model_calls[1]["cost_is_estimated"])
        self.assertEqual(tracker.actual_cost, Decimal("0"))
        self.assertGreater(tracker.accounted_cost, Decimal("0"))

    async def test_negative_usage_is_rejected_and_reserved_conservatively(self):
        provider = FakeProvider(lambda prompt, model, max_tokens: {
            "answer": "invalid",
            "model": model,
            "input_tokens": -1,
            "output_tokens": 0,
            "latency_sec": 0.01,
        })
        estimate = worst_case_cost(PROMPT, PRIMARY_MODEL, 50)
        tracker = BudgetTracker()
        with patch.dict(models._providers, {"groq": provider}):
            with self.assertRaises(InvalidTokenUsageError):
                await models.call_model(
                    PROMPT, model=PRIMARY_MODEL, max_tokens=50,
                    budget_tracker=tracker,
                )
        self.assertEqual(tracker.accounted_cost, estimate)
        self.assertEqual(tracker.model_calls[0]["error_type"], "InvalidTokenUsageError")

    async def test_invalid_budget_values_are_rejected(self):
        for value in (Decimal("-0.01"), Decimal("NaN"), Decimal("Infinity")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    BudgetTracker(value)
                with self.assertRaises(InvalidConstraintError):
                    validate_constraints("balanced", max_cost_usd=value)

    async def test_telemetry_persists_request_totals_and_individual_calls(self):
        supabase = FakeSupabase()
        calls = [{
            "model": PRIMARY_MODEL,
            "input_tokens": 2,
            "output_tokens": 3,
            "cost_usd": Decimal("0.0000009"),
            "actual_cost_usd": Decimal("0.0000009"),
            "estimated_cost_usd": Decimal("0.000075"),
            "cost_is_estimated": False,
            "call_type": "generation",
            "call_succeeded": True,
            "quality_score": 4,
            "quality_passed": True,
        }, {
            "model": STRONGER_MODEL,
            "cost_usd": Decimal("0.0001"),
            "actual_cost_usd": None,
            "estimated_cost_usd": Decimal("0.0001"),
            "cost_is_estimated": True,
            "call_type": "quality_judge",
            "call_succeeded": False,
            "error_type": "RateLimitError",
        }]
        with patch("logger.get_supabase", return_value=supabase):
            request_id = log_full_request(
                PROMPT, "simple", "balanced", Decimal("0.01"), None, calls,
                estimated_cost_usd=Decimal("0.000175"),
                actual_total_cost_usd=Decimal("0.0000009"),
                budget_accounted_cost_usd=Decimal("0.0001009"),
                remaining_budget_usd=Decimal("0.0098991"),
                quality_verification_status="unavailable",
                escalation_attempted=False,
                budget_exceeded=False,
            )
        self.assertEqual(request_id, "request-test")
        request_row = supabase.inserts[0][1]
        self.assertEqual(Decimal(request_row["actual_total_cost_usd"]), Decimal("0.0000009"))
        model_rows = [payload for table, payload in supabase.inserts if table == "model_calls"]
        self.assertEqual(len(model_rows), 2)
        self.assertEqual(model_rows[1]["call_type"], "quality_judge")
        self.assertEqual(Decimal(model_rows[1]["cost_usd"]), Decimal("0.0001"))
        self.assertIsNone(model_rows[1]["actual_cost_usd"])


class SDKCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.sdk = importlib.import_module("llm_cost_autopilot_client")
        self.client_module = importlib.import_module(
            "llm_cost_autopilot_client.client"
        )

    def deserialize(self, payload):
        response = Mock()
        response.json.return_value = payload
        with patch.object(self.client_module.requests, "post", return_value=response):
            return self.sdk.LLMCostAutopilotClient("test-key").ask("prompt")

    def test_sdk_module_import_and_dataclass_construction(self):
        self.assertIsNotNone(self.sdk.LLMCostAutopilotClient)
        result = self.sdk.RoutingResult(
            answer="answer",
            model=PRIMARY_MODEL,
            routing_mode="balanced",
            routing_reason="test",
            total_cost_usd=0.0,
            quality_score=None,
            escalated=False,
            escalation_reason=None,
            request_id="request-id",
        )
        self.assertEqual(result.request_id, "request-id")
        self.assertEqual(result.quality_verification_status, "unavailable")

    def test_sdk_deserializes_legacy_response_without_budget_fields(self):
        result = self.deserialize({
            "answer": "legacy answer",
            "model": PRIMARY_MODEL,
            "routing_mode": "balanced",
            "routing_reason": "legacy",
            "total_cost_usd": 0.001,
            "quality_score": 4,
            "escalated": False,
            "request_id": "legacy-id",
        })

        self.assertEqual(result.answer, "legacy answer")
        self.assertEqual(result.request_id, "legacy-id")
        self.assertIsNone(result.estimated_cost_usd)
        self.assertIsNone(result.actual_total_cost_usd)
        self.assertEqual(result.quality_verification_status, "unavailable")

    def test_sdk_deserializes_new_response_fields(self):
        result = self.deserialize({
            "answer": "new answer",
            "model": PRIMARY_MODEL,
            "routing_mode": "balanced",
            "routing_reason": "new",
            "total_cost_usd": 0.002,
            "quality_score": 5,
            "escalated": True,
            "escalation_reason": "quality",
            "request_id": "new-id",
            "estimated_cost_usd": 0.001,
            "actual_total_cost_usd": 0.002,
            "remaining_budget_usd": 0.003,
            "budget_limit_usd": 0.005,
            "budget_exceeded": False,
            "quality_verification_status": "verified",
            "escalation_attempted": True,
            "model_calls": [{"model": PRIMARY_MODEL}],
        })

        self.assertEqual(result.escalation_reason, "quality")
        self.assertEqual(result.actual_total_cost_usd, 0.002)
        self.assertEqual(result.remaining_budget_usd, 0.003)
        self.assertTrue(result.escalation_attempted)
        self.assertEqual(len(result.model_calls), 1)
        self.assertEqual(result.raw["request_id"], "new-id")

    def test_compare_models_returns_candidates_from_api(self):
        response = Mock()
        response.json.return_value = {
            "answer": "answer",
            "model": PRIMARY_MODEL,
            "routing_mode": "balanced",
            "routing_reason": "test",
            "total_cost_usd": 0.001,
            "quality_score": 4,
            "escalated": False,
            "escalation_reason": None,
            "request_id": "request-id",
            "candidates_considered": [{
                "model": PRIMARY_MODEL,
                "estimated_cost": 0.001,
                "expected_quality": 0.75,
                "avg_latency_sec": 1.0,
            }],
        }
        with patch.object(
            self.client_module.requests, "post", return_value=response
        ):
            candidates = self.sdk.LLMCostAutopilotClient("test-key").compare_models(
                "prompt"
            )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].model, PRIMARY_MODEL)
        self.assertEqual(candidates[0].estimated_cost, 0.001)


if __name__ == "__main__":
    unittest.main()
