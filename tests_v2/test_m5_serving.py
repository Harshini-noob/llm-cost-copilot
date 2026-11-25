"""
tests_v2/test_m5_serving.py

Comprehensive offline unit & integration tests for M5 runtime routing,
OpenAI-compatible serving, ranked fallback, budget enforcement, SQLite decision
logging, outcome feedback, retraining, shadow evaluation, promotion, and rollback.

Zero live LLM API calls; uses DeterministicTestEncoder and mocked provider callers.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from groq import RateLimitError
import httpx
import numpy as np
import pytest

import main
from auth import verify_api_key
from eval.routers.data_prep import ModelOutcomeRecord, PreparedDataset, PromptRecord
from eval.routers.encoders import DeterministicTestEncoder
from eval.routers.knn_router import KNNRouter
from eval.routers.learned_router import LearnedRouter
from routers import (
    KNNRuntimeRouter,
    LearnedLRRuntimeRouter,
    RouterVersionManager,
    resolve_runtime_lambda,
)
from routers.lifecycle import (
    LeakageDetectedError,
    PromotionGateError,
    promote_router,
    retrain_router,
    shadow_evaluate,
)
from telemetry.decision_log import DecisionLogger, normalize_feedback_value


def _make_synthetic_dataset(
    encoder: DeterministicTestEncoder,
    models: list[str],
    n_train: int = 30,
) -> tuple[PreparedDataset, dict[str, float]]:
    prompts: dict[str, PromptRecord] = {}
    embeddings: dict[str, np.ndarray] = {}
    train_records: list[ModelOutcomeRecord] = []
    train_pids: set[str] = set()

    texts = []
    pids = []
    for i in range(n_train):
        pid = f"tr_{i}"
        is_code = i % 2 == 0
        txt = f"def solve_{i}(x): return x + {i}" if is_code else f"Explain concept {i} in history"
        cat = "coding" if is_code else "general"
        prompts[pid] = PromptRecord(prompt_id=pid, prompt_text=txt, task_category=cat)
        train_pids.add(pid)
        pids.append(pid)
        texts.append(txt)

    embs = encoder.encode(texts)
    for pid, vec in zip(pids, embs):
        embeddings[pid] = vec
        rec_p = prompts[pid]
        prompts[pid] = PromptRecord(
            prompt_id=rec_p.prompt_id,
            task_category=rec_p.task_category,
            prompt_text=rec_p.prompt_text,
            prompt_embedding=vec,
            split="train",
        )

    default_costs = [0.00001, 0.00020, 0.00005]
    costs = {m: default_costs[idx % len(default_costs)] for idx, m in enumerate(models)}
    for i, pid in enumerate(pids):
        is_code = i % 2 == 0
        # Model 1 excels on code, Model 0 is cheap & decent on general, Model 2 is balanced
        q_vals = [0.2 if is_code else 0.9, 0.95 if is_code else 0.85, 0.6 if is_code else 0.7]
        q_map = {m: q_vals[idx % len(q_vals)] for idx, m in enumerate(models)}
        for m in models:
            q = q_map[m]
            train_records.append(
                ModelOutcomeRecord(
                    prompt_id=pid,
                    task_category="coding" if is_code else "general",
                    prompt_text=prompts[pid].prompt_text,
                    model_id=m,
                    provider="test",
                    label=q,
                    cost=costs[m],
                    token_usage=None,
                    execution_status="success",
                    evaluation_status="scored",
                    split="train",
                    run_count=1,
                    latency_ms=100.0,
                    prompt_embedding=embeddings[pid],
                    correct_runs=q,
                    evaluated_runs=1,
                    empirical_quality=q,
                )
            )

    ds = PreparedDataset(
        records=train_records,
        prompts=prompts,
        split_info={"train_prompt_count": len(train_pids)},
        sufficiency_report={"status": "synthetic_test"},
    )
    return ds, costs


LIVE_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "gemini-2.5-flash"]


def test_runtime_router_interface_and_bundles(tmp_path: Path) -> None:
    enc = DeterministicTestEncoder(dimension=32)
    ds, costs = _make_synthetic_dataset(enc, LIVE_MODELS)

    knn = KNNRouter(k=5, encoder=enc)
    knn.fit(ds)
    knn_rt = KNNRuntimeRouter(knn, model_costs=costs, router_version="knn-v1", encoder_name="det-32")

    lr = LearnedRouter(encoder=enc, C=1.0, random_state=42)
    lr.fit(ds)
    lr_rt = LearnedLRRuntimeRouter(lr, model_costs=costs, router_version="lr-v1", encoder_name="det-32")

    for rt in (knn_rt, lr_rt):
        dec = rt.route("def foo(x): return x * 2", lam="auto:quality", task_category="coding")
        assert dec.chosen_model in LIVE_MODELS
        assert dec.ranked_candidates[0] == dec.chosen_model
        assert len(dec.ranked_candidates) == 3
        assert dec.lambda_value == 0.0
        assert dec.router_version in ("knn-v1", "lr-v1")
        assert dec.router_latency_ms >= 0.0

    # Serialization round-trip
    knn_dir = knn_rt.save_bundle(tmp_path / "knn-v1")
    lr_dir = lr_rt.save_bundle(tmp_path / "lr-v1")
    knn_loaded = KNNRuntimeRouter.load_bundle(knn_dir, encoder=enc)
    lr_loaded = LearnedLRRuntimeRouter.load_bundle(lr_dir, encoder=enc)

    d1 = knn_rt.route("Explain Rome", lam="auto:balanced")
    d2 = knn_loaded.route("Explain Rome", lam="auto:balanced")
    assert d1.chosen_model == d2.chosen_model
    assert d1.predicted_scores == pytest.approx(d2.predicted_scores)

    l1 = lr_rt.route("Explain Rome", lam="auto:cheap")
    l2 = lr_loaded.route("Explain Rome", lam="auto:cheap")
    assert l1.chosen_model == l2.chosen_model
    assert l1.predicted_scores == pytest.approx(l2.predicted_scores)


def test_lambda_presets_and_tradeoffs() -> None:
    assert resolve_runtime_lambda("auto") == 0.5
    assert resolve_runtime_lambda("auto:quality") == 0.0
    assert resolve_runtime_lambda("auto:balanced") == 0.5
    assert resolve_runtime_lambda("auto:cheap") == 2.0

    enc = DeterministicTestEncoder(dimension=32)
    ds, costs = _make_synthetic_dataset(enc, LIVE_MODELS)
    lr = LearnedRouter(encoder=enc, C=1.0, random_state=42)
    lr.fit(ds)
    rt = LearnedLRRuntimeRouter(lr, model_costs=costs, router_version="lr-v1", encoder_name="det-32")

    dec_q = rt.route("def bar(x): return x", lam="auto:quality", task_category="coding")
    dec_c = rt.route("def bar(x): return x", lam="auto:cheap", task_category="coding")
    # Cheap preset should choose a model with <= expected cost of quality preset
    assert dec_c.expected_costs_usd[dec_c.chosen_model] <= dec_q.expected_costs_usd[dec_q.chosen_model]


def test_missing_model_predictions_and_incremental_knn_addition() -> None:
    enc = DeterministicTestEncoder(dimension=32)
    ds, costs = _make_synthetic_dataset(enc, LIVE_MODELS[:2])
    knn = KNNRouter(k=3, encoder=enc)
    knn.fit(ds)
    rt = KNNRuntimeRouter(knn, model_costs=costs, router_version="knn-v1", encoder_name="det-32")

    # Request includes gemini-2.5-flash which has no predictions yet
    dec = rt.route("Explain gravity", lam="auto", eligible_models=LIVE_MODELS)
    assert dec.predicted_scores["gemini-2.5-flash"] is None
    assert "gemini-2.5-flash" not in dec.ranked_candidates

    # Incrementally add outcomes for gemini-2.5-flash on existing calibration prompt IDs
    new_outcomes = {pid: 0.99 for pid in ds.train_prompt_ids}
    rt.add_model_outcomes("gemini-2.5-flash", new_outcomes, expected_cost_usd=0.00005)
    dec2 = rt.route("Explain gravity", lam="auto:quality", eligible_models=LIVE_MODELS)
    assert dec2.predicted_scores["gemini-2.5-flash"] is not None
    assert dec2.chosen_model == "gemini-2.5-flash"


@pytest.fixture
def isolated_app(tmp_path: Path):
    enc = DeterministicTestEncoder(dimension=32)
    ds, costs = _make_synthetic_dataset(enc, LIVE_MODELS)
    lr = LearnedRouter(encoder=enc, C=1.0, random_state=42)
    lr.fit(ds)
    rt = LearnedLRRuntimeRouter(lr, model_costs=costs, router_version="lr-test-v1", encoder_name="det-32")
    dlog = DecisionLogger(db_path=tmp_path / "decisions.db")

    old_router = getattr(main.app.state, "runtime_router", None)
    old_logger = getattr(main.app.state, "decision_logger", None)
    old_caller = getattr(main.app.state, "model_caller", None)

    main.app.dependency_overrides[verify_api_key] = lambda: "test-key-id"
    main.app.state.runtime_router = rt
    main.app.state.decision_logger = dlog

    yield main.app, dlog, rt

    main.app.dependency_overrides.clear()
    main.app.state.runtime_router = old_router
    main.app.state.decision_logger = old_logger
    main.app.state.model_caller = old_caller


def test_chat_completions_auto_explicit_and_schema(isolated_app) -> None:
    app, dlog, _ = isolated_app

    async def fake_caller(prompt, *, model, max_tokens, allow_fallback, budget_tracker, _reservation):
        await budget_tracker.claim(_reservation)
        res = {
            "answer": f"Mocked reply from {model}",
            "input_tokens": 12,
            "output_tokens": 8,
            "cost_usd": 0.00001,
            "cost_is_estimated": False,
            "latency_sec": 0.05,
        }
        await budget_tracker.record_response(
            _reservation, res, "generation", False, False
        )
        return res

    app.state.model_caller = fake_caller
    client = TestClient(app)

    # Auto routing
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "auto:quality", "messages": [{"role": "user", "content": "def f(x): return x"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] in LIVE_MODELS
    assert body["routing"]["chosen_model"] == body["model"]
    assert body["routing"]["router_version"] == "lr-test-v1"
    assert body["routing"]["actual_cost_usd"] is not None
    assert body["routing"]["fallback_attempts"] == []

    # Check decision log did NOT store raw prompt
    rec = dlog.get_decision(body["id"])
    assert rec is not None
    assert "prompt" not in rec
    assert rec["chosen_model"] == body["model"]

    # Explicit model routing
    resp_exp = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp_exp.status_code == 200
    assert resp_exp.json()["model"] == "openai/gpt-oss-20b"
    assert resp_exp.json()["routing"]["router_version"] == "explicit"

    # Unknown model -> 400
    resp_bad = client.post(
        "/v1/chat/completions",
        json={"model": "nonexistent-model", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp_bad.status_code == 400


def test_ranked_fallback_timeout_and_estimated_cost_separation(isolated_app) -> None:
    app, _, rt = isolated_app
    calls: list[str] = []

    async def failing_then_succeeding_caller(
        prompt, *, model, max_tokens, allow_fallback, budget_tracker, _reservation
    ):
        calls.append(model)
        await budget_tracker.claim(_reservation)
        if len(calls) == 1:
            await budget_tracker.record_failure(_reservation, "generation", False, False, asyncio.TimeoutError("timeout"))
            raise asyncio.TimeoutError("Provider timed out")
        if len(calls) == 2:
            req = httpx.Request("POST", "https://api.groq.com")
            resp = httpx.Response(429, request=req)
            err = RateLimitError("rate limited", response=resp, body=None)
            await budget_tracker.record_failure(_reservation, "generation", False, False, err)
            raise err
        # 3rd candidate succeeds without valid token counts -> cost_is_estimated=True
        res_3 = {
            "answer": f"Recovered on {model}",
            "input_tokens": None,
            "output_tokens": None,
            "cost_usd": 0.0001,
            "cost_is_estimated": True,
            "latency_sec": 0.1,
        }
        await budget_tracker.record_response(
            _reservation, res_3, "generation", False, False
        )
        return res_3

    app.state.model_caller = failing_then_succeeding_caller
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "auto:quality", "messages": [{"role": "user", "content": "def f(x): return x"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(calls) == 3
    assert len(body["routing"]["fallback_attempts"]) == 2
    assert body["routing"]["chosen_model"] == calls[2]
    # Actual cost must be None when cost_is_estimated is True (never fabricate actual usage)
    assert body["routing"]["actual_cost_usd"] is None


def test_budget_enforcement_and_cancellation_propagation(isolated_app) -> None:
    app, _, _ = isolated_app
    client = TestClient(app)

    # Budget too small for any model -> 422
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 500,
            "max_cost_usd": "0.0000001",
        },
    )
    assert resp.status_code == 422

    # Cancellation must propagate
    async def cancelling_caller(*args, **kwargs):
        raise asyncio.CancelledError()

    app.state.model_caller = cancelling_caller
    with pytest.raises((asyncio.CancelledError, concurrent.futures.CancelledError)):
        client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "Hello"}]},
        )


def test_missing_active_router_returns_503(tmp_path: Path) -> None:
    old_router = getattr(main.app.state, "runtime_router", None)
    main.app.dependency_overrides[verify_api_key] = lambda: "test-key-id"
    main.app.state.runtime_router = None

    # Point RouterVersionManager to empty dir temporarily
    import routers.registry as reg_mod
    orig_init = reg_mod.RouterVersionManager.__init__
    reg_mod.RouterVersionManager.__init__ = lambda self, store_dir="": orig_init(self, tmp_path / "empty_store")
    try:
        client = TestClient(main.app)
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "Hello"}]},
        )
        assert resp.status_code == 503
    finally:
        reg_mod.RouterVersionManager.__init__ = orig_init
        main.app.dependency_overrides.clear()
        main.app.state.runtime_router = old_router


def test_outcome_feedback_distinct_types_and_validation(tmp_path: Path) -> None:
    dlog = DecisionLogger(db_path=tmp_path / "fb.db")
    req_id = dlog.log_decision(
        prompt="Test prompt",
        chosen_model="openai/gpt-oss-20b",
        predicted_scores={"openai/gpt-oss-20b": 0.8},
        ranked_candidates=["openai/gpt-oss-20b"],
        lambda_value=0.5,
        router_version="lr-v1",
        expected_cost_usd=0.00001,
        latency_ms=12.0,
        router_latency_ms=1.5,
    )

    fb1 = dlog.record_feedback(req_id, feedback_type="judge_score", raw_value=4)
    fb2 = dlog.record_feedback(req_id, feedback_type="user_feedback", raw_value="thumbs_up")
    fb3 = dlog.record_feedback(req_id, feedback_type="downstream_task_success", raw_value=True)

    assert fb1["normalized_score"] == pytest.approx(0.75)
    assert fb2["normalized_score"] == pytest.approx(1.0)
    assert fb3["normalized_score"] == pytest.approx(1.0)

    rec = dlog.get_decision(req_id)
    assert rec is not None
    assert len(rec["feedback"]) == 3
    assert {f["feedback_type"] for f in rec["feedback"]} == {
        "judge_score",
        "user_feedback",
        "downstream_task_success",
    }

    with pytest.raises(ValueError):
        normalize_feedback_value("judge_score", 6)
    with pytest.raises(ValueError):
        normalize_feedback_value("user_feedback", "maybe")


def test_retraining_leakage_shadow_promotion_and_rollback(tmp_path: Path) -> None:
    enc = DeterministicTestEncoder(dimension=32)
    mgr = RouterVersionManager(store_dir=tmp_path / "routers")

    # Build clean train + val observations (55 val prompts)
    obs: list[dict[str, Any]] = []
    val_prompts: list[dict[str, Any]] = []
    val_outcomes: dict[tuple[str, str], tuple[float, float]] = {}

    for i in range(30):
        for m in LIVE_MODELS:
            obs.append(
                {
                    "prompt_id": f"tr_{i}",
                    "prompt_text": f"Train prompt {i} code def f(): pass" if i % 2 == 0 else f"Train general {i}",
                    "model_id": m,
                    "label": 1.0 if (m == LIVE_MODELS[1] and i % 2 == 0) or (m == LIVE_MODELS[0] and i % 2 == 1) else 0.2,
                    "cost_usd": 0.0002 if m == LIVE_MODELS[1] else 0.00002,
                    "split": "train",
                    "task_category": "coding" if i % 2 == 0 else "general",
                }
            )
    for i in range(55):
        pid = f"val_{i}"
        txt = f"Validation prompt {i} def g(): pass" if i % 2 == 0 else f"Validation history {i}"
        cat = "coding" if i % 2 == 0 else "general"
        val_prompts.append({"prompt_id": pid, "prompt_text": txt, "task_category": cat})
        for m in LIVE_MODELS:
            lbl = 1.0 if (m == LIVE_MODELS[1] and i % 2 == 0) or (m == LIVE_MODELS[0] and i % 2 == 1) else 0.2
            cost = 0.0002 if m == LIVE_MODELS[1] else 0.00002
            obs.append(
                {
                    "prompt_id": pid,
                    "prompt_text": txt,
                    "model_id": m,
                    "label": lbl,
                    "cost_usd": cost,
                    "split": "val",
                    "task_category": cat,
                }
            )
            val_outcomes[(pid, m)] = (lbl, cost)

    # Leakage check: sharing prompt text across train and val raises LeakageDetectedError
    leaky_obs = list(obs) + [
        {
            "prompt_id": "leak_1",
            "prompt_text": obs[0]["prompt_text"],
            "model_id": LIVE_MODELS[0],
            "label": 1.0,
            "cost_usd": 0.00002,
            "split": "val",
        }
    ]
    with pytest.raises(LeakageDetectedError):
        retrain_router(leaky_obs, router_type="knn", new_version="knn-bad", manager=mgr, encoder=enc)

    # Retrain v1 and activate
    rep1 = retrain_router(obs, router_type="knn", new_version="knn-v1", manager=mgr, encoder=enc, encoder_name="det-32")
    assert rep1["active_version_preserved"] is None
    mgr.set_active("knn-v1")

    # Retrain v2 (LR) without overwriting active v1
    rep2 = retrain_router(obs, router_type="learned_lr", new_version="lr-v2", manager=mgr, encoder=enc, encoder_name="det-32")
    assert rep2["active_version_preserved"] == "knn-v1"
    assert mgr.get_active_version() == "knn-v1"

    active_rt = mgr.load_active(encoder=enc)
    cand_rt = mgr.load_version("lr-v2", encoder=enc)

    # Shadow evaluation with labelled outcomes
    shadow = shadow_evaluate(cand_rt, active_rt, val_prompts, labelled_outcomes=val_outcomes)
    assert shadow["predicted_scores_only"] is False
    assert shadow["prompt_count"] == 55

    # Promotion fails closed if min_val_prompts not met
    with pytest.raises(PromotionGateError):
        promote_router("lr-v2", mgr, shadow_report=shadow, min_val_prompts=100, encoder=enc)

    # Promotion succeeds when gates pass
    prom = promote_router("lr-v2", mgr, shadow_report=shadow, min_val_prompts=50, encoder=enc)
    assert prom["active_version"] == "lr-v2"
    assert prom["previous_version"] == "knn-v1"

    # Rollback restores knn-v1
    rb = mgr.rollback()
    assert rb["active_version"] == "knn-v1"
