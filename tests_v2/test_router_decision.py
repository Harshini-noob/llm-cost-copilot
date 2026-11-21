"""
tests_v2/test_router_decision.py

Unit tests for Lambda Utility Decision Engine under Phase M3.
Verifies decision rule U(m) = Q(m) - lambda * C(m), cost normalization,
presets, deterministic tie-breaking, missing data handling, and zero network calls.
"""

from __future__ import annotations

import math
import socket
from typing import Any

import numpy as np
import pytest

from eval.routers.decision import (
    CostNormalizer,
    LambdaDecisionEngine,
    ModelUtilityBreakdown,
    PROVISIONAL_PRESETS,
    RoutingDecision,
    resolve_preset_lambda,
)
from eval.routers.encoders import DeterministicTestEncoder
from eval.routers.knn_router import KNNRouter
from eval.routers.learned_router import LearnedRouter


@pytest.fixture
def mock_cost_normalizer() -> CostNormalizer:
    """Fixture providing three models with distinct cost tiers."""
    costs = {
        "model_fast": 0.0001,   # Cheapest: min_max -> 0.0
        "model_mid": 0.0005,    # Medium: min_max -> 0.444
        "model_smart": 0.0010,  # Most expensive: min_max -> 1.0
    }
    return CostNormalizer(model_costs=costs, strategy="min_max")


def test_cost_normalization_strategies() -> None:
    """Verifies min_max, max_scale, and raw cost normalization behaviors."""
    costs = {"m_cheap": 0.0001, "m_expensive": 0.0005}

    # 1. min_max: cheap -> 0.0, expensive -> 1.0
    norm_minmax = CostNormalizer(costs, strategy="min_max")
    raw_c, norm_c = norm_minmax.normalize("m_cheap")
    assert raw_c == 0.0001
    assert pytest.approx(norm_c) == 0.0

    raw_e, norm_e = norm_minmax.normalize("m_expensive")
    assert raw_e == 0.0005
    assert pytest.approx(norm_e) == 1.0

    # 2. max_scale: cheap -> 0.2, expensive -> 1.0
    norm_max = CostNormalizer(costs, strategy="max_scale")
    _, norm_c2 = norm_max.normalize("m_cheap")
    assert pytest.approx(norm_c2) == 0.2
    _, norm_e2 = norm_max.normalize("m_expensive")
    assert pytest.approx(norm_e2) == 1.0

    # 3. raw: cheap -> 0.0001, expensive -> 0.0005
    norm_raw = CostNormalizer(costs, strategy="raw")
    _, norm_c3 = norm_raw.normalize("m_cheap")
    assert norm_c3 == 0.0001


def test_lambda_zero_selects_highest_quality(mock_cost_normalizer: CostNormalizer) -> None:
    """When lambda = 0.0, the decision engine selects strictly for highest predicted quality."""
    engine = LambdaDecisionEngine(mock_cost_normalizer)
    predictions = {
        "model_fast": 0.70,   # Cheapest
        "model_mid": 0.85,    # Medium
        "model_smart": 0.95,  # Most expensive, highest quality
    }

    decision = engine.select_model(predictions, lambda_value=0.0)
    assert decision.selected_model_id == "model_smart"
    assert decision.predicted_quality == 0.95
    assert decision.utility == 0.95
    assert decision.expected_cost_usd == 0.0010
    assert decision.normalized_cost == 1.0


def test_high_lambda_selects_lowest_cost(mock_cost_normalizer: CostNormalizer) -> None:
    """When lambda is large, the cost penalty dominates and selects the cheapest model."""
    engine = LambdaDecisionEngine(mock_cost_normalizer)
    predictions = {
        "model_fast": 0.70,   # Cheapest (cost=0.0001, norm=0.0)
        "model_mid": 0.85,    # Medium (cost=0.0005, norm=0.444)
        "model_smart": 0.95,  # Most expensive (cost=0.0010, norm=1.0)
    }

    # At lambda = 2.0:
    # U(fast) = 0.70 - 2.0 * 0.0 = 0.70
    # U(mid) = 0.85 - 2.0 * 0.444 = -0.038
    # U(smart) = 0.95 - 2.0 * 1.0 = -1.05
    decision = engine.select_model(predictions, lambda_value=2.0)
    assert decision.selected_model_id == "model_fast"
    assert decision.utility == pytest.approx(0.70)
    assert decision.expected_cost_usd == 0.0001
    assert decision.normalized_cost == 0.0


def test_presets_resolution() -> None:
    """Verifies that auto:quality, auto:balanced, auto:cheap resolve to configured lambda values."""
    assert resolve_preset_lambda("auto:quality") == 0.0
    assert resolve_preset_lambda("auto:balanced") == 0.5
    assert resolve_preset_lambda("auto:cheap") == 2.0
    assert resolve_preset_lambda(1.25) == 1.25

    with pytest.raises(ValueError, match="Unknown routing preset"):
        resolve_preset_lambda("auto:ultra_cheap_nonexistent")


def test_deterministic_tie_breaking() -> None:
    """When utilities are equal, breaks tie in favor of lower cost, then alphabetically."""
    costs = {
        "model_alpha": 0.0002,  # Same utility as beta, but lower cost
        "model_beta": 0.0004,
        "model_gamma": 0.0002,  # Same utility and cost as alpha, but 'alpha' < 'gamma'
    }
    # Raw cost normalization strategy so utility differences are easy to construct
    normalizer = CostNormalizer(costs, strategy="raw")
    engine = LambdaDecisionEngine(normalizer)

    # 1. Tie-break on cost: alpha vs beta with identical utility
    # Let lambda = 1.0. For U to be equal: Q_alpha - C_alpha = Q_beta - C_beta
    # Q_alpha = 0.80, C_alpha = 0.0002 -> U_alpha = 0.7998
    # Q_beta = 0.8002, C_beta = 0.0004 -> U_beta = 0.7998
    preds1 = {"model_alpha": 0.80, "model_beta": 0.8002}
    dec1 = engine.select_model(preds1, lambda_value=1.0)
    assert dec1.selected_model_id == "model_alpha"  # Lower cost wins

    # 2. Tie-break on alphabetical model_id when cost and utility are identical
    # Q_alpha = 0.80, C_alpha = 0.0002 -> U_alpha = 0.80
    # Q_gamma = 0.80, C_gamma = 0.0002 -> U_gamma = 0.80
    preds2 = {"model_alpha": 0.80, "model_gamma": 0.80}
    dec2 = engine.select_model(preds2, lambda_value=0.0)
    assert dec2.selected_model_id == "model_alpha"  # 'model_alpha' < 'model_gamma'


def test_missing_model_predictions_handled_gracefully(mock_cost_normalizer: CostNormalizer) -> None:
    """Models with missing predictions are disqualified without causing failure."""
    engine = LambdaDecisionEngine(mock_cost_normalizer)
    predictions = {
        "model_fast": 0.80,
        "model_mid": None,  # Missing prediction
        # model_smart omitted entirely
    }

    decision = engine.select_model(predictions, lambda_value=0.0)
    assert decision.selected_model_id == "model_fast"
    assert decision.model_utilities["model_mid"].is_eligible is False
    assert decision.model_utilities["model_mid"].disqualification_reason == "missing_prediction"
    assert decision.model_utilities["model_smart"].is_eligible is False


def test_validation_errors(mock_cost_normalizer: CostNormalizer) -> None:
    """Verifies that invalid lambda values and invalid probabilities are rejected."""
    engine = LambdaDecisionEngine(mock_cost_normalizer)

    # Negative lambda
    with pytest.raises(ValueError, match="lambda_value must be a finite, non-negative float"):
        engine.select_model({"model_fast": 0.8}, lambda_value=-0.5)

    # Non-finite lambda
    with pytest.raises(ValueError, match="lambda_value must be a finite, non-negative float"):
        engine.select_model({"model_fast": 0.8}, lambda_value=float("nan"))

    # Invalid probability (> 1.0)
    with pytest.raises(ValueError, match="Predicted quality for model 'model_fast' must be in"):
        engine.select_model({"model_fast": 1.25}, lambda_value=0.0)

    # Invalid probability (< 0.0)
    with pytest.raises(ValueError, match="Predicted quality for model 'model_fast' must be in"):
        engine.select_model({"model_fast": -0.1}, lambda_value=0.0)


def test_batch_decisions(mock_cost_normalizer: CostNormalizer) -> None:
    """Verifies select_models_batch processes multiple queries consistently."""
    engine = LambdaDecisionEngine(mock_cost_normalizer)
    batch_preds = {
        "model_fast": np.array([0.9, 0.4], dtype=np.float32),
        "model_mid": np.array([0.8, 0.7], dtype=np.float32),
        "model_smart": np.array([0.95, 0.9], dtype=np.float32),
    }

    decisions = engine.select_models_batch(batch_preds, lambda_value="auto:quality")
    assert len(decisions) == 2
    assert decisions[0].selected_model_id == "model_smart"
    assert decisions[1].selected_model_id == "model_smart"

    # With high cost sensitivity (auto:cheap)
    decisions_cheap = engine.select_models_batch(batch_preds, lambda_value="auto:cheap")
    assert decisions_cheap[0].selected_model_id == "model_fast"


def test_from_training_dataset_aggregates_per_prompt_first_and_equal_prompt_weight() -> None:
    """Verifies each prompt has equal weight in expected cost, regardless of repeated run count."""
    train_records = [
        # Prompt 1: 10 repeated runs, each costing 0.0001
        type("Rec", (), {"prompt_id": "p1", "model_id": "m1", "cost": 0.0001, "execution_status": "success"})()
        for _ in range(10)
    ] + [
        # Prompt 2: 1 single run, costing 0.0003
        type("Rec", (), {"prompt_id": "p2", "model_id": "m1", "cost": 0.0003, "execution_status": "success"})(),
    ]

    mock_dataset = type(
        "MockDataset",
        (),
        {
            "train_records": train_records,
            "candidate_models": ["m1"],
        },
    )()

    engine = LambdaDecisionEngine.from_training_dataset(mock_dataset, strategy="raw")
    # Expected cost = (0.0001 + 0.0003) / 2 = 0.000200
    # If not aggregated by prompt first, cost would be (10*0.0001 + 1*0.0003)/11 = 0.000118
    raw_cost, _ = engine.cost_normalizer.normalize("m1")
    assert pytest.approx(raw_cost, abs=1e-8) == 0.000200


def test_expected_cost_estimand_attempted_call_scenarios() -> None:
    """Verifies that expected cost estimates expenditure per attempted call across all call types:
    1. Successful scored call with cost -> included
    2. Successful unscored call with cost -> included
    3. Failed call with nonzero cost -> included
    4. Failed call with zero cost -> included as 0.0
    5. Missing cost -> remains missing, never coerced to 0.0
    6. Negative or non-finite costs -> rejected with ValueError
    """
    train_records = [
        # Scenario 1: Successful scored call with cost ($0.0004) on p1
        type("Rec", (), {
            "prompt_id": "p1", "model_id": "m1", "cost": 0.0004,
            "execution_status": "success", "correctness_label": 1
        })(),
        # Scenario 2: Successful unscored call with cost ($0.0002) on p2
        type("Rec", (), {
            "prompt_id": "p2", "model_id": "m1", "cost": 0.0002,
            "execution_status": "success", "correctness_label": None
        })(),
        # Scenario 3: Failed call with nonzero cost ($0.0006, e.g. token billing on timeout) on p3
        type("Rec", (), {
            "prompt_id": "p3", "model_id": "m1", "cost": 0.0006,
            "execution_status": "failed", "correctness_label": None
        })(),
        # Scenario 4: Failed call with legitimate zero cost ($0.0, e.g. pre-call rate limit) on p4
        type("Rec", (), {
            "prompt_id": "p4", "model_id": "m1", "cost": 0.0,
            "execution_status": "failed", "correctness_label": None
        })(),
        # Scenario 5: Missing cost on p5 (must NOT be treated as zero or bias the estimate)
        type("Rec", (), {
            "prompt_id": "p5", "model_id": "m1", "cost": None,
            "execution_status": "failed", "correctness_label": None
        })(),
    ]

    mock_dataset = type(
        "MockDataset",
        (),
        {
            "train_records": train_records,
            "candidate_models": ["m1"],
        },
    )()

    engine = LambdaDecisionEngine.from_training_dataset(mock_dataset, strategy="raw")
    # Prompts with valid costs: p1 (0.0004), p2 (0.0002), p3 (0.0006), p4 (0.0)
    # Prompt p5 has only missing cost (None) -> excluded from prompt_avg_costs
    # Expected cost = (0.0004 + 0.0002 + 0.0006 + 0.0) / 4 = 0.0012 / 4 = 0.000300
    raw_cost, _ = engine.cost_normalizer.normalize("m1")
    assert pytest.approx(raw_cost, abs=1e-8) == 0.000300

    # Scenario 6: Negative or non-finite costs are rejected with ValueError
    neg_dataset = type(
        "MockDataset",
        (),
        {
            "train_records": [
                type("Rec", (), {"prompt_id": "p_err", "model_id": "m1", "cost": -0.001, "execution_status": "success"})()
            ],
            "candidate_models": ["m1"],
        },
    )()
    with pytest.raises(ValueError, match="Costs must be finite and non-negative"):
        LambdaDecisionEngine.from_training_dataset(neg_dataset)

    nan_dataset = type(
        "MockDataset",
        (),
        {
            "train_records": [
                type("Rec", (), {"prompt_id": "p_err", "model_id": "m1", "cost": float("nan"), "execution_status": "success"})()
            ],
            "candidate_models": ["m1"],
        },
    )()
    with pytest.raises(ValueError, match="Costs must be finite and non-negative"):
        LambdaDecisionEngine.from_training_dataset(nan_dataset)


def test_from_training_dataset_strictly_isolates_test_data() -> None:
    """Test records and outcomes have zero influence on training cost estimation."""
    train_records = [
        type("Rec", (), {"prompt_id": "p_train", "model_id": "m1", "cost": 0.0002, "execution_status": "success"})(),
    ]
    test_records = [
        type("Rec", (), {"prompt_id": "p_test", "model_id": "m1", "cost": 0.9999, "execution_status": "success"})(),
    ]

    mock_dataset = type(
        "MockDataset",
        (),
        {
            "train_records": train_records,
            "test_records": test_records,
            "candidate_models": ["m1"],
        },
    )()

    engine = LambdaDecisionEngine.from_training_dataset(mock_dataset, strategy="raw")
    raw_cost, _ = engine.cost_normalizer.normalize("m1")
    assert pytest.approx(raw_cost, abs=1e-8) == 0.0002


def test_missing_and_invalid_costs_handled_explicitly() -> None:
    """Missing training costs raise error unless fallback is provided, and invalid costs raise error."""
    # No training records and no fallback -> raises ValueError
    empty_dataset = type("MockDataset", (), {"train_records": [], "candidate_models": ["m1"]})()
    with pytest.raises(ValueError, match="No valid training costs observed"):
        LambdaDecisionEngine.from_training_dataset(empty_dataset)

    # Valid fallback provided -> uses fallback
    engine = LambdaDecisionEngine.from_training_dataset(empty_dataset, fallback_costs={"m1": 0.0005})
    raw_c, _ = engine.cost_normalizer.normalize("m1")
    assert raw_c == 0.0005

    # Invalid negative fallback cost -> raises ValueError
    with pytest.raises(ValueError, match="Invalid fallback cost"):
        LambdaDecisionEngine.from_training_dataset(empty_dataset, fallback_costs={"m1": -0.001})


def test_zero_network_calls_during_decision(
    monkeypatch: pytest.MonkeyPatch,
    mock_cost_normalizer: CostNormalizer,
) -> None:
    """Guarantees LambdaDecisionEngine operates strictly offline without network socket calls."""
    def blocked_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network socket call attempted during offline decision making!")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)

    engine = LambdaDecisionEngine(mock_cost_normalizer)
    dec = engine.select_model({"model_fast": 0.8, "model_smart": 0.9}, lambda_value=0.5)
    assert dec.selected_model_id in ("model_fast", "model_smart")
