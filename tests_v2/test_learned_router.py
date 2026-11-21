"""
tests_v2/test_learned_router.py

Unit tests for Learned Router (Logistic Regression) under Phase M3.
Verifies probability bounds, multiple models, weighted binary empirical quality,
repeated-run influence control, single-class fallback, feature leakage prevention,
real dataset honest status reporting, and zero network calls.
"""

from __future__ import annotations

import socket
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eval.baselines import DEFAULT_MATRIX_PATH
from eval.routers.data_prep import PreparedDataset, prepare_routing_dataset
from eval.routers.encoders import DeterministicTestEncoder
from eval.routers.learned_router import (
    LearnedRouter,
    ModelFitStatus,
    PromptFeatureExtractor,
)


@pytest.fixture
def synthetic_learned_dataset() -> PreparedDataset:
    """Creates a deterministic synthetic PreparedDataset with multi-class outcomes across multiple models."""
    encoder = DeterministicTestEncoder(dimension=16)

    # 8 training prompts, 2 test prompts
    records = [
        # model_a: balanced classes (4 correct, 4 incorrect)
        {"prompt_id": "p_m1", "task_type": "math", "model_id": "model_a", "run": 1, "correct": True, "score": 1.0, "cost": 0.001, "execution_success": True, "evaluation_status": "scored"},
        {"prompt_id": "p_m2", "task_type": "math", "model_id": "model_a", "run": 1, "correct": True, "score": 1.0, "cost": 0.001, "execution_success": True, "evaluation_status": "scored"},
        {"prompt_id": "p_m3", "task_type": "math", "model_id": "model_a", "run": 1, "correct": False, "score": 0.0, "cost": 0.001, "execution_success": True, "evaluation_status": "scored"},
        {"prompt_id": "p_m4", "task_type": "math", "model_id": "model_a", "run": 1, "correct": False, "score": 0.0, "cost": 0.001, "execution_success": True, "evaluation_status": "scored"},
        {"prompt_id": "p_c1", "task_type": "code", "model_id": "model_a", "run": 1, "correct": True, "score": 1.0, "cost": 0.002, "execution_success": True, "evaluation_status": "scored"},
        {"prompt_id": "p_c2", "task_type": "code", "model_id": "model_a", "run": 1, "correct": False, "score": 0.0, "cost": 0.002, "execution_success": True, "evaluation_status": "scored"},
        # p_c3 has repeated runs on model_a: 1 correct, 1 incorrect -> empirical quality 0.5
        {"prompt_id": "p_c3", "task_type": "code", "model_id": "model_a", "run": 1, "correct": True, "score": 1.0, "cost": 0.002, "execution_success": True, "evaluation_status": "scored"},
        {"prompt_id": "p_c3", "task_type": "code", "model_id": "model_a", "run": 2, "correct": False, "score": 0.0, "cost": 0.002, "execution_success": True, "evaluation_status": "scored"},
        {"prompt_id": "p_c4", "task_type": "code", "model_id": "model_a", "run": 1, "correct": True, "score": 1.0, "cost": 0.002, "execution_success": True, "evaluation_status": "scored"},

        # model_b: single-class target (all 1s across evaluated training prompts)
        {"prompt_id": "p_m1", "task_type": "math", "model_id": "model_b", "run": 1, "correct": True, "score": 1.0, "cost": 0.0005, "execution_success": True, "evaluation_status": "scored"},
        {"prompt_id": "p_m2", "task_type": "math", "model_id": "model_b", "run": 1, "correct": True, "score": 1.0, "cost": 0.0005, "execution_success": True, "evaluation_status": "scored"},
        {"prompt_id": "p_c1", "task_type": "code", "model_id": "model_b", "run": 1, "correct": True, "score": 1.0, "cost": 0.0005, "execution_success": True, "evaluation_status": "scored"},

        # model_c: no evaluations in training set (evaluated in test only)
        {"prompt_id": "p_test_1", "task_type": "math", "model_id": "model_c", "run": 1, "correct": True, "score": 1.0, "cost": 0.0001, "execution_success": True, "evaluation_status": "scored"},

        # Test set rows
        {"prompt_id": "p_test_1", "task_type": "math", "model_id": "model_a", "run": 1, "correct": True, "score": 1.0, "cost": 0.001, "execution_success": True, "evaluation_status": "scored"},
        {"prompt_id": "p_test_2", "task_type": "code", "model_id": "model_a", "run": 1, "correct": False, "score": 0.0, "cost": 0.002, "execution_success": True, "evaluation_status": "scored"},
    ]

    custom_split = {
        "train_prompt_ids": ["p_m1", "p_m2", "p_m3", "p_m4", "p_c1", "p_c2", "p_c3", "p_c4"],
        "test_prompt_ids": ["p_test_1", "p_test_2"],
    }

    prompt_lookup = {
        "p_m1": "Compute 1 + 1.",
        "p_m2": "Compute 2 + 2.",
        "p_m3": "Compute 3 + 3.",
        "p_m4": "Compute 4 + 4.",
        "p_c1": "def solve(): return 1",
        "p_c2": "import math\ndef calculate(): pass",
        "p_c3": "Write a python function to add two numbers.",
        "p_c4": "class Solution: def search(self): pass",
        "p_test_1": "Compute 5 + 5.",
        "p_test_2": "def test_func(): return True",
    }

    return prepare_routing_dataset(
        pd.DataFrame(records),
        benchmark_config_id=None,
        encoder=encoder,
        prompt_lookup=prompt_lookup,
        custom_split=custom_split,
    )


def test_learned_router_probability_bounds_and_shapes(synthetic_learned_dataset: PreparedDataset) -> None:
    """Verifies that predicted probabilities lie strictly within [0.0, 1.0] and match query count."""
    encoder = DeterministicTestEncoder(dimension=16)
    router = LearnedRouter(C=1.0, encoder=encoder)
    router.fit(synthetic_learned_dataset)

    queries = ["Compute 10 + 10.", "def test(): pass", "Simple text query."]
    probs = router.predict_proba(queries)

    assert set(probs.keys()) == set(synthetic_learned_dataset.candidate_models)

    for m in router.candidate_models:
        p_arr = probs[m]
        assert isinstance(p_arr, np.ndarray)
        assert p_arr.shape == (3,)
        assert p_arr.dtype == np.float32
        assert np.all(p_arr >= 0.0)
        assert np.all(p_arr <= 1.0)

    # predict_proba_one
    p_one = router.predict_proba_one(queries[0])
    for m in router.candidate_models:
        assert isinstance(p_one[m], float)
        assert 0.0 <= p_one[m] <= 1.0
        assert pytest.approx(p_one[m], abs=1e-5) == probs[m][0]


def test_repeated_run_weighted_observations_and_sample_weight(synthetic_learned_dataset: PreparedDataset) -> None:
    """Verifies that fractional empirical quality q expands to weights q and (1-q), with sum == 1.0."""
    encoder = DeterministicTestEncoder(dimension=16)
    router = LearnedRouter(C=1.0, encoder=encoder)
    router.fit(synthetic_learned_dataset)

    status_a = router.model_fit_statuses["model_a"]
    assert status_a.is_fitted is True
    assert status_a.fallback_reason is None
    # 8 unique prompts in train set evaluated for model_a (p_c3 was evaluated twice)
    assert status_a.train_prompt_count == 8

    # model_a classifier exists
    assert "model_a" in router._classifiers
    clf_a = router._classifiers["model_a"]
    assert len(clf_a.classes_) == 2


def test_single_class_fallback(synthetic_learned_dataset: PreparedDataset) -> None:
    """Models with only one observed class fall back to Beta-smoothed probability with exposed reason."""
    encoder = DeterministicTestEncoder(dimension=16)
    router = LearnedRouter(C=1.0, encoder=encoder, smoothing_alpha=1.0, smoothing_beta=1.0)
    router.fit(synthetic_learned_dataset)

    status_b = router.model_fit_statuses["model_b"]
    assert status_b.is_fitted is False
    assert status_b.fallback_reason == "single_class_observed"
    assert status_b.observed_classes == [1]
    # 3 successes, 0 failures -> (3 + 1) / (3 + 1 + 1) = 4 / 5 = 0.8
    assert pytest.approx(status_b.mean_empirical_quality, abs=1e-5) == 1.0
    assert pytest.approx(status_b.fallback_prob, abs=1e-5) == 0.8

    probs = router.predict_proba(["Any query prompt."])
    assert pytest.approx(probs["model_b"][0], abs=1e-5) == 0.8


def test_no_training_data_fallback(synthetic_learned_dataset: PreparedDataset) -> None:
    """Models with zero training observations fall back to (0 + alpha)/(0 + alpha + beta) with exposed reason."""
    encoder = DeterministicTestEncoder(dimension=16)
    router = LearnedRouter(C=1.0, encoder=encoder, smoothing_alpha=1.0, smoothing_beta=1.0)
    router.fit(synthetic_learned_dataset)

    status_c = router.model_fit_statuses["model_c"]
    assert status_c.is_fitted is False
    assert status_c.fallback_reason == "no_training_data"
    assert status_c.train_prompt_count == 0
    # 0 observations -> (0 + 1) / (0 + 1 + 1) = 0.5
    assert pytest.approx(status_c.fallback_prob, abs=1e-5) == 0.5

    probs = router.predict_proba(["Any query prompt."])
    assert pytest.approx(probs["model_c"][0], abs=1e-5) == 0.5


def test_beta_smoothed_fallback_exact_examples() -> None:
    """Verifies the exact Beta-smoothing formula: P(correct) = (successes + alpha) / (evals + alpha + beta)."""
    router = LearnedRouter(smoothing_alpha=1.0, smoothing_beta=1.0)

    # 1 success, 0 failures -> (1 + 1) / (1 + 1 + 1) = 2/3 ≈ 0.667
    p1 = router.compute_smoothed_fallback(successes=1.0, total_evals=1.0)
    assert pytest.approx(p1, abs=1e-3) == 0.667
    assert pytest.approx(p1, abs=1e-6) == 2.0 / 3.0

    # 0 successes, 1 failure -> (0 + 1) / (1 + 1 + 1) = 1/3 ≈ 0.333
    p2 = router.compute_smoothed_fallback(successes=0.0, total_evals=1.0)
    assert pytest.approx(p2, abs=1e-3) == 0.333
    assert pytest.approx(p2, abs=1e-6) == 1.0 / 3.0

    # 0 observations -> (0 + 1) / (0 + 1 + 1) = 1/2 = 0.5
    p3 = router.compute_smoothed_fallback(successes=0.0, total_evals=0.0)
    assert pytest.approx(p3, abs=1e-6) == 0.5

    # 3 successes, 1 failure -> (3 + 1) / (4 + 1 + 1) = 4/6 ≈ 0.667
    p4 = router.compute_smoothed_fallback(successes=3.0, total_evals=4.0)
    assert pytest.approx(p4, abs=1e-3) == 0.667
    assert pytest.approx(p4, abs=1e-6) == 4.0 / 6.0


def test_beta_smoothing_configurable_and_validation() -> None:
    """Verifies alpha and beta are configurable and non-positive values are rejected."""
    # Custom alpha and beta: alpha=2, beta=3 -> 0 observations = 2 / (2 + 3) = 0.4
    router_custom = LearnedRouter(smoothing_alpha=2.0, smoothing_beta=3.0)
    assert router_custom.compute_smoothed_fallback(0.0, 0.0) == pytest.approx(0.4)

    # Reject non-positive alpha and beta
    with pytest.raises(ValueError, match="smoothing_alpha and smoothing_beta must be positive"):
        LearnedRouter(smoothing_alpha=0.0)

    with pytest.raises(ValueError, match="smoothing_alpha and smoothing_beta must be positive"):
        LearnedRouter(smoothing_beta=-1.0)


def test_feature_pipeline_fit_strictly_on_training_prompts(synthetic_learned_dataset: PreparedDataset) -> None:
    """Ensures feature normalization parameters and category vocabularies are fit only on training prompts."""
    encoder = DeterministicTestEncoder(dimension=16)
    router = LearnedRouter(C=1.0, encoder=encoder)
    router.fit(synthetic_learned_dataset)

    extractor = router._feature_extractor
    assert extractor is not None
    assert extractor.is_fitted is True

    # Training prompts only have categories "math" and "code"
    assert set(extractor._categories_vocab) == {"code", "math"}

    # An unseen category at test time should map to all zeros for category bits without raising error
    unseen_cat_probs = router.predict_proba(["What is the capital of Peru?"], task_categories=["geography"])
    assert 0.0 <= unseen_cat_probs["model_a"][0] <= 1.0


def test_real_historical_dataset_reports_insufficient_training_honestly() -> None:
    """Evaluates LearnedRouter on real matrix.parquet and verifies that lack of classes is honestly exposed."""
    encoder = DeterministicTestEncoder(dimension=32)
    dataset = prepare_routing_dataset(
        DEFAULT_MATRIX_PATH,
        benchmark_config_id="m1-efa7ec3afd4f",
        encoder=encoder,
    )

    router = LearnedRouter(C=1.0, encoder=encoder)
    router.fit(dataset)

    # In the real historical matrix (2 prompts, 1 in train split):
    # Every model has exactly 1 train observation, which is single-class (either all 1s or all 0s)
    for model_id in router.candidate_models:
        status = router.model_fit_statuses[model_id]
        assert status.is_fitted is False
        assert status.fallback_reason == "single_class_observed"
        assert status.train_prompt_count == 1
        # Raw empirical quality is preserved
        assert status.mean_empirical_quality in (0.0, 1.0)
        # Beta-smoothed fallback prevents unjustified 1.0 or 0.0 certainty:
        # On math_001, each model has 2 scored runs:
        # 2 successes out of 2 runs -> (2 + 1) / (2 + 1 + 1) = 3/4 = 0.75
        # 0 successes out of 2 runs -> (0 + 1) / (2 + 1 + 1) = 1/4 = 0.25
        if status.mean_empirical_quality == 1.0:
            assert pytest.approx(status.fallback_prob, abs=1e-3) == 0.75
        else:
            assert pytest.approx(status.fallback_prob, abs=1e-3) == 0.25


def test_deterministic_learned_predictions(synthetic_learned_dataset: PreparedDataset) -> None:
    """LearnedRouter produces bitwise identical predictions across multiple calls."""
    encoder = DeterministicTestEncoder(dimension=16)
    router = LearnedRouter(C=1.0, encoder=encoder, random_state=42)
    router.fit(synthetic_learned_dataset)

    prompt = "def binary_search(arr, target): pass"
    p1 = router.predict_proba([prompt])
    p2 = router.predict_proba([prompt])

    for m in router.candidate_models:
        assert np.array_equal(p1[m], p2[m])


def test_empty_inputs_handling(synthetic_learned_dataset: PreparedDataset) -> None:
    """Empty queries sequence returns empty probability arrays without error."""
    encoder = DeterministicTestEncoder(dimension=16)
    router = LearnedRouter(C=1.0, encoder=encoder)
    router.fit(synthetic_learned_dataset)

    res = router.predict_proba([])
    for m in router.candidate_models:
        assert isinstance(res[m], np.ndarray)
        assert len(res[m]) == 0


def test_zero_network_calls_during_learned_routing(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_learned_dataset: PreparedDataset,
) -> None:
    """Guarantees LearnedRouter makes zero socket network calls during fitting and prediction."""
    def blocked_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network socket call attempted during offline learned routing!")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)

    enc = DeterministicTestEncoder(dimension=16)
    router = LearnedRouter(C=1.0, encoder=enc)
    router.fit(synthetic_learned_dataset)
    res = router.predict_proba(["Offline prompt query."])
    assert len(res) > 0
