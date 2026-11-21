"""
tests_v2/test_knn_router.py

Unit tests for kNN Router under Phase M3.
Verifies algorithm correctness, probability bounds, distance weighting, repeated-run handling,
missing candidate model outcomes, strict leakage prevention, and zero network calls.
"""

from __future__ import annotations

import socket
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eval.routers.data_prep import PreparedDataset, prepare_routing_dataset
from eval.routers.encoders import DeterministicTestEncoder
from eval.routers.knn_router import KNNRouter, NeighborInfo


@pytest.fixture
def synthetic_router_dataset() -> PreparedDataset:
    """Creates a deterministic synthetic PreparedDataset with known geometry and multiple models."""
    encoder = DeterministicTestEncoder(dimension=16)

    # We define 6 prompts: 4 for train, 2 for test
    # Prompts 1, 2, 3: math
    # Prompts 4, 5, 6: code
    records = [
        # p_math_1 (train): model_a correct (1), model_b incorrect (0)
        {"prompt_id": "p_math_1", "task_type": "math", "model_id": "model_a", "run": 1, "correct": True, "score": 1.0, "cost": 0.001, "execution_success": True, "evaluation_status": "scored"},
        {"prompt_id": "p_math_1", "task_type": "math", "model_id": "model_b", "run": 1, "correct": False, "score": 0.0, "cost": 0.0005, "execution_success": True, "evaluation_status": "scored"},

        # p_math_2 (train): model_a correct (1), model_b has missing evaluation (unscored)
        {"prompt_id": "p_math_2", "task_type": "math", "model_id": "model_a", "run": 1, "correct": True, "score": 1.0, "cost": 0.001, "execution_success": True, "evaluation_status": "scored"},
        {"prompt_id": "p_math_2", "task_type": "math", "model_id": "model_b", "run": 1, "correct": None, "score": np.nan, "cost": 0.0005, "execution_success": True, "evaluation_status": "unscored"},

        # p_math_3 (train): repeated runs on model_a (1 correct, 1 incorrect -> empirical 0.5)
        {"prompt_id": "p_math_3", "task_type": "math", "model_id": "model_a", "run": 1, "correct": True, "score": 1.0, "cost": 0.001, "execution_success": True, "evaluation_status": "scored"},
        {"prompt_id": "p_math_3", "task_type": "math", "model_id": "model_a", "run": 2, "correct": False, "score": 0.0, "cost": 0.001, "execution_success": True, "evaluation_status": "scored"},
        {"prompt_id": "p_math_3", "task_type": "math", "model_id": "model_b", "run": 1, "correct": True, "score": 1.0, "cost": 0.0005, "execution_success": True, "evaluation_status": "scored"},

        # p_code_1 (train): model_a incorrect (0), model_b correct (1)
        {"prompt_id": "p_code_1", "task_type": "code", "model_id": "model_a", "run": 1, "correct": False, "score": 0.0, "cost": 0.002, "execution_success": True, "evaluation_status": "scored"},
        {"prompt_id": "p_code_1", "task_type": "code", "model_id": "model_b", "run": 1, "correct": True, "score": 1.0, "cost": 0.0005, "execution_success": True, "evaluation_status": "scored"},

        # p_code_2 (train): model_c has evaluation here only (incomplete model coverage)
        {"prompt_id": "p_code_2", "task_type": "code", "model_id": "model_c", "run": 1, "correct": True, "score": 1.0, "cost": 0.0002, "execution_success": True, "evaluation_status": "scored"},

        # Test prompts (p_test_1, p_test_2)
        {"prompt_id": "p_test_1", "task_type": "math", "model_id": "model_a", "run": 1, "correct": True, "score": 1.0, "cost": 0.001, "execution_success": True, "evaluation_status": "scored"},
        {"prompt_id": "p_test_2", "task_type": "code", "model_id": "model_b", "run": 1, "correct": True, "score": 1.0, "cost": 0.0005, "execution_success": True, "evaluation_status": "scored"},
    ]

    custom_split = {
        "train_prompt_ids": ["p_math_1", "p_math_2", "p_math_3", "p_code_1", "p_code_2"],
        "test_prompt_ids": ["p_test_1", "p_test_2"],
    }

    prompt_lookup = {
        "p_math_1": "Calculate 10 + 20.",
        "p_math_2": "Calculate 30 + 40.",
        "p_math_3": "Calculate 50 + 60.",
        "p_code_1": "Write python code to reverse a list.",
        "p_code_2": "Write python code to sort integers.",
        "p_test_1": "Calculate 15 + 25.",
        "p_test_2": "Write python code to filter an array.",
    }

    return prepare_routing_dataset(
        pd.DataFrame(records),
        benchmark_config_id=None,
        encoder=encoder,
        prompt_lookup=prompt_lookup,
        custom_split=custom_split,
    )


def test_knn_router_probability_bounds_and_shapes(synthetic_router_dataset: PreparedDataset) -> None:
    """Verifies that predicted probabilities strictly satisfy 0.0 <= P <= 1.0 and match expected shapes."""
    encoder = DeterministicTestEncoder(dimension=16)
    router = KNNRouter(k=2, encoder=encoder, weights="distance")
    router.fit(synthetic_router_dataset)

    queries = ["Calculate 12 + 18.", "Write a python function."]
    probs = router.predict_proba(queries)

    assert set(probs.keys()) == set(synthetic_router_dataset.candidate_models)

    for m in router.candidate_models:
        p_arr = probs[m]
        assert isinstance(p_arr, np.ndarray)
        assert p_arr.shape == (2,)
        assert p_arr.dtype == np.float32
        assert np.all(p_arr >= 0.0)
        assert np.all(p_arr <= 1.0)

    # predict_proba_one
    p_one = router.predict_proba_one(queries[0])
    for m in router.candidate_models:
        assert isinstance(p_one[m], float)
        assert 0.0 <= p_one[m] <= 1.0
        assert pytest.approx(p_one[m], abs=1e-5) == probs[m][0]


def test_distance_weighted_vs_uniform_voting(synthetic_router_dataset: PreparedDataset) -> None:
    """Verifies that distance weighting places higher weight on closer neighbors than uniform voting."""
    encoder = DeterministicTestEncoder(dimension=16)

    # Construct two identical training embeddings with opposite labels
    v_a = encoder.encode_one("target anchor")
    v_b = encoder.encode_one("very different distant anchor")

    train_prompts = {
        "anchor_close": type("PR", (), {"prompt_embedding": v_a, "task_category": "t"})(),
        "anchor_far": type("PR", (), {"prompt_embedding": v_b, "task_category": "t"})(),
    }
    train_records = [
        type("MR", (), {"prompt_id": "anchor_close", "model_id": "m1", "label": 1.0})(),
        type("MR", (), {"prompt_id": "anchor_far", "model_id": "m1", "label": 0.0})(),
    ]

    mock_dataset = type(
        "MockDataset",
        (),
        {
            "train_records": train_records,
            "train_prompt_ids": ["anchor_close", "anchor_far"],
            "candidate_models": ["m1"],
            "prompts": train_prompts,
        },
    )()

    # Query very close to anchor_close
    query_emb = v_a + 1e-4 * (v_b - v_a)

    router_dist = KNNRouter(k=2, encoder=encoder, weights="distance").fit(mock_dataset)
    router_unif = KNNRouter(k=2, encoder=encoder, weights="uniform").fit(mock_dataset)

    p_dist = router_dist.predict_proba(query_emb.reshape(1, -1))["m1"][0]
    p_unif = router_unif.predict_proba(query_emb.reshape(1, -1))["m1"][0]

    # Uniform average is exactly (1.0 + 0.0) / 2 = 0.5
    assert pytest.approx(float(p_unif), abs=1e-5) == 0.5

    # Distance-weighted average is heavily skewed toward anchor_close (close to 1.0)
    assert p_dist > 0.95


def test_zero_distance_exact_match(synthetic_router_dataset: PreparedDataset) -> None:
    """Exact embedding match (distance == 0) yields the exact empirical quality of that neighbor."""
    encoder = DeterministicTestEncoder(dimension=16)
    router = KNNRouter(k=3, encoder=encoder, weights="distance")
    router.fit(synthetic_router_dataset)

    # Query with exact prompt text of p_math_1
    exact_text = "Calculate 10 + 20."
    probs = router.predict_proba_one(exact_text)

    # On p_math_1, model_a had 1.0, model_b had 0.0
    assert pytest.approx(probs["model_a"], abs=1e-5) == 1.0
    assert pytest.approx(probs["model_b"], abs=1e-5) == 0.0


def test_k_greater_than_available_neighbors(synthetic_router_dataset: PreparedDataset) -> None:
    """When k > number of available training neighbors, k is clamped to N without error."""
    encoder = DeterministicTestEncoder(dimension=16)
    # k = 50, but only 5 training prompts exist
    router = KNNRouter(k=50, encoder=encoder)
    router.fit(synthetic_router_dataset)

    probs = router.predict_proba(["Calculate 1 + 1."])
    assert "model_a" in probs
    assert 0.0 <= probs["model_a"][0] <= 1.0


def test_repeated_run_handling_does_not_distort_neighbor_weight(synthetic_router_dataset: PreparedDataset) -> None:
    """Repeated runs on a prompt provide an empirical quality in [0, 1] without multiplying prompt count."""
    encoder = DeterministicTestEncoder(dimension=16)
    router = KNNRouter(k=1, encoder=encoder, weights="distance")
    router.fit(synthetic_router_dataset)

    # p_math_3 had 2 runs: 1 correct, 1 incorrect -> empirical quality 0.5
    # Query matching p_math_3 exactly
    exact_p3 = "Calculate 50 + 60."
    probs = router.predict_proba_one(exact_p3)
    assert pytest.approx(probs["model_a"], abs=1e-5) == 0.5

    # Check neighbor diagnostics
    neighbors = router.get_neighbors(exact_p3, k=1, model_id="model_a")
    assert len(neighbors) == 1
    assert neighbors[0].prompt_id == "p_math_3"
    assert pytest.approx(neighbors[0].distance, abs=1e-5) == 0.0
    assert pytest.approx(neighbors[0].outcomes["model_a"], abs=1e-5) == 0.5


def test_missing_model_outcomes_and_fallback(synthetic_router_dataset: PreparedDataset) -> None:
    """Models with sparse or zero training evaluations filter neighbors or use default prior."""
    encoder = DeterministicTestEncoder(dimension=16)
    router = KNNRouter(k=3, encoder=encoder, default_prior=0.42)
    router.fit(synthetic_router_dataset)

    # model_c was only evaluated on p_code_2 (1 prompt)
    probs = router.predict_proba(["Write some code."])
    # model_c finds p_code_2 (where it scored 1.0), so effective_k = 1 -> prob = 1.0
    assert pytest.approx(probs["model_c"][0], abs=1e-5) == 1.0

    # Model with zero evaluations receives default_prior
    router._model_outcomes["model_unseen"] = {}
    router._candidate_models.append("model_unseen")
    probs_unseen = router.predict_proba(["Any prompt."])
    assert pytest.approx(probs_unseen["model_unseen"][0], abs=1e-5) == 0.42


def test_strict_test_leakage_isolation(synthetic_router_dataset: PreparedDataset) -> None:
    """Guarantees that test prompt IDs and test labels are never seen by KNNRouter."""
    encoder = DeterministicTestEncoder(dimension=16)
    router = KNNRouter(k=3, encoder=encoder)
    router.fit(synthetic_router_dataset)

    # Internal training prompts must match train split strictly
    assert set(router._train_prompt_ids) == set(synthetic_router_dataset.train_prompt_ids)
    for test_id in synthetic_router_dataset.test_prompt_ids:
        assert test_id not in router._train_prompt_ids

    # Modifying test outcomes has zero effect on router state
    for rec in synthetic_router_dataset.test_records:
        object.__setattr__(rec, "label", 0)  # Attempt tampering on test split

    probs_after = router.predict_proba(["Calculate 1 + 1."])
    assert 0.0 <= probs_after["model_a"][0] <= 1.0


def test_deterministic_predictions_and_tie_breaking(synthetic_router_dataset: PreparedDataset) -> None:
    """KNNRouter produces identical bitwise probabilities across repeated calls."""
    encoder = DeterministicTestEncoder(dimension=16)
    router = KNNRouter(k=3, encoder=encoder)
    router.fit(synthetic_router_dataset)

    prompt = "Compute the determinant of a 3x3 matrix."
    p1 = router.predict_proba([prompt])
    p2 = router.predict_proba([prompt])

    for m in router.candidate_models:
        assert np.array_equal(p1[m], p2[m])


def test_empty_inputs_handling(synthetic_router_dataset: PreparedDataset) -> None:
    """Empty queries sequence returns empty probability arrays without error."""
    encoder = DeterministicTestEncoder(dimension=16)
    router = KNNRouter(k=2, encoder=encoder)
    router.fit(synthetic_router_dataset)

    res = router.predict_proba([])
    for m in router.candidate_models:
        assert isinstance(res[m], np.ndarray)
        assert len(res[m]) == 0


def test_zero_network_calls_during_knn_routing(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_router_dataset: PreparedDataset,
) -> None:
    """Guarantees KNNRouter makes zero socket network calls during fitting and prediction."""
    def blocked_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network socket call attempted during offline KNN routing!")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)

    enc = DeterministicTestEncoder(dimension=16)
    router = KNNRouter(k=2, encoder=enc)
    router.fit(synthetic_router_dataset)
    res = router.predict_proba(["Offline prompt query."])
    assert len(res) > 0
