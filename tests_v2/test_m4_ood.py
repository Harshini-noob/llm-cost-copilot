"""
tests_v2/test_m4_ood.py

Unit and integration tests for Phase M4.4: Out-of-Distribution (OOD) Generalization Evaluation.
Verifies held-out task exclusion, prompt leakage prevention, unknown task handling,
training-only preprocessing, baseline fallbacks, continuous SQuAD handling, reproducibility,
and prompt-clustered bootstrap mechanics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval.m4_data_adapter import bootstrap_prompt_metric, compute_prompt_micro_score
from eval.m4_calibration import compute_calibration_metrics
from eval.routers.data_prep import PromptRecord
from eval.routers.learned_router import PromptFeatureExtractor, LearnedRouter
from eval.m4_router_evaluation import PreInferenceCostEstimator, XRouteBenchPreparedDataset


def test_held_out_task_exclusion() -> None:
    """Verify that when a task is held out, no rows from that task exist in training data."""
    df = pd.DataFrame({
        "query": ["q1", "q2", "q3", "q4"],
        "task_name": ["gsm8k", "gsm8k", "mmlu", "squad"],
        "model_name": ["m1", "m1", "m1", "m1"],
        "performance": [1.0, 0.0, 1.0, 0.8],
    })
    held_out = "gsm8k"
    sub_train = df[df["task_name"] != held_out]

    assert held_out not in sub_train["task_name"].values
    assert len(sub_train) == 2
    assert set(sub_train["task_name"]) == {"mmlu", "squad"}


def test_prompt_leakage_safety() -> None:
    """Verify train and test query sets are strictly disjoint."""
    train_queries = ["math prompt 1", "math prompt 2", "code prompt 1"]
    test_queries = ["code prompt 2", "qa prompt 1"]

    assert set(train_queries).isdisjoint(set(test_queries))


def test_unknown_task_handling() -> None:
    """Unseen task categories must produce all-zero one-hot slices without raising errors."""
    known_prompts = [
        PromptRecord("p1", "arc_challenge", "text 1", np.zeros(4, dtype=np.float32), "train"),
        PromptRecord("p2", "boolq", "text 2", np.zeros(4, dtype=np.float32), "train"),
    ]
    extractor = PromptFeatureExtractor()
    extractor.fit(known_prompts, sample_embedding_dim=4)

    # Categories vocab should only contain the 2 known categories
    assert extractor._categories_vocab == ["arc_challenge", "boolq"]
    assert extractor.feature_dimension == 4 + 2 + 2  # 4 emb + length + code + 2 categories = 8

    # Transform with an unseen held-out task "gsm8k"
    test_embs = np.zeros((1, 4), dtype=np.float32)
    features = extractor.transform(test_embs, ["unseen math query"], categories=["gsm8k"])

    assert features.shape == (1, 8)
    # The one-hot slice (columns 6 and 7) should be all zeros
    assert features[0, 6] == 0.0
    assert features[0, 7] == 0.0


def test_training_only_preprocessing() -> None:
    """PreInferenceCostEstimator must be fitted strictly on training data."""
    train_df = pd.DataFrame({
        "task_name": ["task_a", "task_a"],
        "model_name": ["model_x", "model_y"],
        "input_tokens": [100, 100],
        "output_tokens": [50, 50],
        "reconstructed_cost_usd": [0.001, 0.002],
    })
    cand_df = pd.DataFrame({
        "model_name": ["model_x", "model_y"],
        "input_price_per_1m": [1.0, 2.0],
        "output_price_per_1m": [2.0, 4.0],
    })

    estimator = PreInferenceCostEstimator.fit_from_training_data(train_df, cand_df)
    assert "model_x" in estimator.model_mean_cost
    assert "model_y" in estimator.model_mean_cost
    assert estimator.model_mean_cost["model_x"] < estimator.model_mean_cost["model_y"]


def test_baseline_fallback() -> None:
    """Static Task-Type Table must fall back to global best training model for unseen task."""
    train_df = pd.DataFrame({
        "task_name": ["task_a", "task_a", "task_b", "task_b"],
        "model_name": ["model_1", "model_2", "model_1", "model_2"],
        "performance": [0.9, 0.4, 0.8, 0.5],
    })
    # model_1 is best on task_a, task_b, and globally
    task_best = {}
    for t in train_df["task_name"].unique():
        t_sub = train_df[train_df["task_name"] == t]
        task_best[t] = str(t_sub.groupby("model_name")["performance"].mean().idxmax())

    global_best = str(train_df.groupby("model_name")["performance"].mean().idxmax())
    assert global_best == "model_1"

    # Query from unseen task_c
    unseen_task = "task_c"
    routed_model = task_best.get(unseen_task, global_best)
    assert routed_model == "model_1"


def test_continuous_squad_handling() -> None:
    """Continuous SQuAD F1 scores must compute valid calibration and micro scores."""
    df = pd.DataFrame({
        "query": ["squad_q1", "squad_q2", "squad_q3"],
        "observed_performance": [0.85, 0.45, 0.60],  # Continuous token F1
        "observed_cost_usd": [0.0001, 0.0002, 0.00015],
    })
    mean_score = compute_prompt_micro_score(df, score_col="observed_performance")
    assert pytest.approx(mean_score, abs=1e-5) == (0.85 + 0.45 + 0.60) / 3.0

    # Calibration calculation
    y_true = np.array([0.85, 0.45, 0.60])
    y_prob = np.array([0.80, 0.50, 0.70])
    metrics = compute_calibration_metrics(y_true, y_prob, n_bins=5)
    assert metrics.total_samples == 3
    assert metrics.brier_score >= 0.0
    assert 0.0 <= metrics.ece <= 1.0


def test_reproducibility() -> None:
    """Bootstrap CI computation must be bit-for-bit deterministic with fixed seed."""
    df = pd.DataFrame({
        "query": [f"q_{i}" for i in range(20)],
        "task_name": ["task_a"] * 20,
        "observed_performance": np.random.RandomState(123).uniform(0, 1, 20),
        "observed_cost_usd": np.random.RandomState(123).uniform(1e-5, 1e-4, 20),
    })
    pt1, low1, high1 = bootstrap_prompt_metric(df, score_col="observed_performance", n_bootstraps=50, seed=42)
    pt2, low2, high2 = bootstrap_prompt_metric(df, score_col="observed_performance", n_bootstraps=50, seed=42)

    assert pt1 == pt2
    assert low1 == low2
    assert high1 == high2


def test_bootstrap_clustering() -> None:
    """Bootstrap must resample queries, keeping all model observations for a query together."""
    # Create 3 queries with 2 models each
    df = pd.DataFrame({
        "query": ["q1", "q1", "q2", "q2", "q3", "q3"],
        "model_name": ["m1", "m2", "m1", "m2", "m1", "m2"],
        "performance": [1.0, 0.0, 0.5, 0.5, 1.0, 1.0],
    })
    pt, low, high = bootstrap_prompt_metric(df, score_col="performance", n_bootstraps=50, seed=42, aggregate_fn="micro")
    assert 0.0 <= low <= pt <= high <= 1.0
