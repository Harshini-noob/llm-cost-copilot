"""
Unit tests for M2 baseline evaluation strategies (eval/baselines.py).

All tests use synthetic, deterministic data fixtures to ensure no provider
API calls are triggered and matrix.parquet is never modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval.baselines import (
    STATUS_INSUFFICIENT,
    STATUS_SUFFICIENT,
    STRATEGY_DEPLOYABLE,
    STRATEGY_POST_HOC,
    STRATEGY_THEORETICAL,
    aggregate_prompt_model_runs,
    create_stratified_split,
    evaluate_always_best,
    evaluate_always_cheapest,
    evaluate_oracle,
    evaluate_random_matched_cost,
    evaluate_static_task_table,
    plot_cost_vs_quality,
    run_baseline_comparison,
)


def _make_mock_matrix() -> pd.DataFrame:
    """Create synthetic benchmark run records across multiple models and task categories."""
    records = []
    models = ["model_cheap", "model_balanced", "model_premium"]
    costs = {"model_cheap": 0.00001, "model_balanced": 0.00003, "model_premium": 0.00008}
    latencies = {"model_cheap": 300.0, "model_balanced": 600.0, "model_premium": 1200.0}

    # 12 prompts across 6 task categories
    tasks = ["math", "code", "knowledge", "extraction", "summarization", "chit_chat"]
    for task in tasks:
        for p_idx in [1, 2]:
            prompt_id = f"{task}_{p_idx:03d}"
            for m in models:
                # 2 runs per prompt-model pair
                for r in [1, 2]:
                    # Make model_cheap good at chit_chat/summarization, model_premium good at math/code
                    if task in ["math", "code"]:
                        score = 1.0 if m in ["model_balanced", "model_premium"] else 0.0
                    elif task in ["knowledge", "extraction"]:
                        score = 1.0 if m in ["model_balanced", "model_cheap"] else 0.0
                    else:
                        score = 1.0  # all good at simple chat/summary

                    records.append({
                        "prompt_id": prompt_id,
                        "task_type": task,
                        "model_id": m,
                        "provider": "mock_provider",
                        "run": r,
                        "execution_success": True,
                        "cost": costs[m],
                        "latency_ms": latencies[m],
                        "score": float(score),
                        "correct": bool(score >= 1.0),
                        "evaluation_status": "scored",
                        "benchmark_config_id": "mock-config-123",
                    })

    return pd.DataFrame(records)


def test_stratified_train_test_split_prevents_leakage():
    """Verify train/test split has zero overlap and is stratified across all 6 categories."""
    mock_df = _make_mock_matrix()
    prompts = mock_df[["prompt_id", "task_type"]].drop_duplicates().to_dict(orient="records")

    split_info = create_stratified_split(prompts, split_ratio=0.5, seed=42)

    train_ids = set(split_info["train_prompt_ids"])
    test_ids = set(split_info["test_prompt_ids"])

    # Zero leakage
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(train_ids) == 6
    assert len(test_ids) == 6
    assert split_info["sufficiency_status"] == STATUS_SUFFICIENT
    assert len(split_info["test_categories"]) == 6


def test_stratified_train_test_split_flags_insufficient_data():
    """Verify that sparse datasets (like the 2-prompt real matrix) are flagged as INSUFFICIENT_DATA."""
    prompts = [
        {"prompt_id": "math_001", "task_type": "math"},
        {"prompt_id": "math_002", "task_type": "math"},
    ]
    split_info = create_stratified_split(prompts, split_ratio=0.5, seed=42)

    assert split_info["sufficiency_status"] == STATUS_INSUFFICIENT
    assert len(split_info["train_prompt_ids"]) == 1
    assert len(split_info["test_prompt_ids"]) == 1
    assert "INSUFFICIENT_DATA" in split_info["notes"]


def test_aggregate_prompt_model_runs_preserves_failures_and_unscored():
    """Verify execution failures and unscored evaluations are preserved and never scored as 0.0."""
    raw_rows = [
        # Model 1: run 1 and run 2 scored
        {
            "prompt_id": "p1", "task_type": "math", "model_id": "m1", "run": 1,
            "execution_success": True, "cost": 0.01, "latency_ms": 100,
            "evaluation_status": "scored", "score": 1.0, "correct": True,
        },
        {
            "prompt_id": "p1", "task_type": "math", "model_id": "m1", "run": 2,
            "execution_success": True, "cost": 0.01, "latency_ms": 100,
            "evaluation_status": "scored", "score": 1.0, "correct": True,
        },
        # Model 2: execution failure in run 2, cost still incurred in run 1
        {
            "prompt_id": "p1", "task_type": "math", "model_id": "m2", "run": 1,
            "execution_success": True, "cost": 0.02, "latency_ms": 200,
            "evaluation_status": None, "score": np.nan, "correct": None,
        },
        {
            "prompt_id": "p1", "task_type": "math", "model_id": "m2", "run": 2,
            "execution_success": False, "cost": 0.005, "latency_ms": 50,
            "evaluation_status": None, "score": np.nan, "correct": None,
        },
        # Model 3: executed successfully, but unscored (never score 0.0)
        {
            "prompt_id": "p1", "task_type": "math", "model_id": "m3", "run": 1,
            "execution_success": True, "cost": 0.03, "latency_ms": 300,
            "evaluation_status": None, "score": np.nan, "correct": None,
        },
    ]

    agg = aggregate_prompt_model_runs(pd.DataFrame(raw_rows))
    assert len(agg) == 3

    m1_row = agg[agg["model_id"] == "m1"].iloc[0]
    assert bool(m1_row["execution_success"]) is True
    assert m1_row["score"] == 1.0
    assert m1_row["evaluation_status"] == "scored"

    m2_row = agg[agg["model_id"] == "m2"].iloc[0]
    assert bool(m2_row["execution_success"]) is False
    assert m2_row["evaluation_status"] == "execution_failed"
    assert pd.isna(m2_row["score"])
    # Incurred costs are preserved
    assert pytest.approx(m2_row["total_incurred_cost"], 1e-6) == 0.025

    m3_row = agg[agg["model_id"] == "m3"].iloc[0]
    assert bool(m3_row["execution_success"]) is True
    assert m3_row["evaluation_status"] == "unscored"
    assert pd.isna(m3_row["score"])  # NOT 0.0


def test_always_cheapest_selects_lowest_train_cost():
    """Verify Baseline 1 picks the model with lowest average cost on training data."""
    train_data = pd.DataFrame([
        {"prompt_id": "p1", "task_type": "math", "model_id": "m_cheap", "cost": 0.00001, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 0.5, "correct": False},
        {"prompt_id": "p1", "task_type": "math", "model_id": "m_expensive", "cost": 0.00009, "latency_ms": 500, "execution_success": True, "evaluation_status": "scored", "score": 1.0, "correct": True},
    ])
    test_data = pd.DataFrame([
        {"prompt_id": "p2", "task_type": "math", "model_id": "m_cheap", "cost": 0.000012, "latency_ms": 110, "execution_success": True, "evaluation_status": "scored", "score": 0.6, "correct": False},
        {"prompt_id": "p2", "task_type": "math", "model_id": "m_expensive", "cost": 0.000088, "latency_ms": 490, "execution_success": True, "evaluation_status": "scored", "score": 1.0, "correct": True},
    ])

    result = evaluate_always_cheapest(train_data, test_data, ["p2"])

    assert result.strategy_type == STRATEGY_DEPLOYABLE
    assert result.model_selection_distribution == {"m_cheap": 1.0}
    assert pytest.approx(result.mean_cost, 1e-6) == 0.000012
    assert result.mean_quality == 0.6


def test_always_best_identifies_highest_test_quality_and_flags_post_hoc():
    """Verify Baseline 2 selects highest quality on test set and labels as post-hoc reference."""
    test_data = pd.DataFrame([
        {"prompt_id": "p2", "task_type": "math", "model_id": "m_low", "cost": 0.00001, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 0.4, "correct": False},
        {"prompt_id": "p2", "task_type": "math", "model_id": "m_high", "cost": 0.00005, "latency_ms": 300, "execution_success": True, "evaluation_status": "scored", "score": 0.95, "correct": True},
    ])

    result = evaluate_always_best(test_data, ["p2"])

    assert result.strategy_type == STRATEGY_POST_HOC
    assert result.model_selection_distribution == {"m_high": 1.0}
    assert result.mean_quality == 0.95
    assert "Post-hoc test-set reference ceiling" in result.notes


def test_random_matched_cost_achieves_target_and_captures_variability():
    """Verify Baseline 3 matches expected cost, tracks delta, and measures sampling variability."""
    train_data = pd.DataFrame([
        {"prompt_id": "p1", "task_type": "math", "model_id": "m_low", "cost": 0.00001, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 0.5, "correct": False},
        {"prompt_id": "p1", "task_type": "math", "model_id": "m_high", "cost": 0.00005, "latency_ms": 300, "execution_success": True, "evaluation_status": "scored", "score": 1.0, "correct": True},
    ])
    # 10 test prompts to observe variance
    test_rows = []
    for i in range(10):
        test_rows.append({"prompt_id": f"test_{i}", "task_type": "math", "model_id": "m_low", "cost": 0.00001, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 0.5, "correct": False})
        test_rows.append({"prompt_id": f"test_{i}", "task_type": "math", "model_id": "m_high", "cost": 0.00005, "latency_ms": 300, "execution_success": True, "evaluation_status": "scored", "score": 1.0, "correct": True})
    test_data = pd.DataFrame(test_rows)
    test_ids = [f"test_{i}" for i in range(10)]

    target = 0.00003
    result = evaluate_random_matched_cost(train_data, test_data, test_ids, target_cost=target, n_repetitions=30, seed=123)

    assert result.strategy_type == STRATEGY_DEPLOYABLE
    assert pytest.approx(result.target_cost, 1e-6) == target
    assert pytest.approx(result.expected_cost, 1e-6) == target
    assert result.cost_std is not None and result.cost_std >= 0.0
    assert result.cost_delta_from_target is not None
    assert result.relative_cost_deviation is not None
    # Verify delta and relative deviation are tracked accurately
    assert pytest.approx(result.cost_delta_from_target, 1e-6) == abs(result.mean_cost - target)
    assert pytest.approx(result.relative_cost_deviation, 1e-6) == abs(result.mean_cost - target) / target


def test_static_task_type_table_optimizes_quality_over_cost_with_zero_classifier_cost():
    """Verify Baseline 4 selects best Quality/Cost model per task and defaults classifier cost to 0."""
    train_data = pd.DataFrame([
        # Math: m1 has Quality 1.0, cost 0.01 (Value = 100); m2 has Quality 1.0, cost 0.02 (Value = 50) -> Pick m1
        {"prompt_id": "m_1", "task_type": "math", "model_id": "m1", "cost": 0.01, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 1.0, "correct": True},
        {"prompt_id": "m_1", "task_type": "math", "model_id": "m2", "cost": 0.02, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 1.0, "correct": True},
        # Code: m1 has Quality 0.2, cost 0.01 (Value = 20); m2 has Quality 1.0, cost 0.02 (Value = 50) -> Pick m2
        {"prompt_id": "c_1", "task_type": "code", "model_id": "m1", "cost": 0.01, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 0.2, "correct": False},
        {"prompt_id": "c_1", "task_type": "code", "model_id": "m2", "cost": 0.02, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 1.0, "correct": True},
    ])
    test_data = pd.DataFrame([
        {"prompt_id": "m_2", "task_type": "math", "model_id": "m1", "cost": 0.01, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 1.0, "correct": True},
        {"prompt_id": "m_2", "task_type": "math", "model_id": "m2", "cost": 0.02, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 1.0, "correct": True},
        {"prompt_id": "c_2", "task_type": "code", "model_id": "m1", "cost": 0.01, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 0.2, "correct": False},
        {"prompt_id": "c_2", "task_type": "code", "model_id": "m2", "cost": 0.02, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 1.0, "correct": True},
    ])

    result = evaluate_static_task_table(train_data, test_data, ["m_2", "c_2"], objective="quality_over_cost", classifier_cost=0.0)

    assert result.strategy_type == STRATEGY_DEPLOYABLE
    # m1 was routed for math (test cost 0.01), m2 was routed for code (test cost 0.02)
    assert result.model_selection_distribution == {"m1": 0.5, "m2": 0.5}
    assert pytest.approx(result.mean_cost, 1e-6) == 0.015
    assert result.mean_quality == 1.0
    assert "classifier inference cost set to $0.000000" in result.notes


def test_static_task_type_table_includes_configured_classifier_cost():
    """Verify configurable classifier cost is added to each prompt's routing cost."""
    train_data = pd.DataFrame([
        {"prompt_id": "m_1", "task_type": "math", "model_id": "m1", "cost": 0.01, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 1.0, "correct": True},
    ])
    test_data = pd.DataFrame([
        {"prompt_id": "m_2", "task_type": "math", "model_id": "m1", "cost": 0.01, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 1.0, "correct": True},
    ])

    classifier_cost = 0.0005
    result = evaluate_static_task_table(train_data, test_data, ["m_2"], classifier_cost=classifier_cost)

    assert pytest.approx(result.mean_cost, 1e-6) == 0.01 + classifier_cost
    assert f"Classifier inference cost included at ${classifier_cost:.6f}" in result.notes


def test_static_task_type_table_unseen_task_fallback():
    """Verify graceful fallback when test prompt task category was unobserved in training data."""
    train_data = pd.DataFrame([
        {"prompt_id": "m_1", "task_type": "math", "model_id": "m1", "cost": 0.01, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 1.0, "correct": True},
    ])
    test_data = pd.DataFrame([
        # Novel task category unobserved in train
        {"prompt_id": "u_1", "task_type": "novel_category", "model_id": "m1", "cost": 0.01, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 1.0, "correct": True},
    ])

    result = evaluate_static_task_table(train_data, test_data, ["u_1"])
    assert "Unseen task fallback count: 1/1" in result.notes
    assert result.selections[0].selected_model == "m1"


def test_oracle_selects_cheapest_correct_model():
    """Verify Oracle selects the cheapest model that scored correct on each test prompt."""
    test_data = pd.DataFrame([
        # Model 1: incorrect, cheap
        {"prompt_id": "p1", "task_type": "math", "model_id": "m_cheap", "cost": 0.00001, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 0.0, "correct": False},
        # Model 2: correct, balanced
        {"prompt_id": "p1", "task_type": "math", "model_id": "m_balanced", "cost": 0.00003, "latency_ms": 300, "execution_success": True, "evaluation_status": "scored", "score": 1.0, "correct": True},
        # Model 3: correct, expensive
        {"prompt_id": "p1", "task_type": "math", "model_id": "m_premium", "cost": 0.00008, "latency_ms": 800, "execution_success": True, "evaluation_status": "scored", "score": 1.0, "correct": True},
    ])

    result = evaluate_oracle(test_data, ["p1"])

    assert result.strategy_type == STRATEGY_THEORETICAL
    assert result.model_selection_distribution == {"m_balanced": 1.0}
    assert pytest.approx(result.mean_cost, 1e-6) == 0.00003
    assert result.mean_quality == 1.0
    assert "Unanswerable prompts: 0/1" in result.notes


def test_oracle_handles_unanswerable_prompts():
    """Verify Oracle falls back to cheapest candidate when no model answers correctly."""
    test_data = pd.DataFrame([
        # All models incorrect
        {"prompt_id": "p_hard", "task_type": "math", "model_id": "m_cheap", "cost": 0.00001, "latency_ms": 100, "execution_success": True, "evaluation_status": "scored", "score": 0.0, "correct": False},
        {"prompt_id": "p_hard", "task_type": "math", "model_id": "m_premium", "cost": 0.00008, "latency_ms": 800, "execution_success": True, "evaluation_status": "scored", "score": 0.0, "correct": False},
    ])

    result = evaluate_oracle(test_data, ["p_hard"])

    assert result.model_selection_distribution == {"m_cheap": 1.0}
    assert result.mean_quality == 0.0
    assert "Unanswerable prompts: 1/1" in result.notes


def test_plot_generation(tmp_path: Path):
    """Verify cost_vs_quality plot generation with warning annotation for insufficient data."""
    mock_df = _make_mock_matrix()
    train_agg = aggregate_prompt_model_runs(mock_df[mock_df["prompt_id"] == "math_001"])
    test_agg = aggregate_prompt_model_runs(mock_df[mock_df["prompt_id"] == "math_002"])

    res1 = evaluate_always_cheapest(train_agg, test_agg, ["math_002"])
    res2 = evaluate_always_best(test_agg, ["math_002"])
    res3 = evaluate_random_matched_cost(train_agg, test_agg, ["math_002"], target_cost=0.00003, n_repetitions=5)

    plot_file = tmp_path / "test_plot.png"
    out = plot_cost_vs_quality([res1, res2, res3], plot_file, sufficiency_status=STATUS_INSUFFICIENT)

    assert out.exists()
    assert out.stat().st_size > 1000
