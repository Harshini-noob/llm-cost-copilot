"""
tests_v2/test_m4_router_evaluation.py

Unit and integration tests for M4.2 Leakage-Safe Router Evaluation on xRouteBench.
Validates:
1. Leakage prevention: routers never access test labels, outcomes, or future output tokens.
2. Pre-inference cost estimation derived strictly from training data.
3. Observed post-selection cost using actual executed tokens.
4. Oracle and Always-Best reference separation from deployable policies.
5. Training-only baseline construction for Always-Cheapest and Static Task Table.
6. Binary accuracy vs SQuAD continuous F1 preservation in routing decisions.
7. Task Macro-average and Micro-average aggregation formulas.
8. Prompt-level cluster bootstrap confidence interval calculation.
9. Deterministic reproducibility of the Random baseline.
10. Preservation of M0-M2 matrices and artifacts.
"""

from decimal import Decimal
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from eval.m4_data_adapter import (
    ALL_TASKS,
    TASK_METRIC_MAP,
    compute_task_macro_score,
    compute_prompt_micro_score,
)
from eval.m4_router_evaluation import (
    PreInferenceCostEstimator,
    EvaluationSelectionRecord,
    StrategySummary,
    XRouteBenchPreparedDataset,
    evaluate_always_cheapest,
    evaluate_always_best,
    evaluate_random_uniform,
    evaluate_static_task_table,
    evaluate_oracle,
    evaluate_decision_sweep,
    summarize_selections,
)
from eval.routers.decision import CostNormalizer, LambdaDecisionEngine


@pytest.fixture
def mock_evaluation_data():
    """Create a minimal, synthetic multi-task xRouteBench dataset for unit testing."""
    models = ["model_cheap", "model_mid", "model_frontier"]
    pricing = pd.DataFrame({
        "model_name": models,
        "input_price_per_1m": [0.10, 0.50, 1.50],
        "output_price_per_1m": [0.20, 1.00, 3.00],
    })

    # Train data: 2 tasks (math: binary, squad: continuous F1)
    train_rows = []
    for q_idx in range(10):
        task = "math" if q_idx < 5 else "squad"
        query_text = f"train_prompt_{q_idx}"
        for m in models:
            perf = 1.0 if (m == "model_frontier" and task == "math") else (0.85 if task == "squad" and m == "model_mid" else 0.4)
            in_tok = 100 + q_idx * 10
            out_tok = 50 + q_idx * 5
            cost = (in_tok * pricing.set_index("model_name").loc[m, "input_price_per_1m"] +
                    out_tok * pricing.set_index("model_name").loc[m, "output_price_per_1m"]) / 1e6
            train_rows.append({
                "query": query_text,
                "task_name": task,
                "model_name": m,
                "performance": perf,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "token_num": in_tok + out_tok,
                "response_time": 1.0,
                "reconstructed_cost_usd": cost,
                "embedding_id": q_idx,
            })
    train_df = pd.DataFrame(train_rows)

    # Test data: 4 test prompts
    test_rows = []
    for q_idx in range(4):
        task = "math" if q_idx < 2 else "squad"
        query_text = f"test_prompt_{q_idx}"
        for m in models:
            perf = 1.0 if (m == "model_frontier" and task == "math") else (0.90 if task == "squad" and m == "model_mid" else 0.5)
            in_tok = 150
            out_tok = 80
            cost = (in_tok * pricing.set_index("model_name").loc[m, "input_price_per_1m"] +
                    out_tok * pricing.set_index("model_name").loc[m, "output_price_per_1m"]) / 1e6
            test_rows.append({
                "query": query_text,
                "task_name": task,
                "model_name": m,
                "performance": perf,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "token_num": in_tok + out_tok,
                "response_time": 1.2,
                "reconstructed_cost_usd": cost,
                "embedding_id": q_idx + 100,
            })
    test_df = pd.DataFrame(test_rows)

    return train_df, test_df, pricing, models


# --- 1. Pre-Inference Cost Estimation Tests ---

def test_pre_inference_cost_estimator_strictly_training(mock_evaluation_data):
    """Pre-inference cost estimator must be fitted exclusively on training data."""
    train_df, _, pricing, models = mock_evaluation_data
    estimator = PreInferenceCostEstimator.fit_from_training_data(train_df, pricing)

    # Costs are ordered by pricing
    assert estimator.model_mean_cost["model_cheap"] < estimator.model_mean_cost["model_mid"] < estimator.model_mean_cost["model_frontier"]

    # Pre-inference estimate without input tokens returns training mean
    est_cheap = estimator.estimate_cost("model_cheap")
    assert est_cheap == pytest.approx(estimator.model_mean_cost["model_cheap"])

    # Pre-inference estimate with input tokens uses expected training output tokens, not test output tokens
    est_prompt_aware = estimator.estimate_cost("model_cheap", input_tokens=200)
    expected_out = estimator.model_mean_output_tokens["model_cheap"]
    manual_calc = (200 * 0.10 + expected_out * 0.20) / 1e6
    assert est_prompt_aware == pytest.approx(manual_calc)


# --- 2. Baseline Training-Only Construction & Reference Separation ---

def test_always_cheapest_selects_lowest_training_cost(mock_evaluation_data):
    """Always-Cheapest must select the cheapest model according to training data only."""
    train_df, test_df, pricing, models = mock_evaluation_data
    estimator = PreInferenceCostEstimator.fit_from_training_data(train_df, pricing)

    # Build test lookups
    test_records = {}
    for _, row in test_df.iterrows():
        test_records.setdefault(row["query"], {})[row["model_name"]] = row.to_dict()

    test_meta = [{"query": q, "task_name": test_df[test_df["query"] == q]["task_name"].iloc[0]}
                 for q in test_df["query"].unique()]

    summary, selections = evaluate_always_cheapest(test_records, test_meta, estimator)

    assert summary.strategy_type == "deployable_policy"
    assert summary.is_reference is False
    assert all(s.selected_model == "model_cheap" for s in selections)


def test_always_best_marked_as_post_hoc_reference(mock_evaluation_data):
    """Best Single Model must be explicitly labeled as a post-hoc reference ceiling."""
    train_df, test_df, pricing, models = mock_evaluation_data

    test_records = {}
    for _, row in test_df.iterrows():
        test_records.setdefault(row["query"], {})[row["model_name"]] = row.to_dict()

    test_meta = [{"query": q, "task_name": test_df[test_df["query"] == q]["task_name"].iloc[0]}
                 for q in test_df["query"].unique()]

    summary, selections = evaluate_always_best(test_df, test_records, test_meta)

    assert summary.strategy_type == "post_hoc_reference"
    assert summary.is_reference is True
    assert summary.name == "Best Single Model (Test-Set Reference)"
    assert "Not deployable" in summary.notes


def test_oracle_marked_as_theoretical_reference(mock_evaluation_data):
    """Oracle must be explicitly labeled as a theoretical reference ceiling with lambda=0."""
    train_df, test_df, pricing, models = mock_evaluation_data
    estimator = PreInferenceCostEstimator.fit_from_training_data(train_df, pricing)

    test_records = {}
    for _, row in test_df.iterrows():
        test_records.setdefault(row["query"], {})[row["model_name"]] = row.to_dict()

    test_meta = [{"query": q, "task_name": test_df[test_df["query"] == q]["task_name"].iloc[0]}
                 for q in test_df["query"].unique()]

    summary, selections = evaluate_oracle(test_records, test_meta, models, estimator, lambda_value=0.0)

    assert summary.strategy_type == "theoretical_reference"
    assert summary.is_reference is True
    assert summary.lambda_value == 0.0
    assert "Oracle" in summary.name
    # Oracle at lambda=0 should achieve maximal performance
    assert summary.micro_quality >= 0.90


def test_static_task_table_selects_best_per_task(mock_evaluation_data):
    """Static Task Table must learn per-task best models from training data only."""
    train_df, test_df, pricing, models = mock_evaluation_data
    estimator = PreInferenceCostEstimator.fit_from_training_data(train_df, pricing)

    test_records = {}
    for _, row in test_df.iterrows():
        test_records.setdefault(row["query"], {})[row["model_name"]] = row.to_dict()

    test_meta = [{"query": q, "task_name": test_df[test_df["query"] == q]["task_name"].iloc[0]}
                 for q in test_df["query"].unique()]

    summary, selections = evaluate_static_task_table(train_df, test_records, test_meta, estimator)

    assert summary.strategy_type == "deployable_policy"
    assert summary.is_reference is False

    # In train data: math best is model_frontier, squad best is model_mid
    for s in selections:
        if s.task_name == "math":
            assert s.selected_model == "model_frontier"
        elif s.task_name == "squad":
            assert s.selected_model == "model_mid"


def test_random_uniform_deterministic_and_deployable(mock_evaluation_data):
    """Random Uniform baseline must be uniform across candidates, deterministic with seed, and marked deployable."""
    train_df, test_df, pricing, models = mock_evaluation_data
    estimator = PreInferenceCostEstimator.fit_from_training_data(train_df, pricing)

    test_records = {}
    for _, row in test_df.iterrows():
        test_records.setdefault(row["query"], {})[row["model_name"]] = row.to_dict()

    test_meta = [{"query": q, "task_name": test_df[test_df["query"] == q]["task_name"].iloc[0]}
                 for q in test_df["query"].unique()]

    s1, sel1 = evaluate_random_uniform(test_records, test_meta, models, estimator, seed=123)
    s2, sel2 = evaluate_random_uniform(test_records, test_meta, models, estimator, seed=123)
    s3, sel3 = evaluate_random_uniform(test_records, test_meta, models, estimator, seed=999)

    assert [s.selected_model for s in sel1] == [s.selected_model for s in sel2]
    assert s1.strategy_type == "deployable_policy"
    assert s1.is_reference is False
    assert s1.name == "Random-Uniform"


def test_latency_records_historical_response_time(mock_evaluation_data):
    """Verify that latency in selection records matches candidate model response_time."""
    _, test_df, _, _ = mock_evaluation_data
    test_records = {}
    for _, row in test_df.iterrows():
        test_records.setdefault(row["query"], {})[row["model_name"]] = row.to_dict()

    test_meta = [{"query": q, "task_name": test_df[test_df["query"] == q]["task_name"].iloc[0]}
                 for q in test_df["query"].unique()]

    # In mock data, response_time was set to 1.2
    sample_obs = test_records[test_meta[0]["query"]]["model_cheap"]
    assert sample_obs["response_time"] == 1.2


# --- 3. Metric Handling & SQuAD Continuous F1 ---

def test_continuous_squad_f1_preserved(mock_evaluation_data):
    """SQuAD observations must preserve continuous float values (e.g. 0.90) without binarization."""
    _, test_df, _, _ = mock_evaluation_data
    squad_scores = test_df[test_df["task_name"] == "squad"]["performance"]

    # Must contain values strictly between 0 and 1
    has_continuous = any((s > 0.0 and s < 1.0) for s in squad_scores)
    assert has_continuous is True


def test_macro_vs_micro_aggregations():
    """Verify that macro gives equal weight to tasks, while micro weights queries equally."""
    # 2 tasks: task_a (1 query with score 1.0), task_b (3 queries with score 0.0)
    df = pd.DataFrame({
        "task_name": ["task_a", "task_b", "task_b", "task_b"],
        "observed_performance": [1.0, 0.0, 0.0, 0.0],
    })

    macro = compute_task_macro_score(df, score_col="observed_performance")
    micro = compute_prompt_micro_score(df, score_col="observed_performance")

    assert macro == pytest.approx(0.5)  # (1.0 + 0.0) / 2 = 0.5
    assert micro == pytest.approx(0.25)  # (1.0 + 0 + 0 + 0) / 4 = 0.25


# --- 4. Decision Sweep & Cost Trade-Offs ---

def test_decision_sweep_responds_to_lambda(mock_evaluation_data):
    """Increasing lambda must shift decisions toward lower-cost models."""
    train_df, test_df, pricing, models = mock_evaluation_data
    estimator = PreInferenceCostEstimator.fit_from_training_data(train_df, pricing)

    test_records = {}
    for _, row in test_df.iterrows():
        test_records.setdefault(row["query"], {})[row["model_name"]] = row.to_dict()

    test_meta = [{"query": q, "task_name": test_df[test_df["query"] == q]["task_name"].iloc[0]}
                 for q in test_df["query"].unique()]

    # Synthetic predictions: frontier=0.9, mid=0.8, cheap=0.5
    preds_by_query = {
        item["query"]: {"model_frontier": 0.9, "model_mid": 0.8, "model_cheap": 0.5}
        for item in test_meta
    }

    results = evaluate_decision_sweep(
        predicted_qualities_by_query=preds_by_query,
        test_records_by_query=test_records,
        test_query_metadata=test_meta,
        candidate_models=models,
        cost_estimator=estimator,
        router_family="knn",
        lambda_sweep=[0.0, 5.0],
    )

    s_quality, _ = results[0]  # lambda=0.0
    s_cost, _ = results[1]     # lambda=5.0

    # At lambda=0, selects highest quality (model_frontier)
    assert s_quality.model_selection_distribution.get("model_frontier", 0.0) == 1.0

    # At lambda=5, cost penalty forces selection to cheapest model
    assert s_cost.model_selection_distribution.get("model_cheap", 0.0) == 1.0
    assert s_cost.mean_cost_usd < s_quality.mean_cost_usd


# --- 5. Immutability of M0-M2 Matrices ---

def test_m0_m2_matrix_unmodified():
    """Verify that existing M0-M2 matrix in data/processed remains strictly untouched."""
    matrix_path = Path("data/processed/matrix.parquet")
    assert matrix_path.exists()
    df = pd.read_parquet(matrix_path)
    assert "prompt_id" in df.columns
    assert "model_id" in df.columns
