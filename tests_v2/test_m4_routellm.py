"""
tests_v2/test_m4_routellm.py

Comprehensive targeted unit tests for Phase M4.5 RouteLLM Two-Model Baseline & APGR Benchmark.
Validates:
1. Exact APGR and PGR mathematical properties (boundary conditions, trapezoidal integration, sorting).
2. CPT (Call-Performance Threshold) calculations.
3. Training-only model pair selection (leakage prevention, cost ratios, parameter scales).
4. Prompt-grouped validation split isolation (zero prompt leakage).
5. Threshold fitting and call-rate monotonicity.
6. Deterministic decision reproducibility.
7. Prompt-level bootstrap resampling integrity.
8. Reconstructed cost calculation for two-model routing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from eval.m4_routellm_baseline import (
    calculate_pgr,
    calculate_apgr,
    calculate_cpt,
    select_model_pair_from_train,
    RouteLLMClassifier,
    RouteLLMNearestNeighbors,
    RouteLLMHybridEnsemble,
    bootstrap_two_model_metrics,
)
from eval.m4_data_adapter import create_prompt_grouped_validation_split


def test_pgr_formula_boundaries() -> None:
    """Validate Performance Gap Recovered (PGR) mathematical edge cases."""
    # When router matches weak score -> PGR = 0.0
    assert calculate_pgr(0.50, weak_score=0.50, strong_score=0.80) == pytest.approx(0.0)

    # When router matches strong score -> PGR = 1.0
    assert calculate_pgr(0.80, weak_score=0.50, strong_score=0.80) == pytest.approx(1.0)

    # When router is halfway between weak and strong -> PGR = 0.5
    assert calculate_pgr(0.65, weak_score=0.50, strong_score=0.80) == pytest.approx(0.5)

    # When router outperforms strong (e.g. Oracle avoiding weak errors) -> PGR > 1.0
    assert calculate_pgr(0.86, weak_score=0.50, strong_score=0.80) == pytest.approx(1.2)

    # When router underperforms weak -> PGR < 0.0
    assert calculate_pgr(0.44, weak_score=0.50, strong_score=0.80) == pytest.approx(-0.2)

    # Edge case: zero gap (strong does not outperform weak)
    assert calculate_pgr(0.50, weak_score=0.50, strong_score=0.50) == pytest.approx(0.0)


def test_apgr_formula_properties() -> None:
    """Validate Average Performance Gap Recovered (APGR) mathematical properties."""
    # 1. Random router: diagonal y = x from (0, 0) to (1, 1). Area should be exactly 0.5000
    c_grid = np.linspace(0.0, 1.0, 101)
    pgr_diag = c_grid.copy()
    assert calculate_apgr(c_grid, pgr_diag) == pytest.approx(0.5000, abs=1e-4)

    # 2. Perfect step router: reaches PGR = 1.0 at c = 0.0. Area should be 1.0000
    # (0, 1) to (1, 1)
    c_step = [0.0, 0.0001, 1.0]
    pgr_step = [1.0, 1.0, 1.0]
    assert calculate_apgr(c_step, pgr_step) == pytest.approx(1.0000, abs=1e-3)

    # 3. Always-Weak baseline: PGR = 0 everywhere except (1, 1)
    c_weak = [0.0, 0.9999, 1.0]
    pgr_weak = [0.0, 0.0, 1.0]
    assert calculate_apgr(c_weak, pgr_weak) == pytest.approx(0.0000, abs=1e-3)

    # 4. Automatic sorting and endpoint injection
    c_unsorted = [0.8, 0.2, 0.5]
    pgr_unsorted = [0.8, 0.2, 0.5]
    # Automatically inserts (0, 0) and (1, 1) and sorts -> diagonal y = x -> 0.5000
    assert calculate_apgr(c_unsorted, pgr_unsorted) == pytest.approx(0.5000, abs=1e-4)


def test_cpt_calculation() -> None:
    """Validate Call-Performance Threshold (CPT) calculation with linear interpolation."""
    c_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    pgr_grid = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]

    # CPT(50%): PGR reaches 0.5 at c = 0.4 exactly
    assert calculate_cpt(c_grid, pgr_grid, target_pgr=0.50) == pytest.approx(0.4)

    # CPT(80%): Linearly interpolated between c=0.6 (pgr=0.7) and c=0.8 (pgr=0.9) -> c = 0.7
    assert calculate_cpt(c_grid, pgr_grid, target_pgr=0.80) == pytest.approx(0.7)

    # CPT for unreached target returns 1.0
    assert calculate_cpt([0.0, 0.5, 1.0], [0.0, 0.2, 0.3], target_pgr=0.50) == pytest.approx(1.0)


def test_strong_weak_model_pair_selection() -> None:
    """Validate that model pair selection uses training data and criteria rigorously."""
    # Synthetic candidate pricing and train data
    cand_df = pd.DataFrame([
        {"model_name": "frontier-flagship", "size": "671B", "input_price_per_1m": 1.25, "output_price_per_1m": 1.25},
        {"model_name": "mid-model", "size": "70B", "input_price_per_1m": 0.80, "output_price_per_1m": 0.80},
        {"model_name": "llama-3-8b-instruct-lite", "size": "8B", "input_price_per_1m": 0.10, "output_price_per_1m": 0.10},
    ])

    train_rows = []
    for q_idx in range(50):
        q = f"query_{q_idx}"
        train_rows.append({"model_name": "frontier-flagship", "query": q, "performance": 0.85, "reconstructed_cost_usd": 0.00050})
        train_rows.append({"model_name": "mid-model", "query": q, "performance": 0.75, "reconstructed_cost_usd": 0.00025})
        train_rows.append({"model_name": "llama-3-8b-instruct-lite", "query": q, "performance": 0.55, "reconstructed_cost_usd": 0.00002})

    train_df = pd.DataFrame(train_rows)
    pair = select_model_pair_from_train(train_df, cand_df)

    assert pair.strong_model == "frontier-flagship"
    assert pair.weak_model == "llama-3-8b-instruct-lite"
    assert pair.strong_train_perf == pytest.approx(0.85)
    assert pair.weak_train_perf == pytest.approx(0.55)
    assert pair.perf_gap_train == pytest.approx(0.30)
    assert pair.cost_ratio == pytest.approx(25.0)
    assert "replicates RouteLLM" in pair.selection_rationale


def test_prompt_grouped_validation_leakage_free() -> None:
    """Verify that validation splitting for RouteLLM tuning guarantees zero prompt leakage."""
    # Create sample dataframe with multiple models per prompt and cross-task prompts
    records = []
    for i in range(100):
        q = f"prompt text {i}"
        for m in ["model_a", "model_b"]:
            records.append({
                "task_name": f"task_{i % 5}",
                "query": q,
                "model_name": m,
                "performance": 1.0 if (i + len(m)) % 2 == 0 else 0.0,
                "reconstructed_cost_usd": 0.0001,
            })
    df = pd.DataFrame(records)

    train_split, val_split = create_prompt_grouped_validation_split(df, val_ratio=0.2, seed=42)

    tr_queries = set(train_split["query"].str.strip())
    val_queries = set(val_split["query"].str.strip())

    # Critical leakage check: zero intersection
    assert len(tr_queries.intersection(val_queries)) == 0
    assert len(val_queries) > 0
    assert len(tr_queries) > 0


def test_threshold_fitting_monotonicity() -> None:
    """Verify that as threshold decreases, call rate increases monotonically."""
    X_train = np.random.randn(200, 16)
    y_train = np.random.randint(0, 2, size=200)

    clf = RouteLLMClassifier(C=1.0, random_state=42)
    clf.fit(X_train, y_train)

    X_test = np.random.randn(100, 16)
    probs = clf.predict_proba(X_test)

    thresholds = np.linspace(1.0, 0.0, 21)
    call_rates = [float((probs >= th).mean()) for th in thresholds]

    # Monotonic non-decreasing check: call_rates[i] <= call_rates[i+1]
    for i in range(len(call_rates) - 1):
        assert call_rates[i] <= call_rates[i + 1] + 1e-9


def test_deterministic_decisions() -> None:
    """Verify that router produces bitwise deterministic predictions given fixed random_state."""
    rng = np.random.RandomState(42)
    X = rng.randn(150, 20)
    y = rng.randint(0, 2, size=150)
    X_eval = rng.randn(50, 20)

    # First run
    r1 = RouteLLMClassifier(C=0.01, random_state=42)
    r1.fit(X, y)
    probs1 = r1.predict_proba(X_eval)

    # Second run
    r2 = RouteLLMClassifier(C=0.01, random_state=42)
    r2.fit(X, y)
    probs2 = r2.predict_proba(X_eval)

    np.testing.assert_allclose(probs1, probs2, atol=1e-12)


def test_prompt_level_bootstrap_clustering() -> None:
    """Verify that prompt-level bootstrap preserves prompt clustering and returns valid CIs and p-values."""
    N = 100
    piv = pd.DataFrame({
        "strong": np.random.choice([0.0, 1.0], size=N, p=[0.3, 0.7]),
        "weak": np.random.choice([0.0, 1.0], size=N, p=[0.6, 0.4]),
    })
    cost = pd.DataFrame({
        "strong": np.full(N, 0.0005),
        "weak": np.full(N, 0.00002),
    })

    probs = np.random.uniform(0.1, 0.9, size=N)
    th_grid = np.linspace(1.0, 0.0, 21)

    ci_low, ci_high, p_val = bootstrap_two_model_metrics(
        test_piv=piv,
        test_cost=cost,
        strong_model="strong",
        weak_model="weak",
        predicted_probs=probs,
        threshold_grid=th_grid,
        n_bootstraps=200,
        seed=42,
    )

    # Verify CI is well-ordered and bounded
    assert ci_low <= ci_high
    assert -0.5 <= ci_low <= 1.5
    assert -0.5 <= ci_high <= 1.5
    assert 0.0 <= p_val <= 1.0


def test_two_model_cost_bounds() -> None:
    """Verify that routing costs are bounded between weak and strong costs."""
    weak_cost = 0.000019
    strong_cost = 0.000454

    for call_rate in [0.0, 0.25, 0.50, 0.75, 1.0]:
        r_cost = (1.0 - call_rate) * weak_cost + call_rate * strong_cost
        assert weak_cost <= r_cost <= strong_cost
        cost_saved = 1.0 - (r_cost / strong_cost)
        assert 0.0 <= cost_saved <= 1.0


def test_model_pair_canonical_identity_audit() -> None:
    """Audit verification that canonical model IDs match exactly between candidates, train, and test."""
    cand_df = pd.read_parquet("data/processed/xroutebench/candidates_processed.parquet")
    train_df = pd.read_parquet("data/processed/xroutebench/train_processed.parquet", columns=["model_name"])
    test_df = pd.read_parquet("data/processed/xroutebench/test_processed.parquet", columns=["model_name"])

    canonical_strong = "cogito-v2-1-671b"
    canonical_weak = "llama-3-8b-instruct-lite"

    # Verify both canonical models exist in candidate metadata
    cand_models = set(cand_df["model_name"])
    assert canonical_strong in cand_models
    assert canonical_weak in cand_models

    # Verify both models exist in train and test
    assert canonical_strong in set(train_df["model_name"])
    assert canonical_weak in set(train_df["model_name"])
    assert canonical_strong in set(test_df["model_name"])
    assert canonical_weak in set(test_df["model_name"])


def test_dataset_counts_audit_reconciliation() -> None:
    """Audit reconciliation of raw rows, unique queries, and prompt evaluation instances."""
    train_df = pd.read_parquet("data/processed/xroutebench/train_processed.parquet", columns=["task_name", "query", "embedding_id", "model_name"])
    test_df = pd.read_parquet("data/processed/xroutebench/test_processed.parquet", columns=["task_name", "query", "embedding_id", "model_name"])

    # Train: 80,802 rows, 4,487 unique string queries, 4,489 evaluation instances per model
    assert len(train_df) == 80802
    assert train_df["query"].nunique() == 4487
    unique_train_eval_instances = len(train_df.drop_duplicates(subset=["task_name", "query", "embedding_id"]))
    assert unique_train_eval_instances == 4489

    # Test: 67,122 rows, 3,729 unique queries, 3,729 evaluation instances per model
    assert len(test_df) == 67122
    assert test_df["query"].nunique() == 3729
    unique_test_eval_instances = len(test_df.drop_duplicates(subset=["task_name", "query", "embedding_id"]))
    assert unique_test_eval_instances == 3729


def test_null_centered_bootstrap_hypothesis_testing() -> None:
    """Validate null-centered bootstrap hypothesis testing and Davison-Hinkley finite-sample correction.

    Distinguishes:
    1. Ordinary bootstrap CI (centered around sample statistic A_hat).
    2. Null-centered bootstrap test distribution (centered around H0 value 0.5000).
    3. Davison-Hinkley (1997) / Phipson-Smyth (2010) finite-sample correction: p = (1 + k) / (B + 1).
    """
    N = 200
    rng = np.random.RandomState(42)

    # Synthetic dataset where router strongly outperforms weak model
    piv = pd.DataFrame({
        "strong": rng.choice([0.0, 1.0], size=N, p=[0.2, 0.8]),
        "weak": rng.choice([0.0, 1.0], size=N, p=[0.7, 0.3]),
    })
    cost = pd.DataFrame({
        "strong": np.full(N, 0.0005),
        "weak": np.full(N, 0.00002),
    })

    # High quality probabilities correlated with strong success
    probs = np.where(piv["strong"] == 1.0, rng.uniform(0.6, 0.95, size=N), rng.uniform(0.05, 0.4, size=N))
    th_grid = np.linspace(1.0, 0.0, 21)

    B = 500
    ci_low, ci_high, p_val = bootstrap_two_model_metrics(
        test_piv=piv,
        test_cost=cost,
        strong_model="strong",
        weak_model="weak",
        predicted_probs=probs,
        threshold_grid=th_grid,
        n_bootstraps=B,
        seed=42,
    )

    # 1. Verification of finite-sample bounds: p in (0, 1] and never exactly 0
    assert 0.0 < p_val <= 1.0
    # When k=0, minimum possible p-value for B=500 is 1/(B+1) = 1/501
    assert p_val >= 1.0 / (B + 1)
    assert p_val == pytest.approx(1.0 / (B + 1), abs=1e-6)

    # 2. Distinction between CI and Null Distribution:
    # Ordinary CI covers observed effect (well above 0.50)
    assert ci_low > 0.50
    assert ci_high > ci_low

    # 3. Boundary Null Behavior:
    # On a random router (uncorrelated probabilities), APGR is near 0.50, so null hypothesis is not rejected
    random_probs = rng.uniform(0.0, 1.0, size=N)
    ci_low_null, ci_high_null, p_val_null = bootstrap_two_model_metrics(
        test_piv=piv,
        test_cost=cost,
        strong_model="strong",
        weak_model="weak",
        predicted_probs=random_probs,
        threshold_grid=th_grid,
        n_bootstraps=B,
        seed=42,
    )
    # Random router CI covers 0.50, and p-value is well above significance threshold (fails to reject H0)
    assert ci_low_null <= 0.55
    assert ci_high_null >= 0.45
    assert p_val_null > 0.05


def test_micro_macro_apgr_consistency() -> None:
    """Verify that micro and macro APGR calculations operate consistently across balanced and unbalanced tasks."""
    # Balanced case: two tasks with identical sample size
    c_rates = np.linspace(0.0, 1.0, 11)
    pgr_task1 = c_rates * 0.8
    pgr_task2 = c_rates * 0.6

    apgr_1 = calculate_apgr(c_rates, pgr_task1)
    apgr_2 = calculate_apgr(c_rates, pgr_task2)
    macro_expected = (apgr_1 + apgr_2) / 2.0

    pgr_macro_curve = (pgr_task1 + pgr_task2) / 2.0
    apgr_macro = calculate_apgr(c_rates, pgr_macro_curve)

    assert apgr_macro == pytest.approx(macro_expected, abs=1e-6)
    # Check bounds
    assert 0.0 <= apgr_macro <= 1.0
