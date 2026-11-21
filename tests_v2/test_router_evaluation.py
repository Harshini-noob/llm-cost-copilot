"""
tests_v2/test_router_evaluation.py

Unit tests for Comparative Evaluation and Pareto Sweeps under Phase M3.
Verifies prompt-level split isolation, zero test leakage, identical test prompts,
lambda sweep reproducibility, Pareto dominance calculations, honest failure accounting,
insufficient-data flags, artifact preservation, and zero network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
import socket
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eval.baselines import (
    STRATEGY_DEPLOYABLE,
    STRATEGY_POST_HOC,
    STRATEGY_THEORETICAL,
)
from eval.routers.encoders import DeterministicTestEncoder
from eval.router_evaluation import (
    DEFAULT_LAMBDA_SWEEP,
    StrategyEvaluationResult,
    compute_pareto_dominance,
    compute_tradeoff_benchmarks,
    create_evaluation_split,
    run_router_evaluation,
    tune_presets_from_validation,
)


@pytest.fixture
def synthetic_benchmark_df() -> pd.DataFrame:
    """Fixture providing a multi-prompt, multi-model benchmark matrix with scored, failed, and unscored outcomes."""
    records = []
    # 6 prompts across 2 categories (math: 3, code: 3)
    # Models: m_fast ($0.0001), m_mid ($0.0003), m_smart ($0.0008)
    prompts = [
        ("math_101", "math", "Calculate 15 * 14"),
        ("math_102", "math", "Solve for y: 2y + 6 = 20"),
        ("math_103", "math", "Integrate 2x dx from 0 to 4"),
        ("code_101", "code", "Write a python function for fibonacci"),
        ("code_102", "code", "Implement quicksort in python"),
        ("code_103", "code", "Parse JSON string safely in python"),
    ]

    for p_id, cat, text in prompts:
        # m_fast: cheap, low accuracy on math, medium on code
        records.append({
            "prompt_id": p_id,
            "task_type": cat,
            "model_id": "m_fast",
            "provider": "prov_a",
            "cost": 0.0001,
            "latency_ms": 100.0,
            "execution_success": True,
            "evaluation_status": "scored",
            "score": 1.0 if cat == "code" and "101" in p_id else 0.0,
            "correct": (cat == "code" and "101" in p_id),
            "benchmark_config_id": "test_cfg",
        })
        # m_mid: balanced
        records.append({
            "prompt_id": p_id,
            "task_type": cat,
            "model_id": "m_mid",
            "provider": "prov_b",
            "cost": 0.0003,
            "latency_ms": 250.0,
            "execution_success": True,
            "evaluation_status": "scored",
            "score": 1.0 if "101" in p_id or "102" in p_id else 0.0,
            "correct": ("101" in p_id or "102" in p_id),
            "benchmark_config_id": "test_cfg",
        })
        # m_smart: expensive, high accuracy, with one simulated execution failure on code_103
        is_failure = (p_id == "code_103")
        records.append({
            "prompt_id": p_id,
            "task_type": cat,
            "model_id": "m_smart",
            "provider": "prov_c",
            "cost": 0.0008,
            "latency_ms": 600.0,
            "execution_success": not is_failure,
            "evaluation_status": "execution_failed" if is_failure else "scored",
            "score": None if is_failure else 1.0,
            "correct": None if is_failure else True,
            "benchmark_config_id": "test_cfg",
        })

    return pd.DataFrame(records)


def test_split_isolation_and_disjointness() -> None:
    """Verifies train, val, and test partitions are strictly disjoint with zero prompt leakage."""
    prompts = [
        {"prompt_id": f"p_{i}", "task_type": "math" if i < 6 else "code"}
        for i in range(12)
    ]
    split = create_evaluation_split(prompts, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42)

    train_set = set(split["train_prompt_ids"])
    val_set = set(split["val_prompt_ids"])
    test_set = set(split["test_prompt_ids"])

    # Disjointness checks
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0

    # Total coverage
    all_assigned = train_set | val_set | test_set
    assert len(all_assigned) == 12


def test_all_strategies_evaluated_on_exact_same_test_prompts(
    synthetic_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verifies that all 7 strategies (5 baselines + 2 routers) evaluate on the identical test prompts."""
    encoder = DeterministicTestEncoder(dimension=64)
    custom_split = {
        "train_prompt_ids": ["math_101", "math_102", "code_101", "code_102"],
        "val_prompt_ids": [],
        "test_prompt_ids": ["math_103", "code_103"],
        "sufficiency_status": "SUFFICIENT_DATA",
        "notes": "Synthetic split test",
    }

    comp_df, summary, results = run_router_evaluation(
        custom_df=synthetic_benchmark_df,
        custom_split=custom_split,
        encoder=encoder,
        lambda_sweep=[0.0, 0.5, 2.0],
        output_dir=tmp_path,
    )

    expected_test_prompts = sorted(custom_split["test_prompt_ids"])

    for res in results:
        assert res.test_prompt_count == 2
        actual_pids = sorted([s.prompt_id for s in res.selections])
        assert actual_pids == expected_test_prompts, f"Strategy {res.name} evaluated on mismatched prompts!"


def test_reference_versus_deployable_classification(
    synthetic_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verifies that Always-Best and Oracle are marked as reference ceilings, while others are deployable."""
    encoder = DeterministicTestEncoder(dimension=64)
    custom_split = {
        "train_prompt_ids": ["math_101", "math_102", "code_101", "code_102"],
        "val_prompt_ids": [],
        "test_prompt_ids": ["math_103", "code_103"],
        "sufficiency_status": "SUFFICIENT_DATA",
        "notes": "Synthetic split test",
    }

    _, _, results = run_router_evaluation(
        custom_df=synthetic_benchmark_df,
        custom_split=custom_split,
        encoder=encoder,
        lambda_sweep=[0.0, 1.0],
        output_dir=tmp_path,
    )

    for res in results:
        if res.name in ("Always-Best", "Oracle"):
            assert res.is_reference is True
            assert res.strategy_type in (STRATEGY_POST_HOC, STRATEGY_THEORETICAL)
        else:
            assert res.is_reference is False
            assert res.strategy_type == STRATEGY_DEPLOYABLE


def test_pareto_dominance_calculation() -> None:
    """Verifies that Pareto dominance correctly identifies dominated and non-dominated strategies."""
    # Strategy A: Cheap & high quality ($0.0001, Q=0.90) -> should dominate B
    # Strategy B: Expensive & lower quality ($0.0005, Q=0.70) -> dominated
    # Strategy C: Expensive & higher quality ($0.0010, Q=0.98) -> non-dominated trade-off
    r_a = StrategyEvaluationResult(
        name="Strat_A", strategy_type=STRATEGY_DEPLOYABLE, is_reference=False, router_family="test",
        lambda_value=1.0, preset_name=None, mean_cost=0.0001, total_cost=0.0001, mean_quality=0.90,
        quality_std=0.0, mean_latency_ms=100.0, evaluation_coverage=1.0, execution_failure_count=0,
        unscored_count=0, test_prompt_count=10,
    )
    r_b = StrategyEvaluationResult(
        name="Strat_B", strategy_type=STRATEGY_DEPLOYABLE, is_reference=False, router_family="test",
        lambda_value=0.5, preset_name=None, mean_cost=0.0005, total_cost=0.0005, mean_quality=0.70,
        quality_std=0.0, mean_latency_ms=100.0, evaluation_coverage=1.0, execution_failure_count=0,
        unscored_count=0, test_prompt_count=10,
    )
    r_c = StrategyEvaluationResult(
        name="Strat_C", strategy_type=STRATEGY_DEPLOYABLE, is_reference=False, router_family="test",
        lambda_value=0.0, preset_name=None, mean_cost=0.0010, total_cost=0.0010, mean_quality=0.98,
        quality_std=0.0, mean_latency_ms=100.0, evaluation_coverage=1.0, execution_failure_count=0,
        unscored_count=0, test_prompt_count=10,
    )
    # Reference Oracle: ($0.00008, Q=1.00)
    r_oracle = StrategyEvaluationResult(
        name="Oracle", strategy_type=STRATEGY_THEORETICAL, is_reference=True, router_family="baseline",
        lambda_value=None, preset_name=None, mean_cost=0.00008, total_cost=0.00008, mean_quality=1.00,
        quality_std=0.0, mean_latency_ms=100.0, evaluation_coverage=1.0, execution_failure_count=0,
        unscored_count=0, test_prompt_count=10,
    )

    results = [r_a, r_b, r_c, r_oracle]
    compute_pareto_dominance(results)

    # Among deployable: A and C are Pareto efficient, B is dominated
    assert r_a.is_pareto_deployable is True
    assert r_c.is_pareto_deployable is True
    assert r_b.is_pareto_deployable is False

    # Oracle is a reference, so it cannot be pareto_deployable
    assert r_oracle.is_pareto_deployable is False
    # But Oracle is Pareto optimal across all
    assert r_oracle.is_pareto_all is True


def test_lambda_sweep_exposes_selection_changes(
    synthetic_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verifies that increasing lambda shifts router model choices from high-quality to lower-cost models."""
    encoder = DeterministicTestEncoder(dimension=64)
    custom_split = {
        "train_prompt_ids": ["math_101", "math_102", "code_101", "code_102"],
        "val_prompt_ids": [],
        "test_prompt_ids": ["math_103", "code_103"],
        "sufficiency_status": "SUFFICIENT_DATA",
        "notes": "Synthetic split test",
    }

    _, _, results = run_router_evaluation(
        custom_df=synthetic_benchmark_df,
        custom_split=custom_split,
        encoder=encoder,
        lambda_sweep=[0.0, 10.0],
        output_dir=tmp_path,
    )

    res_q = next(r for r in results if r.name == "kNN Router (lambda=0.00)")
    res_c = next(r for r in results if r.name == "kNN Router (lambda=10.00)")

    # At lambda=0.0, cost should be >= cost at lambda=10.0
    assert res_q.mean_cost >= res_c.mean_cost


def test_missing_and_failed_candidates_handled_honestly(
    synthetic_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verifies execution failures and missing candidates are recorded with uninflated quality and honest status."""
    encoder = DeterministicTestEncoder(dimension=64)
    # Include code_103 in test set (which has execution_failed for m_smart)
    custom_split = {
        "train_prompt_ids": ["math_101", "math_102", "code_101", "code_102"],
        "val_prompt_ids": [],
        "test_prompt_ids": ["code_103"],
        "sufficiency_status": "SUFFICIENT_DATA",
        "notes": "Failure accounting test",
    }

    _, _, results = run_router_evaluation(
        custom_df=synthetic_benchmark_df,
        custom_split=custom_split,
        encoder=encoder,
        lambda_sweep=[0.0],
        output_dir=tmp_path,
    )

    for res in results:
        assert res.test_prompt_count == 1
        # Selections must exist and have valid fields
        sel = res.selections[0]
        assert sel.prompt_id == "code_103"
        if not sel.execution_success:
            assert sel.score is None
            assert res.execution_failure_count >= 1


def test_insufficient_data_status_reported_on_real_historical_matrix(tmp_path: Path) -> None:
    """Verifies that running on the real historical matrix marks results as INSUFFICIENT_DATA / EXPLORATORY."""
    encoder = DeterministicTestEncoder(dimension=384)

    comp_df, summary, results = run_router_evaluation(
        benchmark_config_id="m1-efa7ec3afd4f",
        encoder=encoder,
        lambda_sweep=[0.0, 0.5, 2.0],
        output_dir=tmp_path,
    )

    # Must be marked INSUFFICIENT_DATA
    assert summary["sufficiency_status"] == "INSUFFICIENT_DATA"
    assert summary["is_exploratory"] is True
    assert "EXPLORATORY" in summary["sufficiency_notes"]
    assert (comp_df["sufficiency_status"] == "INSUFFICIENT_DATA").all()

    # Presets must be flagged provisional because validation data has 0 prompts
    assert summary["preset_tuning"]["is_empirically_tuned"] is False
    assert summary["preset_tuning"]["status"] == "PROVISIONAL"


def test_m2_baseline_artifacts_are_preserved(tmp_path: Path) -> None:
    """Verifies that running M3 evaluation never overwrites existing M2 baseline artifacts."""
    # Create fake M2 baseline artifacts in tmp_path
    m2_csv = tmp_path / "baseline_comparison.csv"
    m2_csv.write_text("dummy m2 baseline csv content", encoding="utf-8")
    m2_parquet = tmp_path / "baseline_comparison.parquet"
    m2_parquet.write_text("dummy m2 baseline parquet", encoding="utf-8")
    m2_plot = tmp_path / "cost_vs_quality.png"
    m2_plot.write_text("dummy plot", encoding="utf-8")
    m2_split = tmp_path / "train_test_split.json"
    m2_split.write_text("{}", encoding="utf-8")

    encoder = DeterministicTestEncoder(dimension=384)
    run_router_evaluation(
        benchmark_config_id="m1-efa7ec3afd4f",
        encoder=encoder,
        lambda_sweep=[0.0, 1.0],
        output_dir=tmp_path,
    )

    # Check M2 artifacts remain intact with original content
    assert m2_csv.read_text(encoding="utf-8") == "dummy m2 baseline csv content"
    assert m2_parquet.read_text(encoding="utf-8") == "dummy m2 baseline parquet"
    assert m2_plot.read_text(encoding="utf-8") == "dummy plot"
    assert m2_split.read_text(encoding="utf-8") == "{}"

    # Check M3 artifacts were generated
    assert (tmp_path / "router_comparison.csv").exists()
    assert (tmp_path / "router_comparison.parquet").exists()
    assert (tmp_path / "router_pareto_curve.png").exists()
    assert (tmp_path / "router_evaluation_summary.json").exists()


def test_zero_network_calls_during_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Guarantees full comparative evaluation executes strictly offline without network calls."""
    def blocked_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network socket call attempted during offline evaluation!")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)

    encoder = DeterministicTestEncoder(dimension=384)
    comp_df, summary, results = run_router_evaluation(
        benchmark_config_id="m1-efa7ec3afd4f",
        encoder=encoder,
        lambda_sweep=[0.0, 0.5, 2.0],
        output_dir=tmp_path,
    )

    assert len(results) > 0
    assert summary["sufficiency_status"] == "INSUFFICIENT_DATA"


def test_quality_denominator_and_coverage_accounting(
    synthetic_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verifies that quality denominators, evaluation coverage, and coverage warnings are honestly tracked."""
    encoder = DeterministicTestEncoder(dimension=64)
    # code_103 has an execution failure on m_smart
    custom_split = {
        "train_prompt_ids": ["math_101", "math_102", "code_101", "code_102"],
        "val_prompt_ids": [],
        "test_prompt_ids": ["math_103", "code_103"],
        "sufficiency_status": "SUFFICIENT_DATA",
        "notes": "Quality denominator audit test",
    }

    _, summary, results = run_router_evaluation(
        custom_df=synthetic_benchmark_df,
        custom_split=custom_split,
        encoder=encoder,
        lambda_sweep=[0.0],
        output_dir=tmp_path,
    )

    for res in results:
        assert res.test_prompt_count == 2
        scored_count = len([s for s in res.selections if s.score is not None])
        assert res.successful_evaluations == scored_count
        assert res.quality_denominator == scored_count
        expected_coverage = scored_count / 2.0
        assert res.evaluation_coverage == pytest.approx(expected_coverage)

        if expected_coverage < 1.0:
            assert res.coverage_discrepancy_warning is not None
            assert "COVERAGE_DEFICIT" in res.coverage_discrepancy_warning
        else:
            assert res.coverage_discrepancy_warning is None


def test_hyperparameter_selection_audit_and_test_isolation(tmp_path: Path) -> None:
    """Verifies that router hyperparameters are audited and flagged as provisional defaults when val data is insufficient."""
    encoder = DeterministicTestEncoder(dimension=384)
    _, summary, _ = run_router_evaluation(
        benchmark_config_id="m1-efa7ec3afd4f",
        encoder=encoder,
        lambda_sweep=[0.0],
        output_dir=tmp_path,
    )

    audit = summary["router_hyperparameters_audit"]
    assert audit["knn"]["selection_source"] == "provisional_deterministic_default"
    assert audit["knn"]["is_empirically_tuned"] is False
    assert audit["learned_logistic_regression"]["selection_source"] == "provisional_deterministic_default"
    assert audit["learned_logistic_regression"]["is_empirically_tuned"] is False
    assert "Zero test labels" in audit["test_data_isolation"]


def test_matched_tradeoffs_bounded_and_partitions_reference_strategies(
    synthetic_benchmark_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verifies matched-cost and matched-quality calculations operate strictly within observed bounds and segregate references."""
    encoder = DeterministicTestEncoder(dimension=64)
    custom_split = {
        "train_prompt_ids": ["math_101", "math_102", "code_101", "code_102"],
        "val_prompt_ids": [],
        "test_prompt_ids": ["math_103", "code_103"],
        "sufficiency_status": "SUFFICIENT_DATA",
        "notes": "Matched trade-offs audit test",
    }

    _, summary, results = run_router_evaluation(
        custom_df=synthetic_benchmark_df,
        custom_split=custom_split,
        encoder=encoder,
        lambda_sweep=[0.0, 1.0],
        output_dir=tmp_path,
    )

    tradeoffs = summary["tradeoff_benchmarks"]
    assert "methodology" in tradeoffs
    assert tradeoffs["interpolation_policy"] == "strictly_within_observed_range_no_extrapolation"
    assert len(tradeoffs["observed_cost_range_usd"]) == 2
    assert tradeoffs["observed_cost_range_usd"][0] <= tradeoffs["observed_cost_range_usd"][1]

    # Reference strategies must be segregated from deployable alternatives
    deployable_cost_matches = tradeoffs["deployable_at_matched_cost"]
    references_cost_matches = tradeoffs["references_at_matched_cost"]

    for d in deployable_cost_matches:
        assert d["is_reference"] is False
        assert d["strategy_type"] == STRATEGY_DEPLOYABLE

    for r in references_cost_matches:
        assert r["is_reference"] is True
        assert r["strategy_type"] in (STRATEGY_POST_HOC, STRATEGY_THEORETICAL)

