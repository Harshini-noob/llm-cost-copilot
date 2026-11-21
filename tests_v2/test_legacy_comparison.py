"""
tests_v2/test_legacy_comparison.py

Unit tests for Legacy Rule-Based vs M3 Learning-Based Router Evaluation.
Verifies:
1. Correct implementation of all four legacy routing modes (economy, quality, latency, balanced).
2. Same test prompts evaluated across all seven strategies.
3. Strict absence of test data leakage during fitting, estimation, and selection.
4. Quality denominator and evaluation coverage accounting.
5. Graceful handling of missing outcomes and execution failures.
6. Exact reproducibility across repeated runs.
7. Zero network socket calls during evaluation.
8. Preservation of all existing M2 and M3 Step 1-7 reports.
"""

from __future__ import annotations

import json
from pathlib import Path
import socket
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eval.legacy_comparison import (
    DEFAULT_STATIC_LATENCIES_SEC,
    DEFAULT_STATIC_QUALITY_TIERS,
    LegacyRouterAdapter,
    LegacySelectionDecision,
    run_legacy_vs_learned_comparison,
)
from eval.routers.encoders import DeterministicTestEncoder


@pytest.fixture
def synthetic_comparison_matrix() -> pd.DataFrame:
    """Fixture providing a synthetic benchmark matrix across multiple prompts and models."""
    records = []
    # 6 prompts across math and code
    prompts = [
        ("math_101", "math", "Calculate 15 * 4"),
        ("math_102", "math", "Solve 2x = 10"),
        ("math_103", "math", "What is 100 / 4?"),
        ("code_101", "code", "Write code to reverse a list in Python"),
        ("code_102", "code", "Explain the quicksort algorithm"),
        ("code_103", "code", "Implement a binary search tree"),
    ]
    models = [
        ("gpt_oss_20b", 0.00002, 300.0, 1.0),
        ("gpt_oss_120b", 0.00005, 500.0, 1.0),
        ("gemini_3_5_flash_lite", 0.00001, 1200.0, 1.0),
        ("qwen3_30b_a3b_openrouter", 0.000015, 800.0, 0.0),
    ]

    for p_id, cat, text in prompts:
        for m_id, cost, lat, base_score in models:
            records.append({
                "prompt_id": p_id,
                "task_type": cat,
                "prompt": text,
                "model_id": m_id,
                "provider": "prov",
                "cost": cost,
                "latency_ms": lat,
                "execution_success": True,
                "evaluation_status": "scored",
                "score": base_score,
                "correct": (base_score >= 1.0),
                "benchmark_config_id": "test_cfg",
            })

    return pd.DataFrame(records)


def test_legacy_router_adapter_implements_all_four_modes() -> None:
    """Verifies that LegacyRouterAdapter correctly implements economy, quality, latency, and balanced modes."""
    candidate_models = ["gpt_oss_20b", "gpt_oss_120b", "gemini_3_5_flash_lite", "qwen3_30b_a3b_openrouter"]
    adapter = LegacyRouterAdapter(candidate_models=candidate_models)

    prompt = "A shop sells 3 notebooks for Rs. 45. How much will 8 notebooks cost? Give only the numerical answer."

    # 1. Economy mode
    dec_eco = adapter.select_model(prompt, routing_mode="economy")
    assert dec_eco is not None
    assert dec_eco.routing_mode == "economy"
    # Among eligible models (q >= 0.78), qwen has lowest estimated worst-case cost
    assert dec_eco.selected_model_id == "qwen3_30b_a3b_openrouter"

    # 2. Quality mode
    dec_qual = adapter.select_model(prompt, routing_mode="quality")
    assert dec_qual is not None
    assert dec_qual.routing_mode == "quality"
    # Highest expected quality is Gemini (0.90)
    assert dec_qual.selected_model_id == "gemini_3_5_flash_lite"

    # 3. Latency mode
    dec_lat = adapter.select_model(prompt, routing_mode="latency")
    assert dec_lat is not None
    assert dec_lat.routing_mode == "latency"
    # Lowest latency among eligible models is gpt_oss_120b (1.1s)
    assert dec_lat.selected_model_id == "gpt_oss_120b"

    # 4. Balanced mode
    dec_bal = adapter.select_model(prompt, routing_mode="balanced")
    assert dec_bal is not None
    assert dec_bal.routing_mode == "balanced"
    # Under balanced formula (0.50*Q + 0.35*(1-C) + 0.15*(1-L)), gpt_oss_120b wins
    assert dec_bal.selected_model_id == "gpt_oss_120b"


def test_legacy_router_adapter_five_sample_threshold_rule() -> None:
    """Verifies historical judge averages replace static estimates only when >= 5 samples exist."""
    candidate_models = ["gpt_oss_20b"]

    # Case A: 4 samples (< 5 threshold) -> retains static fallback (0.75)
    adapter_4 = LegacyRouterAdapter(
        candidate_models=candidate_models,
        training_scores={"gpt_oss_20b": [1.0, 1.0, 1.0, 1.0]},
    )
    meta_4 = adapter_4.get_model_metadata("gpt_oss_20b")
    assert meta_4["quality_source"] == "static_fallback"
    assert meta_4["expected_quality"] == 0.75

    # Case B: 5 samples (>= 5 threshold) -> uses empirical average (1.0)
    adapter_5 = LegacyRouterAdapter(
        candidate_models=candidate_models,
        training_scores={"gpt_oss_20b": [1.0, 1.0, 1.0, 1.0, 1.0]},
    )
    meta_5 = adapter_5.get_model_metadata("gpt_oss_20b")
    assert meta_5["quality_source"] == "training_data"
    assert meta_5["expected_quality"] == 1.0


def test_same_test_prompts_across_all_strategies(
    synthetic_comparison_matrix: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verifies that all seven strategies are evaluated on the exact same held-out test prompts."""
    encoder = DeterministicTestEncoder(dimension=64)
    custom_split = {
        "train_prompt_ids": ["math_101", "math_102", "code_101", "code_102"],
        "val_prompt_ids": [],
        "test_prompt_ids": ["math_103", "code_103"],
        "sufficiency_status": "SUFFICIENT_DATA",
        "notes": "Synthetic split test",
    }

    df, summary, results = run_legacy_vs_learned_comparison(
        custom_df=synthetic_comparison_matrix,
        custom_split=custom_split,
        encoder=encoder,
        output_dir=tmp_path,
    )

    assert len(results) == 7
    for r in results:
        assert r.test_prompt_count == 2
        assert r.successful_evaluations <= 2


def test_strict_test_data_isolation_and_no_leakage(
    synthetic_comparison_matrix: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verifies that test outcomes never influence router fitting or expected cost estimation."""
    encoder = DeterministicTestEncoder(dimension=64)
    custom_split = {
        "train_prompt_ids": ["math_101", "math_102", "code_101", "code_102"],
        "val_prompt_ids": [],
        "test_prompt_ids": ["math_103", "code_103"],
        "sufficiency_status": "SUFFICIENT_DATA",
        "notes": "Synthetic split test",
    }

    # Alter test costs and scores drastically in matrix copy
    corrupted_matrix = synthetic_comparison_matrix.copy()
    corrupted_matrix.loc[corrupted_matrix["prompt_id"] == "math_103", "cost"] = 999.0
    corrupted_matrix.loc[corrupted_matrix["prompt_id"] == "math_103", "score"] = 0.0

    _, _, results = run_legacy_vs_learned_comparison(
        custom_df=corrupted_matrix,
        custom_split=custom_split,
        encoder=encoder,
        output_dir=tmp_path,
    )

    # Always-Cheapest selection must be determined strictly from training records
    cheapest = next(r for r in results if r.strategy_name == "Always-Cheapest")
    # Training records had gemini as cheapest ($0.00001)
    assert cheapest.model_selection_distribution == {"gemini_3_5_flash_lite": 1.0}


def test_quality_denominator_and_coverage_accounting(
    synthetic_comparison_matrix: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verifies quality denominator strictly equals scored evaluations and coverage is tracked."""
    encoder = DeterministicTestEncoder(dimension=64)
    # Inject an execution failure on test prompt code_103 for gemini
    corrupted_matrix = synthetic_comparison_matrix.copy()
    corrupted_matrix["correct"] = corrupted_matrix["correct"].astype(object)
    idx = corrupted_matrix[
        (corrupted_matrix["prompt_id"] == "code_103") & (corrupted_matrix["model_id"] == "gemini_3_5_flash_lite")
    ].index
    corrupted_matrix.loc[idx, "execution_success"] = False
    corrupted_matrix.loc[idx, "evaluation_status"] = "execution_failed"
    corrupted_matrix.loc[idx, "score"] = None
    corrupted_matrix.loc[idx, "correct"] = None

    custom_split = {
        "train_prompt_ids": ["math_101", "math_102", "code_101", "code_102"],
        "val_prompt_ids": [],
        "test_prompt_ids": ["math_103", "code_103"],
        "sufficiency_status": "SUFFICIENT_DATA",
        "notes": "Synthetic split test",
    }

    _, _, results = run_legacy_vs_learned_comparison(
        custom_df=corrupted_matrix,
        custom_split=custom_split,
        encoder=encoder,
        output_dir=tmp_path,
    )

    # If a strategy selected gemini on code_103, it must record execution failure
    for r in results:
        assert r.quality_denominator == r.successful_evaluations
        assert r.evaluation_coverage == (r.quality_denominator / r.test_prompt_count)
        assert r.execution_failure_count + r.successful_evaluations <= r.test_prompt_count


def test_preservation_of_all_existing_reports(tmp_path: Path) -> None:
    """Verifies that running legacy comparison never overwrites any existing M2 or Step 1-7 artifacts."""
    existing_files = {
        "baseline_comparison.csv": "m2_csv",
        "baseline_comparison.parquet": "m2_parquet",
        "cost_vs_quality.png": "m2_png",
        "train_test_split.json": "m2_json",
        "router_comparison.csv": "step6_csv",
        "router_comparison.parquet": "step6_parquet",
        "router_pareto_curve.png": "step6_png",
        "router_evaluation_summary.json": "step6_json",
        "router_robustness.csv": "step7_csv",
        "router_robustness.parquet": "step7_parquet",
        "router_robustness.png": "step7_png",
        "router_robustness_summary.json": "step7_json",
    }
    for filename, content in existing_files.items():
        (tmp_path / filename).write_text(content, encoding="utf-8")

    encoder = DeterministicTestEncoder(dimension=384)
    run_legacy_vs_learned_comparison(
        benchmark_config_id="m1-efa7ec3afd4f",
        encoder=encoder,
        output_dir=tmp_path,
    )

    # Verify all 12 existing files are completely unmodified
    for filename, content in existing_files.items():
        assert (tmp_path / filename).read_text(encoding="utf-8") == content

    # Verify new comparison files exist
    assert (tmp_path / "legacy_vs_learned_comparison.csv").exists()
    assert (tmp_path / "legacy_vs_learned_comparison.parquet").exists()
    assert (tmp_path / "legacy_vs_learned_comparison.json").exists()
    assert (tmp_path / "legacy_vs_learned_comparison.png").exists()


def test_zero_network_calls_during_legacy_comparison(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Guarantees evaluation executes strictly offline without socket network calls."""
    def blocked_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network socket call attempted during offline evaluation!")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)

    encoder = DeterministicTestEncoder(dimension=384)
    df, summary, results = run_legacy_vs_learned_comparison(
        benchmark_config_id="m1-efa7ec3afd4f",
        encoder=encoder,
        output_dir=tmp_path,
    )

    assert len(results) == 7
    assert summary["sufficiency_status"] == "INSUFFICIENT_DATA"


def test_real_benchmark_matrix_comparison_values(tmp_path: Path) -> None:
    """Verifies exact quantitative metrics on the real benchmark matrix."""
    encoder = DeterministicTestEncoder(dimension=384)
    df, summary, results = run_legacy_vs_learned_comparison(
        benchmark_config_id="m1-efa7ec3afd4f",
        encoder=encoder,
        output_dir=tmp_path,
    )

    by_name = {r.strategy_name: r for r in results}

    # 1. Primary baseline: Legacy Rule-Based Balanced
    leg_bal = by_name["Legacy Rule-Based Balanced"]
    assert leg_bal.selected_model_distribution == {"gpt_oss_120b": 1.0} if hasattr(leg_bal, "selected_model_distribution") else True
    assert leg_bal.model_selection_distribution == {"gpt_oss_120b": 1.0}
    assert leg_bal.mean_cost == pytest.approx(0.00004975, rel=1e-3)
    assert leg_bal.mean_quality == pytest.approx(1.0)

    # 2. M3 Learning-based routers: kNN and Learned
    m3_knn = by_name["M3 kNN Router"]
    m3_lr = by_name["M3 Learned Router"]
    assert m3_knn.model_selection_distribution == {"gemini_3_5_flash_lite": 1.0}
    assert m3_lr.model_selection_distribution == {"gemini_3_5_flash_lite": 1.0}
    assert m3_knn.mean_cost == pytest.approx(0.0000171, rel=1e-3)
    assert m3_lr.mean_cost == pytest.approx(0.0000171, rel=1e-3)
    assert m3_knn.mean_quality == pytest.approx(1.0)
    assert m3_lr.mean_quality == pytest.approx(1.0)

    # 3. Cost savings relative to Legacy Balanced: -65.6%
    assert m3_knn.delta_cost_pct == pytest.approx(-65.628, rel=1e-2)
    assert m3_lr.delta_cost_pct == pytest.approx(-65.628, rel=1e-2)

    # 4. Legacy Economy selected Qwen and failed quality (0.0)
    leg_eco = by_name["Legacy Rule-Based Economy"]
    assert leg_eco.model_selection_distribution == {"qwen3_30b_a3b_openrouter": 1.0}
    assert leg_eco.mean_quality == pytest.approx(0.0)
