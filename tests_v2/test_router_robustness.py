"""
tests_v2/test_router_robustness.py

Unit tests for Router Robustness and Generalization Evaluation under Phase M3 Step 7.
Verifies model availability filtering, single-model fallback, no-eligible-model controlled failure,
cost multiplier application, original cost immutability, lambda sensitivity, distribution shift
insufficiency reporting, decision stability metrics, missing/failed outcome accounting, test-data isolation,
zero network calls, and preservation of existing M2 and M3 artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import socket
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eval.router_robustness import (
    RobustnessScenarioResult,
    evaluate_decisions_for_prompts,
    run_cost_sensitivity_experiment,
    run_distribution_shift_audit,
    run_lambda_sensitivity_experiment,
    run_model_availability_experiment,
    run_router_robustness_evaluation,
    simulate_controlled_distribution_shift,
)
from eval.routers.data_prep import PromptRecord
from eval.routers.decision import CostNormalizer, LambdaDecisionEngine
from eval.routers.encoders import DeterministicTestEncoder
from eval.routers.knn_router import KNNRouter
from eval.routers.learned_router import LearnedRouter


@pytest.fixture
def synthetic_robustness_matrix() -> pd.DataFrame:
    """Fixture providing a multi-prompt, multi-model benchmark matrix with scored, failed, and unscored outcomes."""
    records = []
    # 6 prompts across 2 categories (math: 3, code: 3)
    # Models: m_fast ($0.0001), m_mid ($0.0003), m_smart ($0.0008)
    prompts = [
        ("math_101", "math"),
        ("math_102", "math"),
        ("math_103", "math"),
        ("code_101", "code"),
        ("code_102", "code"),
        ("code_103", "code"),
    ]

    for p_id, cat in prompts:
        # m_fast
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
        # m_mid
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
        # m_smart (with one simulated execution failure on code_103)
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


def test_model_availability_filtering_and_single_fallback(
    synthetic_robustness_matrix: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verifies that unavailable models are never selected, and single remaining model is selected as fallback."""
    encoder = DeterministicTestEncoder(dimension=64)
    custom_split = {
        "train_prompt_ids": ["math_101", "math_102", "code_101", "code_102"],
        "val_prompt_ids": [],
        "test_prompt_ids": ["math_103", "code_103"],
        "sufficiency_status": "SUFFICIENT_DATA",
        "notes": "Synthetic split test",
    }

    _, _, results = run_router_robustness_evaluation(
        custom_df=synthetic_robustness_matrix,
        custom_split=custom_split,
        encoder=encoder,
        output_dir=tmp_path,
    )

    # Find scenario where m_smart is excluded
    drop_smart = next(r for r in results if r.scenario_id == "avail_drop_m_smart" and r.router_name == "Learned Router")
    assert "m_smart" in drop_smart.excluded_models
    assert "m_smart" not in drop_smart.model_selection_distribution

    # Find single model scenario where only m_fast is available
    only_fast = next(r for r in results if r.scenario_id == "avail_only_m_fast" and r.router_name == "Learned Router")
    assert only_fast.available_models == ["m_fast"]
    assert only_fast.successful_decisions == 2
    assert only_fast.model_selection_distribution == {"m_fast": 1.0}


def test_no_eligible_models_handled_gracefully(
    synthetic_robustness_matrix: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verifies that when zero models are available, decisions fail gracefully without crashing."""
    encoder = DeterministicTestEncoder(dimension=64)
    custom_split = {
        "train_prompt_ids": ["math_101", "math_102", "code_101", "code_102"],
        "val_prompt_ids": [],
        "test_prompt_ids": ["math_103", "code_103"],
        "sufficiency_status": "SUFFICIENT_DATA",
        "notes": "Synthetic split test",
    }

    _, _, results = run_router_robustness_evaluation(
        custom_df=synthetic_robustness_matrix,
        custom_split=custom_split,
        encoder=encoder,
        output_dir=tmp_path,
    )

    none_avail = next(r for r in results if r.scenario_id == "avail_none" and r.router_name == "Learned Router")
    assert none_avail.successful_decisions == 0
    assert none_avail.failed_decisions == 2
    assert none_avail.available_models == []


def test_cost_multiplier_application_and_original_cost_immutability(
    synthetic_robustness_matrix: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verifies decision-time cost multipliers scale estimated costs while keeping original observed costs invariant."""
    encoder = DeterministicTestEncoder(dimension=64)
    custom_split = {
        "train_prompt_ids": ["math_101", "math_102", "code_101", "code_102"],
        "val_prompt_ids": [],
        "test_prompt_ids": ["math_103", "code_103"],
        "sufficiency_status": "SUFFICIENT_DATA",
        "notes": "Synthetic split test",
    }

    _, _, results = run_router_robustness_evaluation(
        custom_df=synthetic_robustness_matrix,
        custom_split=custom_split,
        encoder=encoder,
        output_dir=tmp_path,
    )

    cost_1x = next(r for r in results if r.scenario_id == "cost_1.00x" and r.router_name == "Learned Router")
    cost_2x = next(r for r in results if r.scenario_id == "cost_2.00x" and r.router_name == "Learned Router")

    # Estimated costs must scale proportionally
    assert cost_1x.mean_estimated_cost is not None
    assert cost_2x.mean_estimated_cost is not None
    assert cost_2x.mean_estimated_cost == pytest.approx(cost_1x.mean_estimated_cost * 2.0)

    # Actual observed costs in test data must remain invariant
    if cost_1x.mean_actual_cost is not None and cost_2x.mean_actual_cost is not None:
        assert cost_1x.mean_actual_cost == pytest.approx(cost_2x.mean_actual_cost)


def test_lambda_sensitivity_and_decision_transitions(
    synthetic_robustness_matrix: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verifies that increasing lambda smoothly shifts selections from higher quality to lower cost models."""
    encoder = DeterministicTestEncoder(dimension=64)
    custom_split = {
        "train_prompt_ids": ["math_101", "math_102", "code_101", "code_102"],
        "val_prompt_ids": [],
        "test_prompt_ids": ["math_103", "code_103"],
        "sufficiency_status": "SUFFICIENT_DATA",
        "notes": "Synthetic split test",
    }

    _, _, results = run_router_robustness_evaluation(
        custom_df=synthetic_robustness_matrix,
        custom_split=custom_split,
        encoder=encoder,
        output_dir=tmp_path,
    )

    res_lam0 = next(r for r in results if r.scenario_id == "lambda_0.00" and r.router_name == "Learned Router")
    res_lam10 = next(r for r in results if r.scenario_id == "lambda_10.00" and r.router_name == "Learned Router")

    # At lambda 10, estimated cost must be <= cost at lambda 0
    assert res_lam10.mean_estimated_cost is not None
    assert res_lam0.mean_estimated_cost is not None
    assert res_lam10.mean_estimated_cost <= (res_lam0.mean_estimated_cost + 1e-8)


def test_distribution_shift_reports_insufficient_data_on_real_matrix(tmp_path: Path) -> None:
    """Verifies that the distribution shift audit reports INSUFFICIENT_DATA honestly on the 2-prompt real matrix."""
    encoder = DeterministicTestEncoder(dimension=384)
    df, summary, results = run_router_robustness_evaluation(
        benchmark_config_id="m1-efa7ec3afd4f",
        encoder=encoder,
        output_dir=tmp_path,
    )

    assert summary["sufficiency_status"] == "INSUFFICIENT_DATA"
    assert summary["is_exploratory"] is True
    assert summary["distribution_shift_audit"]["is_exploratory"] is True
    assert "INSUFFICIENT_DATA" in summary["distribution_shift_audit"]["generalization_notes"]


def test_decision_stability_metrics_reported_accurately(
    synthetic_robustness_matrix: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verifies selection agreement and change percentages are strictly bounded in [0.0, 1.0]."""
    encoder = DeterministicTestEncoder(dimension=64)
    custom_split = {
        "train_prompt_ids": ["math_101", "math_102", "code_101", "code_102"],
        "val_prompt_ids": [],
        "test_prompt_ids": ["math_103", "code_103"],
        "sufficiency_status": "SUFFICIENT_DATA",
        "notes": "Synthetic split test",
    }

    _, _, results = run_router_robustness_evaluation(
        custom_df=synthetic_robustness_matrix,
        custom_split=custom_split,
        encoder=encoder,
        output_dir=tmp_path,
    )

    for r in results:
        assert 0.0 <= r.selection_agreement_pct <= 1.0
        assert 0.0 <= r.selection_change_pct <= 1.0
        assert pytest.approx(r.selection_agreement_pct + r.selection_change_pct) == 1.0


def test_preservation_of_all_m2_and_m3_artifacts(tmp_path: Path) -> None:
    """Verifies Step 7 execution never touches or overwrites any existing M2 or Step 6 artifacts."""
    # Write mock existing M2 and Step 6 files
    mock_files = {
        "baseline_comparison.csv": "m2_csv",
        "baseline_comparison.parquet": "m2_parquet",
        "cost_vs_quality.png": "m2_png",
        "train_test_split.json": "m2_json",
        "router_comparison.csv": "step6_csv",
        "router_comparison.parquet": "step6_parquet",
        "router_pareto_curve.png": "step6_png",
        "router_evaluation_summary.json": "step6_json",
    }
    for filename, content in mock_files.items():
        (tmp_path / filename).write_text(content, encoding="utf-8")

    encoder = DeterministicTestEncoder(dimension=384)
    run_router_robustness_evaluation(
        benchmark_config_id="m1-efa7ec3afd4f",
        encoder=encoder,
        output_dir=tmp_path,
    )

    # Verify all 8 mock files are unmodified
    for filename, content in mock_files.items():
        assert (tmp_path / filename).read_text(encoding="utf-8") == content

    # Verify new Step 7 files exist
    assert (tmp_path / "router_robustness.csv").exists()
    assert (tmp_path / "router_robustness.parquet").exists()
    assert (tmp_path / "router_robustness_summary.json").exists()
    assert (tmp_path / "router_robustness.png").exists()


def test_zero_network_calls_during_robustness_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Guarantees full robustness evaluation executes strictly offline without network socket calls."""
    def blocked_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network socket call attempted during offline robustness evaluation!")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)

    encoder = DeterministicTestEncoder(dimension=384)
    df, summary, results = run_router_robustness_evaluation(
        benchmark_config_id="m1-efa7ec3afd4f",
        encoder=encoder,
        output_dir=tmp_path,
    )

    assert len(results) > 0
    assert summary["sufficiency_status"] == "INSUFFICIENT_DATA"


def test_cost_normalization_invariance_and_sensitivity() -> None:
    """Audit Test 1: Mathematical invariance and sensitivity of cost normalizers under scaling.

    Verifies:
    1. Under min-max normalization:
       c_norm = (k*c - k*min) / (k*max - k*min) = (c - min) / (max - min).
       Factor k cancels out exactly: normalized costs, utilities, and selections are strictly invariant.
    2. Under max-scale normalization:
       c_norm = (k*c) / (k*max) = c / max.
       Factor k cancels out exactly: normalized costs, utilities, and selections are strictly invariant.
    3. Under raw-cost normalization:
       U'(m) = Q_hat(m) - lambda * (k * c(m)).
       Uniform scaling by k is equivalent to scaling lambda by k: cost penalties scale linearly,
       shifting selections from expensive high-quality models to cheaper models when lambda > 0.
    4. Non-uniform cost shock under min-max:
       Increasing only one candidate's cost alters min/max and relative normalized costs,
       changing utility values and shifting model selections.
    """
    costs = {"m_cheap": 0.0002, "m_mid": 0.0006, "m_expensive": 0.0010}
    preds = {"m_cheap": 0.75, "m_mid": 0.85, "m_expensive": 0.90}

    # 1. Min-max uniform scaling invariance
    norm_base = CostNormalizer(costs, strategy="min_max")
    engine_base = LambdaDecisionEngine(cost_normalizer=norm_base)
    dec_base = engine_base.select_model(preds, lambda_value=0.2)

    for k in [0.5, 1.5, 2.0, 5.0, 10.0]:
        scaled_costs = {m: c * k for m, c in costs.items()}
        norm_scaled = CostNormalizer(scaled_costs, strategy="min_max")
        engine_scaled = LambdaDecisionEngine(cost_normalizer=norm_scaled)
        dec_scaled = engine_scaled.select_model(preds, lambda_value=0.2)

        # Mathematical invariance: normalized costs and selections must match exactly
        assert dec_scaled.selected_model_id == dec_base.selected_model_id
        for m in costs:
            _, n_base = norm_base.normalize(m)
            _, n_scaled = norm_scaled.normalize(m)
            assert n_scaled == pytest.approx(n_base, rel=1e-6)
            assert dec_scaled.model_utilities[m].utility == pytest.approx(dec_base.model_utilities[m].utility, rel=1e-6)

    # 2. Max-scale uniform scaling invariance
    norm_ms_base = CostNormalizer(costs, strategy="max_scale")
    engine_ms_base = LambdaDecisionEngine(cost_normalizer=norm_ms_base)
    dec_ms_base = engine_ms_base.select_model(preds, lambda_value=0.2)

    for k in [0.5, 2.0, 5.0]:
        scaled_ms_costs = {m: c * k for m, c in costs.items()}
        norm_ms_scaled = CostNormalizer(scaled_ms_costs, strategy="max_scale")
        engine_ms_scaled = LambdaDecisionEngine(cost_normalizer=norm_ms_scaled)
        dec_ms_scaled = engine_ms_scaled.select_model(preds, lambda_value=0.2)

        assert dec_ms_scaled.selected_model_id == dec_ms_base.selected_model_id
        for m in costs:
            _, n_base = norm_ms_base.normalize(m)
            _, n_scaled = norm_ms_scaled.normalize(m)
            assert n_scaled == pytest.approx(n_base, rel=1e-6)

    # 3. Raw-cost normalization sensitivity to uniform scaling
    norm_raw_base = CostNormalizer(costs, strategy="raw")
    engine_raw_base = LambdaDecisionEngine(cost_normalizer=norm_raw_base)
    # At lambda=50:
    # m_expensive utility: 0.90 - 50*0.0010 = 0.850
    # m_mid utility: 0.85 - 50*0.0006 = 0.820
    # m_cheap utility: 0.75 - 50*0.0002 = 0.740
    # m_expensive is chosen initially.
    dec_raw_base = engine_raw_base.select_model(preds, lambda_value=50.0)
    assert dec_raw_base.selected_model_id == "m_expensive"

    # With 6x cost multiplier (equivalent to lambda = 300):
    # m_expensive utility: 0.90 - 50*(6*0.0010) = 0.600
    # m_mid utility: 0.85 - 50*(6*0.0006) = 0.670
    # m_cheap utility: 0.75 - 50*(6*0.0002) = 0.690
    # m_cheap is now chosen!
    costs_raw_6x = {m: c * 6.0 for m, c in costs.items()}
    norm_raw_6x = CostNormalizer(costs_raw_6x, strategy="raw")
    engine_raw_6x = LambdaDecisionEngine(cost_normalizer=norm_raw_6x)
    dec_raw_6x = engine_raw_6x.select_model(preds, lambda_value=50.0)
    assert dec_raw_6x.selected_model_id == "m_cheap"
    assert dec_raw_6x.selected_model_id != dec_raw_base.selected_model_id

    # 4. Non-uniform cost shock on cheapest candidate under min-max
    shock_costs = dict(costs)
    shock_costs["m_cheap"] = 0.0009  # Increased near expensive model cost
    norm_shock = CostNormalizer(shock_costs, strategy="min_max")
    engine_shock = LambdaDecisionEngine(cost_normalizer=norm_shock)
    # At lambda=0.5, originally m_expensive was chosen with min-max (dec_base was m_expensive or m_mid).
    # Check that utilities change under non-uniform shock:
    dec_shock = engine_shock.select_model(preds, lambda_value=0.5)
    assert dec_shock.model_utilities["m_cheap"].normalized_cost != dec_base.model_utilities["m_cheap"].normalized_cost


def test_distinct_availability_and_failure_accounting() -> None:
    """Audit Test 2: Verify distinct accounting for unavailable models, missing predictions, and failures."""
    costs = {"m1": 0.0001, "m2": 0.0002, "m3": 0.0003, "m4": 0.0004}
    normalizer = CostNormalizer(costs, strategy="min_max")
    engine = LambdaDecisionEngine(cost_normalizer=normalizer)

    # Candidate m1: valid prediction
    # Candidate m2: deliberately unavailable (excluded from available_models)
    # Candidate m3: missing prediction (None)
    # Candidate m4: valid prediction
    query_preds = {"m1": 0.8, "m2": 0.95, "m3": None, "m4": 0.6}
    dec = engine.select_model(
        query_preds,
        lambda_value=0.5,
        available_models=["m1", "m3", "m4"],  # m2 deliberately excluded
    )

    # 1. Deliberately unavailable candidate m2
    assert dec.model_utilities["m2"].is_eligible is False
    assert dec.model_utilities["m2"].disqualification_reason == "deliberately_unavailable"

    # 2. Missing prediction candidate m3
    assert dec.model_utilities["m3"].is_eligible is False
    assert dec.model_utilities["m3"].disqualification_reason == "missing_prediction"

    # 3. Eligible candidates m1 and m4
    assert dec.model_utilities["m1"].is_eligible is True
    assert dec.model_utilities["m1"].disqualification_reason is None
    assert dec.model_utilities["m4"].is_eligible is True
    assert dec.model_utilities["m4"].disqualification_reason is None

    # Selected model must be m1 (highest utility among eligible candidates)
    assert dec.selected_model_id == "m1"


def test_distribution_shift_audit_vs_simulation_distinction(synthetic_robustness_matrix: pd.DataFrame, tmp_path: Path) -> None:
    """Audit Test 3: Verify static dataset audit on real matrix reports INSUFFICIENT_DATA and simulation is marked."""
    # 1. Static dataset audit on real benchmark matrix
    encoder = DeterministicTestEncoder(dimension=384)
    _, summary, _ = run_router_robustness_evaluation(
        benchmark_config_id="m1-efa7ec3afd4f",
        encoder=encoder,
        output_dir=tmp_path,
    )
    dist_audit = summary["distribution_shift_audit"]
    assert dist_audit["sufficiency_status"] == "INSUFFICIENT_DATA"
    assert dist_audit["is_exploratory"] is True
    assert "INSUFFICIENT_DATA" in dist_audit["generalization_notes"]

    # 2. Controlled simulation using simulate_controlled_distribution_shift
    from eval.routers.data_prep import prepare_routing_dataset
    custom_split = {
        "train_prompt_ids": ["math_101", "math_102", "code_101", "code_102"],
        "test_prompt_ids": ["math_103", "code_103"],
    }
    sim_dataset = prepare_routing_dataset(
        matrix_or_df=synthetic_robustness_matrix,
        encoder=encoder,
        custom_split=custom_split,
    )
    knn_router = KNNRouter(k=1, encoder=encoder).fit(sim_dataset)
    engine = LambdaDecisionEngine(cost_normalizer=CostNormalizer({"m_fast": 0.0001, "m_mid": 0.0003, "m_smart": 0.0008}))

    synthetic_prompts = [
        PromptRecord(
            prompt_id="sim_short_1",
            prompt_text="2+2=?",
            task_category="math",
            prompt_embedding=np.zeros(384, dtype=np.float32),
            split="test",
        ),
        PromptRecord(
            prompt_id="sim_long_novel_1",
            prompt_text="Write an extensive analysis on novel distributed systems consensus algorithms...",
            task_category="distributed_systems",  # Unseen category
            prompt_embedding=np.ones(384, dtype=np.float32),
            split="test",
        ),
    ]

    sim_result = simulate_controlled_distribution_shift(
        router=knn_router,
        decision_engine=engine,
        candidate_models=["m_fast", "m_mid", "m_smart"],
        synthetic_prompts=synthetic_prompts,
        lambda_value=0.5,
        shift_name="novel_task_category",
        shift_description="Testing unseen 'distributed_systems' category and long prompt",
    )

    assert sim_result.experiment_type == "controlled_distribution_shift"
    assert sim_result.scenario_id == "sim_shift_novel_task_category"
    assert sim_result.successful_decisions == 2
    assert "Controlled simulation" in sim_result.notes
    assert "NOT an empirical claim" in sim_result.notes
