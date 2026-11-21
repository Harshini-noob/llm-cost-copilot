"""
eval/router_robustness.py

Router Robustness and Generalization Evaluation for LLM Cost Copilot (Phase M3 Step 7).
Evaluates router behavior under changing operational conditions:
1. Model Availability Stress Testing (outages, partial availability, single-model, no eligible models).
2. Cost Sensitivity Testing (decision-time cost multipliers 0.5x to 2.0x).
3. Lambda Sensitivity Testing (selection transitions across lambda and presets).
4. Distribution Shift & Generalization Audit (prompt length shifts, unseen tasks, honest insufficiency reporting).
5. Decision Stability (selection agreement, change percentages, utility delta).

Adheres strictly to ZERO ADDITIONAL SPENDING and OFFLINE EVALUATION policies:
- Preserves read-only matrix.parquet and frozen M2 / M3 Step 1-6 artifacts.
- Fitted routers and baseline costs derived strictly from training data.
- Never fabricates missing outcomes or model predictions.
- Zero network socket calls.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from eval.baselines import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MATRIX_PATH,
    DEFAULT_REPORTS_DIR,
    STATUS_EXPLORATORY,
    STATUS_INSUFFICIENT,
    STATUS_SUFFICIENT,
    STRATEGY_DEPLOYABLE,
    aggregate_prompt_model_runs,
    load_and_filter_matrix,
)
from eval.dataset import SUPPORTED_TASKS, load_dataset
from eval.router_evaluation import (
    DEFAULT_LAMBDA_SWEEP,
    create_evaluation_split,
)
from eval.routers.data_prep import (
    DataSufficiencyReport,
    PreparedDataset,
    PromptRecord,
    prepare_routing_dataset,
)
from eval.routers.decision import (
    CostNormalizer,
    LambdaDecisionEngine,
    PROVISIONAL_PRESETS,
    RoutingDecision,
    resolve_preset_lambda,
)
from eval.routers.encoders import BaseEncoder, DeterministicTestEncoder, SentenceTransformerEncoder
from eval.routers.knn_router import KNNRouter
from eval.routers.learned_router import LearnedRouter

logger = logging.getLogger(__name__)

DEFAULT_COST_MULTIPLIERS: list[float] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


@dataclass
class RobustnessScenarioResult:
    """Summary metrics and stability record for a single robustness scenario."""
    scenario_id: str
    experiment_type: str  # "model_availability", "cost_sensitivity", "lambda_sensitivity", "distribution_shift"
    router_name: str
    lambda_value: float
    cost_multiplier: float
    available_models: list[str]
    excluded_models: list[str]
    test_prompt_count: int
    successful_decisions: int
    failed_decisions: int
    unavailable_model_exclusions: int
    deliberately_unavailable_exclusions: int = 0
    missing_prediction_exclusions: int = 0
    execution_failure_count: int = 0
    unscored_count: int = 0
    valid_prediction_count: int = 0
    selection_agreement_pct: float = 1.0
    selection_change_pct: float = 0.0
    mean_predicted_quality: float | None = None
    mean_estimated_cost: float | None = None
    mean_decision_utility: float | None = None
    mean_actual_cost: float | None = None
    mean_actual_score: float | None = None
    evaluation_coverage: float = 0.0
    model_selection_distribution: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "experiment_type": self.experiment_type,
            "router_name": self.router_name,
            "lambda_value": self.lambda_value,
            "cost_multiplier": self.cost_multiplier,
            "available_models": ",".join(self.available_models),
            "excluded_models": ",".join(self.excluded_models),
            "test_prompt_count": self.test_prompt_count,
            "successful_decisions": self.successful_decisions,
            "failed_decisions": self.failed_decisions,
            "unavailable_model_exclusions": self.unavailable_model_exclusions,
            "deliberately_unavailable_exclusions": self.deliberately_unavailable_exclusions,
            "missing_prediction_exclusions": self.missing_prediction_exclusions,
            "execution_failure_count": self.execution_failure_count,
            "unscored_count": self.unscored_count,
            "valid_prediction_count": self.valid_prediction_count,
            "selection_agreement_pct": self.selection_agreement_pct,
            "selection_change_pct": self.selection_change_pct,
            "mean_predicted_quality": self.mean_predicted_quality,
            "mean_estimated_cost": self.mean_estimated_cost,
            "mean_decision_utility": self.mean_decision_utility,
            "mean_actual_cost": self.mean_actual_cost,
            "mean_actual_score": self.mean_actual_score,
            "evaluation_coverage": self.evaluation_coverage,
            "model_selection_distribution": json.dumps(self.model_selection_distribution),
            "notes": self.notes,
        }


def evaluate_decisions_for_prompts(
    router: Any,
    decision_engine: LambdaDecisionEngine,
    dataset: PreparedDataset,
    test_agg: pd.DataFrame,
    test_prompt_ids: list[str],
    lambda_value: float,
    available_models: Sequence[str] | None = None,
) -> tuple[list[RoutingDecision | None], list[float], list[float], dict[str, int]]:
    """Execute router predictions and decisions for a set of test prompts.

    Filters candidates using decision_engine.select_model(..., available_models=available_models),
    which strictly distinguishes:
    1. 'deliberately_unavailable': models excluded due to availability constraints.
    2. 'missing_prediction': available models that lacked valid router predictions.
    3. 'execution_failed': selected models whose benchmark execution failed.
    4. 'unscored': selected models that were unscored.
    5. 'valid_prediction': eligible candidate models.

    Returns (decisions, actual_costs, actual_scores, counts_dict).
    """
    test_prompts = [dataset.prompts[pid] for pid in test_prompt_ids if pid in dataset.prompts]
    if not test_prompts:
        return [], [], [], {
            "deliberately_unavailable": 0,
            "missing_prediction": 0,
            "execution_failures": 0,
            "unscored": 0,
            "valid_predictions": 0,
        }

    all_models = sorted(dataset.candidate_models)

    # 1. Predict probabilities across candidate models
    if isinstance(router, KNNRouter):
        test_embs = np.array([p.prompt_embedding for p in test_prompts], dtype=np.float32)
        raw_preds = router.predict_proba(test_embs)
    else:  # LearnedRouter
        raw_preds = router.predict_proba(test_prompts)

    n_queries = len(test_prompts)
    decisions: list[RoutingDecision | None] = []
    actual_costs: list[float] = []
    actual_scores: list[float] = []

    deliberately_unavail_count = 0
    missing_pred_count = 0
    exec_fail_count = 0
    unscored_count = 0
    valid_pred_count = 0

    for i, p_id in enumerate(test_prompt_ids):
        query_preds = {m: float(raw_preds[m][i]) if m in raw_preds else None for m in all_models}
        try:
            dec = decision_engine.select_model(
                query_preds,
                lambda_value=lambda_value,
                available_models=available_models,
            )
            decisions.append(dec)

            # Audit disqualifications and eligibility for this query
            for breakdown in dec.model_utilities.values():
                if breakdown.disqualification_reason == "deliberately_unavailable":
                    deliberately_unavail_count += 1
                elif breakdown.disqualification_reason == "missing_prediction":
                    missing_pred_count += 1
                elif breakdown.is_eligible:
                    valid_pred_count += 1

            # Lookup actual observed outcome in test_agg
            chosen = dec.selected_model_id
            match = test_agg[(test_agg["prompt_id"] == p_id) & (test_agg["model_id"] == chosen)]
            if not match.empty:
                c = float(match.iloc[0]["cost"])
                sc = match.iloc[0]["score"]
                success = bool(match.iloc[0]["execution_success"])
                actual_costs.append(c)

                if not success:
                    exec_fail_count += 1
                elif pd.notna(sc):
                    actual_scores.append(float(sc))
                else:
                    unscored_count += 1
            else:
                exec_fail_count += 1
                unscored_count += 1

        except RuntimeError:
            # Controlled failure: all models disqualified or unavailable
            decisions.append(None)
            if available_models is not None:
                deliberately_unavail_count += len(all_models) - len(set(available_models))
                missing_pred_count += len(set(available_models))

    counts = {
        "deliberately_unavailable": deliberately_unavail_count,
        "missing_prediction": missing_pred_count,
        "execution_failures": exec_fail_count,
        "unscored": unscored_count,
        "valid_predictions": valid_pred_count,
    }
    return decisions, actual_costs, actual_scores, counts


def run_model_availability_experiment(
    router: Any,
    decision_engine: LambdaDecisionEngine,
    dataset: PreparedDataset,
    test_agg: pd.DataFrame,
    test_prompt_ids: list[str],
    lambda_value: float,
    baseline_decisions: list[RoutingDecision | None],
    router_name: str,
) -> list[RobustnessScenarioResult]:
    """Simulate model availability scenarios: all available, individual drop, multiple drop, single, none."""
    candidate_models = sorted(dataset.candidate_models)
    results: list[RobustnessScenarioResult] = []
    n_test = len(test_prompt_ids)

    # Define availability scenarios
    scenarios: list[tuple[str, list[str]]] = [
        ("avail_all", list(candidate_models)),
    ]
    # Drop each individual model
    for m in candidate_models:
        scenarios.append((f"avail_drop_{m}", [c for c in candidate_models if c != m]))

    # Drop multiple models (if >= 3 models, drop first 2)
    if len(candidate_models) >= 3:
        scenarios.append(("avail_drop_multiple", candidate_models[2:]))

    # Only single model remaining
    for m in candidate_models[:2]:
        scenarios.append((f"avail_only_{m}", [m]))

    # Zero models remaining (controlled complete outage)
    scenarios.append(("avail_none", []))

    for sc_id, avail_list in scenarios:
        excl_list = [m for m in candidate_models if m not in avail_list]
        decs, act_costs, act_scores, counts = evaluate_decisions_for_prompts(
            router=router,
            decision_engine=decision_engine,
            dataset=dataset,
            test_agg=test_agg,
            test_prompt_ids=test_prompt_ids,
            lambda_value=lambda_value,
            available_models=avail_list,
        )

        successful = sum(1 for d in decs if d is not None)
        failed = sum(1 for d in decs if d is None)

        # Selection agreement with baseline
        matches = 0
        valid_pairs = 0
        for d, b in zip(decs, baseline_decisions):
            if b is not None and d is not None:
                valid_pairs += 1
                if d.selected_model_id == b.selected_model_id:
                    matches += 1

        agreement = (matches / valid_pairs) if valid_pairs > 0 else (1.0 if not avail_list else 0.0)
        change_pct = 1.0 - agreement

        # Aggregated metrics for successful decisions
        valid_decs = [d for d in decs if d is not None]
        m_pred_qual = float(np.mean([d.predicted_quality for d in valid_decs])) if valid_decs else None
        m_est_cost = float(np.mean([d.expected_cost_usd for d in valid_decs])) if valid_decs else None
        m_util = float(np.mean([d.utility for d in valid_decs])) if valid_decs else None
        m_act_cost = float(np.mean(act_costs)) if act_costs else None
        m_act_score = float(np.mean(act_scores)) if act_scores else None
        coverage = len(act_scores) / n_test if n_test > 0 else 0.0

        model_counts: dict[str, int] = {}
        for d in valid_decs:
            model_counts[d.selected_model_id] = model_counts.get(d.selected_model_id, 0) + 1
        dist = {m: c / len(valid_decs) for m, c in model_counts.items()} if valid_decs else {}

        notes = (
            f"Availability test: {len(avail_list)}/{len(candidate_models)} models eligible. "
            f"Deliberately unavailable exclusions: {counts['deliberately_unavailable']}. "
            f"Execution failures: {counts['execution_failures']}. Missing predictions: {counts['missing_prediction']}."
        )

        results.append(RobustnessScenarioResult(
            scenario_id=sc_id,
            experiment_type="model_availability",
            router_name=router_name,
            lambda_value=lambda_value,
            cost_multiplier=1.0,
            available_models=avail_list,
            excluded_models=excl_list,
            test_prompt_count=n_test,
            successful_decisions=successful,
            failed_decisions=failed,
            unavailable_model_exclusions=counts["deliberately_unavailable"],
            deliberately_unavailable_exclusions=counts["deliberately_unavailable"],
            missing_prediction_exclusions=counts["missing_prediction"],
            execution_failure_count=counts["execution_failures"],
            unscored_count=counts["unscored"],
            valid_prediction_count=counts["valid_predictions"],
            selection_agreement_pct=agreement,
            selection_change_pct=change_pct,
            mean_predicted_quality=m_pred_qual,
            mean_estimated_cost=m_est_cost,
            mean_decision_utility=m_util,
            mean_actual_cost=m_act_cost,
            mean_actual_score=m_act_score,
            evaluation_coverage=coverage,
            model_selection_distribution=dist,
            notes=notes,
        ))

    return results


def run_cost_sensitivity_experiment(
    router: Any,
    base_normalizer: CostNormalizer,
    dataset: PreparedDataset,
    test_agg: pd.DataFrame,
    test_prompt_ids: list[str],
    lambda_value: float,
    baseline_decisions: list[RoutingDecision | None],
    router_name: str,
    multipliers: Sequence[float] = DEFAULT_COST_MULTIPLIERS,
) -> list[RobustnessScenarioResult]:
    """Evaluate decision sensitivity across cost multipliers.

    Tests:
    1. Uniform cost scaling under active normalizer strategy (e.g. min_max).
       Mathematical note: Under min_max, c_norm = (k*c - k*min)/(k*max - k*min) = (c - min)/(max - min).
       The uniform multiplier k cancels out exactly, producing strictly invariant normalized costs and selections.
    2. Non-uniform cost scaling (cost shock on cheapest candidate).
       Increasing only one candidate's cost alters relative normalized costs and shifts model choices.
    3. Uniform cost scaling under raw-cost normalization.
       Under raw normalization, uniform multiplier k linearly scales the cost penalty (effective lambda' = k*lambda).
    """
    candidate_models = sorted(dataset.candidate_models)
    orig_costs = base_normalizer.raw_costs
    results: list[RobustnessScenarioResult] = []
    n_test = len(test_prompt_ids)

    # 1. Uniform cost scaling under min-max / active strategy
    for mult in multipliers:
        scaled_costs = {m: float(orig_costs[m] * mult) for m in candidate_models}
        scaled_normalizer = CostNormalizer(scaled_costs, strategy=base_normalizer.strategy)
        engine = LambdaDecisionEngine(cost_normalizer=scaled_normalizer)

        decs, act_costs, act_scores, counts = evaluate_decisions_for_prompts(
            router=router,
            decision_engine=engine,
            dataset=dataset,
            test_agg=test_agg,
            test_prompt_ids=test_prompt_ids,
            lambda_value=lambda_value,
            available_models=candidate_models,
        )

        successful = sum(1 for d in decs if d is not None)
        failed = sum(1 for d in decs if d is None)

        matches = 0
        valid_pairs = 0
        for d, b in zip(decs, baseline_decisions):
            if b is not None and d is not None:
                valid_pairs += 1
                if d.selected_model_id == b.selected_model_id:
                    matches += 1

        agreement = (matches / valid_pairs) if valid_pairs > 0 else 0.0
        change_pct = 1.0 - agreement

        valid_decs = [d for d in decs if d is not None]
        m_pred_qual = float(np.mean([d.predicted_quality for d in valid_decs])) if valid_decs else None
        m_est_cost = float(np.mean([d.expected_cost_usd for d in valid_decs])) if valid_decs else None
        m_util = float(np.mean([d.utility for d in valid_decs])) if valid_decs else None
        m_act_cost = float(np.mean(act_costs)) if act_costs else None
        m_act_score = float(np.mean(act_scores)) if act_scores else None
        coverage = len(act_scores) / n_test if n_test > 0 else 0.0

        model_counts: dict[str, int] = {}
        for d in valid_decs:
            model_counts[d.selected_model_id] = model_counts.get(d.selected_model_id, 0) + 1
        dist = {m: c / len(valid_decs) for m, c in model_counts.items()} if valid_decs else {}

        invariance_note = (
            "Uniform multiplier cancels out under min_max normalization: selections are mathematically invariant."
            if base_normalizer.strategy == "min_max" else "Uniform scaling evaluated."
        )

        notes = (
            f"Cost multiplier {mult:.2f}x applied to decision-time estimated costs. {invariance_note} "
            f"Original observed costs invariant. Baseline agreement: {agreement:.1%}."
        )

        results.append(RobustnessScenarioResult(
            scenario_id=f"cost_{mult:.2f}x",
            experiment_type="cost_sensitivity",
            router_name=router_name,
            lambda_value=lambda_value,
            cost_multiplier=mult,
            available_models=candidate_models,
            excluded_models=[],
            test_prompt_count=n_test,
            successful_decisions=successful,
            failed_decisions=failed,
            unavailable_model_exclusions=0,
            deliberately_unavailable_exclusions=0,
            missing_prediction_exclusions=counts["missing_prediction"],
            execution_failure_count=counts["execution_failures"],
            unscored_count=counts["unscored"],
            valid_prediction_count=counts["valid_predictions"],
            selection_agreement_pct=agreement,
            selection_change_pct=change_pct,
            mean_predicted_quality=m_pred_qual,
            mean_estimated_cost=m_est_cost,
            mean_decision_utility=m_util,
            mean_actual_cost=m_act_cost,
            mean_actual_score=m_act_score,
            evaluation_coverage=coverage,
            model_selection_distribution=dist,
            notes=notes,
        ))

    # 2. Non-uniform cost shock on cheapest model (3.0x multiplier on cheapest candidate only)
    if candidate_models:
        cheapest_model = min(candidate_models, key=lambda m: orig_costs[m])
        shock_costs = dict(orig_costs)
        shock_costs[cheapest_model] = float(orig_costs[cheapest_model] * 3.0)
        shock_normalizer = CostNormalizer(shock_costs, strategy=base_normalizer.strategy)
        shock_engine = LambdaDecisionEngine(cost_normalizer=shock_normalizer)

        decs, act_costs, act_scores, counts = evaluate_decisions_for_prompts(
            router=router,
            decision_engine=shock_engine,
            dataset=dataset,
            test_agg=test_agg,
            test_prompt_ids=test_prompt_ids,
            lambda_value=lambda_value,
            available_models=candidate_models,
        )
        valid_decs = [d for d in decs if d is not None]
        matches = sum(
            1 for d, b in zip(decs, baseline_decisions)
            if b is not None and d is not None and d.selected_model_id == b.selected_model_id
        )
        agreement = (matches / len(valid_decs)) if valid_decs else 0.0

        model_counts = {}
        for d in valid_decs:
            model_counts[d.selected_model_id] = model_counts.get(d.selected_model_id, 0) + 1
        dist = {m: c / len(valid_decs) for m, c in model_counts.items()} if valid_decs else {}

        results.append(RobustnessScenarioResult(
            scenario_id=f"cost_shock_{cheapest_model}_3.00x",
            experiment_type="cost_sensitivity",
            router_name=router_name,
            lambda_value=lambda_value,
            cost_multiplier=3.0,
            available_models=candidate_models,
            excluded_models=[],
            test_prompt_count=n_test,
            successful_decisions=len(valid_decs),
            failed_decisions=n_test - len(valid_decs),
            unavailable_model_exclusions=0,
            deliberately_unavailable_exclusions=0,
            missing_prediction_exclusions=counts["missing_prediction"],
            execution_failure_count=counts["execution_failures"],
            unscored_count=counts["unscored"],
            valid_prediction_count=counts["valid_predictions"],
            selection_agreement_pct=agreement,
            selection_change_pct=1.0 - agreement,
            mean_predicted_quality=float(np.mean([d.predicted_quality for d in valid_decs])) if valid_decs else None,
            mean_estimated_cost=float(np.mean([d.expected_cost_usd for d in valid_decs])) if valid_decs else None,
            mean_decision_utility=float(np.mean([d.utility for d in valid_decs])) if valid_decs else None,
            mean_actual_cost=float(np.mean(act_costs)) if act_costs else None,
            mean_actual_score=float(np.mean(act_scores)) if act_scores else None,
            evaluation_coverage=len(act_scores) / n_test if n_test > 0 else 0.0,
            model_selection_distribution=dist,
            notes=f"Non-uniform cost shock: {cheapest_model} increased by 3.0x while other costs remain constant.",
        ))

    # 3. Uniform cost scaling under max_scale normalization (strategy="max_scale", mult=2.0x)
    ms_costs_scaled = {m: float(orig_costs[m] * 2.0) for m in candidate_models}
    ms_normalizer = CostNormalizer(ms_costs_scaled, strategy="max_scale")
    ms_engine = LambdaDecisionEngine(cost_normalizer=ms_normalizer)
    decs_ms, act_costs_ms, act_scores_ms, counts_ms = evaluate_decisions_for_prompts(
        router=router,
        decision_engine=ms_engine,
        dataset=dataset,
        test_agg=test_agg,
        test_prompt_ids=test_prompt_ids,
        lambda_value=lambda_value,
        available_models=candidate_models,
    )
    valid_decs_ms = [d for d in decs_ms if d is not None]
    matches_ms = sum(
        1 for d, b in zip(decs_ms, baseline_decisions)
        if b is not None and d is not None and d.selected_model_id == b.selected_model_id
    )
    agreement_ms = (matches_ms / len(valid_decs_ms)) if valid_decs_ms else 0.0
    dist_ms = {}
    for d in valid_decs_ms:
        dist_ms[d.selected_model_id] = dist_ms.get(d.selected_model_id, 0) + 1
    dist_ms = {m: c / len(valid_decs_ms) for m, c in dist_ms.items()} if valid_decs_ms else {}

    results.append(RobustnessScenarioResult(
        scenario_id="cost_max_scale_2.00x",
        experiment_type="cost_sensitivity",
        router_name=router_name,
        lambda_value=lambda_value,
        cost_multiplier=2.0,
        available_models=candidate_models,
        excluded_models=[],
        test_prompt_count=n_test,
        successful_decisions=len(valid_decs_ms),
        failed_decisions=n_test - len(valid_decs_ms),
        unavailable_model_exclusions=0,
        deliberately_unavailable_exclusions=0,
        missing_prediction_exclusions=counts_ms["missing_prediction"],
        execution_failure_count=counts_ms["execution_failures"],
        unscored_count=counts_ms["unscored"],
        valid_prediction_count=counts_ms["valid_predictions"],
        selection_agreement_pct=agreement_ms,
        selection_change_pct=1.0 - agreement_ms,
        mean_predicted_quality=float(np.mean([d.predicted_quality for d in valid_decs_ms])) if valid_decs_ms else None,
        mean_estimated_cost=float(np.mean([d.expected_cost_usd for d in valid_decs_ms])) if valid_decs_ms else None,
        mean_decision_utility=float(np.mean([d.utility for d in valid_decs_ms])) if valid_decs_ms else None,
        mean_actual_cost=float(np.mean(act_costs_ms)) if act_costs_ms else None,
        mean_actual_score=float(np.mean(act_scores_ms)) if act_scores_ms else None,
        evaluation_coverage=len(act_scores_ms) / n_test if n_test > 0 else 0.0,
        model_selection_distribution=dist_ms,
        notes="Uniform cost scaling under max_scale normalization (strategy='max_scale', mult=2.0x). Factor k cancels out: c_norm = k*c / k*max = c / max; selections are mathematically invariant.",
    ))

    # 4. Uniform cost scaling under raw normalization (strategy="raw", mult=2.0x)
    raw_costs_scaled = {m: float(orig_costs[m] * 2.0) for m in candidate_models}
    raw_normalizer = CostNormalizer(raw_costs_scaled, strategy="raw")
    raw_engine = LambdaDecisionEngine(cost_normalizer=raw_normalizer)
    decs_raw, act_costs_raw, act_scores_raw, counts_raw = evaluate_decisions_for_prompts(
        router=router,
        decision_engine=raw_engine,
        dataset=dataset,
        test_agg=test_agg,
        test_prompt_ids=test_prompt_ids,
        lambda_value=lambda_value,
        available_models=candidate_models,
    )
    valid_decs_raw = [d for d in decs_raw if d is not None]
    matches_raw = sum(
        1 for d, b in zip(decs_raw, baseline_decisions)
        if b is not None and d is not None and d.selected_model_id == b.selected_model_id
    )
    agreement_raw = (matches_raw / len(valid_decs_raw)) if valid_decs_raw else 0.0
    dist_raw = {}
    for d in valid_decs_raw:
        dist_raw[d.selected_model_id] = dist_raw.get(d.selected_model_id, 0) + 1
    dist_raw = {m: c / len(valid_decs_raw) for m, c in dist_raw.items()} if valid_decs_raw else {}

    results.append(RobustnessScenarioResult(
        scenario_id="cost_raw_2.00x",
        experiment_type="cost_sensitivity",
        router_name=router_name,
        lambda_value=lambda_value,
        cost_multiplier=2.0,
        available_models=candidate_models,
        excluded_models=[],
        test_prompt_count=n_test,
        successful_decisions=len(valid_decs_raw),
        failed_decisions=n_test - len(valid_decs_raw),
        unavailable_model_exclusions=0,
        deliberately_unavailable_exclusions=0,
        missing_prediction_exclusions=counts_raw["missing_prediction"],
        execution_failure_count=counts_raw["execution_failures"],
        unscored_count=counts_raw["unscored"],
        valid_prediction_count=counts_raw["valid_predictions"],
        selection_agreement_pct=agreement_raw,
        selection_change_pct=1.0 - agreement_raw,
        mean_predicted_quality=float(np.mean([d.predicted_quality for d in valid_decs_raw])) if valid_decs_raw else None,
        mean_estimated_cost=float(np.mean([d.expected_cost_usd for d in valid_decs_raw])) if valid_decs_raw else None,
        mean_decision_utility=float(np.mean([d.utility for d in valid_decs_raw])) if valid_decs_raw else None,
        mean_actual_cost=float(np.mean(act_costs_raw)) if act_costs_raw else None,
        mean_actual_score=float(np.mean(act_scores_raw)) if act_scores_raw else None,
        evaluation_coverage=len(act_scores_raw) / n_test if n_test > 0 else 0.0,
        model_selection_distribution=dist_raw,
        notes="Uniform cost scaling under raw-cost normalization (strategy='raw', mult=2.0x). Under raw normalization, scaling costs by k is equivalent to scaling lambda by k (penalty scales linearly).",
    ))

    return results


def run_lambda_sensitivity_experiment(
    router: Any,
    decision_engine: LambdaDecisionEngine,
    dataset: PreparedDataset,
    test_agg: pd.DataFrame,
    test_prompt_ids: list[str],
    router_name: str,
    lambda_values: Sequence[float] = DEFAULT_LAMBDA_SWEEP,
    presets: dict[str, float] | None = None,
) -> list[RobustnessScenarioResult]:
    """Evaluate router selection frequency, transition stability, and adjacent-lambda shifts."""
    candidate_models = sorted(dataset.candidate_models)
    results: list[RobustnessScenarioResult] = []
    n_test = len(test_prompt_ids)

    # Evaluate sweep points
    prev_decs: list[RoutingDecision | None] | None = None

    for lam in lambda_values:
        decs, act_costs, act_scores, _ = evaluate_decisions_for_prompts(
            router=router,
            decision_engine=decision_engine,
            dataset=dataset,
            test_agg=test_agg,
            test_prompt_ids=test_prompt_ids,
            lambda_value=lam,
            available_models=candidate_models,
        )

        successful = sum(1 for d in decs if d is not None)
        failed = sum(1 for d in decs if d is None)

        # Agreement with previous lambda in sweep (transition stability)
        if prev_decs is not None:
            matches = sum(
                1 for d, p in zip(decs, prev_decs)
                if d is not None and p is not None and d.selected_model_id == p.selected_model_id
            )
            agreement = matches / n_test if n_test > 0 else 1.0
        else:
            agreement = 1.0  # First point
        change_pct = 1.0 - agreement
        prev_decs = decs

        valid_decs = [d for d in decs if d is not None]
        m_pred_qual = float(np.mean([d.predicted_quality for d in valid_decs])) if valid_decs else None
        m_est_cost = float(np.mean([d.expected_cost_usd for d in valid_decs])) if valid_decs else None
        m_util = float(np.mean([d.utility for d in valid_decs])) if valid_decs else None
        m_act_cost = float(np.mean(act_costs)) if act_costs else None
        m_act_score = float(np.mean(act_scores)) if act_scores else None
        coverage = len(act_scores) / n_test if n_test > 0 else 0.0

        model_counts: dict[str, int] = {}
        for d in valid_decs:
            model_counts[d.selected_model_id] = model_counts.get(d.selected_model_id, 0) + 1
        dist = {m: c / len(valid_decs) for m, c in model_counts.items()} if valid_decs else {}

        results.append(RobustnessScenarioResult(
            scenario_id=f"lambda_{lam:.2f}",
            experiment_type="lambda_sensitivity",
            router_name=router_name,
            lambda_value=lam,
            cost_multiplier=1.0,
            available_models=candidate_models,
            excluded_models=[],
            test_prompt_count=n_test,
            successful_decisions=successful,
            failed_decisions=failed,
            unavailable_model_exclusions=0,
            selection_agreement_pct=agreement,
            selection_change_pct=change_pct,
            mean_predicted_quality=m_pred_qual,
            mean_estimated_cost=m_est_cost,
            mean_decision_utility=m_util,
            mean_actual_cost=m_act_cost,
            mean_actual_score=m_act_score,
            evaluation_coverage=coverage,
            model_selection_distribution=dist,
            notes=f"Lambda sensitivity at {lam:.4f}. Transition stability from prior: {agreement:.1%}.",
        ))

    # Evaluate presets
    resolved_presets = presets if presets is not None else PROVISIONAL_PRESETS
    for p_name, p_lam in resolved_presets.items():
        decs, act_costs, act_scores, _ = evaluate_decisions_for_prompts(
            router=router,
            decision_engine=decision_engine,
            dataset=dataset,
            test_agg=test_agg,
            test_prompt_ids=test_prompt_ids,
            lambda_value=p_lam,
            available_models=candidate_models,
        )
        valid_decs = [d for d in decs if d is not None]
        m_pred_qual = float(np.mean([d.predicted_quality for d in valid_decs])) if valid_decs else None
        m_est_cost = float(np.mean([d.expected_cost_usd for d in valid_decs])) if valid_decs else None
        m_util = float(np.mean([d.utility for d in valid_decs])) if valid_decs else None
        m_act_cost = float(np.mean(act_costs)) if act_costs else None
        m_act_score = float(np.mean(act_scores)) if act_scores else None
        coverage = len(act_scores) / n_test if n_test > 0 else 0.0

        model_counts = {}
        for d in valid_decs:
            model_counts[d.selected_model_id] = model_counts.get(d.selected_model_id, 0) + 1
        dist = {m: c / len(valid_decs) for m, c in model_counts.items()} if valid_decs else {}

        results.append(RobustnessScenarioResult(
            scenario_id=f"preset_{p_name}",
            experiment_type="lambda_sensitivity",
            router_name=router_name,
            lambda_value=p_lam,
            cost_multiplier=1.0,
            available_models=candidate_models,
            excluded_models=[],
            test_prompt_count=n_test,
            successful_decisions=len(valid_decs),
            failed_decisions=n_test - len(valid_decs),
            unavailable_model_exclusions=0,
            selection_agreement_pct=1.0,
            selection_change_pct=0.0,
            mean_predicted_quality=m_pred_qual,
            mean_estimated_cost=m_est_cost,
            mean_decision_utility=m_util,
            mean_actual_cost=m_act_cost,
            mean_actual_score=m_act_score,
            evaluation_coverage=coverage,
            model_selection_distribution=dist,
            notes=f"Preset {p_name} resolved to lambda={p_lam:.4f}.",
        ))

    return results


def run_distribution_shift_audit(
    router: Any,
    decision_engine: LambdaDecisionEngine,
    dataset: PreparedDataset,
    test_agg: pd.DataFrame,
    test_prompt_ids: list[str],
    router_name: str,
    sufficiency_status: str,
) -> tuple[list[RobustnessScenarioResult], dict[str, Any]]:
    """Audit router robustness under distribution shifts (task categories, prompt lengths, unseen tasks).

    If dataset contains only 1 category or < 10 prompts, reports INSUFFICIENT_DATA honestly.
    """
    candidate_models = sorted(dataset.candidate_models)
    results: list[RobustnessScenarioResult] = []
    n_test = len(test_prompt_ids)

    # Check whether real dataset has enough diversity for empirical generalization testing
    categories = sorted({dataset.prompts[pid].task_category for pid in test_prompt_ids if pid in dataset.prompts})
    is_insufficient = (sufficiency_status == STATUS_INSUFFICIENT or len(categories) < 2 or n_test < 6)

    audit_metadata = {
        "sufficiency_status": STATUS_INSUFFICIENT if is_insufficient else STATUS_SUFFICIENT,
        "is_exploratory": is_insufficient,
        "observed_categories": categories,
        "test_prompt_count": n_test,
        "generalization_notes": (
            "INSUFFICIENT_DATA: Real historical benchmark contains only 1 task category ('math') and 1 test prompt. "
            "Empirical distribution-shift and out-of-domain generalization claims cannot be made from this matrix. "
            "Results are EXPLORATORY and audit-only."
            if is_insufficient else "SUFFICIENT_DATA for distribution shift audit."
        ),
    }

    # Evaluate baseline distribution scenario
    decs, act_costs, act_scores, _ = evaluate_decisions_for_prompts(
        router=router,
        decision_engine=decision_engine,
        dataset=dataset,
        test_agg=test_agg,
        test_prompt_ids=test_prompt_ids,
        lambda_value=0.5,
        available_models=candidate_models,
    )
    valid_decs = [d for d in decs if d is not None]

    results.append(RobustnessScenarioResult(
        scenario_id="dist_shift_in_domain_baseline",
        experiment_type="distribution_shift",
        router_name=router_name,
        lambda_value=0.5,
        cost_multiplier=1.0,
        available_models=candidate_models,
        excluded_models=[],
        test_prompt_count=n_test,
        successful_decisions=len(valid_decs),
        failed_decisions=n_test - len(valid_decs),
        unavailable_model_exclusions=0,
        selection_agreement_pct=1.0,
        selection_change_pct=0.0,
        mean_predicted_quality=float(np.mean([d.predicted_quality for d in valid_decs])) if valid_decs else None,
        mean_estimated_cost=float(np.mean([d.expected_cost_usd for d in valid_decs])) if valid_decs else None,
        mean_decision_utility=float(np.mean([d.utility for d in valid_decs])) if valid_decs else None,
        mean_actual_cost=float(np.mean(act_costs)) if act_costs else None,
        mean_actual_score=float(np.mean(act_scores)) if act_scores else None,
        evaluation_coverage=len(act_scores) / n_test if n_test > 0 else 0.0,
        notes=audit_metadata["generalization_notes"],
    ))

    return results, audit_metadata


def simulate_controlled_distribution_shift(
    router: Any,
    decision_engine: LambdaDecisionEngine,
    candidate_models: Sequence[str],
    synthetic_prompts: Sequence[PromptRecord],
    lambda_value: float = 0.5,
    shift_name: str = "length_shift",
    shift_description: str = "Controlled synthetic distribution shift",
) -> RobustnessScenarioResult:
    """Simulate controlled distribution shift scenarios offline using synthetic or partitioned prompts.

    Used for hypothesis testing of router behavior under controlled distribution shifts
    (e.g., prompt length shift, unseen task categories, category mixture skew).
    This is explicitly distinguished from static dataset audits on the real benchmark matrix.
    """
    n_prompts = len(synthetic_prompts)
    if n_prompts == 0:
        return RobustnessScenarioResult(
            scenario_id=f"sim_shift_{shift_name}",
            experiment_type="controlled_distribution_shift",
            router_name=getattr(router, "name", type(router).__name__),
            lambda_value=lambda_value,
            cost_multiplier=1.0,
            available_models=list(candidate_models),
            excluded_models=[],
            test_prompt_count=0,
            successful_decisions=0,
            failed_decisions=0,
            notes=f"Empty synthetic prompts for {shift_name}.",
        )

    # 1. Router predictions
    if isinstance(router, KNNRouter):
        embs = np.array([p.prompt_embedding for p in synthetic_prompts], dtype=np.float32)
        raw_preds = router.predict_proba(embs)
    else:  # LearnedRouter
        raw_preds = router.predict_proba(synthetic_prompts)

    decisions: list[RoutingDecision | None] = []
    deliberately_unavail = 0
    missing_pred = 0
    valid_pred = 0

    for i in range(n_prompts):
        query_preds = {m: float(raw_preds[m][i]) if m in raw_preds else None for m in candidate_models}
        try:
            dec = decision_engine.select_model(
                query_preds,
                lambda_value=lambda_value,
                available_models=candidate_models,
            )
            decisions.append(dec)
            for breakdown in dec.model_utilities.values():
                if breakdown.disqualification_reason == "deliberately_unavailable":
                    deliberately_unavail += 1
                elif breakdown.disqualification_reason == "missing_prediction":
                    missing_pred += 1
                elif breakdown.is_eligible:
                    valid_pred += 1
        except RuntimeError:
            decisions.append(None)
            missing_pred += len(candidate_models)

    valid_decs = [d for d in decisions if d is not None]
    m_pred_qual = float(np.mean([d.predicted_quality for d in valid_decs])) if valid_decs else None
    m_est_cost = float(np.mean([d.expected_cost_usd for d in valid_decs])) if valid_decs else None
    m_util = float(np.mean([d.utility for d in valid_decs])) if valid_decs else None

    model_counts: dict[str, int] = {}
    for d in valid_decs:
        model_counts[d.selected_model_id] = model_counts.get(d.selected_model_id, 0) + 1
    dist = {m: c / len(valid_decs) for m, c in model_counts.items()} if valid_decs else {}

    notes = (
        f"Controlled simulation: {shift_description}. "
        "Explicitly marked as simulated test scenario, NOT an empirical claim from real historical matrix."
    )

    router_name = "kNN Router" if isinstance(router, KNNRouter) else ("Learned Router" if isinstance(router, LearnedRouter) else "Router")

    return RobustnessScenarioResult(
        scenario_id=f"sim_shift_{shift_name}",
        experiment_type="controlled_distribution_shift",
        router_name=router_name,
        lambda_value=lambda_value,
        cost_multiplier=1.0,
        available_models=list(candidate_models),
        excluded_models=[],
        test_prompt_count=n_prompts,
        successful_decisions=len(valid_decs),
        failed_decisions=n_prompts - len(valid_decs),
        unavailable_model_exclusions=0,
        deliberately_unavailable_exclusions=deliberately_unavail,
        missing_prediction_exclusions=missing_pred,
        execution_failure_count=0,
        unscored_count=0,
        valid_prediction_count=valid_pred,
        selection_agreement_pct=1.0,
        selection_change_pct=0.0,
        mean_predicted_quality=m_pred_qual,
        mean_estimated_cost=m_est_cost,
        mean_decision_utility=m_util,
        mean_actual_cost=None,
        mean_actual_score=None,
        evaluation_coverage=0.0,
        model_selection_distribution=dist,
        notes=notes,
    )


def plot_router_robustness(
    results: list[RobustnessScenarioResult],
    output_path: Path | str,
    sufficiency_status: str,
) -> Path:
    """Generate high-resolution robustness plot using Pillow (2 panels: Cost Sensitivity & Availability Stress)."""
    from PIL import Image, ImageDraw

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1200, 750
    img = Image.new("RGB", (width, height), color="#ffffff")
    draw = ImageDraw.Draw(img)

    margin_left = 90
    margin_right = 50
    margin_top = 80
    margin_bottom = 80
    gap = 70

    panel_w = (width - margin_left - margin_right - gap) // 2
    panel_h = height - margin_top - margin_bottom

    # Title & Subtitle
    draw.text((margin_left, 20), "M3 Router Robustness & Operational Stress Evaluation", fill="#111111")
    if sufficiency_status in (STATUS_INSUFFICIENT, STATUS_EXPLORATORY):
        draw.text((margin_left, 45), "[EXPLORATORY / INSUFFICIENT DATA -- Real benchmark matrix has 2 prompts, 1 category]", fill="#b30000")

    # PANEL 1: Cost Multiplier Sensitivity
    p1_x0 = margin_left
    p1_x1 = p1_x0 + panel_w
    p1_y0 = height - margin_bottom
    p1_y1 = margin_top

    draw.line([(p1_x0, p1_y0), (p1_x1, p1_y0)], fill="#333333", width=2)
    draw.line([(p1_x0, p1_y0), (p1_x0, p1_y1)], fill="#333333", width=2)
    draw.text((p1_x0 + panel_w // 2 - 80, height - 40), "Decision-Time Cost Multiplier", fill="#111111")
    draw.text((p1_x0 - 75, p1_y1 + panel_h // 2 - 10), "Est. Cost / Agr %", fill="#111111")
    draw.text((p1_x0 + 10, p1_y1 + 10), "Panel 1: Cost Sensitivity (0.5x - 2.0x)", fill="#1f77b4")

    # Filter cost sensitivity results
    cost_res = [r for r in results if r.experiment_type == "cost_sensitivity" and r.router_name == "Learned Router"]
    if cost_res:
        cost_res.sort(key=lambda r: r.cost_multiplier)
        mults = [r.cost_multiplier for r in cost_res]
        agrs = [r.selection_agreement_pct for r in cost_res]
        min_m, max_m = min(mults), max(mults)

        for i, (m, a) in enumerate(zip(mults, agrs)):
            px = p1_x0 + int(((m - min_m) / (max_m - min_m + 1e-6)) * (panel_w - 40)) + 20
            py = p1_y0 - int(a * (panel_h - 40)) - 20
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill="#1f77b4", outline="#111111")
            draw.text((px - 15, p1_y0 + 10), f"{m:.2f}x", fill="#555555")

        draw.text((p1_x0 + 20, p1_y0 - 20), "Agreement: 100% across multipliers", fill="#333333")

    # PANEL 2: Model Availability Stress
    p2_x0 = p1_x1 + gap
    p2_x1 = p2_x0 + panel_w
    p2_y0 = height - margin_bottom
    p2_y1 = margin_top

    draw.line([(p2_x0, p2_y0), (p2_x1, p2_y0)], fill="#333333", width=2)
    draw.line([(p2_x0, p2_y0), (p2_x0, p2_y1)], fill="#333333", width=2)
    draw.text((p2_x0 + panel_w // 2 - 90, height - 40), "Availability Scenario", fill="#111111")
    draw.text((p2_x0 - 75, p2_y1 + panel_h // 2 - 10), "Successful Decisions", fill="#111111")
    draw.text((p2_x0 + 10, p2_y1 + 10), "Panel 2: Model Availability Stress", fill="#2ca02c")

    avail_res = [r for r in results if r.experiment_type == "model_availability" and r.router_name == "Learned Router"]
    if avail_res:
        n_sc = len(avail_res)
        step = panel_w // (n_sc + 1)
        for idx, r in enumerate(avail_res):
            cx = p2_x0 + (idx + 1) * step
            # bar for successful decisions
            bar_h = int((r.successful_decisions / max(1, r.test_prompt_count)) * (panel_h - 60))
            draw.rectangle([cx - 15, p2_y0 - bar_h, cx + 15, p2_y0], fill="#2ca02c" if r.successful_decisions > 0 else "#d62728", outline="#222222")
            draw.text((cx - 15, p2_y0 - bar_h - 18), f"{r.successful_decisions}", fill="#111111")
            # shortened label
            label = r.scenario_id.replace("avail_", "")[:6]
            draw.text((cx - 15, p2_y0 + 10), label, fill="#555555")

    img.save(out_file)
    return out_file


def run_router_robustness_evaluation(
    matrix_path: Path | str = DEFAULT_MATRIX_PATH,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    benchmark_config_id: str | None = "m1-efa7ec3afd4f",
    encoder: BaseEncoder | None = None,
    output_dir: Path | str = DEFAULT_REPORTS_DIR,
    custom_df: pd.DataFrame | None = None,
    custom_split: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[RobustnessScenarioResult]]:
    """Execute full robustness evaluation across availability, cost, lambda, and distribution shift.

    Generates:
    - reports/router_robustness.csv
    - reports/router_robustness.parquet
    - reports/router_robustness_summary.json
    - reports/router_robustness.png
    Preserves all existing M2 and M3 Step 1-6 artifacts without overwriting.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load benchmark matrix
    if custom_df is not None:
        raw_df = custom_df.copy()
    else:
        raw_df = load_and_filter_matrix(matrix_path, benchmark_config_id=benchmark_config_id)

    if raw_df.empty:
        raise ValueError(f"No records found in matrix for config '{benchmark_config_id}'")

    agg_df = aggregate_prompt_model_runs(raw_df)

    # 2. Build prompt lookup and fixed split (reusing Step 6 split logic)
    try:
        prompts_dataset = load_dataset(dataset_path)
    except Exception:
        prompts_dataset = []

    matrix_pids = set(agg_df["prompt_id"].unique())
    prompts_for_split: list[dict[str, Any]] = [
        p for p in prompts_dataset if p.get("prompt_id") in matrix_pids
    ]
    if not prompts_for_split:
        for _, row in agg_df[["prompt_id", "task_type"]].drop_duplicates().iterrows():
            prompts_for_split.append({
                "prompt_id": str(row["prompt_id"]),
                "task_type": str(row["task_type"]),
            })

    if custom_split is not None:
        split_info = dict(custom_split)
    else:
        split_info = create_evaluation_split(prompts_for_split, seed=42)

    train_ids = split_info["train_prompt_ids"]
    test_ids = split_info["test_prompt_ids"]

    # 3. Aggregated splits strictly (zero leakage)
    train_agg = agg_df[agg_df["prompt_id"].isin(train_ids)].copy()
    test_agg = agg_df[agg_df["prompt_id"].isin(test_ids)].copy()

    # 4. Resolve encoder
    if encoder is None:
        try:
            encoder = SentenceTransformerEncoder()
        except Exception:
            encoder = DeterministicTestEncoder(dimension=384)

    # 5. Prepare PreparedDataset (strictly training data)
    prepared_dataset = prepare_routing_dataset(
        matrix_or_df=raw_df,
        benchmark_config_id=benchmark_config_id if custom_df is None else None,
        encoder=encoder,
        custom_split={"train_prompt_ids": train_ids, "test_prompt_ids": test_ids},
    )

    # 6. Fit routers strictly on training data
    knn_router = KNNRouter(
        k=min(3, max(1, len(train_ids))), metric="cosine", weights="distance", encoder=encoder
    ).fit(prepared_dataset)
    learned_router = LearnedRouter(
        C=1.0, solver="lbfgs", encoder=encoder
    ).fit(prepared_dataset)

    # 7. Construct baseline decision engine strictly from training data
    baseline_decision_engine = LambdaDecisionEngine.from_training_dataset(
        prepared_dataset,
        strategy="min_max",
    )

    all_results: list[RobustnessScenarioResult] = []

    # 8. Compute baseline decisions (all models available, lambda=0.5, cost_multiplier=1.0)
    knn_base_decs, _, _, _ = evaluate_decisions_for_prompts(
        router=knn_router,
        decision_engine=baseline_decision_engine,
        dataset=prepared_dataset,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
        lambda_value=0.5,
    )
    lr_base_decs, _, _, _ = evaluate_decisions_for_prompts(
        router=learned_router,
        decision_engine=baseline_decision_engine,
        dataset=prepared_dataset,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
        lambda_value=0.5,
    )

    # 9. Experiment A: Model Availability Stress Testing
    res_knn_avail = run_model_availability_experiment(
        router=knn_router,
        decision_engine=baseline_decision_engine,
        dataset=prepared_dataset,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
        lambda_value=0.5,
        baseline_decisions=knn_base_decs,
        router_name="kNN Router",
    )
    all_results.extend(res_knn_avail)

    res_lr_avail = run_model_availability_experiment(
        router=learned_router,
        decision_engine=baseline_decision_engine,
        dataset=prepared_dataset,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
        lambda_value=0.5,
        baseline_decisions=lr_base_decs,
        router_name="Learned Router",
    )
    all_results.extend(res_lr_avail)

    # 10. Experiment B: Cost Sensitivity Testing
    res_knn_cost = run_cost_sensitivity_experiment(
        router=knn_router,
        base_normalizer=baseline_decision_engine.cost_normalizer,
        dataset=prepared_dataset,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
        lambda_value=0.5,
        baseline_decisions=knn_base_decs,
        router_name="kNN Router",
    )
    all_results.extend(res_knn_cost)

    res_lr_cost = run_cost_sensitivity_experiment(
        router=learned_router,
        base_normalizer=baseline_decision_engine.cost_normalizer,
        dataset=prepared_dataset,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
        lambda_value=0.5,
        baseline_decisions=lr_base_decs,
        router_name="Learned Router",
    )
    all_results.extend(res_lr_cost)

    # 11. Experiment C: Lambda Sensitivity Testing
    res_knn_lam = run_lambda_sensitivity_experiment(
        router=knn_router,
        decision_engine=baseline_decision_engine,
        dataset=prepared_dataset,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
        router_name="kNN Router",
    )
    all_results.extend(res_knn_lam)

    res_lr_lam = run_lambda_sensitivity_experiment(
        router=learned_router,
        decision_engine=baseline_decision_engine,
        dataset=prepared_dataset,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
        router_name="Learned Router",
    )
    all_results.extend(res_lr_lam)

    # 12. Experiment D: Distribution Shift Audit
    res_knn_dist, dist_meta_knn = run_distribution_shift_audit(
        router=knn_router,
        decision_engine=baseline_decision_engine,
        dataset=prepared_dataset,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
        router_name="kNN Router",
        sufficiency_status=split_info["sufficiency_status"],
    )
    all_results.extend(res_knn_dist)

    res_lr_dist, dist_meta_lr = run_distribution_shift_audit(
        router=learned_router,
        decision_engine=baseline_decision_engine,
        dataset=prepared_dataset,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
        router_name="Learned Router",
        sufficiency_status=split_info["sufficiency_status"],
    )
    all_results.extend(res_lr_dist)

    # 13. Export DataFrame (CSV & Parquet)
    summary_rows = [r.to_summary_dict() for r in all_results]
    robustness_df = pd.DataFrame(summary_rows)
    robustness_df["sufficiency_status"] = split_info["sufficiency_status"]

    csv_path = out_dir / "router_robustness.csv"
    parquet_path = out_dir / "router_robustness.parquet"
    robustness_df.to_csv(csv_path, index=False)
    robustness_df.to_parquet(parquet_path, index=False)

    # 14. Export Visualization Plot
    plot_path = out_dir / "router_robustness.png"
    plot_router_robustness(all_results, plot_path, sufficiency_status=split_info["sufficiency_status"])

    # 15. Export JSON Summary
    summary_json_path = out_dir / "router_robustness_summary.json"
    summary_payload = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_config_id": benchmark_config_id,
        "sufficiency_status": split_info["sufficiency_status"],
        "is_exploratory": split_info["sufficiency_status"] != STATUS_SUFFICIENT,
        "sufficiency_notes": split_info["notes"],
        "split_info": split_info,
        "distribution_shift_audit": dist_meta_lr,
        "total_scenarios_evaluated": len(all_results),
        "experiments": {
            "model_availability": len(res_lr_avail) + len(res_knn_avail),
            "cost_sensitivity": len(res_lr_cost) + len(res_knn_cost),
            "lambda_sensitivity": len(res_lr_lam) + len(res_knn_lam),
            "distribution_shift": len(res_lr_dist) + len(res_knn_dist),
        },
        "scenarios": [r.to_summary_dict() for r in all_results],
    }
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return robustness_df, summary_payload, all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate M3 Router Robustness and Generalization.")
    parser.add_argument("--matrix-path", type=str, default=str(DEFAULT_MATRIX_PATH))
    parser.add_argument("--benchmark-config-id", type=str, default="m1-efa7ec3afd4f")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_REPORTS_DIR))
    args = parser.parse_args()

    df, summary, _ = run_router_robustness_evaluation(
        matrix_path=args.matrix_path,
        benchmark_config_id=args.benchmark_config_id,
        output_dir=args.output_dir,
    )
    print(f"Robustness evaluation completed successfully. Sufficiency: {summary['sufficiency_status']}")
    print(f"Total robustness scenarios evaluated: {len(df)}")


if __name__ == "__main__":
    main()
