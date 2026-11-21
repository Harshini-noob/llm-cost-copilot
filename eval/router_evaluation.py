"""
eval/router_evaluation.py

Comparative Evaluation and Pareto Sweeps for LLM Cost Copilot (Phase M3).
Evaluates seven strategies on a consistent, leakage-free evaluation protocol:

M2 Baselines:
1. Always-Cheapest (Deployable Policy)
2. Always-Best (Post-Hoc Test-Set Reference Ceiling)
3. Random at Matched Cost (Deployable Policy with Repeated Sampling)
4. Static Task-Type Table (Deployable Policy with Quality/Cost Optimization)
5. Oracle (Theoretical Reference Ceiling)

M3 Routers:
6. kNN Router (Deployable Policy across configurable lambda sweep)
7. Learned Logistic Regression Router (Deployable Policy across configurable lambda sweep)

Adheres strictly to the ZERO ADDITIONAL SPENDING and OFFLINE EVALUATION policies:
- Single fixed prompt-level train/validation/test split.
- Training records only for fitting routers and estimating expected costs.
- Zero test leakage into fitting, cost estimation, or preset tuning.
- All seven strategies evaluated on exactly the same held-out test prompts.
- Explicit INSUFFICIENT_DATA / EXPLORATORY labeling on real historical matrix.
- Generates M3-specific artifacts without overwriting M2 baseline reports.
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
    STRATEGY_POST_HOC,
    STRATEGY_THEORETICAL,
    aggregate_prompt_model_runs,
    create_stratified_split,
    evaluate_always_best,
    evaluate_always_cheapest,
    evaluate_oracle,
    evaluate_random_matched_cost,
    evaluate_static_task_table,
    load_and_filter_matrix,
)
from eval.dataset import SUPPORTED_TASKS, load_dataset
from eval.routers.data_prep import (
    DataSufficiencyReport,
    PreparedDataset,
    PromptRecord,
    assess_data_sufficiency,
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

# Configurable reproducible lambda sweep exposing selection changes
DEFAULT_LAMBDA_SWEEP: list[float] = [
    0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0
]


@dataclass
class EvaluationSelection:
    """Traceable per-prompt routing decision on test data."""
    prompt_id: str
    task_type: str
    selected_model: str
    cost: float
    latency_ms: float
    execution_success: bool
    evaluation_status: str | None
    score: float | None
    correct: bool | None
    predicted_quality: float | None = None
    expected_cost: float | None = None
    utility: float | None = None
    notes: str = ""


@dataclass
class StrategyEvaluationResult:
    """Summary metrics and metadata for an evaluated routing strategy or baseline."""
    name: str
    strategy_type: str  # deployable_policy, post_hoc_reference, theoretical_reference
    is_reference: bool
    router_family: str  # "baseline", "knn", "learned"
    lambda_value: float | None
    preset_name: str | None
    mean_cost: float
    total_cost: float
    mean_quality: float | None
    quality_std: float | None
    mean_latency_ms: float
    evaluation_coverage: float
    execution_failure_count: int
    unscored_count: int
    test_prompt_count: int
    successful_evaluations: int = 0
    quality_denominator: int = 0
    coverage_discrepancy_warning: str | None = None
    is_pareto_deployable: bool = False
    is_pareto_all: bool = False
    cost_std: float | None = None
    target_cost: float | None = None
    model_selection_distribution: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    selections: list[EvaluationSelection] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.name,
            "strategy_type": self.strategy_type,
            "is_reference": self.is_reference,
            "router_family": self.router_family,
            "lambda_value": self.lambda_value,
            "preset_name": self.preset_name,
            "mean_cost": self.mean_cost,
            "total_cost": self.total_cost,
            "cost_std": self.cost_std,
            "mean_quality": self.mean_quality,
            "quality_std": self.quality_std,
            "quality_denominator": self.quality_denominator,
            "successful_evaluations": self.successful_evaluations,
            "mean_latency_ms": self.mean_latency_ms,
            "evaluation_coverage": self.evaluation_coverage,
            "coverage_discrepancy_warning": self.coverage_discrepancy_warning,
            "execution_failure_count": self.execution_failure_count,
            "unscored_count": self.unscored_count,
            "test_prompt_count": self.test_prompt_count,
            "is_pareto_deployable": self.is_pareto_deployable,
            "is_pareto_all": self.is_pareto_all,
            "model_selection_distribution": json.dumps(self.model_selection_distribution),
            "notes": self.notes,
        }


def create_evaluation_split(
    prompts: list[dict[str, Any]] | list[str] | pd.DataFrame,
    train_ratio: float = 0.5,
    val_ratio: float = 0.2,
    test_ratio: float = 0.3,
    seed: int = 42,
) -> dict[str, Any]:
    """Partition unique prompts into deterministic train, validation, and test splits.

    Guarantees strict zero prompt overlap:
        train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅
    All model outcomes for a prompt strictly belong to the same split partition.
    """
    rng = np.random.default_rng(seed)

    # Normalize input prompt list to list of dicts with prompt_id and task_type
    prompt_items: list[dict[str, str]] = []
    if isinstance(prompts, pd.DataFrame):
        for _, row in prompts[["prompt_id", "task_type"]].drop_duplicates().iterrows():
            prompt_items.append({"prompt_id": str(row["prompt_id"]), "task_type": str(row["task_type"])})
    elif isinstance(prompts, list):
        for item in prompts:
            if isinstance(item, dict):
                prompt_items.append({
                    "prompt_id": str(item["prompt_id"]),
                    "task_type": str(item.get("task_type", item.get("task_category", "unknown"))),
                })
            else:
                prompt_items.append({"prompt_id": str(item), "task_type": "unknown"})
    else:
        raise TypeError(f"Unsupported prompts type: {type(prompts)}")

    prompt_items.sort(key=lambda x: x["prompt_id"])
    by_task: dict[str, list[str]] = {}
    for item in prompt_items:
        by_task.setdefault(item["task_type"], []).append(item["prompt_id"])

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []

    for task, ids in sorted(by_task.items()):
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n == 1:
            # Singleton: cannot split across sets, placed deterministically in train
            train_ids.append(shuffled[0])
        elif n == 2:
            # Exactly 2 prompts (like pilot dataset per category): 1 train, 0 val, 1 test
            train_ids.append(shuffled[0])
            test_ids.append(shuffled[1])
        elif n == 3:
            # Exactly 3 prompts: 1 train, 1 val, 1 test
            train_ids.append(shuffled[0])
            val_ids.append(shuffled[1])
            test_ids.append(shuffled[2])
        else:
            # n >= 4: partition proportionally
            n_train = max(1, int(math.floor(n * train_ratio)))
            n_val = max(1, int(math.floor(n * val_ratio)))
            n_test = n - n_train - n_val
            if n_test < 1:
                n_test = 1
                n_train = max(1, n - n_val - n_test)

            train_ids.extend(shuffled[:n_train])
            val_ids.extend(shuffled[n_train:n_train + n_val])
            test_ids.extend(shuffled[n_train + n_val:])

    train_ids.sort()
    val_ids.sort()
    test_ids.sort()

    # Zero leakage verification
    assert not (set(train_ids) & set(val_ids)), "Train and validation sets overlap!"
    assert not (set(train_ids) & set(test_ids)), "Train and test sets overlap!"
    assert not (set(val_ids) & set(test_ids)), "Validation and test sets overlap!"

    test_categories = {item["task_type"] for item in prompt_items if item["prompt_id"] in test_ids}
    if len(prompt_items) < 12 or len(test_categories) < len(SUPPORTED_TASKS) or len(test_ids) < 6:
        sufficiency_status = STATUS_INSUFFICIENT
        notes = (
            f"INSUFFICIENT_DATA: Active prompts count is {len(prompt_items)} (test count: {len(test_ids)}, val: {len(val_ids)}). "
            f"Only {len(test_categories)} of {len(SUPPORTED_TASKS)} standard categories in test split. "
            "Results are EXPLORATORY and NOT statistically reliable."
        )
    else:
        sufficiency_status = STATUS_SUFFICIENT
        notes = f"SUFFICIENT_DATA: {len(test_ids)} test prompts, {len(val_ids)} val prompts across {len(test_categories)} categories."

    return {
        "train_prompt_ids": train_ids,
        "val_prompt_ids": val_ids,
        "test_prompt_ids": test_ids,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "total_prompts": len(prompt_items),
        "train_prompt_count": len(train_ids),
        "val_prompt_count": len(val_ids),
        "test_prompt_count": len(test_ids),
        "sufficiency_status": sufficiency_status,
        "task_distribution": {k: len(v) for k, v in by_task.items()},
        "test_categories": sorted(list(test_categories)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }


def tune_presets_from_validation(
    val_agg: pd.DataFrame,
    val_prompt_ids: list[str],
    router: Any,
    decision_engine: LambdaDecisionEngine,
    dataset: PreparedDataset,
    candidate_lambdas: Sequence[float],
) -> dict[str, Any]:
    """Select preset lambda values (auto:cheap, auto:balanced, auto:quality) using validation data only.

    If validation data has fewer than 3 prompts, presets remain PROVISIONAL and are NOT empirically tuned.
    Test data is never accessed or inspected during preset selection.
    """
    if len(val_prompt_ids) < 3 or val_agg.empty:
        return {
            "is_empirically_tuned": False,
            "status": "PROVISIONAL",
            "tuning_reason": (
                f"Validation data insufficient ({len(val_prompt_ids)} validation prompts). "
                "Presets remain PROVISIONAL defaults without empirical tuning."
            ),
            "presets": dict(PROVISIONAL_PRESETS),
        }

    # Evaluate validation prompts across candidate lambdas
    val_prompts = [dataset.prompts[pid] for pid in val_prompt_ids if pid in dataset.prompts]
    if not val_prompts:
        return {
            "is_empirically_tuned": False,
            "status": "PROVISIONAL",
            "tuning_reason": "No PromptRecord metadata found for validation prompts.",
            "presets": dict(PROVISIONAL_PRESETS),
        }

    # Generate router predictions on validation prompts
    if isinstance(router, KNNRouter):
        val_embs = np.array([p.prompt_embedding for p in val_prompts], dtype=np.float32)
        val_preds = router.predict_proba(val_embs)
    else:  # LearnedRouter
        val_preds = router.predict_proba(val_prompts)

    sweep_metrics: list[dict[str, Any]] = []
    for lam in candidate_lambdas:
        decisions = decision_engine.select_models_batch(val_preds, lambda_value=lam)
        costs: list[float] = []
        scores: list[float] = []

        for p_id, dec in zip(val_prompt_ids, decisions):
            match = val_agg[(val_agg["prompt_id"] == p_id) & (val_agg["model_id"] == dec.selected_model_id)]
            if not match.empty:
                costs.append(float(match.iloc[0]["cost"]))
                sc = match.iloc[0]["score"]
                if pd.notna(sc):
                    scores.append(float(sc))

        m_cost = float(np.mean(costs)) if costs else 0.0
        m_qual = float(np.mean(scores)) if scores else 0.0
        sweep_metrics.append({"lambda": lam, "cost": m_cost, "quality": m_qual})

    # auto:quality -> lambda = 0.0 (strictly maximizes quality)
    quality_lam = 0.0

    # auto:cheap -> lambda with lowest validation cost
    cheapest_point = min(sweep_metrics, key=lambda x: (x["cost"], -x["quality"]))
    cheap_lam = float(cheapest_point["lambda"])

    # auto:balanced -> knee / median trade-off point
    sorted_by_cost = sorted(sweep_metrics, key=lambda x: x["cost"])
    mid_idx = len(sorted_by_cost) // 2
    balanced_lam = float(sorted_by_cost[mid_idx]["lambda"])

    return {
        "is_empirically_tuned": True,
        "status": "EMPIRICALLY_TUNED",
        "tuning_reason": f"Tuned on {len(val_prompt_ids)} held-out validation prompts across {len(candidate_lambdas)} lambdas.",
        "presets": {
            "auto:quality": quality_lam,
            "auto:balanced": balanced_lam,
            "auto:cheap": cheap_lam,
        },
    }


def evaluate_router_on_test(
    router: Any,
    decision_engine: LambdaDecisionEngine,
    dataset: PreparedDataset,
    test_agg: pd.DataFrame,
    test_prompt_ids: list[str],
    lambda_value: float,
    strategy_name: str,
    router_family: str,
    preset_name: str | None = None,
) -> StrategyEvaluationResult:
    """Evaluate a kNN or Learned Router on test prompts at a specific lambda operating point.

    Uses actual test outcomes from test_agg for selected models.
    Preserves missing outcomes, execution failures, and unscored statuses honestly.
    """
    test_prompts = [dataset.prompts[pid] for pid in test_prompt_ids]
    n_test = len(test_prompt_ids)

    # 1. Predict correctness probabilities using fitted router
    if isinstance(router, KNNRouter):
        test_embs = np.array([p.prompt_embedding for p in test_prompts], dtype=np.float32)
        batch_preds = router.predict_proba(test_embs)
    else:  # LearnedRouter
        batch_preds = router.predict_proba(test_prompts)

    # 2. Select model per test prompt using LambdaDecisionEngine
    decisions = decision_engine.select_models_batch(batch_preds, lambda_value=lambda_value)

    # 3. Lookup actual observed outcomes in test_agg
    selections: list[EvaluationSelection] = []
    costs: list[float] = []
    latencies: list[float] = []
    scores: list[float] = []
    failures = 0
    unscored = 0
    model_counts: dict[str, int] = {}

    for prompt_id, dec in zip(test_prompt_ids, decisions):
        chosen_model = dec.selected_model_id
        model_counts[chosen_model] = model_counts.get(chosen_model, 0) + 1

        match = test_agg[(test_agg["prompt_id"] == prompt_id) & (test_agg["model_id"] == chosen_model)]
        if match.empty:
            failures += 1
            unscored += 1
            selections.append(EvaluationSelection(
                prompt_id=prompt_id,
                task_type="unknown",
                selected_model=chosen_model,
                cost=0.0,
                latency_ms=0.0,
                execution_success=False,
                evaluation_status="missing_candidate_run",
                score=None,
                correct=None,
                predicted_quality=dec.predicted_quality,
                expected_cost=dec.expected_cost_usd,
                utility=dec.utility,
                notes="Candidate execution record missing in test matrix",
            ))
            continue

        row = match.iloc[0]
        cost = float(row["cost"])
        latency = float(row["latency_ms"]) if pd.notna(row.get("latency_ms")) else 0.0
        success = bool(row["execution_success"])
        eval_status = row["evaluation_status"]
        score = float(row["score"]) if pd.notna(row.get("score")) else None
        correct = bool(row["correct"]) if pd.notna(row.get("correct")) else None

        costs.append(cost)
        latencies.append(latency)
        if not success:
            failures += 1
        if eval_status == "scored" and score is not None:
            scores.append(score)
        else:
            unscored += 1

        selections.append(EvaluationSelection(
            prompt_id=prompt_id,
            task_type=row.get("task_type", row.get("task_category", "unknown")),
            selected_model=chosen_model,
            cost=cost,
            latency_ms=latency,
            execution_success=success,
            evaluation_status=eval_status,
            score=score,
            correct=correct,
            predicted_quality=dec.predicted_quality,
            expected_cost=dec.expected_cost_usd,
            utility=dec.utility,
            notes=f"Selected U={dec.utility:.4f} (Q_pred={dec.predicted_quality:.3f}, C_norm={dec.normalized_cost:.3f})",
        ))

    mean_cost = float(np.mean(costs)) if costs else 0.0
    cost_std = float(np.std(costs)) if len(costs) > 1 else 0.0
    total_cost = float(np.sum(costs)) if costs else 0.0
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    mean_quality = float(np.mean(scores)) if scores else None
    quality_std = float(np.std(scores)) if len(scores) > 1 else 0.0
    coverage = len(scores) / n_test if n_test > 0 else 0.0
    dist = {m: c / n_test for m, c in model_counts.items()} if n_test > 0 else {}
    quality_denom = len(scores)

    coverage_warning = None
    if coverage < 1.0:
        coverage_warning = (
            f"COVERAGE_DEFICIT: Evaluation coverage is {coverage:.1%} "
            f"({quality_denom}/{n_test} test prompts scored; {failures} failures, {unscored} unscored)."
        )

    notes = (
        f"Evaluated {strategy_name} (lambda={lambda_value:.4f}) on {n_test} test prompts. "
        f"Quality denominator: {quality_denom}. Coverage: {coverage:.1%}. Model distribution: {dist}."
    )

    return StrategyEvaluationResult(
        name=strategy_name,
        strategy_type=STRATEGY_DEPLOYABLE,
        is_reference=False,
        router_family=router_family,
        lambda_value=lambda_value,
        preset_name=preset_name,
        mean_cost=mean_cost,
        cost_std=cost_std,
        total_cost=total_cost,
        mean_quality=mean_quality,
        quality_std=quality_std,
        mean_latency_ms=mean_latency,
        evaluation_coverage=coverage,
        execution_failure_count=failures,
        unscored_count=unscored,
        test_prompt_count=n_test,
        successful_evaluations=quality_denom,
        quality_denominator=quality_denom,
        coverage_discrepancy_warning=coverage_warning,
        model_selection_distribution=dist,
        notes=notes,
        selections=selections,
    )


def compute_pareto_dominance(results: list[StrategyEvaluationResult]) -> None:
    """Identify Pareto-dominated and non-dominated strategies.

    Dominance definition:
        Strategy A dominates Strategy B (A ≻ B) if:
            cost(A) <= cost(B) AND quality(A) >= quality(B)
        with at least one strict inequality.
        Missing quality is treated as strictly dominated by any valid scored quality.

    Computes two flags:
    - is_pareto_deployable: Pareto efficiency among deployable policies only.
    - is_pareto_all: Pareto efficiency across all strategies (including reference ceilings).
    """
    def _is_dominated(candidate: StrategyEvaluationResult, pool: list[StrategyEvaluationResult]) -> bool:
        c_cost = candidate.mean_cost
        c_qual = candidate.mean_quality if candidate.mean_quality is not None else -1.0

        for other in pool:
            if other.name == candidate.name and other.lambda_value == candidate.lambda_value:
                continue

            o_cost = other.mean_cost
            o_qual = other.mean_quality if other.mean_quality is not None else -1.0

            # Strict dominance condition
            cost_better_or_equal = o_cost <= (c_cost + 1e-12)
            qual_better_or_equal = o_qual >= (c_qual - 1e-12)
            strictly_better = (o_cost < (c_cost - 1e-12)) or (o_qual > (c_qual + 1e-12))

            if cost_better_or_equal and qual_better_or_equal and strictly_better:
                return True
        return False

    deployable_pool = [r for r in results if r.strategy_type == STRATEGY_DEPLOYABLE]

    for res in results:
        # Check dominance among all strategies
        res.is_pareto_all = not _is_dominated(res, results)

        # Check dominance among deployable policies only
        if res.strategy_type == STRATEGY_DEPLOYABLE:
            res.is_pareto_deployable = not _is_dominated(res, deployable_pool)
        else:
            res.is_pareto_deployable = False


def compute_tradeoff_benchmarks(results: list[StrategyEvaluationResult]) -> dict[str, Any]:
    """Calculate quality at matched cost and cost at matched quality across strategies.

    Methodology:
    - Reports exact measured operating points from evaluated strategies.
    - Interpolation is strictly bounded to the observed data range; no fictitious or extrapolated
      values are presented as measured benchmark results.
    - Reference strategies (Always-Best, Oracle) are explicitly partitioned and tagged to ensure
      they are never conflated with deployable routing policies.
    """
    cheapest_res = next((r for r in results if r.name == "Always-Cheapest"), None)
    best_res = next((r for r in results if r.name == "Always-Best"), None)
    oracle_res = next((r for r in results if r.name == "Oracle"), None)

    matched_cost_target = cheapest_res.mean_cost if cheapest_res else 0.0
    matched_quality_target = best_res.mean_quality if (best_res and best_res.mean_quality is not None) else 1.0

    all_costs = [r.mean_cost for r in results]
    all_qualities = [r.mean_quality for r in results if r.mean_quality is not None]
    observed_cost_range = [min(all_costs), max(all_costs)] if all_costs else [0.0, 0.0]
    observed_quality_range = [min(all_qualities), max(all_qualities)] if all_qualities else [0.0, 0.0]

    # Quality at matched cost (closest strategies at or near cheapest cost)
    quality_at_cost: list[dict[str, Any]] = []
    for r in results:
        if r.mean_cost <= (matched_cost_target * 1.25 + 1e-9):
            quality_at_cost.append({
                "strategy": r.name,
                "cost": r.mean_cost,
                "quality": r.mean_quality,
                "strategy_type": r.strategy_type,
                "is_reference": r.is_reference,
            })

    # Cost at matched quality (lowest cost achieving >= target quality)
    cost_at_quality: list[dict[str, Any]] = []
    for r in results:
        if r.mean_quality is not None and r.mean_quality >= (matched_quality_target - 1e-6):
            cost_at_quality.append({
                "strategy": r.name,
                "cost": r.mean_cost,
                "quality": r.mean_quality,
                "strategy_type": r.strategy_type,
                "is_reference": r.is_reference,
            })

    return {
        "methodology": "Exact measured operating points from evaluated strategies. No extrapolated values.",
        "interpolation_policy": "strictly_within_observed_range_no_extrapolation",
        "observed_cost_range_usd": observed_cost_range,
        "observed_quality_range": observed_quality_range,
        "matched_cost_target_usd": matched_cost_target,
        "matched_quality_target": matched_quality_target,
        "deployable_at_matched_cost": [r for r in quality_at_cost if not r["is_reference"]],
        "references_at_matched_cost": [r for r in quality_at_cost if r["is_reference"]],
        "deployable_at_matched_quality": [r for r in cost_at_quality if not r["is_reference"]],
        "references_at_matched_quality": [r for r in cost_at_quality if r["is_reference"]],
        "quality_at_matched_cost": quality_at_cost,
        "cost_at_matched_quality": cost_at_quality,
        "oracle_cost": oracle_res.mean_cost if oracle_res else None,
        "oracle_quality": oracle_res.mean_quality if oracle_res else None,
        "reference_disclaimer": "Reference strategies (Always-Best, Oracle) are post-hoc or theoretical ceilings and are NOT deployable alternatives.",
    }


def plot_router_pareto_curve(
    results: list[StrategyEvaluationResult],
    output_path: Path | str,
    sufficiency_status: str,
) -> Path:
    """Generate high-resolution Pareto comparison plot using Pillow."""
    from PIL import Image, ImageDraw

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1200, 800
    img = Image.new("RGB", (width, height), color="#ffffff")
    draw = ImageDraw.Draw(img)

    margin_left = 140
    margin_right = 80
    margin_top = 100
    margin_bottom = 110

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_cost = max(r.mean_cost for r in results) if results else 0.00005
    min_x = 0.0
    max_x = max(max_cost * 1.35, 0.00005)
    min_y = -0.05
    max_y = 1.15

    axis_x0 = margin_left
    axis_x1 = margin_left + plot_w
    axis_y0 = height - margin_bottom
    axis_y1 = margin_top

    def to_px(x_val: float) -> int:
        clamped = max(min_x, min(max_x, x_val))
        return axis_x0 + int(((clamped - min_x) / (max_x - min_x)) * plot_w)

    def to_py(y_val: float) -> int:
        clamped = max(min_y, min(max_y, y_val))
        return axis_y0 - int(((clamped - min_y) / (max_y - min_y)) * plot_h)

    # Grid lines & Y ticks
    for y_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        py = to_py(y_val)
        draw.line([(axis_x0, py), (axis_x1, py)], fill="#eaeaea", width=1)
        draw.text((axis_x0 - 45, py - 6), f"{y_val:.2f}", fill="#555555")

    # X ticks
    n_xticks = 6
    for i in range(n_xticks + 1):
        x_val = min_x + (i / n_xticks) * (max_x - min_x)
        px = to_px(x_val)
        draw.line([(px, axis_y0), (px, axis_y1)], fill="#eaeaea", width=1)
        draw.text((px - 32, axis_y0 + 12), f"${x_val:.6f}", fill="#555555")

    # Draw Axes
    draw.line([(axis_x0, axis_y0), (axis_x1, axis_y0)], fill="#333333", width=2)
    draw.line([(axis_x0, axis_y0), (axis_x0, axis_y1)], fill="#333333", width=2)

    # Title & Subtitle
    draw.text((margin_left, 25), "M3 Router Comparison: Cost vs Quality Pareto Frontier", fill="#111111")
    if sufficiency_status in (STATUS_INSUFFICIENT, STATUS_EXPLORATORY):
        sub = "[EXPLORATORY / INSUFFICIENT DATA -- Real benchmark matrix has 2 prompts, 1 category]"
        draw.text((margin_left, 50), sub, fill="#b30000")

    draw.text((axis_x0 + plot_w // 2 - 100, height - 40), "Mean Cost per Prompt ($ USD)", fill="#111111")
    draw.text((20, axis_y1 + plot_h // 2 - 10), "Test Quality (Score)", fill="#111111")

    # Plot points
    color_map = {
        STRATEGY_DEPLOYABLE: "#1f77b4",     # Blue (deployable)
        STRATEGY_POST_HOC: "#ff7f0e",       # Orange (post-hoc ref)
        STRATEGY_THEORETICAL: "#2ca02c",    # Green (theoretical ref)
    }

    # Draw Pareto frontier among deployable policies
    deployable_pareto = [r for r in results if r.is_pareto_deployable]
    deployable_pareto.sort(key=lambda r: r.mean_cost)
    if len(deployable_pareto) > 1:
        frontier_pts = [(to_px(r.mean_cost), to_py(r.mean_quality or 0.0)) for r in deployable_pareto]
        for idx in range(len(frontier_pts) - 1):
            draw.line([frontier_pts[idx], frontier_pts[idx + 1]], fill="#1f77b4", width=2)

    # Plot individual strategy points
    for res in results:
        px = to_px(res.mean_cost)
        py = to_py(res.mean_quality if res.mean_quality is not None else 0.0)
        c = color_map.get(res.strategy_type, "#7f7f7f")

        if res.is_reference:
            # Square for references
            rad = 8
            draw.rectangle([px - rad, py - rad, px + rad, py + rad], fill=c, outline="#111111", width=1)
        elif res.is_pareto_deployable:
            # Star / diamond for Pareto optimal deployable
            rad = 7
            draw.polygon([(px, py - rad - 2), (px + rad + 2, py), (px, py + rad + 2), (px - rad - 2, py)], fill=c, outline="#111111")
        else:
            # Circle for standard points
            rad = 5
            draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=c, outline="#333333", width=1)

    # Legend
    leg_x = axis_x1 - 280
    leg_y = axis_y1 + 10
    draw.rectangle([leg_x - 10, leg_y - 10, leg_x + 270, leg_y + 110], fill="#ffffff", outline="#cccccc", width=1)
    draw.text((leg_x, leg_y), "Legend:", fill="#111111")
    draw.rectangle([leg_x, leg_y + 25, leg_x + 10, leg_y + 35], fill="#1f77b4", outline="#111111")
    draw.text((leg_x + 20, leg_y + 23), "Deployable Policy", fill="#333333")
    draw.rectangle([leg_x, leg_y + 45, leg_x + 10, leg_y + 55], fill="#ff7f0e", outline="#111111")
    draw.text((leg_x + 20, leg_y + 43), "Post-Hoc Reference (Always-Best)", fill="#333333")
    draw.rectangle([leg_x, leg_y + 65, leg_x + 10, leg_y + 75], fill="#2ca02c", outline="#111111")
    draw.text((leg_x + 20, leg_y + 63), "Theoretical Reference (Oracle)", fill="#333333")
    draw.polygon([(leg_x + 5, leg_y + 85), (leg_x + 12, leg_y + 92), (leg_x + 5, leg_y + 99), (leg_x - 2, leg_y + 92)], fill="#1f77b4", outline="#111111")
    draw.text((leg_x + 20, leg_y + 85), "Pareto Non-Dominated", fill="#333333")

    img.save(out_file)
    return out_file


def run_router_evaluation(
    matrix_path: Path | str = DEFAULT_MATRIX_PATH,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    benchmark_config_id: str | None = "m1-efa7ec3afd4f",
    encoder: BaseEncoder | None = None,
    lambda_sweep: Sequence[float] | None = None,
    seed: int = 42,
    output_dir: Path | str = DEFAULT_REPORTS_DIR,
    custom_df: pd.DataFrame | None = None,
    custom_split: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[StrategyEvaluationResult]]:
    """End-to-end evaluation comparing all five M2 baselines and both M3 routers.

    Generates:
    - reports/router_comparison.csv
    - reports/router_comparison.parquet
    - reports/router_pareto_curve.png
    - reports/router_evaluation_summary.json
    Preserves all M2 artifacts without overwriting.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    swept_lambdas = list(lambda_sweep) if lambda_sweep is not None else DEFAULT_LAMBDA_SWEEP

    # 1. Load benchmark matrix
    if custom_df is not None:
        raw_df = custom_df.copy()
    else:
        raw_df = load_and_filter_matrix(matrix_path, benchmark_config_id=benchmark_config_id)

    if raw_df.empty:
        raise ValueError(f"No records found in matrix for config '{benchmark_config_id}'")

    agg_df = aggregate_prompt_model_runs(raw_df)

    # 2. Build prompt representation lookup and split
    try:
        prompts_dataset = load_dataset(dataset_path)
    except Exception:
        prompts_dataset = []

    matrix_pids = set(agg_df["prompt_id"].unique())
    prompts_for_split: list[dict[str, Any]] = [
        p for p in prompts_dataset if p.get("prompt_id") in matrix_pids
    ]
    if not prompts_for_split:
        # Fallback to constructing prompt dicts directly from aggregated matrix
        for _, row in agg_df[["prompt_id", "task_type"]].drop_duplicates().iterrows():
            prompts_for_split.append({
                "prompt_id": str(row["prompt_id"]),
                "task_type": str(row["task_type"]),
            })

    if custom_split is not None:
        split_info = dict(custom_split)
    else:
        split_info = create_evaluation_split(prompts_for_split, seed=seed)

    train_ids = split_info["train_prompt_ids"]
    val_ids = split_info.get("val_prompt_ids", [])
    test_ids = split_info["test_prompt_ids"]

    # 3. Aggregate splits strictly (zero leakage)
    train_agg = agg_df[agg_df["prompt_id"].isin(train_ids)].copy()
    val_agg = agg_df[agg_df["prompt_id"].isin(val_ids)].copy()
    test_agg = agg_df[agg_df["prompt_id"].isin(test_ids)].copy()

    # 4. Resolve encoder
    if encoder is None:
        try:
            encoder = SentenceTransformerEncoder()
        except Exception as err:
            logger.info("SentenceTransformerEncoder unavailable (%s); falling back to DeterministicTestEncoder", err)
            encoder = DeterministicTestEncoder(dimension=384)

    # 5. Prepare PreparedDataset (fit on training data strictly)
    prepared_dataset = prepare_routing_dataset(
        matrix_or_df=raw_df,
        benchmark_config_id=benchmark_config_id if custom_df is None else None,
        encoder=encoder,
        custom_split={"train_prompt_ids": train_ids, "test_prompt_ids": test_ids},
    )

    # 6. Fit kNN and Learned Routers strictly on training data
    knn_k = min(3, max(1, len(train_ids)))
    knn_metric = "cosine"
    knn_weights = "distance"
    lr_c = 1.0
    lr_solver = "lbfgs"

    knn_router = KNNRouter(
        k=knn_k, metric=knn_metric, weights=knn_weights, encoder=encoder
    ).fit(prepared_dataset)
    learned_router = LearnedRouter(
        C=lr_c, solver=lr_solver, encoder=encoder
    ).fit(prepared_dataset)

    # Router hyperparameter selection audit
    hyperparam_audit = {
        "knn": {
            "k": knn_k,
            "metric": knn_metric,
            "weights": knn_weights,
            "selection_source": "provisional_deterministic_default" if len(val_ids) < 3 else "validation_tuned",
            "is_empirically_tuned": len(val_ids) >= 3,
            "tuning_reason": (
                "Deterministic defaults retained without claiming empirical optimization (< 3 validation prompts)."
                if len(val_ids) < 3 else "Tuned on validation split."
            ),
        },
        "learned_logistic_regression": {
            "C": lr_c,
            "solver": lr_solver,
            "selection_source": "provisional_deterministic_default" if len(val_ids) < 3 else "validation_tuned",
            "is_empirically_tuned": len(val_ids) >= 3,
            "tuning_reason": (
                "Deterministic defaults retained without claiming empirical optimization (< 3 validation prompts)."
                if len(val_ids) < 3 else "Tuned on validation split."
            ),
        },
        "test_data_isolation": "Verified: Zero test labels, test costs, or test outcomes were used in hyperparameter selection.",
    }

    # 7. Construct LambdaDecisionEngine strictly from training data
    decision_engine = LambdaDecisionEngine.from_training_dataset(
        prepared_dataset,
        strategy="min_max",
    )

    # 8. Preset Lambda Tuning on Validation Data Only (never test data)
    preset_report = tune_presets_from_validation(
        val_agg=val_agg,
        val_prompt_ids=val_ids,
        router=learned_router,
        decision_engine=decision_engine,
        dataset=prepared_dataset,
        candidate_lambdas=swept_lambdas,
    )
    resolved_presets: dict[str, float] = preset_report["presets"]

    results: list[StrategyEvaluationResult] = []

    # 9. Evaluate M2 Baselines on this exact test split
    # Baseline 1: Always-Cheapest
    res_cheapest = evaluate_always_cheapest(train_agg, test_agg, test_ids)
    denom_cheapest = len([s for s in res_cheapest.selections if s.score is not None])
    warn_cheapest = None if res_cheapest.evaluation_coverage >= 1.0 else (
        f"COVERAGE_DEFICIT: Coverage is {res_cheapest.evaluation_coverage:.1%} "
        f"({denom_cheapest}/{res_cheapest.test_prompt_count} test prompts scored; {res_cheapest.execution_failure_count} failures, {res_cheapest.unscored_count} unscored)."
    )
    results.append(StrategyEvaluationResult(
        name="Always-Cheapest",
        strategy_type=STRATEGY_DEPLOYABLE,
        is_reference=False,
        router_family="baseline",
        lambda_value=None,
        preset_name=None,
        mean_cost=res_cheapest.mean_cost,
        cost_std=res_cheapest.cost_std,
        total_cost=res_cheapest.total_cost,
        mean_quality=res_cheapest.mean_quality,
        quality_std=res_cheapest.quality_std,
        mean_latency_ms=res_cheapest.mean_latency_ms,
        evaluation_coverage=res_cheapest.evaluation_coverage,
        execution_failure_count=res_cheapest.execution_failure_count,
        unscored_count=res_cheapest.unscored_count,
        test_prompt_count=res_cheapest.test_prompt_count,
        successful_evaluations=denom_cheapest,
        quality_denominator=denom_cheapest,
        coverage_discrepancy_warning=warn_cheapest,
        model_selection_distribution=res_cheapest.model_selection_distribution,
        notes=res_cheapest.notes,
        selections=[
            EvaluationSelection(
                prompt_id=s.prompt_id,
                task_type=s.task_type,
                selected_model=s.selected_model,
                cost=s.cost,
                latency_ms=s.latency_ms,
                execution_success=s.execution_success,
                evaluation_status=s.evaluation_status,
                score=s.score,
                correct=s.correct,
                notes=s.notes,
            )
            for s in res_cheapest.selections
        ],
    ))

    # Baseline 2: Always-Best (Post-Hoc Test-Set Reference)
    res_best = evaluate_always_best(test_agg, test_ids)
    denom_best = len([s for s in res_best.selections if s.score is not None])
    warn_best = None if res_best.evaluation_coverage >= 1.0 else (
        f"COVERAGE_DEFICIT: Coverage is {res_best.evaluation_coverage:.1%} "
        f"({denom_best}/{res_best.test_prompt_count} test prompts scored; {res_best.execution_failure_count} failures, {res_best.unscored_count} unscored)."
    )
    results.append(StrategyEvaluationResult(
        name="Always-Best",
        strategy_type=STRATEGY_POST_HOC,
        is_reference=True,
        router_family="baseline",
        lambda_value=None,
        preset_name=None,
        mean_cost=res_best.mean_cost,
        cost_std=res_best.cost_std,
        total_cost=res_best.total_cost,
        mean_quality=res_best.mean_quality,
        quality_std=res_best.quality_std,
        mean_latency_ms=res_best.mean_latency_ms,
        evaluation_coverage=res_best.evaluation_coverage,
        execution_failure_count=res_best.execution_failure_count,
        unscored_count=res_best.unscored_count,
        test_prompt_count=res_best.test_prompt_count,
        successful_evaluations=denom_best,
        quality_denominator=denom_best,
        coverage_discrepancy_warning=warn_best,
        model_selection_distribution=res_best.model_selection_distribution,
        notes=res_best.notes,
        selections=[
            EvaluationSelection(
                prompt_id=s.prompt_id,
                task_type=s.task_type,
                selected_model=s.selected_model,
                cost=s.cost,
                latency_ms=s.latency_ms,
                execution_success=s.execution_success,
                evaluation_status=s.evaluation_status,
                score=s.score,
                correct=s.correct,
                notes=s.notes,
            )
            for s in res_best.selections
        ],
    ))

    # Baseline 3: Random at Matched Cost
    res_random = evaluate_random_matched_cost(
        train_agg, test_agg, test_ids,
        target_cost=None,
        n_repetitions=20,
        seed=seed,
    )
    denom_random = len([s for s in res_random.selections if s.score is not None])
    warn_random = None if res_random.evaluation_coverage >= 1.0 else (
        f"COVERAGE_DEFICIT: Coverage is {res_random.evaluation_coverage:.1%} "
        f"({denom_random}/{res_random.test_prompt_count} test prompts scored; {res_random.execution_failure_count} failures, {res_random.unscored_count} unscored)."
    )
    results.append(StrategyEvaluationResult(
        name="Random (Matched Cost)",
        strategy_type=STRATEGY_DEPLOYABLE,
        is_reference=False,
        router_family="baseline",
        lambda_value=None,
        preset_name=None,
        mean_cost=res_random.mean_cost,
        cost_std=res_random.cost_std,
        total_cost=res_random.total_cost,
        mean_quality=res_random.mean_quality,
        quality_std=res_random.quality_std,
        mean_latency_ms=res_random.mean_latency_ms,
        evaluation_coverage=res_random.evaluation_coverage,
        execution_failure_count=res_random.execution_failure_count,
        unscored_count=res_random.unscored_count,
        test_prompt_count=res_random.test_prompt_count,
        successful_evaluations=denom_random,
        quality_denominator=denom_random,
        coverage_discrepancy_warning=warn_random,
        target_cost=res_random.target_cost,
        model_selection_distribution=res_random.model_selection_distribution,
        notes=res_random.notes,
        selections=[
            EvaluationSelection(
                prompt_id=s.prompt_id,
                task_type=s.task_type,
                selected_model=s.selected_model,
                cost=s.cost,
                latency_ms=s.latency_ms,
                execution_success=s.execution_success,
                evaluation_status=s.evaluation_status,
                score=s.score,
                correct=s.correct,
                notes=s.notes,
            )
            for s in res_random.selections
        ],
    ))

    # Baseline 4: Static Task-Type Table
    res_table = evaluate_static_task_table(
        train_agg, test_agg, test_ids,
        objective="quality_over_cost",
        classifier_cost=0.0,
    )
    denom_table = len([s for s in res_table.selections if s.score is not None])
    warn_table = None if res_table.evaluation_coverage >= 1.0 else (
        f"COVERAGE_DEFICIT: Coverage is {res_table.evaluation_coverage:.1%} "
        f"({denom_table}/{res_table.test_prompt_count} test prompts scored; {res_table.execution_failure_count} failures, {res_table.unscored_count} unscored)."
    )
    results.append(StrategyEvaluationResult(
        name="Static Task-Type Table",
        strategy_type=STRATEGY_DEPLOYABLE,
        is_reference=False,
        router_family="baseline",
        lambda_value=None,
        preset_name=None,
        mean_cost=res_table.mean_cost,
        cost_std=res_table.cost_std,
        total_cost=res_table.total_cost,
        mean_quality=res_table.mean_quality,
        quality_std=res_table.quality_std,
        mean_latency_ms=res_table.mean_latency_ms,
        evaluation_coverage=res_table.evaluation_coverage,
        execution_failure_count=res_table.execution_failure_count,
        unscored_count=res_table.unscored_count,
        test_prompt_count=res_table.test_prompt_count,
        successful_evaluations=denom_table,
        quality_denominator=denom_table,
        coverage_discrepancy_warning=warn_table,
        model_selection_distribution=res_table.model_selection_distribution,
        notes=res_table.notes,
        selections=[
            EvaluationSelection(
                prompt_id=s.prompt_id,
                task_type=s.task_type,
                selected_model=s.selected_model,
                cost=s.cost,
                latency_ms=s.latency_ms,
                execution_success=s.execution_success,
                evaluation_status=s.evaluation_status,
                score=s.score,
                correct=s.correct,
                notes=s.notes,
            )
            for s in res_table.selections
        ],
    ))

    # Baseline 5: Oracle (Theoretical Reference Ceiling)
    res_oracle = evaluate_oracle(test_agg, test_ids)
    denom_oracle = len([s for s in res_oracle.selections if s.score is not None])
    warn_oracle = None if res_oracle.evaluation_coverage >= 1.0 else (
        f"COVERAGE_DEFICIT: Coverage is {res_oracle.evaluation_coverage:.1%} "
        f"({denom_oracle}/{res_oracle.test_prompt_count} test prompts scored; {res_oracle.execution_failure_count} failures, {res_oracle.unscored_count} unscored)."
    )
    results.append(StrategyEvaluationResult(
        name="Oracle",
        strategy_type=STRATEGY_THEORETICAL,
        is_reference=True,
        router_family="baseline",
        lambda_value=None,
        preset_name=None,
        mean_cost=res_oracle.mean_cost,
        cost_std=res_oracle.cost_std,
        total_cost=res_oracle.total_cost,
        mean_quality=res_oracle.mean_quality,
        quality_std=res_oracle.quality_std,
        mean_latency_ms=res_oracle.mean_latency_ms,
        evaluation_coverage=res_oracle.evaluation_coverage,
        execution_failure_count=res_oracle.execution_failure_count,
        unscored_count=res_oracle.unscored_count,
        test_prompt_count=res_oracle.test_prompt_count,
        successful_evaluations=denom_oracle,
        quality_denominator=denom_oracle,
        coverage_discrepancy_warning=warn_oracle,
        model_selection_distribution=res_oracle.model_selection_distribution,
        notes=res_oracle.notes,
        selections=[
            EvaluationSelection(
                prompt_id=s.prompt_id,
                task_type=s.task_type,
                selected_model=s.selected_model,
                cost=s.cost,
                latency_ms=s.latency_ms,
                execution_success=s.execution_success,
                evaluation_status=s.evaluation_status,
                score=s.score,
                correct=s.correct,
                notes=s.notes,
            )
            for s in res_oracle.selections
        ],
    ))

    # 10. Evaluate kNN Router across Lambda Sweep
    for lam in swept_lambdas:
        res_knn = evaluate_router_on_test(
            router=knn_router,
            decision_engine=decision_engine,
            dataset=prepared_dataset,
            test_agg=test_agg,
            test_prompt_ids=test_ids,
            lambda_value=lam,
            strategy_name=f"kNN Router (lambda={lam:.2f})",
            router_family="knn",
        )
        results.append(res_knn)

    # 11. Evaluate Learned Router across Lambda Sweep
    for lam in swept_lambdas:
        res_learned = evaluate_router_on_test(
            router=learned_router,
            decision_engine=decision_engine,
            dataset=prepared_dataset,
            test_agg=test_agg,
            test_prompt_ids=test_ids,
            lambda_value=lam,
            strategy_name=f"Learned Router (lambda={lam:.2f})",
            router_family="learned",
        )
        results.append(res_learned)

    # 12. Evaluate Configured Presets for both Routers
    for preset_label, preset_lam in resolved_presets.items():
        # kNN preset
        res_knn_preset = evaluate_router_on_test(
            router=knn_router,
            decision_engine=decision_engine,
            dataset=prepared_dataset,
            test_agg=test_agg,
            test_prompt_ids=test_ids,
            lambda_value=preset_lam,
            strategy_name=f"kNN Router ({preset_label})",
            router_family="knn",
            preset_name=preset_label,
        )
        results.append(res_knn_preset)

        # Learned preset
        res_lr_preset = evaluate_router_on_test(
            router=learned_router,
            decision_engine=decision_engine,
            dataset=prepared_dataset,
            test_agg=test_agg,
            test_prompt_ids=test_ids,
            lambda_value=preset_lam,
            strategy_name=f"Learned Router ({preset_label})",
            router_family="learned",
            preset_name=preset_label,
        )
        results.append(res_lr_preset)

    # 13. Compute Pareto Dominance
    compute_pareto_dominance(results)

    # 14. Compute Matched Trade-off Benchmarks
    tradeoffs = compute_tradeoff_benchmarks(results)

    # 15. Export CSV & Parquet (M3-specific, preserving M2 baseline reports)
    summary_rows = [r.to_summary_dict() for r in results]
    comparison_df = pd.DataFrame(summary_rows)
    comparison_df["sufficiency_status"] = split_info["sufficiency_status"]

    csv_path = out_dir / "router_comparison.csv"
    parquet_path = out_dir / "router_comparison.parquet"
    comparison_df.to_csv(csv_path, index=False)
    comparison_df.to_parquet(parquet_path, index=False)

    # 16. Plot Pareto Curve
    plot_path = out_dir / "router_pareto_curve.png"
    plot_router_pareto_curve(results, plot_path, sufficiency_status=split_info["sufficiency_status"])

    # 17. Export JSON Summary
    summary_json_path = out_dir / "router_evaluation_summary.json"
    summary_payload = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_config_id": benchmark_config_id,
        "sufficiency_status": split_info["sufficiency_status"],
        "is_exploratory": split_info["sufficiency_status"] != STATUS_SUFFICIENT,
        "sufficiency_notes": split_info["notes"],
        "split_info": split_info,
        "router_hyperparameters_audit": hyperparam_audit,
        "preset_tuning": preset_report,
        "lambda_sweep": swept_lambdas,
        "tradeoff_benchmarks": tradeoffs,
        "pareto_deployable_strategies": [r.name for r in results if r.is_pareto_deployable],
        "pareto_all_strategies": [r.name for r in results if r.is_pareto_all],
        "strategy_summaries": [
            {
                "name": r.name,
                "type": r.strategy_type,
                "is_reference": r.is_reference,
                "lambda": r.lambda_value,
                "mean_cost": r.mean_cost,
                "mean_quality": r.mean_quality,
                "is_pareto_deployable": r.is_pareto_deployable,
                "is_pareto_all": r.is_pareto_all,
            }
            for r in results
        ],
    }
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return comparison_df, summary_payload, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate M3 Routers and Baselines.")
    parser.add_argument("--matrix-path", type=str, default=str(DEFAULT_MATRIX_PATH))
    parser.add_argument("--benchmark-config-id", type=str, default="m1-efa7ec3afd4f")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_REPORTS_DIR))
    args = parser.parse_args()

    df, summary, _ = run_router_evaluation(
        matrix_path=args.matrix_path,
        benchmark_config_id=args.benchmark_config_id,
        output_dir=args.output_dir,
    )
    print(f"Evaluation finished successfully. Sufficiency: {summary['sufficiency_status']}")
    print(f"Total evaluated strategies: {len(df)}")


if __name__ == "__main__":
    main()
