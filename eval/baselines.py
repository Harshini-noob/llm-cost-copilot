"""
Baseline Evaluation Engine for LLM Cost Copilot (Phase M2).

Evaluates five baseline strategies:
1. Always-Cheapest (Deployable Policy)
2. Always-Best (Post-Hoc Test-Set Reference Ceiling)
3. Random at Matched Cost (Deployable Policy with Repeated Sampling)
4. Static Task-Type Table (Deployable Policy with Quality/Cost Optimization)
5. Oracle (Theoretical Reference Ceiling)

Complies with failure accounting, test leakage prevention, zero-cost classifier
defaults for metadata labels, and real-data sufficiency flags.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eval.dataset import SUPPORTED_TASKS, load_dataset

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_PATH = PROJECT_ROOT / "data" / "processed" / "matrix.parquet"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "pilot_prompts.jsonl"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

STRATEGY_DEPLOYABLE = "deployable_policy"
STRATEGY_POST_HOC = "post_hoc_reference"
STRATEGY_THEORETICAL = "theoretical_reference"

STATUS_SUFFICIENT = "SUFFICIENT_DATA"
STATUS_INSUFFICIENT = "INSUFFICIENT_DATA"
STATUS_EXPLORATORY = "EXPLORATORY"


@dataclass
class BaselineSelection:
    """Detailed record of a model selection for a single test prompt."""
    prompt_id: str
    task_type: str
    selected_model: str
    cost: float
    latency_ms: float
    execution_success: bool
    evaluation_status: str | None
    score: float | None
    correct: bool | None
    notes: str = ""


@dataclass
class BaselineResult:
    """Summary metrics for a baseline evaluation strategy."""
    name: str
    strategy_type: str
    mean_cost: float
    total_cost: float
    mean_quality: float | None
    mean_latency_ms: float
    evaluation_coverage: float
    execution_failure_count: int
    unscored_count: int
    test_prompt_count: int
    target_cost: float | None = None
    expected_cost: float | None = None
    cost_delta_from_target: float | None = None
    relative_cost_deviation: float | None = None
    cost_std: float | None = None
    quality_std: float | None = None
    model_selection_distribution: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    selections: list[BaselineSelection] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "baseline_name": self.name,
            "strategy_type": self.strategy_type,
            "mean_cost": self.mean_cost,
            "total_cost": self.total_cost,
            "mean_quality": self.mean_quality,
            "quality_std": self.quality_std,
            "mean_latency_ms": self.mean_latency_ms,
            "evaluation_coverage": self.evaluation_coverage,
            "execution_failure_count": self.execution_failure_count,
            "unscored_count": self.unscored_count,
            "test_prompt_count": self.test_prompt_count,
            "target_cost": self.target_cost,
            "expected_cost": self.expected_cost,
            "cost_delta_from_target": self.cost_delta_from_target,
            "relative_cost_deviation": self.relative_cost_deviation,
            "cost_std": self.cost_std,
            "model_selection_distribution": json.dumps(self.model_selection_distribution),
            "notes": self.notes,
        }


def load_and_filter_matrix(
    matrix_path: Path | str = DEFAULT_MATRIX_PATH,
    benchmark_config_id: str | None = "m1-efa7ec3afd4f",
) -> pd.DataFrame:
    """Load benchmark matrix without silently dropping failures or unevaluated records."""
    path = Path(matrix_path)
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")

    df = pd.read_parquet(path)

    if benchmark_config_id is not None:
        if "benchmark_config_id" in df.columns:
            filtered = df[df["benchmark_config_id"] == benchmark_config_id].copy()
            if filtered.empty:
                logger.warning(
                    "No records matched benchmark_config_id=%s; returning empty DataFrame",
                    benchmark_config_id,
                )
            return filtered

    return df.copy()


def aggregate_prompt_model_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple runs for each (prompt_id, model_id) pair.
    
    Preserves execution failure status, captures actual costs incurred,
    and accurately separates evaluated scores from missing/unscored evaluations.
    """
    if df.empty:
        return pd.DataFrame()

    records = []
    grouped = df.groupby(["prompt_id", "model_id"], as_index=False)

    for (prompt_id, model_id), group in grouped:
        task_type = group["task_type"].iloc[0] if "task_type" in group.columns else "unknown"
        provider = group["provider"].iloc[0] if "provider" in group.columns else "unknown"

        run_count = len(group)
        execution_successes = group["execution_success"].tolist() if "execution_success" in group.columns else [True] * run_count
        execution_success_rate = float(sum(bool(x) for x in execution_successes) / run_count)
        all_execution_success = bool(execution_success_rate == 1.0)

        mean_cost = float(group["cost"].mean()) if "cost" in group.columns else 0.0
        total_incurred_cost = float(group["cost"].sum()) if "cost" in group.columns else 0.0
        mean_latency = float(group["latency_ms"].mean()) if "latency_ms" in group.columns else 0.0

        # Evaluation accounting
        # Filter to runs with scored evaluations
        scored_mask = (group["evaluation_status"] == "scored") & (~group["score"].isna()) if "evaluation_status" in group.columns and "score" in group.columns else pd.Series([False] * run_count)
        scored_runs = group[scored_mask]

        if len(scored_runs) > 0:
            mean_score = float(scored_runs["score"].mean())
            majority_correct = bool((scored_runs["score"] >= 1.0).sum() > (len(scored_runs) / 2))
            eval_status = "scored"
        else:
            mean_score = None
            majority_correct = None
            if not all_execution_success:
                eval_status = "execution_failed"
            elif "evaluation_status" in group.columns and (group["evaluation_status"] == "error").any():
                eval_status = "evaluation_error"
            else:
                eval_status = "unscored"

        records.append({
            "prompt_id": prompt_id,
            "task_type": task_type,
            "model_id": model_id,
            "provider": provider,
            "run_count": run_count,
            "execution_success": all_execution_success,
            "execution_success_rate": execution_success_rate,
            "cost": mean_cost,
            "total_incurred_cost": total_incurred_cost,
            "latency_ms": mean_latency,
            "score": mean_score,
            "correct": majority_correct,
            "evaluation_status": eval_status,
        })

    return pd.DataFrame(records)


def create_stratified_split(
    prompts: list[dict[str, Any]] | list[str] | pd.DataFrame,
    split_ratio: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Partition prompts into deterministic train and test sets stratified by task_type.
    
    Guarantees zero test prompt leakage into train set.
    """
    rng = np.random.default_rng(seed)

    # Normalize prompt list to list of dicts with prompt_id and task_type
    prompt_items: list[dict[str, str]] = []
    if isinstance(prompts, pd.DataFrame):
        for _, row in prompts[["prompt_id", "task_type"]].drop_duplicates().iterrows():
            prompt_items.append({"prompt_id": str(row["prompt_id"]), "task_type": str(row["task_type"])})
    elif isinstance(prompts, list):
        for item in prompts:
            if isinstance(item, dict):
                prompt_items.append({
                    "prompt_id": str(item["prompt_id"]),
                    "task_type": str(item.get("task_type", "unknown")),
                })
            else:
                prompt_items.append({"prompt_id": str(item), "task_type": "unknown"})
    else:
        raise TypeError(f"Unsupported prompts type: {type(prompts)}")

    # Sort deterministically before grouping
    prompt_items.sort(key=lambda x: x["prompt_id"])

    # Group by task_type
    by_task: dict[str, list[str]] = {}
    for item in prompt_items:
        by_task.setdefault(item["task_type"], []).append(item["prompt_id"])

    train_ids: list[str] = []
    test_ids: list[str] = []

    for task, ids in sorted(by_task.items()):
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n = len(shuffled)
        if n == 1:
            # Singleton category: cannot stratify evenly. Deterministically place in train.
            train_ids.append(shuffled[0])
        elif n == 2:
            # Exactly 2 prompts (like pilot dataset per category): exactly 1 train, 1 test
            train_ids.append(shuffled[0])
            test_ids.append(shuffled[1])
        else:
            n_train = max(1, min(n - 1, int(round(n * split_ratio))))
            train_ids.extend(shuffled[:n_train])
            test_ids.extend(shuffled[n_train:])

    train_ids.sort()
    test_ids.sort()

    # Integrity verification: zero overlap
    overlap = set(train_ids).intersection(set(test_ids))
    if overlap:
        raise ValueError(f"Integrity error: train and test sets overlap on {overlap}")

    # Coverage and data sufficiency assessment
    all_categories_in_test = set(by_task.keys())
    test_categories = {item["task_type"] for item in prompt_items if item["prompt_id"] in test_ids}
    
    if len(prompt_items) < 12 or len(test_categories) < len(SUPPORTED_TASKS) or len(test_ids) < 6:
        sufficiency_status = STATUS_INSUFFICIENT
        notes = (
            f"INSUFFICIENT_DATA: Active prompts count is {len(prompt_items)} (test count: {len(test_ids)}). "
            f"Only {len(test_categories)} of {len(SUPPORTED_TASKS)} standard task categories represented in test split. "
            "Results are EXPLORATORY and NOT statistically reliable."
        )
    else:
        sufficiency_status = STATUS_SUFFICIENT
        notes = f"SUFFICIENT_DATA: {len(test_ids)} test prompts across {len(test_categories)} categories."

    return {
        "train_prompt_ids": train_ids,
        "test_prompt_ids": test_ids,
        "split_ratio": split_ratio,
        "seed": seed,
        "sufficiency_status": sufficiency_status,
        "total_prompts": len(prompt_items),
        "train_prompt_count": len(train_ids),
        "test_prompt_count": len(test_ids),
        "task_distribution": {k: len(v) for k, v in by_task.items()},
        "test_categories": sorted(list(test_categories)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }


def evaluate_always_cheapest(
    train_agg: pd.DataFrame,
    test_agg: pd.DataFrame,
    test_prompt_ids: list[str],
) -> BaselineResult:
    """
    Baseline 1: Always-Cheapest (Deployable Policy).
    
    Identifies the model with the lowest mean cost on training data and routes all test prompts to it.
    """
    # Learn cheapest model from training data only
    if train_agg.empty:
        raise ValueError("Cannot evaluate Always-Cheapest with empty training data.")

    # Compute mean cost per model on train
    train_costs = train_agg.groupby("model_id")["cost"].mean()
    cheapest_model = train_costs.idxmin()

    # Route all test prompts to cheapest_model
    selections: list[BaselineSelection] = []
    costs: list[float] = []
    latencies: list[float] = []
    scores: list[float] = []
    failures = 0
    unscored = 0

    for prompt_id in test_prompt_ids:
        match = test_agg[(test_agg["prompt_id"] == prompt_id) & (test_agg["model_id"] == cheapest_model)]
        if match.empty:
            failures += 1
            unscored += 1
            selections.append(BaselineSelection(
                prompt_id=prompt_id,
                task_type="unknown",
                selected_model=cheapest_model,
                cost=0.0,
                latency_ms=0.0,
                execution_success=False,
                evaluation_status="missing_candidate_run",
                score=None,
                correct=None,
                notes="Candidate execution record missing in benchmark matrix",
            ))
            continue

        row = match.iloc[0]
        cost = float(row["cost"])
        latency = float(row["latency_ms"])
        success = bool(row["execution_success"])
        eval_status = row["evaluation_status"]
        score = float(row["score"]) if pd.notna(row["score"]) else None
        correct = bool(row["correct"]) if pd.notna(row["correct"]) else None

        costs.append(cost)
        latencies.append(latency)
        if not success:
            failures += 1
        if eval_status == "scored" and score is not None:
            scores.append(score)
        else:
            unscored += 1

        selections.append(BaselineSelection(
            prompt_id=prompt_id,
            task_type=row["task_type"],
            selected_model=cheapest_model,
            cost=cost,
            latency_ms=latency,
            execution_success=success,
            evaluation_status=eval_status,
            score=score,
            correct=correct,
        ))

    n_test = len(test_prompt_ids)
    mean_cost = float(np.mean(costs)) if costs else 0.0
    total_cost = float(np.sum(costs)) if costs else 0.0
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    mean_quality = float(np.mean(scores)) if scores else None
    coverage = len(scores) / n_test if n_test > 0 else 0.0

    return BaselineResult(
        name="Always-Cheapest",
        strategy_type=STRATEGY_DEPLOYABLE,
        mean_cost=mean_cost,
        total_cost=total_cost,
        mean_quality=mean_quality,
        mean_latency_ms=mean_latency,
        evaluation_coverage=coverage,
        execution_failure_count=failures,
        unscored_count=unscored,
        test_prompt_count=n_test,
        model_selection_distribution={cheapest_model: 1.0},
        notes=f"Selected model '{cheapest_model}' with train mean cost ${train_costs[cheapest_model]:.6f}.",
        selections=selections,
    )


def evaluate_always_best(
    test_agg: pd.DataFrame,
    test_prompt_ids: list[str],
) -> BaselineResult:
    """
    Baseline 2: Always-Best (Post-Hoc Test-Set Reference Ceiling).
    
    Selects the single model that achieves the highest quality on the test set.
    Clearly labeled as post-hoc reference, not a deployable train-time selection policy.
    """
    if test_agg.empty:
        raise ValueError("Cannot evaluate Always-Best with empty test data.")

    # Only evaluate candidate models across test prompts
    candidate_metrics = []
    models = sorted(test_agg["model_id"].unique())

    for model_id in models:
        m_subset = test_agg[(test_agg["model_id"] == model_id) & (test_agg["prompt_id"].isin(test_prompt_ids))]
        scored = m_subset[(m_subset["evaluation_status"] == "scored") & (~m_subset["score"].isna())]
        quality = float(scored["score"].mean()) if not scored.empty else -1.0
        cost = float(m_subset["cost"].mean()) if not m_subset.empty else 0.0
        candidate_metrics.append({
            "model_id": model_id,
            "quality": quality,
            "cost": cost,
        })

    # Pick model with highest test quality, tie-breaking by lower cost
    candidate_metrics.sort(key=lambda x: (-x["quality"], x["cost"]))
    best_model = candidate_metrics[0]["model_id"]

    selections: list[BaselineSelection] = []
    costs: list[float] = []
    latencies: list[float] = []
    scores: list[float] = []
    failures = 0
    unscored = 0

    for prompt_id in test_prompt_ids:
        match = test_agg[(test_agg["prompt_id"] == prompt_id) & (test_agg["model_id"] == best_model)]
        if match.empty:
            failures += 1
            unscored += 1
            selections.append(BaselineSelection(
                prompt_id=prompt_id,
                task_type="unknown",
                selected_model=best_model,
                cost=0.0,
                latency_ms=0.0,
                execution_success=False,
                evaluation_status="missing_candidate_run",
                score=None,
                correct=None,
            ))
            continue

        row = match.iloc[0]
        cost = float(row["cost"])
        latency = float(row["latency_ms"])
        success = bool(row["execution_success"])
        eval_status = row["evaluation_status"]
        score = float(row["score"]) if pd.notna(row["score"]) else None
        correct = bool(row["correct"]) if pd.notna(row["correct"]) else None

        costs.append(cost)
        latencies.append(latency)
        if not success:
            failures += 1
        if eval_status == "scored" and score is not None:
            scores.append(score)
        else:
            unscored += 1

        selections.append(BaselineSelection(
            prompt_id=prompt_id,
            task_type=row["task_type"],
            selected_model=best_model,
            cost=cost,
            latency_ms=latency,
            execution_success=success,
            evaluation_status=eval_status,
            score=score,
            correct=correct,
        ))

    n_test = len(test_prompt_ids)
    mean_cost = float(np.mean(costs)) if costs else 0.0
    total_cost = float(np.sum(costs)) if costs else 0.0
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    mean_quality = float(np.mean(scores)) if scores else None
    coverage = len(scores) / n_test if n_test > 0 else 0.0

    return BaselineResult(
        name="Always-Best",
        strategy_type=STRATEGY_POST_HOC,
        mean_cost=mean_cost,
        total_cost=total_cost,
        mean_quality=mean_quality,
        mean_latency_ms=mean_latency,
        evaluation_coverage=coverage,
        execution_failure_count=failures,
        unscored_count=unscored,
        test_prompt_count=n_test,
        model_selection_distribution={best_model: 1.0},
        notes="Post-hoc test-set reference ceiling; not a deployable train-time selection policy.",
        selections=selections,
    )


def evaluate_random_matched_cost(
    train_agg: pd.DataFrame,
    test_agg: pd.DataFrame,
    test_prompt_ids: list[str],
    target_cost: float | None = None,
    n_repetitions: int = 20,
    seed: int = 42,
) -> BaselineResult:
    """
    Baseline 3: Random at Matched Cost (Deployable Policy).
    
    Randomly selects models using a reproducible seed to match a target expected cost.
    Measures and reports sampling variability over multiple repetitions.
    """
    if train_agg.empty or test_agg.empty:
        raise ValueError("Cannot evaluate Random at Matched Cost with empty data.")

    # Calculate mean train cost per candidate model
    train_costs = train_agg.groupby("model_id")["cost"].mean().to_dict()
    models = sorted(train_costs.keys())
    if not models:
        raise ValueError("No models found in training data.")

    costs_array = np.array([train_costs[m] for m in models], dtype=float)
    min_cost = float(np.min(costs_array))
    max_cost = float(np.max(costs_array))

    if target_cost is None:
        target_cost = float(np.mean(costs_array))

    # Compute probability weights over models matching target expected cost
    # Convex combination between cheapest and most expensive or uniform interpolation
    weights = np.zeros(len(models), dtype=float)
    if target_cost <= min_cost:
        min_idx = int(np.argmin(costs_array))
        weights[min_idx] = 1.0
        expected_cost = min_cost
    elif target_cost >= max_cost:
        max_idx = int(np.argmax(costs_array))
        weights[max_idx] = 1.0
        expected_cost = max_cost
    else:
        # Interpolate between min and max model to match target cost exactly in expectation
        min_idx = int(np.argmin(costs_array))
        max_idx = int(np.argmax(costs_array))
        alpha = (max_cost - target_cost) / (max_cost - min_cost)
        weights[min_idx] = alpha
        weights[max_idx] = 1.0 - alpha
        expected_cost = float(np.sum(weights * costs_array))

    # Run n_repetitions
    n_test = len(test_prompt_ids)
    rep_mean_costs: list[float] = []
    rep_qualities: list[float] = []
    rep_latencies: list[float] = []
    all_selections_counts: dict[str, int] = {m: 0 for m in models}
    
    first_rep_selections: list[BaselineSelection] = []
    total_failures = 0
    total_unscored = 0

    for rep in range(n_repetitions):
        rng = np.random.default_rng(seed + rep)
        cur_costs: list[float] = []
        cur_scores: list[float] = []
        cur_latencies: list[float] = []

        for prompt_id in test_prompt_ids:
            chosen_model = str(rng.choice(models, p=weights))
            all_selections_counts[chosen_model] += 1

            match = test_agg[(test_agg["prompt_id"] == prompt_id) & (test_agg["model_id"] == chosen_model)]
            if match.empty:
                cur_costs.append(0.0)
                cur_latencies.append(0.0)
                if rep == 0:
                    total_failures += 1
                    total_unscored += 1
                    first_rep_selections.append(BaselineSelection(
                        prompt_id=prompt_id,
                        task_type="unknown",
                        selected_model=chosen_model,
                        cost=0.0,
                        latency_ms=0.0,
                        execution_success=False,
                        evaluation_status="missing_candidate_run",
                        score=None,
                        correct=None,
                    ))
                continue

            row = match.iloc[0]
            cost = float(row["cost"])
            latency = float(row["latency_ms"])
            success = bool(row["execution_success"])
            eval_status = row["evaluation_status"]
            score = float(row["score"]) if pd.notna(row["score"]) else None
            correct = bool(row["correct"]) if pd.notna(row["correct"]) else None

            cur_costs.append(cost)
            cur_latencies.append(latency)
            if eval_status == "scored" and score is not None:
                cur_scores.append(score)
            elif rep == 0:
                total_unscored += 1

            if not success and rep == 0:
                total_failures += 1

            if rep == 0:
                first_rep_selections.append(BaselineSelection(
                    prompt_id=prompt_id,
                    task_type=row["task_type"],
                    selected_model=chosen_model,
                    cost=cost,
                    latency_ms=latency,
                    execution_success=success,
                    evaluation_status=eval_status,
                    score=score,
                    correct=correct,
                ))

        rep_mean_costs.append(float(np.mean(cur_costs)) if cur_costs else 0.0)
        rep_latencies.append(float(np.mean(cur_latencies)) if cur_latencies else 0.0)
        if cur_scores:
            rep_qualities.append(float(np.mean(cur_scores)))

    realized_mean_cost = float(np.mean(rep_mean_costs))
    cost_std = float(np.std(rep_mean_costs)) if len(rep_mean_costs) > 1 else 0.0
    cost_delta = abs(realized_mean_cost - target_cost)
    rel_cost_deviation = cost_delta / target_cost if (target_cost is not None and target_cost > 0) else 0.0

    realized_mean_quality = float(np.mean(rep_qualities)) if rep_qualities else None
    quality_std = float(np.std(rep_qualities)) if len(rep_qualities) > 1 else 0.0
    mean_latency = float(np.mean(rep_latencies))

    total_choices = sum(all_selections_counts.values())
    distribution = {m: count / total_choices for m, count in all_selections_counts.items() if count > 0}
    coverage = len(first_rep_selections) - total_unscored
    coverage_rate = coverage / n_test if n_test > 0 else 0.0

    notes = (
        f"Target cost: ${target_cost:.6f}, Expected cost: ${expected_cost:.6f}, "
        f"Realized mean cost: ${realized_mean_cost:.6f} +/- ${cost_std:.6f} (delta: ${cost_delta:.6f}, rel_dev: {rel_cost_deviation:.2%}). "
        f"Evaluated over {n_repetitions} repetitions."
    )

    return BaselineResult(
        name="Random (Matched Cost)",
        strategy_type=STRATEGY_DEPLOYABLE,
        mean_cost=realized_mean_cost,
        total_cost=realized_mean_cost * n_test,
        mean_quality=realized_mean_quality,
        mean_latency_ms=mean_latency,
        evaluation_coverage=coverage_rate,
        execution_failure_count=total_failures,
        unscored_count=total_unscored,
        test_prompt_count=n_test,
        target_cost=target_cost,
        expected_cost=expected_cost,
        cost_delta_from_target=cost_delta,
        relative_cost_deviation=rel_cost_deviation,
        cost_std=cost_std,
        quality_std=quality_std,
        model_selection_distribution=distribution,
        notes=notes,
        selections=first_rep_selections,
    )


def evaluate_static_task_table(
    train_agg: pd.DataFrame,
    test_agg: pd.DataFrame,
    test_prompt_ids: list[str],
    objective: str = "quality_over_cost",
    classifier_cost: float = 0.0,
) -> BaselineResult:
    """
    Baseline 4: Static Task-Type Table (Deployable Policy).
    
    Learns a mapping from task_type -> best-value model from training data only.
    Default objective is Quality / MeanCost. Includes configurable classifier cost.
    """
    if train_agg.empty or test_agg.empty:
        raise ValueError("Cannot evaluate Static Task Table with empty data.")

    # Learn table per task_type on training data
    task_table: dict[str, str] = {}
    train_tasks = train_agg["task_type"].unique()

    def _calc_value(q: float, c: float) -> float:
        if objective == "quality_over_cost":
            safe_c = max(c, 1e-9)
            return q / safe_c
        elif objective == "quality_only":
            return q
        elif objective == "cost_only":
            return -c
        else:
            raise ValueError(f"Unknown objective: {objective}")

    for task in train_tasks:
        sub = train_agg[train_agg["task_type"] == task]
        model_values = []
        for model_id, m_group in sub.groupby("model_id"):
            scored = m_group[(m_group["evaluation_status"] == "scored") & (~m_group["score"].isna())]
            q = float(scored["score"].mean()) if not scored.empty else 0.0
            c = float(m_group["cost"].mean()) if not m_group.empty else 0.0
            val = _calc_value(q, c)
            model_values.append((val, -c, q, model_id))

        # Sort descending by value, tie break by lower cost (higher -c), then higher quality
        model_values.sort(reverse=True)
        task_table[task] = model_values[0][3]

    # Global train fallback for unseen test tasks
    overall_values = []
    for model_id, m_group in train_agg.groupby("model_id"):
        scored = m_group[(m_group["evaluation_status"] == "scored") & (~m_group["score"].isna())]
        q = float(scored["score"].mean()) if not scored.empty else 0.0
        c = float(m_group["cost"].mean()) if not m_group.empty else 0.0
        val = _calc_value(q, c)
        overall_values.append((val, -c, q, model_id))
    overall_values.sort(reverse=True)
    overall_fallback_model = overall_values[0][3]

    # Apply to test prompts
    selections: list[BaselineSelection] = []
    costs: list[float] = []
    latencies: list[float] = []
    scores: list[float] = []
    failures = 0
    unscored = 0
    fallback_count = 0
    model_counts: dict[str, int] = {}

    for prompt_id in test_prompt_ids:
        # Determine prompt task_type from test_agg
        p_rows = test_agg[test_agg["prompt_id"] == prompt_id]
        task_type = p_rows["task_type"].iloc[0] if not p_rows.empty else "unknown"

        if task_type in task_table:
            chosen_model = task_table[task_type]
            used_fallback = False
        else:
            chosen_model = overall_fallback_model
            used_fallback = True
            fallback_count += 1

        model_counts[chosen_model] = model_counts.get(chosen_model, 0) + 1

        match = test_agg[(test_agg["prompt_id"] == prompt_id) & (test_agg["model_id"] == chosen_model)]
        if match.empty:
            failures += 1
            unscored += 1
            selections.append(BaselineSelection(
                prompt_id=prompt_id,
                task_type=task_type,
                selected_model=chosen_model,
                cost=classifier_cost,
                latency_ms=0.0,
                execution_success=False,
                evaluation_status="missing_candidate_run",
                score=None,
                correct=None,
                notes=f"Missing candidate run. Fallback={used_fallback}",
            ))
            costs.append(classifier_cost)
            continue

        row = match.iloc[0]
        base_cost = float(row["cost"])
        total_prompt_cost = base_cost + classifier_cost
        latency = float(row["latency_ms"])
        success = bool(row["execution_success"])
        eval_status = row["evaluation_status"]
        score = float(row["score"]) if pd.notna(row["score"]) else None
        correct = bool(row["correct"]) if pd.notna(row["correct"]) else None

        costs.append(total_prompt_cost)
        latencies.append(latency)
        if not success:
            failures += 1
        if eval_status == "scored" and score is not None:
            scores.append(score)
        else:
            unscored += 1

        selections.append(BaselineSelection(
            prompt_id=prompt_id,
            task_type=task_type,
            selected_model=chosen_model,
            cost=total_prompt_cost,
            latency_ms=latency,
            execution_success=success,
            evaluation_status=eval_status,
            score=score,
            correct=correct,
            notes=f"Table={chosen_model}; Fallback={used_fallback}",
        ))

    n_test = len(test_prompt_ids)
    mean_cost = float(np.mean(costs)) if costs else 0.0
    total_cost = float(np.sum(costs)) if costs else 0.0
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    mean_quality = float(np.mean(scores)) if scores else None
    coverage = len(scores) / n_test if n_test > 0 else 0.0
    dist = {m: c / n_test for m, c in model_counts.items()} if n_test > 0 else {}

    classifier_note = (
        "Task labels supplied via benchmark metadata; classifier inference cost set to $0.000000."
        if classifier_cost == 0.0
        else f"Classifier inference cost included at ${classifier_cost:.6f} per prompt."
    )
    notes = (
        f"Learned mapping: {task_table}. Objective: '{objective}'. "
        f"{classifier_note} Unseen task fallback count: {fallback_count}/{n_test}."
    )

    return BaselineResult(
        name="Static Task-Type Table",
        strategy_type=STRATEGY_DEPLOYABLE,
        mean_cost=mean_cost,
        total_cost=total_cost,
        mean_quality=mean_quality,
        mean_latency_ms=mean_latency,
        evaluation_coverage=coverage,
        execution_failure_count=failures,
        unscored_count=unscored,
        test_prompt_count=n_test,
        model_selection_distribution=dist,
        notes=notes,
        selections=selections,
    )


def evaluate_oracle(
    test_agg: pd.DataFrame,
    test_prompt_ids: list[str],
) -> BaselineResult:
    """
    Baseline 5: Oracle (Theoretical Reference Ceiling).
    
    For each test prompt, picks the cheapest model that answered correctly (score >= 1.0).
    If no model answered correctly or evaluated, falls back to the cheapest available candidate
    and tracks the unanswerable prompt.
    """
    if test_agg.empty:
        raise ValueError("Cannot evaluate Oracle with empty test data.")

    selections: list[BaselineSelection] = []
    costs: list[float] = []
    latencies: list[float] = []
    scores: list[float] = []
    failures = 0
    unscored = 0
    unanswerable_count = 0
    model_counts: dict[str, int] = {}

    for prompt_id in test_prompt_ids:
        candidates = test_agg[test_agg["prompt_id"] == prompt_id]
        if candidates.empty:
            failures += 1
            unscored += 1
            unanswerable_count += 1
            selections.append(BaselineSelection(
                prompt_id=prompt_id,
                task_type="unknown",
                selected_model="none",
                cost=0.0,
                latency_ms=0.0,
                execution_success=False,
                evaluation_status="missing_candidates",
                score=None,
                correct=None,
                notes="No candidate models available in test data",
            ))
            continue

        # Correct candidates: execution_success == True, evaluation_status == 'scored', score >= 1.0 (or correct == True)
        correct_candidates = candidates[
            (candidates["execution_success"]) &
            (candidates["evaluation_status"] == "scored") &
            ((candidates["score"] >= 1.0) | (candidates["correct"] == True))
        ]

        if not correct_candidates.empty:
            # Pick cheapest correct model
            chosen_row = correct_candidates.sort_values(by=["cost", "latency_ms"]).iloc[0]
            unanswerable = False
        else:
            # Fallback to cheapest candidate among available
            chosen_row = candidates.sort_values(by=["cost", "latency_ms"]).iloc[0]
            unanswerable = True
            unanswerable_count += 1

        chosen_model = str(chosen_row["model_id"])
        model_counts[chosen_model] = model_counts.get(chosen_model, 0) + 1

        cost = float(chosen_row["cost"])
        latency = float(chosen_row["latency_ms"])
        success = bool(chosen_row["execution_success"])
        eval_status = chosen_row["evaluation_status"]
        score = float(chosen_row["score"]) if pd.notna(chosen_row["score"]) else None
        correct = bool(chosen_row["correct"]) if pd.notna(chosen_row["correct"]) else None

        costs.append(cost)
        latencies.append(latency)
        if not success:
            failures += 1
        if eval_status == "scored" and score is not None:
            scores.append(score)
        else:
            unscored += 1

        selections.append(BaselineSelection(
            prompt_id=prompt_id,
            task_type=chosen_row["task_type"],
            selected_model=chosen_model,
            cost=cost,
            latency_ms=latency,
            execution_success=success,
            evaluation_status=eval_status,
            score=score,
            correct=correct,
            notes="Correct cheapest model" if not unanswerable else "Unanswerable prompt: fallback to cheapest candidate",
        ))

    n_test = len(test_prompt_ids)
    mean_cost = float(np.mean(costs)) if costs else 0.0
    total_cost = float(np.sum(costs)) if costs else 0.0
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    mean_quality = float(np.mean(scores)) if scores else None
    coverage = len(scores) / n_test if n_test > 0 else 0.0
    dist = {m: c / n_test for m, c in model_counts.items()} if n_test > 0 else {}

    notes = (
        "Theoretical reference ceiling; optimal post-hoc cheapest correct choice per prompt. "
        f"Unanswerable prompts: {unanswerable_count}/{n_test}."
    )

    return BaselineResult(
        name="Oracle",
        strategy_type=STRATEGY_THEORETICAL,
        mean_cost=mean_cost,
        total_cost=total_cost,
        mean_quality=mean_quality,
        mean_latency_ms=mean_latency,
        evaluation_coverage=coverage,
        execution_failure_count=failures,
        unscored_count=unscored,
        test_prompt_count=n_test,
        model_selection_distribution=dist,
        notes=notes,
        selections=selections,
    )


def _plot_with_matplotlib(
    results: list[BaselineResult],
    out_file: Path,
    sufficiency_status: str,
) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=150)

    # Strategy classification colors
    color_map = {
        STRATEGY_DEPLOYABLE: "#1f77b4",     # Blue
        STRATEGY_POST_HOC: "#ff7f0e",       # Orange
        STRATEGY_THEORETICAL: "#2ca02c",    # Green
    }
    # Marker map per baseline
    marker_map = {
        "Always-Cheapest": "o",
        "Always-Best": "s",
        "Random (Matched Cost)": "^",
        "Static Task-Type Table": "D",
        "Oracle": "*",
    }
    size_map = {
        "Oracle": 230,
        "Always-Best": 140,
        "Static Task-Type Table": 90,
        "Always-Cheapest": 45,
        "Random (Matched Cost)": 110,
    }

    # Group baselines by coordinate to detect overlaps
    coord_groups: dict[tuple[float, float], list[BaselineResult]] = {}
    for res in results:
        x = res.mean_cost
        y = res.mean_quality if res.mean_quality is not None else 0.0
        key = (round(x, 8), round(y, 4))
        coord_groups.setdefault(key, []).append(res)

    # Determine X limit including error bars comfortably
    max_cost_with_err = max(r.mean_cost + (r.cost_std or 0.0) for r in results) if results else 0.00005
    x_limit_high = max(max_cost_with_err * 1.30, 0.00005)

    # Plot markers at exact data coordinates (no artificial jitter)
    # When multiple baselines share the exact coordinate, plot concentric markers sorted by size descending
    for (gx, gy), group in coord_groups.items():
        sorted_group = sorted(group, key=lambda r: size_map.get(r.name, 100), reverse=True)
        for res in sorted_group:
            c = color_map.get(res.strategy_type, "#7f7f7f")
            m = marker_map.get(res.name, "o")
            s = size_map.get(res.name, 100)

            if res.cost_std or res.quality_std:
                ax.errorbar(
                    gx, gy,
                    xerr=res.cost_std if res.cost_std else None,
                    yerr=res.quality_std if res.quality_std else None,
                    fmt=m,
                    color=c,
                    ecolor="#555555",
                    elinewidth=1.6,
                    capsize=5,
                    markersize=9,
                    zorder=6,
                )
            else:
                ax.scatter(
                    [gx], [gy],
                    color=c,
                    marker=m,
                    s=s,
                    edgecolors="#222222",
                    linewidths=1.0,
                    zorder=5,
                )

    # Callout annotations with leader lines
    for (gx, gy), group in coord_groups.items():
        if len(group) > 1:
            # Multi-baseline callout card for coincident points
            lines = [f"Coincident Baselines (${gx:.6f}, Q:{gy:.2f}):"]
            for r in group:
                strat_tag = "Deployable" if r.strategy_type == STRATEGY_DEPLOYABLE else ("Post-Hoc Ref" if r.strategy_type == STRATEGY_POST_HOC else "Theoretical Ref")
                lines.append(f"  • {r.name} [{strat_tag}]")
            card_text = "\n".join(lines)
            
            offset_xy = (-35, -110) if gy > 0.5 else (-35, 50)
            ax.annotate(
                card_text,
                xy=(gx, gy),
                xytext=offset_xy,
                textcoords="offset points",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#1f77b4", alpha=0.95, lw=1.2),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2, connectionstyle="arc3,rad=0.15"),
                zorder=10,
            )
        else:
            r = group[0]
            strat_tag = "Deployable" if r.strategy_type == STRATEGY_DEPLOYABLE else ("Post-Hoc Ref" if r.strategy_type == STRATEGY_POST_HOC else "Theoretical Ref")
            if r.cost_std:
                card_text = f"{r.name} [{strat_tag}]\n(${gx:.6f} ± ${r.cost_std:.6f}, Q:{gy:.2f})"
            else:
                card_text = f"{r.name} [{strat_tag}]\n(${gx:.6f}, Q:{gy:.2f})"

            offset_xy = (45, -55) if gy > 0.5 else (45, 40)
            ax.annotate(
                card_text,
                xy=(gx, gy),
                xytext=offset_xy,
                textcoords="offset points",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#888888", alpha=0.95, lw=1.0),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2, connectionstyle="arc3,rad=-0.15"),
                zorder=10,
            )

    # Formatting X and Y axes
    ax.set_xlabel("Mean Cost per Prompt ($ USD)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Test Quality (Accuracy / Score)", fontsize=11, fontweight="bold")
    ax.set_xlim(0.0, x_limit_high)
    ax.set_ylim(-0.08, 1.15)
    
    # Format X axis with 6 decimal places (shows exact microdollar differences)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: f"${val:.6f}"))
    ax.tick_params(axis="x", rotation=15)

    # Title & Subtitle
    if sufficiency_status in (STATUS_INSUFFICIENT, STATUS_EXPLORATORY):
        ax.set_title(
            "Baseline Comparison: Cost vs Quality\n[EXPLORATORY / INSUFFICIENT DATA -- 1 Category (math), 1 Test Prompt]",
            fontsize=12,
            fontweight="bold",
            color="#b30000",
            pad=12,
        )
    else:
        ax.set_title("Baseline Comparison: Cost vs Quality", fontsize=13, fontweight="bold", pad=12)

    # Custom Legend for Baselines
    legend_handles = []
    for name in ["Always-Cheapest", "Always-Best", "Random (Matched Cost)", "Static Task-Type Table", "Oracle"]:
        res_matches = [r for r in results if r.name == name]
        strat = res_matches[0].strategy_type if res_matches else STRATEGY_DEPLOYABLE
        c = color_map.get(strat, "#7f7f7f")
        m = marker_map.get(name, "o")
        legend_handles.append(Line2D([0], [0], marker=m, color="w", label=f"{name}", markerfacecolor=c, markeredgecolor="#222222", markersize=9))

    ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=8.5, title="Baselines")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    return out_file


def _plot_with_pillow(
    results: list[BaselineResult],
    out_file: Path,
    sufficiency_status: str,
) -> Path:
    from PIL import Image, ImageDraw

    width, height = 1100, 750
    img = Image.new("RGB", (width, height), color="#ffffff")
    draw = ImageDraw.Draw(img)

    margin_left = 130
    margin_right = 80
    margin_top = 90
    margin_bottom = 100

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    max_cost_with_err = max(r.mean_cost + (r.cost_std or 0.0) for r in results) if results else 0.00005
    min_x = 0.0
    max_x = max(max_cost_with_err * 1.30, 0.00005)
    min_y = -0.05
    max_y = 1.10

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

    # Grid lines & Y Ticks (0.00 to 1.00)
    for y_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        py = to_py(y_val)
        draw.line([(axis_x0, py), (axis_x1, py)], fill="#e8e8e8", width=1)
        draw.text((axis_x0 - 45, py - 6), f"{y_val:.2f}", fill="#555555")

    # X Ticks with 6 decimal places (shows exact microdollar values)
    n_xticks = 6
    for i in range(n_xticks + 1):
        x_val = min_x + (i / n_xticks) * (max_x - min_x)
        px = to_px(x_val)
        draw.line([(px, axis_y0), (px, axis_y1)], fill="#e8e8e8", width=1)
        draw.text((px - 32, axis_y0 + 12), f"${x_val:.6f}", fill="#555555")

    # Axes
    draw.line([(axis_x0, axis_y0), (axis_x1, axis_y0)], fill="#333333", width=2)
    draw.line([(axis_x0, axis_y0), (axis_x0, axis_y1)], fill="#333333", width=2)

    # Title & Subtitle
    draw.text((margin_left, 20), "Baseline Comparison: Cost vs Quality", fill="#111111")
    if sufficiency_status in (STATUS_INSUFFICIENT, STATUS_EXPLORATORY):
        sub = "[EXPLORATORY / INSUFFICIENT DATA -- 1 Category (math), 1 Test Prompt]"
        draw.text((margin_left, 42), sub, fill="#b30000")

    # Axis Labels
    draw.text((axis_x0 + plot_w // 2 - 90, height - 40), "Mean Cost per Prompt ($ USD)", fill="#111111")
    draw.text((20, axis_y1 + plot_h // 2 - 10), "Test Quality", fill="#111111")

    # Strategy colors
    color_map = {
        STRATEGY_DEPLOYABLE: "#1f77b4",     # Blue
        STRATEGY_POST_HOC: "#ff7f0e",       # Orange
        STRATEGY_THEORETICAL: "#2ca02c",    # Green
    }

    # Group by coordinate
    coord_groups: dict[tuple[float, float], list[BaselineResult]] = {}
    for res in results:
        x = res.mean_cost
        y = res.mean_quality if res.mean_quality is not None else 0.0
        key = (round(x, 8), round(y, 4))
        coord_groups.setdefault(key, []).append(res)

    def draw_marker(cx: int, cy: int, baseline_name: str, strat: str, radius: int) -> None:
        c = color_map.get(strat, "#7f7f7f")
        if baseline_name == "Always-Cheapest":
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=c, outline="#222222", width=1)
        elif baseline_name == "Always-Best":
            draw.rectangle([cx - radius, cy - radius, cx + radius, cy + radius], fill=c, outline="#222222", width=1)
        elif baseline_name == "Random (Matched Cost)":
            draw.polygon([(cx, cy - radius - 2), (cx + radius + 2, cy + radius), (cx - radius - 2, cy + radius)], fill=c, outline="#222222")
        elif "Task" in baseline_name or "Table" in baseline_name:
            draw.polygon([(cx, cy - radius - 2), (cx + radius + 2, cy), (cx, cy + radius + 2), (cx - radius - 2, cy)], fill=c, outline="#222222")
        elif baseline_name == "Oracle":
            pts = []
            for k in range(8):
                r_k = radius + 3 if k % 2 == 0 else radius - 2
                ang = k * (3.14159 / 4.0)
                pts.append((int(cx + r_k * np.cos(ang)), int(cy + r_k * np.sin(ang))))
            draw.polygon(pts, fill=c, outline="#222222")
        else:
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=c, outline="#222222")

    # Plot markers (concentric if coincident at exact coordinates)
    for (gx, gy), group in coord_groups.items():
        px = to_px(gx)
        py = to_py(gy)

        if len(group) > 1:
            radii = [15, 11, 7, 4]
            order = ["Oracle", "Always-Best", "Static Task-Type Table", "Always-Cheapest"]
            sorted_group = sorted(group, key=lambda r: order.index(r.name) if r.name in order else 99)
            for idx, res in enumerate(sorted_group):
                rad = radii[min(idx, len(radii) - 1)]
                draw_marker(px, py, res.name, res.strategy_type, rad)
        else:
            res = group[0]
            if res.cost_std:
                x_left = to_px(gx - res.cost_std)
                x_right = to_px(gx + res.cost_std)
                draw.line([(x_left, py), (x_right, py)], fill="#555555", width=2)
                draw.line([(x_left, py - 5), (x_left, py + 5)], fill="#555555", width=2)
                draw.line([(x_right, py - 5), (x_right, py + 5)], fill="#555555", width=2)
            draw_marker(px, py, res.name, res.strategy_type, 7)

    # Callout annotations with leader lines
    for (gx, gy), group in coord_groups.items():
        px = to_px(gx)
        py = to_py(gy)

        if len(group) > 1:
            card_w = 265
            card_h = 24 + len(group) * 18 + 8
            card_x0 = max(axis_x0 + 10, px - 110)
            card_y0 = py + 45

            # Leader line from point to top of card
            draw.line([(px, py), (card_x0 + 100, card_y0)], fill="#444444", width=2)
            draw.rectangle([card_x0, card_y0, card_x0 + card_w, card_y0 + card_h], fill="#ffffff", outline="#1f77b4", width=2)
            header = f"Coincident Baselines (${gx:.6f}, Q:{gy:.2f}):"
            draw.text((card_x0 + 10, card_y0 + 8), header, fill="#111111")

            for idx, r in enumerate(group):
                strat_tag = "Deployable" if r.strategy_type == STRATEGY_DEPLOYABLE else ("Post-Hoc Ref" if r.strategy_type == STRATEGY_POST_HOC else "Theoretical Ref")
                line_y = card_y0 + 30 + idx * 18
                draw_marker(card_x0 + 18, line_y + 6, r.name, r.strategy_type, 4)
                draw.text((card_x0 + 28, line_y), f"{r.name} [{strat_tag}]", fill="#222222")
        else:
            r = group[0]
            strat_tag = "Deployable" if r.strategy_type == STRATEGY_DEPLOYABLE else ("Post-Hoc Ref" if r.strategy_type == STRATEGY_POST_HOC else "Theoretical Ref")
            card_w = 245
            card_h = 52
            card_x0 = px + 40
            card_y0 = py + 45

            draw.line([(px, py), (card_x0, card_y0 + 15)], fill="#444444", width=2)
            draw.rectangle([card_x0, card_y0, card_x0 + card_w, card_y0 + card_h], fill="#ffffff", outline="#888888", width=1)
            draw_marker(card_x0 + 18, card_y0 + 16, r.name, r.strategy_type, 5)
            draw.text((card_x0 + 28, card_y0 + 10), f"{r.name} [{strat_tag}]", fill="#111111")
            if r.cost_std:
                draw.text((card_x0 + 28, card_y0 + 28), f"${gx:.6f} +/- ${r.cost_std:.6f}, Q:{gy:.2f}", fill="#555555")
            else:
                draw.text((card_x0 + 28, card_y0 + 28), f"${gx:.6f}, Q:{gy:.2f}", fill="#555555")

    # Legend in bottom right
    leg_x0 = width - margin_right - 220
    leg_y0 = height - margin_bottom - 130
    draw.rectangle([leg_x0, leg_y0, leg_x0 + 210, leg_y0 + 120], fill="#fdfdfd", outline="#cccccc", width=1)
    draw.text((leg_x0 + 10, leg_y0 + 6), "Baselines Legend", fill="#333333")

    legend_items = [
        ("Always-Cheapest", STRATEGY_DEPLOYABLE),
        ("Always-Best", STRATEGY_POST_HOC),
        ("Random (Matched Cost)", STRATEGY_DEPLOYABLE),
        ("Static Task Table", STRATEGY_DEPLOYABLE),
        ("Oracle", STRATEGY_THEORETICAL),
    ]
    for idx, (name, strat) in enumerate(legend_items):
        ly = leg_y0 + 26 + idx * 18
        draw_marker(leg_x0 + 18, ly + 6, name, strat, 4)
        draw.text((leg_x0 + 30, ly), name, fill="#333333")

    img.save(out_file, format="PNG")
    return out_file


def plot_cost_vs_quality(
    results: list[BaselineResult],
    output_path: Path | str,
    sufficiency_status: str = STATUS_SUFFICIENT,
) -> Path:
    """
    Generate a clean, single cost-vs-quality comparison plot.
    
    Uses matplotlib if available; otherwise falls back to Pillow.
    Includes warning annotations when data coverage is insufficient/exploratory.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        return _plot_with_matplotlib(results, out_file, sufficiency_status)
    except ImportError:
        return _plot_with_pillow(results, out_file, sufficiency_status)


def run_baseline_comparison(
    matrix_path: Path | str = DEFAULT_MATRIX_PATH,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    benchmark_config_id: str | None = "m1-efa7ec3afd4f",
    split_ratio: float = 0.5,
    seed: int = 42,
    target_cost: float | None = None,
    random_repetitions: int = 20,
    task_objective: str = "quality_over_cost",
    classifier_cost: float = 0.0,
    output_dir: Path | str = DEFAULT_REPORTS_DIR,
) -> tuple[pd.DataFrame, dict[str, Any], list[BaselineResult]]:
    """
    Run end-to-end baseline comparison across all five strategies.
    
    Generates comparison summary, split metadata JSON, and cost-vs-quality plot.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load benchmark matrix
    raw_df = load_and_filter_matrix(matrix_path, benchmark_config_id=benchmark_config_id)
    if raw_df.empty:
        raise ValueError(f"No records found in matrix for config '{benchmark_config_id}'")

    # 2. Aggregate runs
    agg_df = aggregate_prompt_model_runs(raw_df)

    # 3. Load pilot prompts for reference and split
    prompts_dataset = load_dataset(dataset_path)

    # Determine prompts present in the active benchmark matrix
    matrix_prompt_ids = set(agg_df["prompt_id"].unique())
    prompts_for_split = [p for p in prompts_dataset if p["prompt_id"] in matrix_prompt_ids]

    # Create stratified partition
    split_info = create_stratified_split(prompts_for_split, split_ratio=split_ratio, seed=seed)

    train_ids = split_info["train_prompt_ids"]
    test_ids = split_info["test_prompt_ids"]

    # Save train/test split metadata
    split_file = out_dir / "train_test_split.json"
    with split_file.open("w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    # Split aggregated records strictly by prompt_ids (zero leakage)
    train_agg = agg_df[agg_df["prompt_id"].isin(train_ids)].copy()
    test_agg = agg_df[agg_df["prompt_id"].isin(test_ids)].copy()

    # 4. Evaluate each of the 5 baselines
    results: list[BaselineResult] = []

    # Baseline 1: Always-Cheapest
    res_cheapest = evaluate_always_cheapest(train_agg, test_agg, test_ids)
    results.append(res_cheapest)

    # Baseline 2: Always-Best (Post-Hoc)
    res_best = evaluate_always_best(test_agg, test_ids)
    results.append(res_best)

    # Baseline 3: Random at Matched Cost
    res_random = evaluate_random_matched_cost(
        train_agg, test_agg, test_ids,
        target_cost=target_cost,
        n_repetitions=random_repetitions,
        seed=seed,
    )
    results.append(res_random)

    # Baseline 4: Static Task-Type Table
    res_table = evaluate_static_task_table(
        train_agg, test_agg, test_ids,
        objective=task_objective,
        classifier_cost=classifier_cost,
    )
    results.append(res_table)

    # Baseline 5: Oracle
    res_oracle = evaluate_oracle(test_agg, test_ids)
    results.append(res_oracle)

    # 5. Compile comparison summary table
    summary_rows = [r.to_summary_dict() for r in results]
    comparison_df = pd.DataFrame(summary_rows)

    # Export CSV & Parquet
    csv_path = out_dir / "baseline_comparison.csv"
    parquet_path = out_dir / "baseline_comparison.parquet"
    comparison_df.to_csv(csv_path, index=False)
    comparison_df.to_parquet(parquet_path, index=False)

    # 6. Plot cost vs quality
    plot_path = out_dir / "cost_vs_quality.png"
    plot_cost_vs_quality(results, plot_path, sufficiency_status=split_info["sufficiency_status"])

    return comparison_df, split_info, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate M2 Baselines on LLM Cost Copilot Benchmark Data.")
    parser.add_argument("--matrix-path", type=str, default=str(DEFAULT_MATRIX_PATH), help="Path to matrix.parquet")
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH), help="Path to pilot_prompts.jsonl")
    parser.add_argument("--config-id", type=str, default="m1-efa7ec3afd4f", help="Benchmark configuration ID")
    parser.add_argument("--split-ratio", type=float, default=0.5, help="Train split ratio (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split and sampling")
    parser.add_argument("--target-cost", type=float, default=None, help="Target cost for Random Matched Cost baseline")
    parser.add_argument("--random-repetitions", type=int, default=20, help="Number of repetitions for random sampling")
    parser.add_argument("--task-objective", type=str, default="quality_over_cost", choices=["quality_over_cost", "quality_only", "cost_only"])
    parser.add_argument("--classifier-cost", type=float, default=0.0, help="Task classifier inference cost in USD per prompt (default: 0.0)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_REPORTS_DIR), help="Output directory for reports and plots")

    args = parser.parse_args()

    comparison_df, split_info, results = run_baseline_comparison(
        matrix_path=args.matrix_path,
        dataset_path=args.dataset_path,
        benchmark_config_id=args.config_id,
        split_ratio=args.split_ratio,
        seed=args.seed,
        target_cost=args.target_cost,
        random_repetitions=args.random_repetitions,
        task_objective=args.task_objective,
        classifier_cost=args.classifier_cost,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 78)
    print(f"M2 BASELINE COMPARISON RESULTS (Status: {split_info['sufficiency_status']})")
    print("=" * 78)
    if split_info["sufficiency_status"] == STATUS_INSUFFICIENT:
        print(f"WARNING: {split_info['notes']}\n")

    cols_to_print = [
        "baseline_name",
        "strategy_type",
        "mean_cost",
        "mean_quality",
        "mean_latency_ms",
        "evaluation_coverage",
        "execution_failure_count",
    ]
    print(comparison_df[cols_to_print].to_string(index=False))
    print("=" * 78)
    print(f"Artifacts saved to: {Path(args.output_dir).resolve()}")
    print("  - train_test_split.json")
    print("  - baseline_comparison.csv")
    print("  - baseline_comparison.parquet")
    print("  - cost_vs_quality.png")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
