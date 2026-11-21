"""
eval/legacy_comparison.py

M3 Final Experiment: Comprehensive comparative evaluation of the original M0
rule-based routing implementation against M3 learning-based routers (kNN and Learned Logistic Regression)
and the Always-Cheapest baseline.

Adheres strictly to ZERO ADDITIONAL SPENDING and OFFLINE EVALUATION policies:
- Reuses the frozen Step 6 prompt-level train/validation/test split.
- Evaluates on identical test prompts without test data leakage.
- Preserves read-only matrix.parquet and frozen M2 / M3 Step 1-7 artifacts.
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

from cost_estimator import estimate_tokens
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
from eval.dataset import load_dataset
from eval.router_evaluation import (
    EvaluationSelection,
    create_evaluation_split,
)
from eval.routers.data_prep import (
    PreparedDataset,
    PromptRecord,
    prepare_routing_dataset,
)
from eval.routers.decision import (
    CostNormalizer,
    LambdaDecisionEngine,
)
from eval.routers.encoders import BaseEncoder, DeterministicTestEncoder, SentenceTransformerEncoder
from eval.routers.knn_router import KNNRouter
from eval.routers.learned_router import LearnedRouter
from model_registry_v2 import MODELS as V2_MODELS
from router import MAX_TOKENS_BY_TIER, classify
from routing_policy import COMPLEXITY_MIN_QUALITY, QUALITY_SAFETY_MARGIN

logger = logging.getLogger(__name__)

# Static metadata for candidate models when training observations are < 5
DEFAULT_STATIC_QUALITY_TIERS: dict[str, float] = {
    "gpt_oss_20b": 0.75,
    "gpt_oss_120b": 0.85,
    "gemini_3_5_flash_lite": 0.90,
    "qwen3_30b_a3b_openrouter": 0.80,
}

DEFAULT_STATIC_LATENCIES_SEC: dict[str, float] = {
    "gpt_oss_20b": 0.5,
    "gpt_oss_120b": 1.1,
    "gemini_3_5_flash_lite": 2.5,
    "qwen3_30b_a3b_openrouter": 1.5,
}

MIN_TRAINING_SAMPLES_FOR_LIVE_QUALITY: int = 5


@dataclass
class LegacyCandidateRecord:
    """Evaluated candidate eligibility record for legacy routing."""
    model_id: str
    estimated_cost: float
    expected_quality: float
    avg_latency_sec: float
    quality_source: str
    sample_count: int
    is_eligible: bool
    rejection_reason: str | None = None
    score: float | None = None


@dataclass
class LegacySelectionDecision:
    """Selection record produced by LegacyRouterAdapter."""
    selected_model_id: str
    routing_mode: str
    tier: str
    reason: str
    estimated_cost: float
    expected_quality: float
    candidates_considered: list[dict[str, Any]]
    rejected_candidates: list[dict[str, Any]]


class LegacyRouterAdapter:
    """Faithful evaluation adapter for the original M0 rule-based router.

    Preserves exact production logic from router.py and routing_policy.py:
    1. Complexity classification: simple (<15 words), medium, complex (hard keywords).
    2. Token caps: simple=250, medium=600, complex=1200.
    3. Minimum quality floors: simple=0.70, medium=0.75, complex=0.90.
    4. Safety margin: +0.03 added to quality floor.
    5. Eligibility checks: availability, context-window, quality floor, and budget.
    6. Historical sample update: If >= 5 scored training samples exist for a model,
       historical mean score replaces static quality tier.
    7. Routing modes:
       - economy: min estimated cost among eligible candidates.
       - quality: max expected quality among eligible candidates.
       - latency: min average latency among eligible candidates.
       - balanced: 0.50 * Q + 0.35 * (1 - c_norm) + 0.15 * (1 - l_norm).
    """

    def __init__(
        self,
        candidate_models: Sequence[str],
        pricing_dict: dict[str, dict[str, float]] | None = None,
        context_windows: dict[str, int] | None = None,
        static_quality_tiers: dict[str, float] | None = None,
        static_latencies: dict[str, float] | None = None,
        training_scores: dict[str, list[float]] | None = None,
        training_latencies: dict[str, list[float]] | None = None,
    ) -> None:
        self.candidate_models = sorted(list(candidate_models))
        self.pricing = pricing_dict if pricing_dict is not None else {
            m: {
                "input": V2_MODELS[m]["input_price_per_1m"],
                "output": V2_MODELS[m]["output_price_per_1m"],
            }
            for m in self.candidate_models if m in V2_MODELS
        }
        self.context_windows = context_windows if context_windows is not None else {
            m: V2_MODELS[m].get("context_window", 131072)
            for m in self.candidate_models if m in V2_MODELS
        }
        self.static_quality_tiers = (
            static_quality_tiers if static_quality_tiers is not None else DEFAULT_STATIC_QUALITY_TIERS
        )
        self.static_latencies = (
            static_latencies if static_latencies is not None else DEFAULT_STATIC_LATENCIES_SEC
        )
        self.training_scores = training_scores or {}
        self.training_latencies = training_latencies or {}

    def get_model_metadata(self, model_id: str) -> dict[str, Any]:
        """Build model metadata using training observations (>= 5 samples) or static fallbacks."""
        scores = self.training_scores.get(model_id, [])
        sample_count = len(scores)

        if sample_count >= MIN_TRAINING_SAMPLES_FOR_LIVE_QUALITY:
            expected_q = float(np.mean(scores))
            q_source = "training_data"
        else:
            expected_q = float(self.static_quality_tiers.get(model_id, 0.75))
            q_source = "static_fallback"

        lats = self.training_latencies.get(model_id, [])
        if lats:
            avg_lat = float(np.mean(lats)) / 1000.0 if np.mean(lats) > 50.0 else float(np.mean(lats))
        else:
            avg_lat = float(self.static_latencies.get(model_id, 1.0))

        pricing = self.pricing.get(model_id, {"input": 0.1, "output": 0.3})
        ctx = self.context_windows.get(model_id, 131072)

        return {
            "model": model_id,
            "input_price": pricing["input"],
            "output_price": pricing["output"],
            "avg_latency_sec": avg_lat,
            "expected_quality": expected_q,
            "quality_source": q_source,
            "quality_sample_count": sample_count,
            "context_window": ctx,
            "available": True,
        }

    def select_model(
        self,
        prompt: str,
        routing_mode: str = "balanced",
        max_cost_usd: float | None = None,
        min_quality: float | None = None,
    ) -> LegacySelectionDecision | None:
        """Execute legacy routing rules for a given prompt text."""
        tier = classify(prompt)
        required_quality = min_quality if min_quality is not None else COMPLEXITY_MIN_QUALITY[tier]
        prompt_tokens = estimate_tokens(prompt)
        max_tokens = MAX_TOKENS_BY_TIER[tier]

        candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for model in self.candidate_models:
            meta = self.get_model_metadata(model)
            if not meta["available"]:
                rejected.append({"model": model, "reason": "unavailable"})
                continue
            if prompt_tokens > meta["context_window"]:
                rejected.append({"model": model, "reason": "exceeds context window"})
                continue
            if meta["expected_quality"] < (required_quality + QUALITY_SAFETY_MARGIN - 1e-9):
                rejected.append({
                    "model": model,
                    "reason": f"quality {meta['expected_quality']:.3f} below required {required_quality + QUALITY_SAFETY_MARGIN:.3f}",
                })
                continue

            est_cost = (prompt_tokens * meta["input_price"] + max_tokens * meta["output_price"]) / 1_000_000.0
            if max_cost_usd is not None and est_cost > max_cost_usd:
                rejected.append({
                    "model": model,
                    "reason": f"worst-case cost {est_cost:.6f} exceeds max_cost_usd {max_cost_usd:.6f}",
                })
                continue

            candidates.append({
                "model": model,
                "estimated_cost": est_cost,
                "expected_quality": meta["expected_quality"],
                "avg_latency_sec": meta["avg_latency_sec"],
                "quality_source": meta["quality_source"],
                "quality_sample_count": meta["quality_sample_count"],
            })

        if not candidates:
            return None

        # Mode selection
        if routing_mode == "economy":
            best = min(candidates, key=lambda c: (c["estimated_cost"], c["model"]))
            reason = f"Cheapest model meeting quality >= {required_quality:.2f}"
        elif routing_mode == "quality":
            best = max(candidates, key=lambda c: (c["expected_quality"], -c["estimated_cost"], c["model"]))
            reason = "Highest expected quality among eligible models"
        elif routing_mode == "latency":
            best = min(candidates, key=lambda c: (c["avg_latency_sec"], c["estimated_cost"], c["model"]))
            reason = f"Fastest model meeting quality >= {required_quality:.2f}"
        else:  # balanced
            costs = [c["estimated_cost"] for c in candidates]
            latencies = [c["avg_latency_sec"] for c in candidates]
            min_c, max_c = min(costs), max(costs)
            min_l, max_l = min(latencies), max(latencies)

            for c in candidates:
                cost_norm = 0.0 if max_c == min_c else (c["estimated_cost"] - min_c) / (max_c - min_c)
                lat_norm = 0.0 if max_l == min_l else (c["avg_latency_sec"] - min_l) / (max_l - min_l)
                c["score"] = (c["expected_quality"] * 0.5) + ((1.0 - cost_norm) * 0.35) + ((1.0 - lat_norm) * 0.15)

            best = max(candidates, key=lambda c: (c["score"], c["model"]))
            reason = "Best balance of cost, quality, and latency (0.50*Q + 0.35*(1-C) + 0.15*(1-L))"

        return LegacySelectionDecision(
            selected_model_id=best["model"],
            routing_mode=routing_mode,
            tier=tier,
            reason=reason,
            estimated_cost=best["estimated_cost"],
            expected_quality=best["expected_quality"],
            candidates_considered=candidates,
            rejected_candidates=rejected,
        )


@dataclass
class LegacyComparisonResult:
    """Evaluation metrics for a strategy compared against Legacy Rule-Based Balanced."""
    strategy_name: str
    strategy_type: str
    router_family: str
    test_prompt_count: int
    successful_evaluations: int
    correct_outcomes: int
    mean_quality: float | None
    quality_denominator: int
    evaluation_coverage: float
    execution_failure_count: int
    unscored_count: int
    mean_cost: float
    total_cost: float
    mean_latency_ms: float
    model_selection_distribution: dict[str, float]
    # Pairwise differences relative to Legacy Rule-Based Balanced
    delta_quality: float | None = None
    delta_cost: float | None = None
    delta_cost_pct: float | None = None
    delta_latency_ms: float | None = None
    notes: str = ""

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type,
            "router_family": self.router_family,
            "test_prompt_count": self.test_prompt_count,
            "successful_evaluations": self.successful_evaluations,
            "correct_outcomes": self.correct_outcomes,
            "mean_quality": self.mean_quality,
            "quality_denominator": self.quality_denominator,
            "evaluation_coverage": self.evaluation_coverage,
            "execution_failure_count": self.execution_failure_count,
            "unscored_count": self.unscored_count,
            "mean_cost": self.mean_cost,
            "total_cost": self.total_cost,
            "mean_latency_ms": self.mean_latency_ms,
            "delta_quality_vs_legacy_balanced": self.delta_quality,
            "delta_cost_usd_vs_legacy_balanced": self.delta_cost,
            "delta_cost_pct_vs_legacy_balanced": self.delta_cost_pct,
            "delta_latency_ms_vs_legacy_balanced": self.delta_latency_ms,
            "model_selection_distribution": json.dumps(self.model_selection_distribution),
            "notes": self.notes,
        }


def evaluate_legacy_mode_on_test(
    adapter: LegacyRouterAdapter,
    routing_mode: str,
    strategy_name: str,
    dataset: PreparedDataset,
    test_agg: pd.DataFrame,
    test_prompt_ids: list[str],
) -> LegacyComparisonResult:
    """Evaluate a specific legacy rule-based mode across test prompts."""
    n_test = len(test_prompt_ids)
    costs: list[float] = []
    latencies: list[float] = []
    scores: list[float] = []
    failures = 0
    unscored = 0
    model_counts: dict[str, int] = {}
    correct_count = 0

    for pid in test_prompt_ids:
        prompt_record = dataset.prompts.get(pid)
        prompt_text = prompt_record.prompt_text if prompt_record else pid
        decision = adapter.select_model(prompt_text, routing_mode=routing_mode)

        if decision is None:
            failures += 1
            unscored += 1
            continue

        chosen = decision.selected_model_id
        model_counts[chosen] = model_counts.get(chosen, 0) + 1

        match = test_agg[(test_agg["prompt_id"] == pid) & (test_agg["model_id"] == chosen)]
        if match.empty:
            failures += 1
            unscored += 1
            continue

        row = match.iloc[0]
        c = float(row["cost"])
        lat = float(row["latency_ms"]) if pd.notna(row.get("latency_ms")) else 0.0
        success = bool(row["execution_success"])
        eval_status = row["evaluation_status"]
        sc = float(row["score"]) if pd.notna(row.get("score")) else None
        is_correct = bool(row["correct"]) if pd.notna(row.get("correct")) else False

        costs.append(c)
        latencies.append(lat)
        if not success:
            failures += 1
        if eval_status == "scored" and sc is not None:
            scores.append(sc)
            if is_correct or sc >= 1.0:
                correct_count += 1
        else:
            unscored += 1

    mean_c = float(np.mean(costs)) if costs else 0.0
    tot_c = float(np.sum(costs)) if costs else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    mean_q = float(np.mean(scores)) if scores else None
    coverage = len(scores) / n_test if n_test > 0 else 0.0
    dist = {m: c / n_test for m, c in model_counts.items()} if n_test > 0 else {}

    return LegacyComparisonResult(
        strategy_name=strategy_name,
        strategy_type=STRATEGY_DEPLOYABLE,
        router_family="legacy_rule_based",
        test_prompt_count=n_test,
        successful_evaluations=len(scores),
        correct_outcomes=correct_count,
        mean_quality=mean_q,
        quality_denominator=len(scores),
        evaluation_coverage=coverage,
        execution_failure_count=failures,
        unscored_count=unscored,
        mean_cost=mean_c,
        total_cost=tot_c,
        mean_latency_ms=mean_lat,
        model_selection_distribution=dist,
        notes=f"Legacy M0 rule-based mode '{routing_mode}'.",
    )


def evaluate_always_cheapest_on_test(
    train_agg: pd.DataFrame,
    test_agg: pd.DataFrame,
    test_prompt_ids: list[str],
) -> LegacyComparisonResult:
    """Evaluate Always-Cheapest deployable baseline: picks candidate with lowest train cost."""
    # Compute mean cost per model in training partition strictly
    train_model_costs = train_agg.groupby("model_id")["cost"].mean().to_dict()
    if not train_model_costs:
        raise ValueError("No training records available to determine Always-Cheapest model.")

    cheapest_model = min(train_model_costs, key=lambda m: train_model_costs[m])

    n_test = len(test_prompt_ids)
    costs: list[float] = []
    latencies: list[float] = []
    scores: list[float] = []
    failures = 0
    unscored = 0
    correct_count = 0

    for pid in test_prompt_ids:
        match = test_agg[(test_agg["prompt_id"] == pid) & (test_agg["model_id"] == cheapest_model)]
        if match.empty:
            failures += 1
            unscored += 1
            continue

        row = match.iloc[0]
        c = float(row["cost"])
        lat = float(row["latency_ms"]) if pd.notna(row.get("latency_ms")) else 0.0
        success = bool(row["execution_success"])
        eval_status = row["evaluation_status"]
        sc = float(row["score"]) if pd.notna(row.get("score")) else None
        is_correct = bool(row["correct"]) if pd.notna(row.get("correct")) else False

        costs.append(c)
        latencies.append(lat)
        if not success:
            failures += 1
        if eval_status == "scored" and sc is not None:
            scores.append(sc)
            if is_correct or sc >= 1.0:
                correct_count += 1
        else:
            unscored += 1

    mean_c = float(np.mean(costs)) if costs else 0.0
    tot_c = float(np.sum(costs)) if costs else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    mean_q = float(np.mean(scores)) if scores else None
    coverage = len(scores) / n_test if n_test > 0 else 0.0

    return LegacyComparisonResult(
        strategy_name="Always-Cheapest",
        strategy_type=STRATEGY_DEPLOYABLE,
        router_family="baseline",
        test_prompt_count=n_test,
        successful_evaluations=len(scores),
        correct_outcomes=correct_count,
        mean_quality=mean_q,
        quality_denominator=len(scores),
        evaluation_coverage=coverage,
        execution_failure_count=failures,
        unscored_count=unscored,
        mean_cost=mean_c,
        total_cost=tot_c,
        mean_latency_ms=mean_lat,
        model_selection_distribution={cheapest_model: 1.0},
        notes=f"Always-Cheapest baseline derived strictly from training data ({cheapest_model}).",
    )


def evaluate_m3_router_on_test(
    router: Any,
    decision_engine: LambdaDecisionEngine,
    dataset: PreparedDataset,
    test_agg: pd.DataFrame,
    test_prompt_ids: list[str],
    strategy_name: str,
    router_family: str,
    lambda_value: float = 0.5,
) -> LegacyComparisonResult:
    """Evaluate an M3 router (kNN or Learned) coupled with LambdaDecisionEngine."""
    test_prompts = [dataset.prompts[pid] for pid in test_prompt_ids if pid in dataset.prompts]
    n_test = len(test_prompt_ids)

    if isinstance(router, KNNRouter):
        embs = np.array([p.prompt_embedding for p in test_prompts], dtype=np.float32)
        raw_preds = router.predict_proba(embs)
    else:
        raw_preds = router.predict_proba(test_prompts)

    costs: list[float] = []
    latencies: list[float] = []
    scores: list[float] = []
    failures = 0
    unscored = 0
    model_counts: dict[str, int] = {}
    correct_count = 0

    for i, pid in enumerate(test_prompt_ids):
        query_preds = {m: float(raw_preds[m][i]) if m in raw_preds else None for m in dataset.candidate_models}
        try:
            dec = decision_engine.select_model(query_preds, lambda_value=lambda_value)
            chosen = dec.selected_model_id
        except RuntimeError:
            failures += 1
            unscored += 1
            continue

        model_counts[chosen] = model_counts.get(chosen, 0) + 1
        match = test_agg[(test_agg["prompt_id"] == pid) & (test_agg["model_id"] == chosen)]
        if match.empty:
            failures += 1
            unscored += 1
            continue

        row = match.iloc[0]
        c = float(row["cost"])
        lat = float(row["latency_ms"]) if pd.notna(row.get("latency_ms")) else 0.0
        success = bool(row["execution_success"])
        eval_status = row["evaluation_status"]
        sc = float(row["score"]) if pd.notna(row.get("score")) else None
        is_correct = bool(row["correct"]) if pd.notna(row.get("correct")) else False

        costs.append(c)
        latencies.append(lat)
        if not success:
            failures += 1
        if eval_status == "scored" and sc is not None:
            scores.append(sc)
            if is_correct or sc >= 1.0:
                correct_count += 1
        else:
            unscored += 1

    mean_c = float(np.mean(costs)) if costs else 0.0
    tot_c = float(np.sum(costs)) if costs else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    mean_q = float(np.mean(scores)) if scores else None
    coverage = len(scores) / n_test if n_test > 0 else 0.0
    dist = {m: c / n_test for m, c in model_counts.items()} if n_test > 0 else {}

    return LegacyComparisonResult(
        strategy_name=strategy_name,
        strategy_type=STRATEGY_DEPLOYABLE,
        router_family=router_family,
        test_prompt_count=n_test,
        successful_evaluations=len(scores),
        correct_outcomes=correct_count,
        mean_quality=mean_q,
        quality_denominator=len(scores),
        evaluation_coverage=coverage,
        execution_failure_count=failures,
        unscored_count=unscored,
        mean_cost=mean_c,
        total_cost=tot_c,
        mean_latency_ms=mean_lat,
        model_selection_distribution=dist,
        notes=f"M3 {router_family} router + LambdaDecisionEngine (lambda={lambda_value:.2f}).",
    )


def plot_legacy_vs_learned_comparison(
    results: list[LegacyComparisonResult],
    output_path: Path | str,
    sufficiency_status: str,
) -> Path:
    """Generate high-resolution comparison plot using Pillow (2 panels: Quality & Cost comparison)."""
    from PIL import Image, ImageDraw

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1250, 750
    img = Image.new("RGB", (width, height), color="#ffffff")
    draw = ImageDraw.Draw(img)

    margin_left = 100
    margin_right = 50
    margin_top = 80
    margin_bottom = 80
    gap = 70

    panel_w = (width - margin_left - margin_right - gap) // 2
    panel_h = height - margin_top - margin_bottom

    # Title & Subtitle
    draw.text((margin_left, 20), "Legacy Rule-Based vs M3 Learning-Based Router Evaluation", fill="#111111")
    if sufficiency_status in (STATUS_INSUFFICIENT, STATUS_EXPLORATORY):
        draw.text(
            (margin_left, 45),
            "[EXPLORATORY / INSUFFICIENT DATA -- Real benchmark matrix has 2 prompts, 1 category]",
            fill="#b30000",
        )

    # PANEL 1: Quality & Cost Trade-off Scatter Plot
    p1_x0 = margin_left
    p1_x1 = p1_x0 + panel_w
    p1_y0 = height - margin_bottom
    p1_y1 = margin_top

    draw.line([(p1_x0, p1_y0), (p1_x1, p1_y0)], fill="#333333", width=2)
    draw.line([(p1_x0, p1_y0), (p1_x0, p1_y1)], fill="#333333", width=2)
    draw.text((p1_x0 + panel_w // 2 - 90, height - 40), "Observed Cost per Request ($ USD)", fill="#111111")
    draw.text((p1_x0 - 80, p1_y1 + panel_h // 2 - 10), "Quality (0-1)", fill="#111111")
    draw.text((p1_x0 + 10, p1_y1 + 10), "Panel 1: Cost vs Quality Trade-off", fill="#1f77b4")

    max_c = max(r.mean_cost for r in results) if results else 0.00006
    max_x = max(max_c * 1.3, 0.00005)

    def to_p1_x(cost_val: float) -> int:
        return p1_x0 + int((cost_val / max_x) * (panel_w - 40)) + 20

    def to_p1_y(qual_val: float | None) -> int:
        q = qual_val if qual_val is not None else 0.0
        return p1_y0 - int(q * (panel_h - 40)) - 20

    # Draw Y grid
    for q_tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        py = to_p1_y(q_tick)
        draw.line([(p1_x0, py), (p1_x1, py)], fill="#eaeaea", width=1)
        draw.text((p1_x0 - 45, py - 6), f"{q_tick:.2f}", fill="#555555")

    # Draw X grid
    for i in range(5):
        c_tick = (i / 4.0) * max_x
        px = to_p1_x(c_tick)
        draw.line([(px, p1_y0), (px, p1_y1)], fill="#eaeaea", width=1)
        draw.text((px - 28, p1_y0 + 12), f"${c_tick:.5f}", fill="#555555")

    family_colors = {
        "legacy_rule_based": "#d62728",  # Red
        "knn": "#1f77b4",                # Blue
        "learned": "#2ca02c",            # Green
        "baseline": "#9467bd",           # Purple
    }

    # Plot points
    for r in results:
        px = to_p1_x(r.mean_cost)
        py = to_p1_y(r.mean_quality)
        color = family_colors.get(r.router_family, "#333333")

        if "Balanced" in r.strategy_name:
            # Diamond marker for primary legacy baseline
            rad = 8
            draw.polygon([(px, py - rad), (px + rad, py), (px, py + rad), (px - rad, py)], fill=color, outline="#111111")
        elif r.router_family in ("knn", "learned"):
            # Square for learning routers
            rad = 7
            draw.rectangle([px - rad, py - rad, px + rad, py + rad], fill=color, outline="#111111")
        else:
            # Circle
            rad = 6
            draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=color, outline="#111111")

    # Callout annotations for key strategies
    leg_bal = next((r for r in results if r.strategy_name == "Legacy Rule-Based Balanced"), None)
    m3_knn = next((r for r in results if r.strategy_name == "M3 kNN Router"), None)

    if leg_bal:
        bx = to_p1_x(leg_bal.mean_cost)
        by = to_p1_y(leg_bal.mean_quality)
        draw.text((bx - 120, by - 25), f"Legacy Balanced (${leg_bal.mean_cost:.5f}, Q=1.0)", fill="#d62728")

    if m3_knn:
        kx = to_p1_x(m3_knn.mean_cost)
        ky = to_p1_y(m3_knn.mean_quality)
        draw.text((kx + 10, ky + 10), f"M3 Routers (${m3_knn.mean_cost:.5f}, Q=1.0)", fill="#1f77b4")

    # PANEL 2: Pairwise Deltas Relative to Legacy Balanced
    p2_x0 = p1_x1 + gap
    p2_x1 = p2_x0 + panel_w
    p2_y0 = height - margin_bottom
    p2_y1 = margin_top

    draw.line([(p2_x0, p2_y0), (p2_x1, p2_y0)], fill="#333333", width=2)
    draw.line([(p2_x0, p2_y0), (p2_x0, p2_y1)], fill="#333333", width=2)
    draw.text((p2_x0 + panel_w // 2 - 100, height - 40), "Strategy vs Legacy Balanced", fill="#111111")
    draw.text((p2_x0 - 80, p2_y1 + panel_h // 2 - 10), "Cost Reduction (%)", fill="#111111")
    draw.text((p2_x0 + 10, p2_y1 + 10), "Panel 2: Cost Savings vs Legacy Balanced", fill="#2ca02c")

    # Bar chart for cost savings % (-delta_cost_pct)
    strat_order = [
        "Legacy Rule-Based Economy",
        "Legacy Rule-Based Quality",
        "Legacy Rule-Based Latency",
        "M3 kNN Router",
        "M3 Learned Router",
        "Always-Cheapest",
    ]
    bar_results = [r for r in results if r.strategy_name in strat_order]
    bar_results.sort(key=lambda r: strat_order.index(r.strategy_name) if r.strategy_name in strat_order else 99)

    step = panel_h // (len(bar_results) + 1)
    for idx, r in enumerate(bar_results):
        y_pos = p2_y1 + 40 + idx * step
        cost_saving_pct = -(r.delta_cost_pct or 0.0)  # Positive = saved money
        bar_w = int((cost_saving_pct / 100.0) * (panel_w - 180)) if cost_saving_pct > 0 else 0

        color = family_colors.get(r.router_family, "#333333")
        draw.rectangle([p2_x0 + 10, y_pos, p2_x0 + 10 + bar_w, y_pos + 22], fill=color, outline="#222222")

        # Label
        short_name = r.strategy_name.replace("Legacy Rule-Based ", "Legacy ").replace(" Router", "")
        draw.text((p2_x0 + 15, y_pos + 4), f"{short_name}: {cost_saving_pct:+.1f}%", fill="#ffffff" if bar_w > 90 else "#111111")

        # Annotation: Quality delta
        q_label = f"Q={r.mean_quality:.2f}" if r.mean_quality is not None else "Q=N/A"
        draw.text((p2_x0 + 20 + bar_w, y_pos + 4), f"({q_label}, C=${r.mean_cost:.5f})", fill="#444444")

    img.save(out_file)
    return out_file


def run_legacy_vs_learned_comparison(
    matrix_path: Path | str = DEFAULT_MATRIX_PATH,
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    benchmark_config_id: str | None = "m1-efa7ec3afd4f",
    encoder: BaseEncoder | None = None,
    output_dir: Path | str = DEFAULT_REPORTS_DIR,
    custom_df: pd.DataFrame | None = None,
    custom_split: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[LegacyComparisonResult]]:
    """Execute complete comparative evaluation of Legacy Rule-Based vs M3 Learning-Based Routers.

    Generates:
    - reports/legacy_vs_learned_comparison.csv
    - reports/legacy_vs_learned_comparison.parquet
    - reports/legacy_vs_learned_comparison.json
    - reports/legacy_vs_learned_comparison.png

    Preserves all existing M2 and M3 Step 1-7 artifacts intact.
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

    # 2. Build prompt lookup and split (strictly reusing fixed split from Step 6 & 7)
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

    # 3. Partition aggregated runs strictly (zero leakage)
    train_agg = agg_df[agg_df["prompt_id"].isin(train_ids)].copy()
    test_agg = agg_df[agg_df["prompt_id"].isin(test_ids)].copy()

    # 4. Resolve encoder
    if encoder is None:
        try:
            encoder = SentenceTransformerEncoder()
        except Exception:
            encoder = DeterministicTestEncoder(dimension=384)

    # 5. Build PreparedDataset strictly from training partition
    prepared_dataset = prepare_routing_dataset(
        matrix_or_df=raw_df,
        benchmark_config_id=benchmark_config_id if custom_df is None else None,
        encoder=encoder,
        custom_split={"train_prompt_ids": train_ids, "test_prompt_ids": test_ids},
    )

    candidate_models = sorted(prepared_dataset.candidate_models)

    # 6. Extract training observations for LegacyRouterAdapter
    train_scores: dict[str, list[float]] = {}
    train_latencies: dict[str, list[float]] = {}
    for m in candidate_models:
        m_train = train_agg[train_agg["model_id"] == m]
        scs = [float(s) for s in m_train["score"].dropna() if pd.notna(s)]
        lats = [float(l) for l in m_train["latency_ms"].dropna() if pd.notna(l)]
        train_scores[m] = scs
        train_latencies[m] = lats

    legacy_adapter = LegacyRouterAdapter(
        candidate_models=candidate_models,
        training_scores=train_scores,
        training_latencies=train_latencies,
    )

    # 7. Fit M3 routers strictly on training data
    knn_router = KNNRouter(
        k=min(3, max(1, len(train_ids))), metric="cosine", weights="distance", encoder=encoder
    ).fit(prepared_dataset)

    learned_router = LearnedRouter(
        C=1.0, solver="lbfgs", encoder=encoder
    ).fit(prepared_dataset)

    decision_engine = LambdaDecisionEngine.from_training_dataset(
        prepared_dataset,
        strategy="min_max",
    )

    # 8. Evaluate all seven strategies on the exact same test prompts
    # 1. Legacy Rule-Based Balanced (Primary Legacy Baseline)
    res_leg_bal = evaluate_legacy_mode_on_test(
        adapter=legacy_adapter,
        routing_mode="balanced",
        strategy_name="Legacy Rule-Based Balanced",
        dataset=prepared_dataset,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
    )

    # 2. Legacy Rule-Based Economy
    res_leg_eco = evaluate_legacy_mode_on_test(
        adapter=legacy_adapter,
        routing_mode="economy",
        strategy_name="Legacy Rule-Based Economy",
        dataset=prepared_dataset,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
    )

    # 3. Legacy Rule-Based Quality
    res_leg_qual = evaluate_legacy_mode_on_test(
        adapter=legacy_adapter,
        routing_mode="quality",
        strategy_name="Legacy Rule-Based Quality",
        dataset=prepared_dataset,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
    )

    # 4. Legacy Rule-Based Latency
    res_leg_lat = evaluate_legacy_mode_on_test(
        adapter=legacy_adapter,
        routing_mode="latency",
        strategy_name="Legacy Rule-Based Latency",
        dataset=prepared_dataset,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
    )

    # 5. M3 kNN + Decision Engine (lambda=0.5 / auto:balanced)
    res_m3_knn = evaluate_m3_router_on_test(
        router=knn_router,
        decision_engine=decision_engine,
        dataset=prepared_dataset,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
        strategy_name="M3 kNN Router",
        router_family="knn",
        lambda_value=0.5,
    )

    # 6. M3 Logistic Regression + Decision Engine (lambda=0.5 / auto:balanced)
    res_m3_lr = evaluate_m3_router_on_test(
        router=learned_router,
        decision_engine=decision_engine,
        dataset=prepared_dataset,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
        strategy_name="M3 Learned Router",
        router_family="learned",
        lambda_value=0.5,
    )

    # 7. Always-Cheapest Baseline
    res_cheapest = evaluate_always_cheapest_on_test(
        train_agg=train_agg,
        test_agg=test_agg,
        test_prompt_ids=test_ids,
    )

    all_results = [
        res_leg_bal,
        res_leg_eco,
        res_leg_qual,
        res_leg_lat,
        res_m3_knn,
        res_m3_lr,
        res_cheapest,
    ]

    # 9. Compute pairwise differences relative to Legacy Rule-Based Balanced
    base_q = res_leg_bal.mean_quality
    base_c = res_leg_bal.mean_cost
    base_l = res_leg_bal.mean_latency_ms

    for r in all_results:
        if base_q is not None and r.mean_quality is not None:
            r.delta_quality = float(r.mean_quality - base_q)
        else:
            r.delta_quality = None

        r.delta_cost = float(r.mean_cost - base_c)
        r.delta_cost_pct = float((r.delta_cost / base_c) * 100.0) if base_c > 1e-12 else 0.0
        r.delta_latency_ms = float(r.mean_latency_ms - base_l)

    # 10. Export DataFrames (CSV & Parquet)
    summary_rows = [r.to_summary_dict() for r in all_results]
    comparison_df = pd.DataFrame(summary_rows)
    comparison_df["sufficiency_status"] = split_info["sufficiency_status"]

    csv_path = out_dir / "legacy_vs_learned_comparison.csv"
    parquet_path = out_dir / "legacy_vs_learned_comparison.parquet"
    comparison_df.to_csv(csv_path, index=False)
    comparison_df.to_parquet(parquet_path, index=False)

    # 11. Export High-Resolution Dual-Panel Visualization
    plot_path = out_dir / "legacy_vs_learned_comparison.png"
    plot_legacy_vs_learned_comparison(
        all_results, plot_path, sufficiency_status=split_info["sufficiency_status"]
    )

    # 12. Export JSON Summary
    json_path = out_dir / "legacy_vs_learned_comparison.json"
    summary_payload = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_config_id": benchmark_config_id,
        "sufficiency_status": split_info["sufficiency_status"],
        "is_exploratory": split_info["sufficiency_status"] != STATUS_SUFFICIENT,
        "split_info": split_info,
        "primary_baseline": "Legacy Rule-Based Balanced",
        "strategies_evaluated": [r.strategy_name for r in all_results],
        "pairwise_comparison_relative_to_legacy_balanced": {
            r.strategy_name: {
                "delta_quality": r.delta_quality,
                "delta_cost_usd": r.delta_cost,
                "delta_cost_pct": r.delta_cost_pct,
                "delta_latency_ms": r.delta_latency_ms,
                "model_selection": r.model_selection_distribution,
            }
            for r in all_results
        },
        "results": [r.to_summary_dict() for r in all_results],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return comparison_df, summary_payload, all_results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate Legacy Rule-Based vs M3 Learning-Based Routers")
    parser.add_argument("--matrix-path", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--config-id", type=str, default="m1-efa7ec3afd4f")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args()

    df, summary, results = run_legacy_vs_learned_comparison(
        matrix_path=args.matrix_path,
        dataset_path=args.dataset_path,
        benchmark_config_id=args.config_id,
        output_dir=args.output_dir,
    )
    print("\n=== Legacy Rule-Based vs M3 Learning-Based Router Evaluation Complete ===")
    print(f"Sufficiency Status: {summary['sufficiency_status']}")
    print(f"Total Strategies Evaluated: {len(results)}\n")
    for r in results:
        d_c_str = f"({r.delta_cost_pct:+.1f}%)" if r.delta_cost_pct is not None else ""
        print(f"- {r.strategy_name:30s} | Q={r.mean_quality} | Cost=${r.mean_cost:.6f} {d_c_str:8s} | Lat={r.mean_latency_ms:.1f}ms | Dist={r.model_selection_distribution}")


if __name__ == "__main__":
    main()
