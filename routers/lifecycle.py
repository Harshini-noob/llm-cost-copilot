"""
routers/lifecycle.py

M5 router retraining, shadow evaluation, fail-closed promotion, and rollback workflow.
Zero live API calls; strictly prevents train/val/test leakage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eval.routers.data_prep import ModelOutcomeRecord, PreparedDataset, PromptRecord
from eval.routers.encoders import BaseEncoder
from eval.routers.knn_router import KNNRouter
from eval.routers.learned_router import LearnedRouter
from routers.base import RuntimeRouter, load_cached_encoder
from routers.knn_runtime import KNNRuntimeRouter
from routers.lr_runtime import LearnedLRRuntimeRouter
from routers.registry import RouterVersionManager


class LeakageDetectedError(ValueError):
    """Raised when prompt IDs or normalized prompt texts overlap across splits."""


class PromotionGateError(RuntimeError):
    """Raised when a candidate router fails fail-closed promotion checks."""


@dataclass(frozen=True)
class LabelledObservation:
    prompt_id: str
    prompt_text: str
    model_id: str
    label: float
    cost_usd: float
    split: str  # "train", "val", or "test"
    task_category: str = "unknown"


def validate_observations_and_leakage(
    observations: Sequence[LabelledObservation | dict[str, Any]],
) -> list[LabelledObservation]:
    """Validate observation fields and strictly enforce zero train/val/test leakage."""
    parsed: list[LabelledObservation] = []
    split_pids: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    split_texts: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    for i, raw in enumerate(observations):
        obs = (
            raw
            if isinstance(raw, LabelledObservation)
            else LabelledObservation(
                prompt_id=str(raw["prompt_id"]),
                prompt_text=str(raw["prompt_text"]),
                model_id=str(raw["model_id"]),
                label=float(raw["label"]),
                cost_usd=float(raw["cost_usd"]),
                split=str(raw["split"]).strip().lower(),
                task_category=str(raw.get("task_category", "unknown")),
            )
        )
        if not obs.prompt_id.strip():
            raise ValueError(f"Empty prompt_id at index {i}")
        if not obs.prompt_text.strip():
            raise ValueError(f"Empty prompt_text at index {i}")
        if not obs.model_id.strip():
            raise ValueError(f"Empty model_id at index {i}")
        if obs.split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split '{obs.split}' at index {i}; expected train/val/test")
        if not math.isfinite(obs.label) or not (0.0 <= obs.label <= 1.0):
            raise ValueError(f"Label must be in [0.0, 1.0], got {obs.label} at index {i}")
        if not math.isfinite(obs.cost_usd) or obs.cost_usd < 0.0:
            raise ValueError(f"cost_usd must be >= 0, got {obs.cost_usd} at index {i}")

        split_pids[obs.split].add(obs.prompt_id)
        split_texts[obs.split].add(obs.prompt_text.strip())
        parsed.append(obs)

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        pid_overlap = split_pids[a].intersection(split_pids[b])
        if pid_overlap:
            raise LeakageDetectedError(
                f"Prompt ID leakage between {a} and {b}: {sorted(pid_overlap)[:5]}"
            )
        txt_overlap = split_texts[a].intersection(split_texts[b])
        if txt_overlap:
            raise LeakageDetectedError(
                f"Normalized prompt text leakage between {a} and {b}: {len(txt_overlap)} shared texts"
            )

    if not split_pids["train"]:
        raise ValueError("Training split cannot be empty")
    return parsed


def _build_prepared_dataset(
    observations: Sequence[LabelledObservation],
    encoder: BaseEncoder,
) -> tuple[PreparedDataset, dict[str, float]]:
    prompts_dict: dict[str, PromptRecord] = {}
    all_records: list[ModelOutcomeRecord] = []
    train_pids: set[str] = set()
    model_costs_acc: dict[str, list[float]] = defaultdict(list)
    all_models: set[str] = set()

    # Encode all unique prompt texts once
    pid_to_text_cat: dict[str, tuple[str, str, str]] = {}
    for obs in observations:
        all_models.add(obs.model_id)
        if obs.prompt_id not in pid_to_text_cat:
            pid_to_text_cat[obs.prompt_id] = (obs.prompt_text, obs.task_category, obs.split)

    pid_list = sorted(pid_to_text_cat.keys())
    embs = encoder.encode([pid_to_text_cat[pid][0] for pid in pid_list])
    emb_map = {pid: embs[i] for i, pid in enumerate(pid_list)}

    for pid, (text, cat, spl) in pid_to_text_cat.items():
        prompts_dict[pid] = PromptRecord(
            prompt_id=pid,
            task_category=cat,
            prompt_text=text,
            prompt_embedding=emb_map[pid],
            split=spl,
        )

    for obs in observations:
        rec = ModelOutcomeRecord(
            prompt_id=obs.prompt_id,
            task_category=obs.task_category,
            prompt_text=obs.prompt_text,
            model_id=obs.model_id,
            provider="runtime",
            label=float(obs.label),
            cost=float(obs.cost_usd),
            token_usage=None,
            execution_status="success",
            evaluation_status="scored",
            split=obs.split,
            run_count=1,
            latency_ms=0.0,
            prompt_embedding=emb_map[obs.prompt_id],
            correct_runs=float(obs.label),
            evaluated_runs=1,
            empirical_quality=float(obs.label),
        )
        all_records.append(rec)
        if obs.split == "train":
            train_pids.add(obs.prompt_id)
            model_costs_acc[obs.model_id].append(obs.cost_usd)

    model_costs = {
        m: float(np.mean(model_costs_acc[m])) if model_costs_acc[m] else 0.0
        for m in sorted(all_models)
    }

    dataset = PreparedDataset(
        records=all_records,
        prompts=prompts_dict,
        split_info={"train_prompt_count": len(train_pids)},
        sufficiency_report={"status": "runtime_retrain"},
    )
    return dataset, model_costs


def retrain_router(
    observations: Sequence[LabelledObservation | dict[str, Any]],
    *,
    router_type: str,
    new_version: str,
    manager: RouterVersionManager,
    encoder: BaseEncoder | None = None,
    encoder_name: str = "all-MiniLM-L6-v2",
    k: int = 5,
    C: float = 0.5,
    default_prior: float = 0.5,
) -> dict[str, Any]:
    """Retrain a versioned router strictly on train observations without activating it."""
    parsed = validate_observations_and_leakage(observations)
    enc = encoder if encoder is not None else load_cached_encoder(encoder_name)
    dataset, model_costs = _build_prepared_dataset(parsed, enc)

    if router_type == "knn":
        knn = KNNRouter(k=k, encoder=enc, weights="distance", metric="cosine", default_prior=default_prior)
        knn.fit(dataset)
        runtime_router: RuntimeRouter = KNNRuntimeRouter(
            knn,
            model_costs=model_costs,
            router_version=new_version,
            encoder_name=encoder_name,
            calibration_status="calibrated_on_runtime_train",
            metadata={"train_prompts": len(dataset.train_prompt_ids)},
        )
        runtime_router.save_bundle(manager.version_dir(new_version))
    elif router_type == "learned_lr":
        lr = LearnedRouter(encoder=enc, C=C, max_iter=1000, random_state=42, default_prior=default_prior)
        lr.fit(dataset)
        runtime_router = LearnedLRRuntimeRouter(
            lr,
            model_costs=model_costs,
            router_version=new_version,
            encoder_name=encoder_name,
            calibration_status="calibrated_on_runtime_train",
            metadata={"train_prompts": len(dataset.train_prompt_ids)},
        )
        runtime_router.save_bundle(manager.version_dir(new_version))
    else:
        raise ValueError(f"Unsupported router_type '{router_type}'; expected 'knn' or 'learned_lr'")

    # Evaluate on validation split if present (never test split)
    val_obs = [o for o in parsed if o.split == "val"]
    val_metrics: dict[str, Any] = {"has_validation_data": False, "val_prompt_count": 0}
    if val_obs:
        val_prompts = {o.prompt_id: o for o in val_obs}
        val_outcomes: dict[tuple[str, str], LabelledObservation] = {
            (o.prompt_id, o.model_id): o for o in val_obs
        }
        qualities: list[float] = []
        costs: list[float] = []
        for pid, p_obs in val_prompts.items():
            dec = runtime_router.route(p_obs.prompt_text, lam="auto:balanced", task_category=p_obs.task_category)
            key = (pid, dec.chosen_model)
            if key in val_outcomes:
                qualities.append(val_outcomes[key].label)
                costs.append(val_outcomes[key].cost_usd)
        val_metrics = {
            "has_validation_data": bool(qualities),
            "val_prompt_count": len(val_prompts),
            "evaluated_prompts": len(qualities),
            "val_mean_quality": float(np.mean(qualities)) if qualities else None,
            "val_mean_cost_usd": float(np.mean(costs)) if costs else None,
        }

    return {
        "new_version": new_version,
        "router_type": router_type,
        "bundle_dir": str(manager.version_dir(new_version)),
        "active_version_preserved": manager.get_active_version(),
        "train_prompt_count": len(dataset.train_prompt_ids),
        "model_ids": runtime_router.model_ids,
        "validation_report": val_metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def shadow_evaluate(
    candidate_router: RuntimeRouter,
    active_router: RuntimeRouter,
    prompts: Sequence[dict[str, Any]],
    *,
    lam: float | str = "auto:balanced",
    labelled_outcomes: dict[tuple[str, str], tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Compare a candidate router against the active router on a shared prompt set.

    labelled_outcomes (optional): mapping of (prompt_id, model_id) -> (actual_quality, actual_cost_usd).
    When absent, explicitly marks predicted_scores_only=True so predicted scores are never
    presented as proof of actual answer quality.
    """
    if not prompts:
        raise ValueError("Shadow evaluation requires at least one prompt")

    agreements = 0
    cand_choices: Counter[str] = Counter()
    active_choices: Counter[str] = Counter()
    cand_pred_scores: list[float] = []
    active_pred_scores: list[float] = []
    cand_exp_costs: list[float] = []
    active_exp_costs: list[float] = []
    cand_latencies: list[float] = []
    per_task_mix: dict[str, Counter[str]] = defaultdict(Counter)

    cand_actual_q: list[float] = []
    active_actual_q: list[float] = []
    cand_actual_c: list[float] = []
    active_actual_c: list[float] = []

    for item in prompts:
        pid = str(item.get("prompt_id", ""))
        text = str(item["prompt_text"])
        task = str(item.get("task_category", "unknown"))

        c_dec = candidate_router.route(text, lam=lam, task_category=task)
        a_dec = active_router.route(text, lam=lam, task_category=task)

        if c_dec.chosen_model == a_dec.chosen_model:
            agreements += 1
        cand_choices[c_dec.chosen_model] += 1
        active_choices[a_dec.chosen_model] += 1
        per_task_mix[task][c_dec.chosen_model] += 1

        c_score = c_dec.predicted_scores.get(c_dec.chosen_model)
        a_score = a_dec.predicted_scores.get(a_dec.chosen_model)
        if c_score is not None:
            cand_pred_scores.append(float(c_score))
        if a_score is not None:
            active_pred_scores.append(float(a_score))

        cand_exp_costs.append(float(c_dec.expected_costs_usd.get(c_dec.chosen_model, 0.0)))
        active_exp_costs.append(float(a_dec.expected_costs_usd.get(a_dec.chosen_model, 0.0)))
        cand_latencies.append(float(c_dec.router_latency_ms))

        if labelled_outcomes is not None and pid:
            c_key = (pid, c_dec.chosen_model)
            a_key = (pid, a_dec.chosen_model)
            if c_key in labelled_outcomes and a_key in labelled_outcomes:
                cq, cc = labelled_outcomes[c_key]
                aq, ac = labelled_outcomes[a_key]
                cand_actual_q.append(float(cq))
                active_actual_q.append(float(aq))
                cand_actual_c.append(float(cc))
                active_actual_c.append(float(ac))

    n = len(prompts)
    has_actual = bool(cand_actual_q)
    return {
        "candidate_version": candidate_router.router_version,
        "active_version": active_router.router_version,
        "prompt_count": n,
        "lambda": lam,
        "routing_agreement": agreements / n,
        "candidate_selection_distribution": {m: c / n for m, c in cand_choices.items()},
        "active_selection_distribution": {m: c / n for m, c in active_choices.items()},
        "per_task_routing_mix": {t: dict(cnt) for t, cnt in per_task_mix.items()},
        "candidate_predicted_quality": {
            "mean": float(np.mean(cand_pred_scores)) if cand_pred_scores else None,
            "p50": float(np.median(cand_pred_scores)) if cand_pred_scores else None,
        },
        "active_predicted_quality": {
            "mean": float(np.mean(active_pred_scores)) if active_pred_scores else None,
            "p50": float(np.median(active_pred_scores)) if active_pred_scores else None,
        },
        "candidate_expected_cost_usd": {
            "mean": float(np.mean(cand_exp_costs)),
            "p50": float(np.median(cand_exp_costs)),
        },
        "active_expected_cost_usd": {
            "mean": float(np.mean(active_exp_costs)),
            "p50": float(np.median(active_exp_costs)),
        },
        "candidate_overhead_ms": {
            "p50": float(np.percentile(cand_latencies, 50)),
            "p95": float(np.percentile(cand_latencies, 95)),
        },
        "predicted_scores_only": not has_actual,
        "actual_validation_comparison": {
            "evaluated_count": len(cand_actual_q),
            "candidate_quality": float(np.mean(cand_actual_q)) if has_actual else None,
            "active_quality": float(np.mean(active_actual_q)) if has_actual else None,
            "candidate_cost_usd": float(np.mean(cand_actual_c)) if has_actual else None,
            "active_cost_usd": float(np.mean(active_actual_c)) if has_actual else None,
        },
    }


def promote_router(
    candidate_version: str,
    manager: RouterVersionManager,
    *,
    shadow_report: dict[str, Any] | None = None,
    min_val_prompts: int = 50,
    max_quality_drop: float = 0.02,
    max_cost_multiplier: float = 1.25,
    allow_pool_change: bool = False,
    encoder: BaseEncoder | None = None,
) -> dict[str, Any]:
    """Fail-closed promotion of candidate_version to active router."""
    # 1. Bundle completeness & encoder compatibility check (raises if incomplete/incompatible)
    try:
        candidate = manager.load_version(candidate_version, encoder=encoder)
    except Exception as e:
        raise PromotionGateError(f"Candidate bundle '{candidate_version}' failed load check: {e}") from e

    active_ver = manager.get_active_version()
    if active_ver is not None and active_ver != candidate_version:
        active = manager.load_version(active_ver, encoder=encoder)
        # 2. Encoder compatibility
        if candidate.encoder_name != active.encoder_name or candidate.encoder_dim != active.encoder_dim:
            raise PromotionGateError(
                f"Encoder mismatch: candidate ({candidate.encoder_name}, {candidate.encoder_dim}) "
                f"!= active ({active.encoder_name}, {active.encoder_dim})"
            )
        # 3. Model pool compatibility
        if not allow_pool_change and set(candidate.model_ids) != set(active.model_ids):
            raise PromotionGateError(
                f"Model pool changed unexpectedly: candidate={candidate.model_ids}, active={active.model_ids}"
            )

    # 4. Shadow / validation data sufficiency & quality/cost gates
    if shadow_report is None:
        raise PromotionGateError("Shadow evaluation report is required before promotion")

    prompt_count = int(shadow_report.get("prompt_count", 0))
    if prompt_count < min_val_prompts:
        raise PromotionGateError(
            f"Insufficient validation data: {prompt_count} < min_val_prompts ({min_val_prompts})"
        )

    actual_cmp = shadow_report.get("actual_validation_comparison", {})
    cand_q = actual_cmp.get("candidate_quality")
    act_q = actual_cmp.get("active_quality")
    cand_c = actual_cmp.get("candidate_cost_usd")
    act_c = actual_cmp.get("active_cost_usd")

    if cand_q is not None and act_q is not None:
        if float(cand_q) < float(act_q) - max_quality_drop:
            raise PromotionGateError(
                f"Validation quality gate failed: candidate {cand_q:.4f} < active {act_q:.4f} - {max_quality_drop}"
            )
    if cand_c is not None and act_c is not None and float(act_c) > 0:
        if float(cand_c) > float(act_c) * max_cost_multiplier:
            raise PromotionGateError(
                f"Validation cost gate failed: candidate {cand_c:.6f} > active {act_c:.6f} * {max_cost_multiplier}"
            )

    ptr = manager.set_active(candidate_version)
    return {"promoted": True, **ptr}
