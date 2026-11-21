"""
eval/routers/data_prep.py

Data preparation, label sanitization, and leakage-free dataset splitting for M3.
Adheres strictly to the ZERO ADDITIONAL SPENDING and OFFLINE EVALUATION policies.

Key guarantees:
1. Strict prompt-level splitting (zero test prompt leakage into train sets).
2. Honest label accounting: correct -> 1, incorrect -> 0, missing/unscored/failed -> None.
3. Distinguishes provider/execution failures from incorrect outputs.
4. Preserves missing values and actual costs; never fabricates outcomes.
5. Handles multi-model coverage and incomplete evaluations.
6. Assesses data sufficiency, marking exploratory / insufficient sample sizes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from eval.baselines import DEFAULT_MATRIX_PATH, create_stratified_split, load_and_filter_matrix
from eval.dataset import load_dataset as load_pilot_dataset
from eval.routers.encoders import BaseEncoder

logger = logging.getLogger(__name__)

STATUS_SUFFICIENT = "SUFFICIENT_DATA"
STATUS_INSUFFICIENT = "INSUFFICIENT_DATA"
MIN_PROMPTS_FOR_ML = 10
MIN_CATEGORIES_FOR_ML = 2


@dataclass(frozen=True)
class PromptRecord:
    """Metadata and representation for a unique benchmark prompt."""
    prompt_id: str
    task_category: str
    prompt_text: str
    prompt_embedding: np.ndarray | None = None
    split: str = "unassigned"  # "train", "test", or "unassigned"


@dataclass(frozen=True)
class ModelOutcomeRecord:
    """Structured representation of an evaluated model outcome for a prompt."""
    prompt_id: str
    task_category: str
    prompt_text: str
    model_id: str
    provider: str
    label: float | int | None  # Empirical probability in [0, 1], or None if missing / failed / unscored
    cost: float | None  # Actual incurred cost in USD, None if unrecorded
    token_usage: dict[str, int] | None  # input_tokens, output_tokens, total_tokens
    execution_status: str  # "success" or "failed"
    evaluation_status: str  # "scored", "unscored", "evaluation_error", "execution_failed"
    split: str  # "train" or "test"
    run_count: int = 1
    latency_ms: float | None = None
    prompt_embedding: np.ndarray | None = None
    correct_runs: int = 0
    evaluated_runs: int = 0
    empirical_quality: float | None = None
    run_outcomes: tuple[int | None, ...] = ()


@dataclass(frozen=True)
class DataSufficiencyReport:
    """Audit report assessing whether the dataset is sufficient for statistical router training."""
    status: str
    is_sufficient_for_ml: bool
    unique_prompts: int
    task_categories: list[str]
    evaluated_outcomes: int
    correct_labels: int
    incorrect_labels: int
    execution_failures: int
    missing_or_unscored: int
    model_coverage: dict[str, int]
    train_prompt_count: int
    test_prompt_count: int
    notes: str


@dataclass
class PreparedDataset:
    """Container for prepared training and testing router data."""
    records: list[ModelOutcomeRecord]
    prompts: dict[str, PromptRecord]
    split_info: dict[str, Any]
    sufficiency_report: DataSufficiencyReport

    @property
    def train_records(self) -> list[ModelOutcomeRecord]:
        return [r for r in self.records if r.split == "train"]

    @property
    def test_records(self) -> list[ModelOutcomeRecord]:
        return [r for r in self.records if r.split == "test"]

    @property
    def candidate_models(self) -> list[str]:
        return sorted({r.model_id for r in self.records})

    @property
    def train_prompt_ids(self) -> list[str]:
        return sorted({r.prompt_id for r in self.train_records})

    @property
    def test_prompt_ids(self) -> list[str]:
        return sorted({r.prompt_id for r in self.test_records})

    def get_train_data(
        self, model_id: str
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Extract training features and labels for a specific candidate model.

        Filters strictly to scored outcomes (label is not None) belonging to the train split.
        Returns (X, y, prompt_ids) where y has dtype float32 (empirical quality in [0, 1]).
        """
        model_records = [
            r for r in self.train_records if r.model_id == model_id and r.label is not None
        ]
        if not model_records:
            return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float32), []

        embeddings = [r.prompt_embedding for r in model_records]
        if any(e is None for e in embeddings):
            raise ValueError(f"Prompt embeddings not populated for model {model_id} training records.")

        X = np.asarray(embeddings, dtype=np.float32)
        y = np.asarray([r.label for r in model_records], dtype=np.float32)
        p_ids = [r.prompt_id for r in model_records]
        return X, y, p_ids

    def get_test_data(
        self, model_id: str
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Extract test features and labels for a specific candidate model.

        Filters strictly to scored outcomes (label is not None) belonging to the test split.
        Returns (X, y, prompt_ids) where y has dtype float32 (empirical quality in [0, 1]).
        """
        model_records = [
            r for r in self.test_records if r.model_id == model_id and r.label is not None
        ]
        if not model_records:
            return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float32), []

        embeddings = [r.prompt_embedding for r in model_records]
        if any(e is None for e in embeddings):
            raise ValueError(f"Prompt embeddings not populated for model {model_id} test records.")

        X = np.asarray(embeddings, dtype=np.float32)
        y = np.asarray([r.label for r in model_records], dtype=np.float32)
        p_ids = [r.prompt_id for r in model_records]
        return X, y, p_ids

    def to_dataframe(self, include_embeddings: bool = False) -> pd.DataFrame:
        """Convert outcome records to a Pandas DataFrame."""
        rows = []
        for r in self.records:
            d = {
                "prompt_id": r.prompt_id,
                "task_category": r.task_category,
                "prompt_text": r.prompt_text,
                "model_id": r.model_id,
                "provider": r.provider,
                "label": r.label,
                "cost": r.cost,
                "input_tokens": r.token_usage.get("input_tokens") if r.token_usage else None,
                "output_tokens": r.token_usage.get("output_tokens") if r.token_usage else None,
                "total_tokens": r.token_usage.get("total_tokens") if r.token_usage else None,
                "execution_status": r.execution_status,
                "evaluation_status": r.evaluation_status,
                "split": r.split,
                "run_count": r.run_count,
                "latency_ms": r.latency_ms,
            }
            if include_embeddings:
                d["prompt_embedding"] = r.prompt_embedding
            rows.append(d)
        return pd.DataFrame(rows)


def load_prompt_text_lookup(custom_lookup: dict[str, str] | None = None) -> dict[str, str]:
    """Build a mapping from prompt_id to raw prompt text."""
    lookup: dict[str, str] = {}
    try:
        pilot_data = load_pilot_dataset()
        for item in pilot_data:
            pid = item.get("prompt_id")
            ptext = item.get("prompt")
            if pid and ptext:
                lookup[str(pid)] = str(ptext)
    except Exception as err:
        logger.warning("Could not load pilot dataset prompts: %s", err)

    if custom_lookup:
        lookup.update(custom_lookup)

    return lookup


def assess_data_sufficiency(
    records: Sequence[ModelOutcomeRecord],
    train_prompt_ids: Sequence[str],
    test_prompt_ids: Sequence[str],
    min_prompts: int = MIN_PROMPTS_FOR_ML,
    min_categories: int = MIN_CATEGORIES_FOR_ML,
) -> DataSufficiencyReport:
    """Assess whether dataset size and diversity suffice for machine learning routers."""
    unique_prompts = sorted({r.prompt_id for r in records})
    categories = sorted({r.task_category for r in records})

    evaluated = [r for r in records if r.label is not None]
    correct_count = sum(1 for r in evaluated if r.label == 1)
    incorrect_count = sum(1 for r in evaluated if r.label == 0)
    failed_count = sum(1 for r in records if r.execution_status == "failed")
    missing_count = sum(1 for r in records if r.label is None)

    # Coverage: count of unique prompts with valid evaluated labels per model
    model_cov: dict[str, int] = {}
    for r in evaluated:
        model_cov[r.model_id] = model_cov.get(r.model_id, 0) + 1

    prompt_count = len(unique_prompts)
    category_count = len(categories)

    if (
        prompt_count < min_prompts
        or category_count < min_categories
        or len(test_prompt_ids) < 1
        or len(train_prompt_ids) < 2
    ):
        status = STATUS_INSUFFICIENT
        is_sufficient = False
        notes = (
            f"INSUFFICIENT_DATA: Active prompts count is {prompt_count} across {category_count} "
            f"category(ies). Minimum {min_prompts} prompts and {min_categories} categories required "
            "for reliable statistical learning. Router training and evaluation are EXPLORATORY."
        )
    else:
        status = STATUS_SUFFICIENT
        is_sufficient = True
        notes = f"SUFFICIENT_DATA: {prompt_count} prompts across {category_count} categories."

    return DataSufficiencyReport(
        status=status,
        is_sufficient_for_ml=is_sufficient,
        unique_prompts=prompt_count,
        task_categories=categories,
        evaluated_outcomes=len(evaluated),
        correct_labels=correct_count,
        incorrect_labels=incorrect_count,
        execution_failures=failed_count,
        missing_or_unscored=missing_count,
        model_coverage=model_cov,
        train_prompt_count=len(train_prompt_ids),
        test_prompt_count=len(test_prompt_ids),
        notes=notes,
    )


def prepare_routing_dataset(
    matrix_or_df: pd.DataFrame | Path | str | None = None,
    *,
    benchmark_config_id: str | None = "m1-efa7ec3afd4f",
    encoder: BaseEncoder | None = None,
    prompt_lookup: dict[str, str] | None = None,
    split_ratio: float = 0.5,
    seed: int = 42,
    custom_split: dict[str, list[str]] | None = None,
    aggregate_runs: bool = True,
) -> PreparedDataset:
    """Load, sanitize, split, and represent benchmark outcomes for router training and eval.

    Parameters
    ----------
    matrix_or_df : pd.DataFrame | Path | str | None
        Input DataFrame or path to Parquet benchmark matrix. Defaults to DEFAULT_MATRIX_PATH.
    benchmark_config_id : str | None
        Filter to benchmark_config_id (e.g. 'm1-efa7ec3afd4f'). If None, loads all rows.
    encoder : BaseEncoder | None
        Optional text encoder (e.g. SentenceTransformerEncoder or DeterministicTestEncoder).
    prompt_lookup : dict[str, str] | None
        Optional dictionary mapping prompt_id -> prompt text string.
    split_ratio : float
        Ratio of prompts allocated to train split (default 0.5).
    seed : int
        Random seed for stratified train/test split.
    custom_split : dict[str, list[str]] | None
        Explicit dictionary with "train_prompt_ids" and "test_prompt_ids" if overriding.
    aggregate_runs : bool
        If True (default), aggregates multiple runs of the same (prompt_id, model_id) pair.

    Returns
    -------
    PreparedDataset
        Self-contained dataset with train/test records, prompt representations, and audit report.
    """
    # 1. Load source DataFrame without mutating the source file
    if matrix_or_df is None or isinstance(matrix_or_df, (str, Path)):
        path = Path(matrix_or_df) if matrix_or_df is not None else DEFAULT_MATRIX_PATH
        raw_df = load_and_filter_matrix(path, benchmark_config_id=benchmark_config_id)
    elif isinstance(matrix_or_df, pd.DataFrame):
        raw_df = matrix_or_df.copy()
        if benchmark_config_id is not None and "benchmark_config_id" in raw_df.columns:
            filtered = raw_df[raw_df["benchmark_config_id"] == benchmark_config_id]
            if not filtered.empty:
                raw_df = filtered.copy()
    else:
        raise TypeError(f"Unsupported matrix_or_df type: {type(matrix_or_df)}")

    if raw_df.empty:
        raise ValueError("Source benchmark DataFrame is empty; cannot prepare routing dataset.")

    # 2. Build prompt text mapping
    text_map = load_prompt_text_lookup(prompt_lookup)

    # 3. Aggregate runs per (prompt_id, model_id) or keep row-level
    if aggregate_runs:
        df_records = _aggregate_runs_with_tokens(raw_df)
    else:
        df_records = _process_row_level_records(raw_df)

    # 4. Resolve train/test prompt split with strict zero leakage
    unique_prompt_items = []
    for _, row in df_records[["prompt_id", "task_category"]].drop_duplicates().iterrows():
        unique_prompt_items.append({
            "prompt_id": str(row["prompt_id"]),
            "task_type": str(row["task_category"]),
        })

    if custom_split is not None:
        train_pids = sorted(custom_split["train_prompt_ids"])
        test_pids = sorted(custom_split["test_prompt_ids"])
        split_info = {
            "train_prompt_ids": train_pids,
            "test_prompt_ids": test_pids,
            "split_ratio": split_ratio,
            "seed": seed,
            "custom": True,
        }
    else:
        split_info = create_stratified_split(
            unique_prompt_items,
            split_ratio=split_ratio,
            seed=seed,
        )
        train_pids = split_info["train_prompt_ids"]
        test_pids = split_info["test_prompt_ids"]

    # Strict leakage assertion: disjoint sets
    leakage = set(train_pids).intersection(set(test_pids))
    if leakage:
        raise ValueError(f"Data leakage detected! Train and test prompt IDs overlap on {leakage}")

    train_set = set(train_pids)
    test_set = set(test_pids)

    # 5. Precompute prompt embeddings once per unique prompt
    prompt_records_map: dict[str, PromptRecord] = {}
    unique_prompts = sorted(df_records["prompt_id"].unique())

    prompt_texts = []
    for pid in unique_prompts:
        ptext = text_map.get(pid, f"Prompt {pid}")
        prompt_texts.append(ptext)

    embeddings_array: np.ndarray | None = None
    if encoder is not None:
        embeddings_array = encoder.encode(prompt_texts)

    for i, pid in enumerate(unique_prompts):
        sub_df = df_records[df_records["prompt_id"] == pid]
        task_cat = sub_df["task_category"].iloc[0]
        ptext = prompt_texts[i]
        emb = embeddings_array[i] if embeddings_array is not None else None

        split_label = "train" if pid in train_set else ("test" if pid in test_set else "unassigned")
        prompt_records_map[pid] = PromptRecord(
            prompt_id=pid,
            task_category=task_cat,
            prompt_text=ptext,
            prompt_embedding=emb,
            split=split_label,
        )

    # 6. Build structured ModelOutcomeRecord objects
    outcome_records: list[ModelOutcomeRecord] = []
    for _, row in df_records.iterrows():
        pid = str(row["prompt_id"])
        pr = prompt_records_map[pid]

        record_split = "train" if pid in train_set else ("test" if pid in test_set else "unassigned")

        raw_label = row["label"]
        clean_label = float(raw_label) if pd.notna(raw_label) and raw_label is not None else None

        raw_cost = row["cost"]
        clean_cost = float(raw_cost) if pd.notna(raw_cost) and raw_cost is not None else None

        raw_latency = row.get("latency_ms")
        clean_latency = float(raw_latency) if pd.notna(raw_latency) and raw_latency is not None else None

        raw_emp = row.get("empirical_quality")
        clean_emp = float(raw_emp) if pd.notna(raw_emp) and raw_emp is not None else None

        outcome_records.append(
            ModelOutcomeRecord(
                prompt_id=pid,
                task_category=pr.task_category,
                prompt_text=pr.prompt_text,
                model_id=str(row["model_id"]),
                provider=str(row.get("provider", "unknown")),
                label=clean_label,
                cost=clean_cost,
                token_usage=row["token_usage"],
                execution_status=str(row["execution_status"]),
                evaluation_status=str(row["evaluation_status"]),
                split=record_split,
                run_count=int(row.get("run_count", 1)),
                latency_ms=clean_latency,
                prompt_embedding=pr.prompt_embedding,
                correct_runs=int(row.get("correct_runs", 1 if clean_label == 1.0 else 0)),
                evaluated_runs=int(row.get("evaluated_runs", 1 if clean_label is not None else 0)),
                empirical_quality=clean_emp,
                run_outcomes=tuple(row.get("run_outcomes", (1 if clean_label == 1.0 else (0 if clean_label == 0.0 else None),))),
            )
        )

    # 7. Audit data sufficiency
    sufficiency = assess_data_sufficiency(
        outcome_records,
        train_prompt_ids=train_pids,
        test_prompt_ids=test_pids,
    )

    return PreparedDataset(
        records=outcome_records,
        prompts=prompt_records_map,
        split_info=split_info,
        sufficiency_report=sufficiency,
    )


def _aggregate_runs_with_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Group multiple runs of (prompt_id, model_id) preserving tokens, costs, and labels."""
    records = []
    grouped = df.groupby(["prompt_id", "model_id"], as_index=False)

    for (prompt_id, model_id), group in grouped:
        task_category = (
            str(group["task_type"].iloc[0])
            if "task_type" in group.columns
            else str(group.get("task_category", pd.Series(["unknown"])).iloc[0])
        )
        provider = str(group["provider"].iloc[0]) if "provider" in group.columns else "unknown"
        run_count = len(group)

        # Execution status
        exec_success_list = (
            group["execution_success"].tolist()
            if "execution_success" in group.columns
            else [True] * run_count
        )
        all_exec_success = all(bool(x) for x in exec_success_list)
        exec_status = "success" if all_exec_success else "failed"

        # Costs and Latency
        mean_cost = float(group["cost"].mean()) if "cost" in group.columns and not group["cost"].isna().all() else None
        mean_latency = float(group["latency_ms"].mean()) if "latency_ms" in group.columns and not group["latency_ms"].isna().all() else None

        # Token usage
        token_usage: dict[str, int] | None = None
        has_in = "input_tokens" in group.columns and not group["input_tokens"].isna().all()
        has_out = "output_tokens" in group.columns and not group["output_tokens"].isna().all()
        if has_in and has_out:
            in_tok = int(round(float(group["input_tokens"].mean())))
            out_tok = int(round(float(group["output_tokens"].mean())))
            token_usage = {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
            }

        # Evaluation label resolution
        # Identify individual run outcomes
        run_outcomes: list[int | None] = []
        has_eval_status = "evaluation_status" in group.columns
        has_score = "score" in group.columns
        has_correct = "correct" in group.columns

        for _, run_row in group.iterrows():
            exec_ok = bool(run_row.get("execution_success", True)) if "execution_success" in group.columns else True
            ev_status = str(run_row.get("evaluation_status", "")) if has_eval_status else ""
            sc = run_row.get("score") if has_score else None
            corr = run_row.get("correct") if has_correct else None

            if not exec_ok:
                run_outcomes.append(None)
            elif ev_status == "scored" or (pd.notna(sc) and ev_status != "error"):
                if pd.notna(sc):
                    run_outcomes.append(1 if float(sc) >= 1.0 else 0)
                elif pd.notna(corr):
                    run_outcomes.append(1 if bool(corr) else 0)
                else:
                    run_outcomes.append(None)
            else:
                run_outcomes.append(None)

        evaluated_runs = sum(1 for o in run_outcomes if o is not None)
        correct_runs = sum(1 for o in run_outcomes if o == 1)

        if evaluated_runs > 0:
            eval_status = "scored"
            empirical_quality = float(correct_runs / evaluated_runs)
            label = empirical_quality
        else:
            empirical_quality = None
            label = None
            if not all_exec_success:
                eval_status = "execution_failed"
            elif has_eval_status and (group["evaluation_status"] == "error").any():
                eval_status = "evaluation_error"
            else:
                eval_status = "unscored"

        records.append({
            "prompt_id": str(prompt_id),
            "model_id": str(model_id),
            "task_category": task_category,
            "provider": provider,
            "run_count": run_count,
            "execution_status": exec_status,
            "evaluation_status": eval_status,
            "label": label,
            "correct_runs": correct_runs,
            "evaluated_runs": evaluated_runs,
            "empirical_quality": empirical_quality,
            "run_outcomes": tuple(run_outcomes),
            "cost": mean_cost,
            "latency_ms": mean_latency,
            "token_usage": token_usage,
        })

    return pd.DataFrame(records)


def _process_row_level_records(df: pd.DataFrame) -> pd.DataFrame:
    """Process unaggregated benchmark rows preserving per-row metrics and labels."""
    records = []
    for _, row in df.iterrows():
        prompt_id = str(row["prompt_id"])
        model_id = str(row["model_id"])
        task_category = str(row.get("task_type", row.get("task_category", "unknown")))
        provider = str(row.get("provider", "unknown"))

        exec_success = bool(row.get("execution_success", True))
        exec_status = "success" if exec_success else "failed"

        cost = float(row["cost"]) if pd.notna(row.get("cost")) else None
        latency = float(row["latency_ms"]) if pd.notna(row.get("latency_ms")) else None

        token_usage = None
        if pd.notna(row.get("input_tokens")) and pd.notna(row.get("output_tokens")):
            in_tok = int(row["input_tokens"])
            out_tok = int(row["output_tokens"])
            token_usage = {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
            }

        eval_status = str(row.get("evaluation_status", "unscored"))
        score = row.get("score")
        correct = row.get("correct")

        if eval_status == "scored" or (pd.notna(score) and eval_status != "error"):
            eval_status = "scored"
            if pd.notna(score):
                label = 1 if float(score) >= 1.0 else 0
            elif pd.notna(correct):
                label = 1 if bool(correct) else 0
            else:
                label = None
                eval_status = "unscored"
        else:
            label = None
            if not exec_success:
                eval_status = "execution_failed"
            elif eval_status == "error" or pd.notna(row.get("evaluation_error")):
                eval_status = "evaluation_error"
            else:
                eval_status = "unscored"

        records.append({
            "prompt_id": prompt_id,
            "model_id": model_id,
            "task_category": task_category,
            "provider": provider,
            "run_count": 1,
            "execution_status": exec_status,
            "evaluation_status": eval_status,
            "label": label,
            "cost": cost,
            "latency_ms": latency,
            "token_usage": token_usage,
        })

    return pd.DataFrame(records)
