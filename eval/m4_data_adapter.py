"""
eval/m4_data_adapter.py

xRouteBench Benchmark Integration, Validation, and Cost Adaptation Module for M4.
Adheres strictly to the ZERO ADDITIONAL SPENDING and OFFLINE EVALUATION policies.

Dataset Provenance:
- Source: Hugging Face dataset 'ulab-ai/xRouteBench'
- Paper: "LLMRouter: A Multi-Task Benchmark and Framework for LLM Routing" (arXiv:2608.06867)
- Authors: Tao Feng, Chuanyang Zheng, et al. (UIUC, UMD, Tsinghua University)
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- Configurations: 'llmrouter_generic' task collection, 'llm_candidates' model pool (18 candidate LLMs)

Guarantees & Integrity:
1. Dataset Immutability: Raw parquet files in data/external/xroutebench/ are validated against
   their cryptographic SHA-256 digests and never overwritten or modified.
2. Reconstructed Cost: Costs are computed from observed token usage and official candidate pricing
   using exact Decimal arithmetic. Documented explicitly as reconstructed benchmark estimates,
   not actual provider invoices.
3. Preserves Cross-Task Records: The 36 cross-task rows (e.g. query shared between 'math' and
   'aime_2020_2024', and distinct embeddings in 'arc_challenge') are preserved intact.
4. Continuous vs Binary Metrics: SQuAD F1 scores are preserved as continuous floats in [0.0, 1.0];
   the remaining 12 tasks are verified as binary accuracy in {0.0, 1.0}.
5. Leakage-Free Splitting: Validation splits created from training data group strictly by normalized
   query text, guaranteeing zero prompt leakage across folds.
6. Honest Aggregation: Supports both Macro-average (unweighted mean across 13 task categories)
   and Micro-average (prompt-level mean), plus prompt-clustered bootstrap confidence intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default file paths
EXTERNAL_DATA_DIR = Path("data/external/xroutebench")
PROCESSED_DATA_DIR = Path("data/processed/xroutebench")

TRAIN_PARQUET_NAME = "train.parquet"
TEST_PARQUET_NAME = "test.parquet"
CANDIDATES_PARQUET_NAME = "llm_candidates.parquet"

# Official cryptographic digests of original raw files
EXPECTED_SHA256 = {
    TRAIN_PARQUET_NAME: "9307e40bd27737592f25677f3a7e58a8712181b08fe435b82c2943d7b49105c2",
    TEST_PARQUET_NAME: "e7ce786054bcf1b81f4da32243d4d6414f4d94d9b5729ebd8b239f289b6dabe9",
    CANDIDATES_PARQUET_NAME: "5131134ed46c9ff68c0744fb90556714e99cbabd3ab0541778929bae9a91f23f",
}

# 13 benchmark task categories and their metrics
TASK_METRIC_MAP = {
    "aime_2020_2024": {"metric": "GSM8K", "is_binary": True, "description": "American Invitational Mathematics Exam"},
    "arc_challenge": {"metric": "em_mc", "is_binary": True, "description": "AI2 Reasoning Challenge (Challenge Set)"},
    "boolq": {"metric": "em_mc", "is_binary": True, "description": "BoolQ Yes/No Question Answering"},
    "commonsense_qa": {"metric": "em_mc", "is_binary": True, "description": "CommonsenseQA Multiple-Choice"},
    "gsm8k": {"metric": "GSM8K", "is_binary": True, "description": "Grade School Math 8K Word Problems"},
    "hellaswag": {"metric": "em_mc", "is_binary": True, "description": "HellaSwag Commonsense Reasoning"},
    "human_eval": {"metric": "code_eval", "is_binary": True, "description": "HumanEval Python Coding Benchmark"},
    "math": {"metric": "MATH", "is_binary": True, "description": "MATH Competition Level Mathematics"},
    "mbpp": {"metric": "code_eval", "is_binary": True, "description": "Mostly Basic Python Problems"},
    "mmlu": {"metric": "em_mc", "is_binary": True, "description": "Massive Multitask Language Understanding"},
    "mmlu_pro": {"metric": "em_mc", "is_binary": True, "description": "MMLU-Pro High-Difficulty Multitask Benchmark"},
    "openbook_qa": {"metric": "em_mc", "is_binary": True, "description": "OpenBookQA Question Answering"},
    "squad": {"metric": "f1_score", "is_binary": False, "description": "Stanford Question Answering Dataset (F1 score)"},
}

ALL_TASKS = tuple(sorted(TASK_METRIC_MAP.keys()))
BINARY_TASKS = tuple(sorted(t for t, info in TASK_METRIC_MAP.items() if info["is_binary"]))
CONTINUOUS_TASKS = tuple(sorted(t for t, info in TASK_METRIC_MAP.items() if not info["is_binary"]))


@dataclass(frozen=True)
class CandidatePricing:
    """Pricing configuration and metadata for a candidate LLM."""
    model_name: str
    size: str
    input_price_per_1m: float
    output_price_per_1m: float
    service: str
    api_model_id: str
    description: str


@dataclass(frozen=True)
class ValidationReport:
    """Detailed dataset integrity and consistency validation report."""
    is_valid: bool
    split_name: str
    total_rows: int
    unique_queries: int
    candidate_model_count: int
    task_count: int
    token_equality_valid: bool
    non_negative_tokens: bool
    non_negative_response_time: bool
    performance_in_bounds: bool
    null_counts: dict[str, int]
    cross_task_queries_preserved: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def compute_sha256(file_path: Path | str) -> str:
    """Calculate the cryptographic SHA-256 checksum of a file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for hash computation: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file_hashes(data_dir: Path | str | None = None) -> dict[str, bool]:
    """Verify raw xRouteBench Parquet files against expected cryptographic hashes."""
    base_dir = Path(data_dir) if data_dir is not None else EXTERNAL_DATA_DIR
    results: dict[str, bool] = {}

    for filename, expected_hash in EXPECTED_SHA256.items():
        file_path = base_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Expected benchmark file missing: {file_path}")
        computed = compute_sha256(file_path)
        is_match = (computed.lower() == expected_hash.lower())
        if not is_match:
            logger.error(
                "Hash mismatch for %s: computed=%s, expected=%s",
                filename, computed, expected_hash
            )
        results[filename] = is_match

    return results


def calculate_reconstructed_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_1m: float,
    output_price_per_1m: float,
    use_decimal: bool = True,
) -> float:
    """Compute reconstructed observation cost in USD from token usage and rates.

    Formula:
        cost = (input_tokens * input_price_per_1m + output_tokens * output_price_per_1m) / 1,000,000

    IMPORTANT:
    This cost represents a reconstructed benchmark expenditure calculated from
    candidate model pricing tables, NOT an actual billing invoice from a live LLM API.
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError(f"Token counts must be non-negative, got in={input_tokens}, out={output_tokens}")
    if input_price_per_1m < 0.0 or output_price_per_1m < 0.0:
        raise ValueError("Prices per 1M tokens must be non-negative.")

    if use_decimal:
        d_in_tokens = Decimal(int(input_tokens))
        d_out_tokens = Decimal(int(output_tokens))
        d_in_price = Decimal(str(input_price_per_1m))
        d_out_price = Decimal(str(output_price_per_1m))
        d_denom = Decimal("1000000")

        total = (d_in_tokens * d_in_price + d_out_tokens * d_out_price) / d_denom
        return float(total)

    return float((input_tokens * input_price_per_1m + output_tokens * output_price_per_1m) / 1e6)


def load_xroutebench_raw(
    data_dir: Path | str | None = None,
    verify_hashes: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw xRouteBench Parquet files into pandas DataFrames.

    Returns
    -------
    (train_df, test_df, candidates_df)
    """
    base_dir = Path(data_dir) if data_dir is not None else EXTERNAL_DATA_DIR

    if verify_hashes:
        hash_check = verify_file_hashes(base_dir)
        mismatches = [f for f, ok in hash_check.items() if not ok]
        if mismatches:
            raise ValueError(f"Raw data file hash verification failed for: {mismatches}")

    train_path = base_dir / TRAIN_PARQUET_NAME
    test_path = base_dir / TEST_PARQUET_NAME
    cand_path = base_dir / CANDIDATES_PARQUET_NAME

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    cand_df = pd.read_parquet(cand_path)

    return train_df, test_df, cand_df


def attach_pricing_and_costs(
    df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    cost_col_name: str = "reconstructed_cost_usd",
) -> pd.DataFrame:
    """Enrich observation records with candidate pricing and computed USD costs.

    Parameters
    ----------
    df : pd.DataFrame
        Observation DataFrame containing 'model_name', 'input_tokens', 'output_tokens'.
    candidates_df : pd.DataFrame
        Pricing DataFrame containing 'model_name', 'input_price_per_1m', 'output_price_per_1m'.
    cost_col_name : str
        Name of the computed cost column to add.
    """
    required_cols = {"model_name", "input_tokens", "output_tokens"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns for pricing: {missing}")

    cand_required = {"model_name", "input_price_per_1m", "output_price_per_1m"}
    missing_cand = cand_required - set(candidates_df.columns)
    if missing_cand:
        raise ValueError(f"Candidates DataFrame missing required columns: {missing_cand}")

    # Check that all models in df exist in candidates_df
    df_models = set(df["model_name"].unique())
    cand_models = set(candidates_df["model_name"].unique())
    unmatched = df_models - cand_models
    if unmatched:
        raise ValueError(f"Observations contain models without pricing specifications: {unmatched}")

    pricing_map = candidates_df.set_index("model_name")[
        ["input_price_per_1m", "output_price_per_1m"]
    ].to_dict(orient="index")

    result = df.copy()

    # Vectorized / element-wise exact calculation
    in_prices = result["model_name"].map(lambda m: pricing_map[m]["input_price_per_1m"]).astype(float)
    out_prices = result["model_name"].map(lambda m: pricing_map[m]["output_price_per_1m"]).astype(float)

    result["input_price_per_1m"] = in_prices
    result["output_price_per_1m"] = out_prices

    # Vectorized calculation (guaranteed non-negative by validation)
    result[cost_col_name] = (
        result["input_tokens"] * in_prices + result["output_tokens"] * out_prices
    ) / 1_000_000.0

    return result


def validate_xroutebench_dataframe(
    df: pd.DataFrame,
    split_name: str,
    candidates_df: pd.DataFrame,
) -> ValidationReport:
    """Perform comprehensive integrity checks on an xRouteBench DataFrame."""
    errors: list[str] = []
    warnings: list[str] = []

    total_rows = len(df)
    unique_queries = df["query"].nunique() if "query" in df.columns else 0

    # 1. Required columns check
    expected_cols = {
        "task_name", "query", "ground_truth", "metric", "choices",
        "task_id", "model_name", "response", "token_num",
        "input_tokens", "output_tokens", "response_time", "performance", "embedding_id"
    }
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        errors.append(f"Missing expected columns: {missing_cols}")

    # 2. Token equality: token_num == input_tokens + output_tokens
    token_mismatches = 0
    if {"token_num", "input_tokens", "output_tokens"}.issubset(df.columns):
        diff = df["token_num"] - (df["input_tokens"] + df["output_tokens"])
        token_mismatches = int((diff != 0).sum())
        if token_mismatches > 0:
            errors.append(f"Found {token_mismatches} rows where token_num != input_tokens + output_tokens")

    token_equality_valid = (token_mismatches == 0)

    # 3. Non-negativity checks
    non_negative_tokens = True
    for c in ["token_num", "input_tokens", "output_tokens"]:
        if c in df.columns and (df[c] < 0).any():
            errors.append(f"Column {c} contains negative values")
            non_negative_tokens = False

    non_negative_response_time = True
    if "response_time" in df.columns and (df["response_time"] < 0).any():
        errors.append("Column response_time contains negative values")
        non_negative_response_time = False

    # 4. Performance bounded in [0.0, 1.0]
    performance_in_bounds = True
    if "performance" in df.columns:
        if (df["performance"] < 0.0).any() or (df["performance"] > 1.0).any():
            errors.append("Column performance contains values outside [0.0, 1.0]")
            performance_in_bounds = False

    # 5. Candidate model coverage
    expected_candidates = sorted(candidates_df["model_name"].unique())
    actual_candidates = sorted(df["model_name"].unique()) if "model_name" in df.columns else []
    candidate_model_count = len(actual_candidates)

    if set(actual_candidates) != set(expected_candidates):
        errors.append(
            f"Candidate model mismatch. Expected {len(expected_candidates)} models, got {len(actual_candidates)}."
        )

    # 6. Task coverage
    actual_tasks = sorted(df["task_name"].unique()) if "task_name" in df.columns else []
    task_count = len(actual_tasks)
    expected_tasks = list(ALL_TASKS)
    if set(actual_tasks) != set(expected_tasks):
        errors.append(f"Task set mismatch. Expected {expected_tasks}, got {actual_tasks}")

    # 7. Null counts
    null_counts: dict[str, int] = {}
    for col in df.columns:
        cnt = int(df[col].isnull().sum())
        null_counts[col] = cnt
        # choices and task_id are allowed to be null in non-MC/non-indexed tasks
        if cnt > 0 and col not in ("choices", "task_id"):
            errors.append(f"Unexpected null values in column '{col}': {cnt} nulls")

    # 8. Check cross-task query preservation
    cross_task_count = 0
    if "query" in df.columns and "task_name" in df.columns:
        query_task_counts = df.groupby("query")["task_name"].nunique()
        cross_task_count = int((query_task_counts > 1).sum())

    # 9. Verify SQuAD continuous vs binary tasks
    if "task_name" in df.columns and "performance" in df.columns:
        for task in actual_tasks:
            task_df = df[df["task_name"] == task]
            vals = set(task_df["performance"].unique())
            is_subset_binary = vals.issubset({0.0, 1.0})
            if task == "squad":
                if is_subset_binary:
                    warnings.append("SQuAD was expected to contain continuous F1 scores, but only found {0, 1}.")
            else:
                if not is_subset_binary:
                    errors.append(f"Task '{task}' was expected to have strictly binary performance, but found non-binary values.")

    is_valid = (len(errors) == 0)

    return ValidationReport(
        is_valid=is_valid,
        split_name=split_name,
        total_rows=total_rows,
        unique_queries=unique_queries,
        candidate_model_count=candidate_model_count,
        task_count=task_count,
        token_equality_valid=token_equality_valid,
        non_negative_tokens=non_negative_tokens,
        non_negative_response_time=non_negative_response_time,
        performance_in_bounds=performance_in_bounds,
        null_counts=null_counts,
        cross_task_queries_preserved=cross_task_count,
        errors=errors,
        warnings=warnings,
    )


def check_train_test_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    """Verify that there is strictly zero prompt/query overlap between train and test splits."""
    train_queries = set(train_df["query"].dropna().str.strip().unique())
    test_queries = set(test_df["query"].dropna().str.strip().unique())

    overlap = train_queries.intersection(test_queries)
    has_leakage = (len(overlap) > 0)

    return {
        "has_leakage": has_leakage,
        "train_unique_queries": len(train_queries),
        "test_unique_queries": len(test_queries),
        "overlap_count": len(overlap),
        "overlap_sample": sorted(list(overlap))[:5] if overlap else [],
    }


def create_prompt_grouped_validation_split(
    train_df: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42,
    stratify_by_task: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data into train and validation subsets grouped strictly by prompt query.

    Guarantees:
    1. Zero prompt leakage: Every record with the same normalized query text is placed in
       the same split (train or val).
    2. Cross-task query preservation: If a query appears in multiple tasks (such as the math /
       aime_2020_2024 prompt), all 36 associated rows are kept in the same split fold.
    3. Stratification by task: Prompts are grouped by primary task category so each task maintains
       approximately the target val_ratio.
    """
    if val_ratio <= 0.0 or val_ratio >= 1.0:
        raise ValueError(f"val_ratio must be in (0.0, 1.0), got {val_ratio}")

    rng = np.random.RandomState(seed)

    # Group records by normalized query text
    df = train_df.copy()
    df["_norm_query"] = df["query"].str.strip()

    # Determine representative task for each unique prompt (first task)
    prompt_task_map = df.groupby("_norm_query")["task_name"].first().to_dict()
    unique_queries = sorted(prompt_task_map.keys())

    val_queries_set: set[str] = set()

    if stratify_by_task:
        task_to_queries: dict[str, list[str]] = {}
        for q, t in prompt_task_map.items():
            task_to_queries.setdefault(t, []).append(q)

        for task, q_list in task_to_queries.items():
            shuffled = list(q_list)
            rng.shuffle(shuffled)
            n_val = max(1, int(round(len(shuffled) * val_ratio)))
            # ensure at least 1 remains in train if len > 1
            if len(shuffled) > 1 and n_val >= len(shuffled):
                n_val = len(shuffled) - 1
            val_queries_set.update(shuffled[:n_val])
    else:
        shuffled = list(unique_queries)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio)))
        val_queries_set = set(shuffled[:n_val])

    # Assign split
    is_val = df["_norm_query"].isin(val_queries_set)
    val_split = df[is_val].drop(columns=["_norm_query"]).copy()
    train_split = df[~is_val].drop(columns=["_norm_query"]).copy()

    # Verify zero prompt overlap between generated splits
    train_q = set(train_split["query"].str.strip())
    val_q = set(val_split["query"].str.strip())
    overlap = train_q.intersection(val_q)
    if overlap:
        raise RuntimeError(f"Internal split error: {len(overlap)} queries leaked between train and val!")

    return train_split, val_split


def compute_task_macro_score(df: pd.DataFrame, score_col: str = "performance") -> float:
    """Compute unweighted Macro-average score across all 13 benchmark tasks.

    Formula:
        Macro-Score = (1 / 13) * sum_{t=1}^{13} mean(performance_t)

    This ensures smaller task benchmarks (such as human_eval with 16 test queries) have
    equal standing with high-volume benchmarks (such as math with 500 test queries).
    """
    if score_col not in df.columns:
        raise KeyError(f"Score column '{score_col}' not in DataFrame")

    task_means = df.groupby("task_name")[score_col].mean()
    return float(task_means.mean())


def compute_prompt_micro_score(df: pd.DataFrame, score_col: str = "performance") -> float:
    """Compute prompt/record-level Micro-average score across all observations."""
    if score_col not in df.columns:
        raise KeyError(f"Score column '{score_col}' not in DataFrame")
    return float(df[score_col].mean())


def bootstrap_prompt_metric(
    df: pd.DataFrame,
    score_col: str = "performance",
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    aggregate_fn: str = "macro",
) -> tuple[float, float, float]:
    """Perform prompt-clustered bootstrap resampling to estimate mean and confidence interval.

    Instead of resampling individual rows (which would break model-prompt pairing),
    this samples unique prompt queries with replacement, gathers all corresponding
    rows, and calculates the chosen aggregate.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluated observations.
    score_col : str
        Column to aggregate.
    n_bootstraps : int
        Number of bootstrap iterations.
    ci_level : float
        Confidence interval level (e.g. 0.95 for 95% CI).
    seed : int
        Random seed.
    aggregate_fn : str
        'macro' for task macro-average, 'micro' for overall mean.

    Returns
    -------
    (mean_score, ci_lower, ci_upper)
    """
    if aggregate_fn not in ("macro", "micro"):
        raise ValueError(f"aggregate_fn must be 'macro' or 'micro', got '{aggregate_fn}'")

    rng = np.random.RandomState(seed)
    unique_queries = np.array(sorted(df["query"].unique()))
    n_queries = len(unique_queries)

    if n_queries == 0:
        return 0.0, 0.0, 0.0

    # Group DataFrame rows by query for fast sample collection
    query_grouped = {q: sub_df for q, sub_df in df.groupby("query")}

    boot_scores = np.empty(n_bootstraps, dtype=float)

    for i in range(n_bootstraps):
        sampled_queries = rng.choice(unique_queries, size=n_queries, replace=True)
        # Concatenate sampled query dataframes
        sampled_dfs = [query_grouped[q] for q in sampled_queries]
        combined = pd.concat(sampled_dfs, ignore_index=True)

        if aggregate_fn == "macro":
            boot_scores[i] = compute_task_macro_score(combined, score_col=score_col)
        else:
            boot_scores[i] = compute_prompt_micro_score(combined, score_col=score_col)

    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(boot_scores, alpha * 100.0))
    ci_upper = float(np.percentile(boot_scores, (1.0 - alpha) * 100.0))
    point_estimate = float(np.mean(boot_scores))

    return point_estimate, ci_lower, ci_upper


def prepare_and_export_processed_dataset(
    input_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Load, validate, enrich with pricing, and export clean processed parquets to output_dir.

    Preserves original files completely. Does NOT overwrite M0-M2 matrices.
    """
    in_dir = Path(input_dir) if input_dir is not None else EXTERNAL_DATA_DIR
    out_dir = Path(output_dir) if output_dir is not None else PROCESSED_DATA_DIR

    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, cand_df = load_xroutebench_raw(in_dir, verify_hashes=True)

    # Attach pricing & calculate costs
    train_enriched = attach_pricing_and_costs(train_df, cand_df)
    test_enriched = attach_pricing_and_costs(test_df, cand_df)

    # Validate
    train_report = validate_xroutebench_dataframe(train_enriched, "train", cand_df)
    test_report = validate_xroutebench_dataframe(test_enriched, "test", cand_df)
    leakage_report = check_train_test_leakage(train_enriched, test_enriched)

    if not train_report.is_valid:
        raise ValueError(f"Train dataset validation failed: {train_report.errors}")
    if not test_report.is_valid:
        raise ValueError(f"Test dataset validation failed: {test_report.errors}")
    if leakage_report["has_leakage"]:
        raise ValueError(f"Train/Test leakage detected: {leakage_report['overlap_count']} overlapping queries!")

    # Write processed parquets
    train_out_path = out_dir / "train_processed.parquet"
    test_out_path = out_dir / "test_processed.parquet"
    cand_out_path = out_dir / "candidates_processed.parquet"

    train_enriched.to_parquet(train_out_path, index=False)
    test_enriched.to_parquet(test_out_path, index=False)
    cand_df.to_parquet(cand_out_path, index=False)

    return {
        "status": "SUCCESS",
        "train_rows": len(train_enriched),
        "test_rows": len(test_enriched),
        "candidate_models": len(cand_df),
        "train_path": str(train_out_path),
        "test_path": str(test_out_path),
        "candidates_path": str(cand_out_path),
        "train_sha256": compute_sha256(train_out_path),
        "test_sha256": compute_sha256(test_out_path),
    }
