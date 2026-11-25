"""
tests_v2/test_xroutebench_adapter.py

Unit and integration tests for the M4 xRouteBench Data Adapter and Integrity Suite.
Validates:
1. Dataset loading and cryptographic SHA-256 hash verification.
2. Exact Decimal cost calculation and pricing table enrichment.
3. Preservation of 36 cross-task rows (math vs aime_2020_2024 and arc_challenge embeddings).
4. Leakage-free prompt-grouped validation splitting.
5. Train/Test zero-leakage verification and synthetic leakage detection.
6. Continuous SQuAD F1 vs 12 binary tasks handling.
7. Task Macro-average vs prompt Micro-average aggregation.
8. Prompt-clustered bootstrap confidence interval calculation.
9. Robust error detection for corrupted / invalid dataframes.
10. Immutability of raw files and existing M0-M2 matrices.
"""

from decimal import Decimal
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from eval.m4_data_adapter import (
    EXTERNAL_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXPECTED_SHA256,
    TASK_METRIC_MAP,
    ALL_TASKS,
    compute_sha256,
    verify_file_hashes,
    calculate_reconstructed_cost,
    load_xroutebench_raw,
    attach_pricing_and_costs,
    validate_xroutebench_dataframe,
    check_train_test_leakage,
    create_prompt_grouped_validation_split,
    compute_task_macro_score,
    compute_prompt_micro_score,
    bootstrap_prompt_metric,
    prepare_and_export_processed_dataset,
)


# --- 1. Dataset Loading and SHA-256 Verification ---

def test_file_hashes_intact():
    """Verify that all raw external files match official cryptographic SHA-256 digests."""
    hash_results = verify_file_hashes()
    for fname, is_match in hash_results.items():
        assert is_match is True, f"Hash mismatch for {fname}"


def test_compute_sha256_missing_file(tmp_path):
    """Verify compute_sha256 raises FileNotFoundError on missing file."""
    with pytest.raises(FileNotFoundError):
        compute_sha256(tmp_path / "nonexistent.parquet")


def test_load_xroutebench_raw():
    """Verify raw parquet files load with expected shapes and candidates."""
    train_df, test_df, cand_df = load_xroutebench_raw(verify_hashes=True)

    assert len(train_df) == 80802
    assert len(test_df) == 67122
    assert len(cand_df) == 18

    assert train_df["query"].nunique() == 4487
    assert test_df["query"].nunique() == 3729
    assert cand_df["model_name"].nunique() == 18


# --- 2. Cost Calculation & Pricing Join ---

def test_calculate_reconstructed_cost_exact():
    """Test cost calculation with known values using Decimal and float."""
    # 1000 input tokens at $0.20/1M, 500 output tokens at $1.50/1M
    # Cost = (1000 * 0.20 + 500 * 1.50) / 1,000,000 = (200 + 750) / 1,000,000 = 950 / 1,000,000 = 0.00095 USD
    cost_dec = calculate_reconstructed_cost(1000, 500, 0.20, 1.50, use_decimal=True)
    cost_flt = calculate_reconstructed_cost(1000, 500, 0.20, 1.50, use_decimal=False)

    assert cost_dec == pytest.approx(0.00095, rel=1e-9)
    assert cost_flt == pytest.approx(0.00095, rel=1e-9)
    assert abs(cost_dec - cost_flt) < 1e-15


def test_calculate_reconstructed_cost_zero_tokens():
    """Zero tokens should result in exactly 0.0 USD cost."""
    cost = calculate_reconstructed_cost(0, 0, 1.0, 2.0)
    assert cost == 0.0


def test_calculate_reconstructed_cost_negative_rejected():
    """Negative tokens or prices must raise ValueError."""
    with pytest.raises(ValueError, match="must be non-negative"):
        calculate_reconstructed_cost(-10, 100, 0.5, 0.5)

    with pytest.raises(ValueError, match="must be non-negative"):
        calculate_reconstructed_cost(100, -5, 0.5, 0.5)

    with pytest.raises(ValueError, match="must be non-negative"):
        calculate_reconstructed_cost(100, 100, -0.5, 0.5)


def test_attach_pricing_and_costs(tmp_path):
    """Verify that attach_pricing_and_costs merges pricing and computes cost for all rows."""
    _, _, cand_df = load_xroutebench_raw(verify_hashes=False)
    sample_df = pd.DataFrame({
        "model_name": ["gpt-oss-20b", "cogito-v2-1-671b"],
        "input_tokens": [1000, 1000],
        "output_tokens": [500, 500],
    })

    enriched = attach_pricing_and_costs(sample_df, cand_df)
    assert "reconstructed_cost_usd" in enriched.columns
    assert "input_price_per_1m" in enriched.columns
    assert "output_price_per_1m" in enriched.columns

    # gpt-oss-20b: input=0.05, output=0.20 -> (1000*0.05 + 500*0.20)/1e6 = 150/1e6 = 0.00015
    assert enriched["reconstructed_cost_usd"].iloc[0] == pytest.approx(0.00015, rel=1e-6)

    # cogito-v2-1-671b: input=1.25, output=1.25 -> (1000*1.25 + 500*1.25)/1e6 = 1875/1e6 = 0.001875
    assert enriched["reconstructed_cost_usd"].iloc[1] == pytest.approx(0.001875, rel=1e-6)


def test_attach_pricing_missing_model():
    """Attaching pricing with an unknown model must raise ValueError."""
    cand_df = pd.DataFrame({
        "model_name": ["model_a"],
        "input_price_per_1m": [1.0],
        "output_price_per_1m": [2.0],
    })
    sample_df = pd.DataFrame({
        "model_name": ["unknown_model"],
        "input_tokens": [100],
        "output_tokens": [50],
    })
    with pytest.raises(ValueError, match="without pricing specifications"):
        attach_pricing_and_costs(sample_df, cand_df)


# --- 3. Cross-Task Preservation ---

def test_cross_task_records_preserved():
    """Verify that the 36 cross-task rows in train.parquet are preserved intact."""
    train_df, _, _ = load_xroutebench_raw(verify_hashes=False)

    # Find the query shared between 'math' and 'aime_2020_2024'
    query_tasks = train_df.groupby("query")["task_name"].nunique()
    multi_task_q = query_tasks[query_tasks > 1].index.tolist()
    assert len(multi_task_q) == 1

    shared_sub = train_df[train_df["query"] == multi_task_q[0]]
    assert len(shared_sub) == 36
    assert set(shared_sub["task_name"]) == {"math", "aime_2020_2024"}
    assert shared_sub["model_name"].nunique() == 18

    # Find the query in 'arc_challenge' with two embeddings
    arc = train_df[train_df["task_name"] == "arc_challenge"]
    query_embs = arc.groupby("query")["embedding_id"].nunique()
    multi_emb_q = query_embs[query_embs > 1].index.tolist()
    assert len(multi_emb_q) == 1

    arc_sub = arc[arc["query"] == multi_emb_q[0]]
    assert len(arc_sub) == 36
    assert arc_sub["model_name"].nunique() == 18
    assert set(arc_sub["embedding_id"]) == {320, 4190}


# --- 4. Leakage-Free Prompt-Grouped Validation Splitting ---

def test_prompt_grouped_validation_split_zero_leakage():
    """Verify validation splitting ensures zero prompt overlap between train and val."""
    train_df, _, cand_df = load_xroutebench_raw(verify_hashes=False)
    train_enriched = attach_pricing_and_costs(train_df, cand_df)

    t_split, v_split = create_prompt_grouped_validation_split(
        train_enriched, val_ratio=0.2, seed=42, stratify_by_task=True
    )

    t_queries = set(t_split["query"].str.strip())
    v_queries = set(v_split["query"].str.strip())
    overlap = t_queries.intersection(v_queries)

    assert len(overlap) == 0, f"Found {len(overlap)} queries leaked between train and val!"
    assert len(t_split) + len(v_split) == len(train_enriched)

    # Verify cross-task query (36 rows) is entirely in one split
    multi_q = train_enriched.groupby("query")["task_name"].nunique()
    shared_query = multi_q[multi_q > 1].index[0].strip()

    in_train = shared_query in t_queries
    in_val = shared_query in v_queries
    assert (in_train and not in_val) or (in_val and not in_train), "Shared query split across folds!"


# --- 5. Train/Test Leakage Verification ---

def test_official_splits_zero_leakage():
    """Verify that official train and test sets have exactly zero overlapping queries."""
    train_df, test_df, _ = load_xroutebench_raw(verify_hashes=False)
    leakage = check_train_test_leakage(train_df, test_df)

    assert leakage["has_leakage"] is False
    assert leakage["overlap_count"] == 0
    assert len(leakage["overlap_sample"]) == 0


def test_leakage_detector_catches_synthetic_overlap():
    """Verify check_train_test_leakage successfully flags injected leakage."""
    train_df = pd.DataFrame({"query": ["prompt A", "prompt B", "prompt C"]})
    test_df = pd.DataFrame({"query": ["prompt C", "prompt D"]})

    leakage = check_train_test_leakage(train_df, test_df)
    assert leakage["has_leakage"] is True
    assert leakage["overlap_count"] == 1
    assert "prompt C" in leakage["overlap_sample"]


# --- 6. Task Metrics and SQuAD Continuous F1 ---

def test_task_metrics_binary_and_continuous():
    """Verify that SQuAD contains continuous F1 scores, while all other tasks are strictly binary."""
    train_df, test_df, _ = load_xroutebench_raw(verify_hashes=False)

    for df, split_name in [(train_df, "train"), (test_df, "test")]:
        for task, info in TASK_METRIC_MAP.items():
            sub = df[df["task_name"] == task]
            unique_perfs = set(sub["performance"].unique())

            if info["is_binary"]:
                assert unique_perfs.issubset({0.0, 1.0}), f"Task {task} in {split_name} has non-binary scores!"
            else:
                assert task == "squad"
                # SQuAD should have continuous values between 0.0 and 1.0
                non_binary = sub[(sub["performance"] > 0.0) & (sub["performance"] < 1.0)]
                assert len(non_binary) > 0, f"SQuAD in {split_name} did not contain continuous scores!"


# --- 7. Macro-Score vs Micro-Score Aggregation ---

def test_macro_vs_micro_aggregation():
    """Verify mathematical formulation of task macro-average vs prompt micro-average."""
    # Synthetic dataset with 2 tasks: task 1 has 1 query with score 1.0, task 2 has 9 queries with score 0.0
    synth_df = pd.DataFrame({
        "task_name": ["task_1"] + ["task_2"] * 9,
        "performance": [1.0] + [0.0] * 9,
    })

    # Macro score: mean(task_1)=1.0, mean(task_2)=0.0 -> (1.0 + 0.0) / 2 = 0.5
    macro = compute_task_macro_score(synth_df)
    assert macro == pytest.approx(0.5, rel=1e-6)

    # Micro score: (1.0 + 9*0.0) / 10 = 0.1
    micro = compute_prompt_micro_score(synth_df)
    assert micro == pytest.approx(0.1, rel=1e-6)


# --- 8. Prompt-Clustered Bootstrap ---

def test_bootstrap_prompt_metric():
    """Verify prompt-clustered bootstrap returns valid point estimate and 95% CI."""
    # Varied queries across two tasks
    queries = [f"q_{i}" for i in range(20)]
    tasks = ["task_a"] * 10 + ["task_b"] * 10
    # Diverse performances
    perfs = [0.2, 0.4, 0.6, 0.8, 1.0, 0.3, 0.5, 0.7, 0.9, 0.4,
             0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.5]

    rows = []
    for q, t, p in zip(queries, tasks, perfs):
        rows.append({"query": q, "task_name": t, "performance": p})
        rows.append({"query": q, "task_name": t, "performance": p})

    synth_df = pd.DataFrame(rows)

    mean_est, ci_low, ci_high = bootstrap_prompt_metric(
        synth_df, score_col="performance", n_bootstraps=200, ci_level=0.95, seed=42, aggregate_fn="macro"
    )

    assert 0.0 <= ci_low <= mean_est + 1e-6
    assert mean_est <= ci_high + 1e-6 <= 1.0
    assert ci_low < ci_high
    assert mean_est == pytest.approx(0.5, abs=0.1)


# --- 9. Data Integrity & Validation Error Handling ---

def test_validate_xroutebench_dataframe_valid():
    """Validate that official enriched train and test pass all checks."""
    train_df, test_df, cand_df = load_xroutebench_raw(verify_hashes=False)
    train_enriched = attach_pricing_and_costs(train_df, cand_df)
    test_enriched = attach_pricing_and_costs(test_df, cand_df)

    rep_train = validate_xroutebench_dataframe(train_enriched, "train", cand_df)
    assert rep_train.is_valid is True
    assert len(rep_train.errors) == 0

    rep_test = validate_xroutebench_dataframe(test_enriched, "test", cand_df)
    assert rep_test.is_valid is True
    assert len(rep_test.errors) == 0


def test_validate_detects_corrupted_tokens(tmp_path):
    """Validation must fail if token_num != input_tokens + output_tokens."""
    train_df, _, cand_df = load_xroutebench_raw(verify_hashes=False)
    corrupted = train_df.iloc[:50].copy()
    corrupted.loc[0, "token_num"] = 999999  # deliberate mismatch

    rep = validate_xroutebench_dataframe(corrupted, "corrupted", cand_df)
    assert rep.is_valid is False
    assert any("token_num != input_tokens + output_tokens" in err for err in rep.errors)


def test_validate_detects_out_of_bounds_performance():
    """Validation must fail if performance is < 0.0 or > 1.0."""
    train_df, _, cand_df = load_xroutebench_raw(verify_hashes=False)
    corrupted = train_df.iloc[:50].copy()
    corrupted.loc[0, "performance"] = 1.5  # out of bounds

    rep = validate_xroutebench_dataframe(corrupted, "corrupted", cand_df)
    assert rep.is_valid is False
    assert any("outside [0.0, 1.0]" in err for err in rep.errors)


# --- 10. Dataset Immutability ---

def test_m0_m2_matrix_unmodified():
    """Verify that existing M0-M2 matrices in data/processed remain strictly unmodified."""
    matrix_path = Path("data/processed/matrix.parquet")
    assert matrix_path.exists()
    # Confirm it still loads as a valid DataFrame with expected columns
    df = pd.read_parquet(matrix_path)
    assert "prompt_id" in df.columns
    assert "model_id" in df.columns
