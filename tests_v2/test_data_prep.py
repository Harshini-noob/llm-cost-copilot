"""
tests_v2/test_data_prep.py

Unit tests for training data preparation, label sanitization, and leakage prevention in M3.
Adheres strictly to the ZERO ADDITIONAL SPENDING and OFFLINE EVALUATION policies.
Uses synthetic fixtures for algorithmic validation and tests real matrix loading safely.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import socket
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eval.baselines import DEFAULT_MATRIX_PATH
from eval.routers.data_prep import (
    STATUS_INSUFFICIENT,
    STATUS_SUFFICIENT,
    assess_data_sufficiency,
    prepare_routing_dataset,
)
from eval.routers.encoders import DeterministicTestEncoder


@pytest.fixture
def synthetic_benchmark_df() -> pd.DataFrame:
    """Creates a deterministic synthetic benchmark DataFrame with multiple models, tasks, and edge cases."""
    rows = [
        # prompt 1: math, both models succeed
        {
            "prompt_id": "synth_math_1",
            "task_type": "math",
            "model_id": "model_fast",
            "provider": "prov_a",
            "run": 1,
            "correct": True,
            "score": 1.0,
            "cost": 0.0001,
            "input_tokens": 50,
            "output_tokens": 20,
            "execution_success": True,
            "evaluation_status": "scored",
        },
        {
            "prompt_id": "synth_math_1",
            "task_type": "math",
            "model_id": "model_smart",
            "provider": "prov_b",
            "run": 1,
            "correct": True,
            "score": 1.0,
            "cost": 0.0005,
            "input_tokens": 50,
            "output_tokens": 30,
            "execution_success": True,
            "evaluation_status": "scored",
        },
        # prompt 2: math, fast model fails execution, smart model is correct
        {
            "prompt_id": "synth_math_2",
            "task_type": "math",
            "model_id": "model_fast",
            "provider": "prov_a",
            "run": 1,
            "correct": None,
            "score": np.nan,
            "cost": 0.00002,  # Incurred partial cost before connection drop
            "input_tokens": 40,
            "output_tokens": 0,
            "execution_success": False,
            "error": "ConnectionReset",
            "evaluation_status": "execution_failed",
        },
        {
            "prompt_id": "synth_math_2",
            "task_type": "math",
            "model_id": "model_smart",
            "provider": "prov_b",
            "run": 1,
            "correct": True,
            "score": 1.0,
            "cost": 0.0006,
            "input_tokens": 40,
            "output_tokens": 25,
            "execution_success": True,
            "evaluation_status": "scored",
        },
        # prompt 3: code, fast model incorrect, smart model correct
        {
            "prompt_id": "synth_code_1",
            "task_type": "code",
            "model_id": "model_fast",
            "provider": "prov_a",
            "run": 1,
            "correct": False,
            "score": 0.0,
            "cost": 0.0002,
            "input_tokens": 100,
            "output_tokens": 40,
            "execution_success": True,
            "evaluation_status": "scored",
        },
        {
            "prompt_id": "synth_code_1",
            "task_type": "code",
            "model_id": "model_smart",
            "provider": "prov_b",
            "run": 1,
            "correct": True,
            "score": 1.0,
            "cost": 0.0008,
            "input_tokens": 100,
            "output_tokens": 50,
            "execution_success": True,
            "evaluation_status": "scored",
        },
        # prompt 4: code, fast model unscored/pending, smart model not evaluated (incomplete coverage)
        {
            "prompt_id": "synth_code_2",
            "task_type": "code",
            "model_id": "model_fast",
            "provider": "prov_a",
            "run": 1,
            "correct": None,
            "score": np.nan,
            "cost": 0.0002,
            "input_tokens": 90,
            "output_tokens": 30,
            "execution_success": True,
            "evaluation_status": "unscored",
        },
    ]
    return pd.DataFrame(rows)


def test_label_handling_and_status_distinction(synthetic_benchmark_df: pd.DataFrame) -> None:
    """Verifies correct is 1, incorrect is 0, and execution failures or unscored evaluations are None."""
    dataset = prepare_routing_dataset(
        synthetic_benchmark_df,
        split_ratio=0.5,
        seed=42,
        benchmark_config_id=None,
    )

    records_by_key = {(r.prompt_id, r.model_id): r for r in dataset.records}

    # Correct -> label 1
    rec_m1_fast = records_by_key[("synth_math_1", "model_fast")]
    assert rec_m1_fast.label == 1
    assert rec_m1_fast.execution_status == "success"
    assert rec_m1_fast.evaluation_status == "scored"
    assert rec_m1_fast.cost == pytest.approx(0.0001)
    assert rec_m1_fast.token_usage == {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}

    # Incorrect -> label 0
    rec_c1_fast = records_by_key[("synth_code_1", "model_fast")]
    assert rec_c1_fast.label == 0
    assert rec_c1_fast.execution_status == "success"
    assert rec_c1_fast.evaluation_status == "scored"

    # Execution failure -> label None (NEVER silently 0)
    rec_m2_fast = records_by_key[("synth_math_2", "model_fast")]
    assert rec_m2_fast.label is None
    assert rec_m2_fast.execution_status == "failed"
    assert rec_m2_fast.evaluation_status == "execution_failed"
    assert rec_m2_fast.cost == pytest.approx(0.00002)  # Incurred cost preserved

    # Unscored -> label None
    rec_c2_fast = records_by_key[("synth_code_2", "model_fast")]
    assert rec_c2_fast.label is None
    assert rec_c2_fast.execution_status == "success"
    assert rec_c2_fast.evaluation_status == "unscored"


def test_strict_prompt_level_splitting_prevents_leakage(synthetic_benchmark_df: pd.DataFrame) -> None:
    """Ensures prompts are partitioned strictly without leakage: all model rows stay in prompt split."""
    encoder = DeterministicTestEncoder(dimension=32)
    dataset = prepare_routing_dataset(
        synthetic_benchmark_df,
        split_ratio=0.5,
        seed=42,
        encoder=encoder,
        benchmark_config_id=None,
    )

    train_ids = set(dataset.train_prompt_ids)
    test_ids = set(dataset.test_prompt_ids)

    # Disjoint prompt sets
    assert train_ids.isdisjoint(test_ids)
    assert train_ids.union(test_ids) == {"synth_math_1", "synth_math_2", "synth_code_1", "synth_code_2"}

    # Verify every outcome record matches its prompt's split
    for r in dataset.records:
        if r.prompt_id in train_ids:
            assert r.split == "train"
        else:
            assert r.split == "test"

    # Verify get_train_data and get_test_data isolation
    X_train, y_train, pids_train = dataset.get_train_data("model_smart")
    for pid in pids_train:
        assert pid in train_ids
        assert pid not in test_ids


def test_incomplete_model_coverage_and_missing_values(synthetic_benchmark_df: pd.DataFrame) -> None:
    """Verifies that missing model evaluations are handled without fabricating values."""
    dataset = prepare_routing_dataset(
        synthetic_benchmark_df,
        split_ratio=0.5,
        seed=42,
        benchmark_config_id=None,
    )

    # model_smart was never run on synth_code_2
    smart_prompts = {r.prompt_id for r in dataset.records if r.model_id == "model_smart"}
    assert "synth_code_2" not in smart_prompts

    # fast model was run on all 4 prompts
    fast_prompts = {r.prompt_id for r in dataset.records if r.model_id == "model_fast"}
    assert fast_prompts == {"synth_math_1", "synth_math_2", "synth_code_1", "synth_code_2"}


def test_prompt_embeddings_attached_deterministically(synthetic_benchmark_df: pd.DataFrame) -> None:
    """Verifies prompt embeddings are attached to PromptRecord and ModelOutcomeRecord."""
    encoder = DeterministicTestEncoder(dimension=64)
    prompt_lookup = {
        "synth_math_1": "Calculate 15 * 12.",
        "synth_math_2": "Calculate 84 / 7.",
        "synth_code_1": "Write a python function to reverse a list.",
        "synth_code_2": "Parse a date string in ISO format.",
    }
    dataset = prepare_routing_dataset(
        synthetic_benchmark_df,
        encoder=encoder,
        prompt_lookup=prompt_lookup,
        split_ratio=0.5,
        benchmark_config_id=None,
    )

    for pid, pr in dataset.prompts.items():
        assert pr.prompt_embedding is not None
        assert pr.prompt_embedding.shape == (64,)
        assert pr.prompt_text == prompt_lookup[pid]

    for rec in dataset.records:
        assert rec.prompt_embedding is not None
        assert rec.prompt_embedding.shape == (64,)
        assert np.array_equal(rec.prompt_embedding, dataset.prompts[rec.prompt_id].prompt_embedding)


def test_data_sufficiency_assessment_marks_small_data_insufficient() -> None:
    """Data sufficiency flags INSUFFICIENT_DATA / exploratory for small sets and SUFFICIENT for large sets."""
    # Small dataset (< 10 prompts)
    small_records = [
        prepare_routing_dataset(
            pd.DataFrame([
                {
                    "prompt_id": f"p_{i}",
                    "task_type": "math",
                    "model_id": "m1",
                    "correct": True,
                    "score": 1.0,
                    "cost": 0.001,
                    "execution_success": True,
                    "evaluation_status": "scored",
                }
                for i in range(4)
            ]),
            benchmark_config_id=None,
        ).sufficiency_report
    ][0]

    assert small_records.status == STATUS_INSUFFICIENT
    assert small_records.is_sufficient_for_ml is False
    assert "INSUFFICIENT_DATA" in small_records.notes

    # Large dataset (>= 12 prompts across multiple categories)
    large_df = pd.DataFrame([
        {
            "prompt_id": f"prompt_{i}",
            "task_type": "math" if i < 6 else "code",
            "model_id": "m1",
            "correct": bool(i % 2 == 0),
            "score": 1.0 if (i % 2 == 0) else 0.0,
            "cost": 0.001,
            "execution_success": True,
            "evaluation_status": "scored",
        }
        for i in range(14)
    ])
    large_dataset = prepare_routing_dataset(large_df, benchmark_config_id=None)
    assert large_dataset.sufficiency_report.status == STATUS_SUFFICIENT
    assert large_dataset.sufficiency_report.is_sufficient_for_ml is True


def test_real_historical_matrix_loading_and_sufficiency() -> None:
    """Safely loads real historical matrix.parquet without mutation and flags INSUFFICIENT_DATA."""
    # Record initial hash of matrix.parquet
    parquet_path = Path(DEFAULT_MATRIX_PATH)
    assert parquet_path.exists()
    initial_hash = hashlib.sha256(parquet_path.read_bytes()).hexdigest()

    encoder = DeterministicTestEncoder(dimension=32)
    dataset = prepare_routing_dataset(
        parquet_path,
        benchmark_config_id="m1-efa7ec3afd4f",
        encoder=encoder,
    )

    # 1. Source file must remain byte-for-byte unchanged
    final_hash = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
    assert initial_hash == final_hash

    # 2. Covers exactly the 2 real math prompts
    assert sorted(dataset.prompts.keys()) == ["math_001", "math_002"]
    assert dataset.candidate_models == [
        "gemini_3_5_flash_lite",
        "gpt_oss_120b",
        "gpt_oss_20b",
        "qwen3_30b_a3b_openrouter",
    ]

    # 3. Must be flagged as INSUFFICIENT_DATA / EXPLORATORY
    report = dataset.sufficiency_report
    assert report.status == STATUS_INSUFFICIENT
    assert report.is_sufficient_for_ml is False
    assert "INSUFFICIENT_DATA" in report.notes
    assert report.unique_prompts == 2
    assert report.task_categories == ["math"]

    # 4. Prompt text correctly resolved from pilot dataset
    pr1 = dataset.prompts["math_001"]
    assert "notebooks" in pr1.prompt_text.lower()
    pr2 = dataset.prompts["math_002"]
    assert "solve for x" in pr2.prompt_text.lower()


def test_repeated_run_handling_uses_empirical_probability() -> None:
    """Repeated runs compute correct_runs / evaluated_runs and preserve failures as unavailable."""
    repeated_df = pd.DataFrame([
        # Prompt 1, model_fast: 3 runs (1 correct, 1 incorrect, 1 execution failure)
        {
            "prompt_id": "p_rep_1",
            "task_type": "math",
            "model_id": "model_fast",
            "run": 1,
            "correct": True,
            "score": 1.0,
            "execution_success": True,
            "evaluation_status": "scored",
            "cost": 0.0001,
        },
        {
            "prompt_id": "p_rep_1",
            "task_type": "math",
            "model_id": "model_fast",
            "run": 2,
            "correct": False,
            "score": 0.0,
            "execution_success": True,
            "evaluation_status": "scored",
            "cost": 0.0001,
        },
        {
            "prompt_id": "p_rep_1",
            "task_type": "math",
            "model_id": "model_fast",
            "run": 3,
            "correct": None,
            "score": np.nan,
            "execution_success": False,
            "error": "RateLimit",
            "evaluation_status": "execution_failed",
            "cost": 0.0,
        },
    ])

    dataset = prepare_routing_dataset(
        repeated_df,
        benchmark_config_id=None,
    )
    assert len(dataset.records) == 1
    rec = dataset.records[0]

    assert rec.run_count == 3
    assert rec.evaluated_runs == 2
    assert rec.correct_runs == 1
    assert rec.empirical_quality == pytest.approx(0.5)
    assert rec.label == pytest.approx(0.5)
    assert rec.run_outcomes == (1, 0, None)
    assert rec.execution_status == "failed"  # At least one run failed


def test_zero_network_calls_during_data_prep(monkeypatch: pytest.MonkeyPatch, synthetic_benchmark_df: pd.DataFrame) -> None:
    """Guarantees prepare_routing_dataset makes strictly zero socket connections."""
    def blocked_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network socket call attempted during offline data preparation!")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)

    enc = DeterministicTestEncoder(dimension=16)
    dataset = prepare_routing_dataset(
        synthetic_benchmark_df,
        encoder=enc,
        benchmark_config_id=None,
    )
    assert len(dataset.records) > 0
