"""Blocked optional normalization helpers for RouterBench's precomputed tables.

Actual RouterBench ingestion remains blocked pending data-rights clarification.
This scaffold accepts an already trusted DataFrame only; it deliberately does
not download or deserialize the Hub's pickle files.
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROUTERBENCH_DATASET = "withmartian/routerbench"
ROUTERBENCH_REVISION = "784021482c3f320c6619ed4b3bb3b41a21424fcb"
ROUTERBENCH_ROOT = PROJECT_ROOT / "data" / "external" / "routerbench"
RAW_DATA_DIR = ROUTERBENCH_ROOT / "raw"
NORMALIZED_DATA_DIR = ROUTERBENCH_ROOT / "normalized"
NORMALIZED_SCHEMA_VERSION = "1.0"

SOURCE_FILES = {
    0: "routerbench_0shot.pkl",
    5: "routerbench_5shot.pkl",
}
SOURCE_FILE_SHA256 = {
    0: "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d",
    5: "fbd7d3d16fba2759a18fa0ad44409d3e0e92ba80d03d90827eed1a6d084c9ffe",
}

NORMALIZED_COLUMNS = [
    "schema_version",
    "dataset_name",
    "dataset_revision",
    "source_file",
    "source_file_sha256",
    "shot_count",
    "source_sample_id",
    "source_eval_name",
    "task_category",
    "source_prompt",
    "source_model_id",
    "source_response",
    "source_response_column",
    "source_performance_value",
    "source_performance_column",
    "source_performance_metric",
    "source_cost_value",
    "source_cost_column",
    "source_cost_currency",
    "source_cost_method",
    "source_oracle_model_id",
]

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


def normalized_output_path(shot_count: int) -> Path:
    """Return the separate, ignored output location for a shot variant."""
    if isinstance(shot_count, bool) or shot_count not in SOURCE_FILES:
        raise ValueError("shot_count must be 0 or 5.")
    return NORMALIZED_DATA_DIR / f"routerbench_{shot_count}shot.normalized.parquet"


def _validate_provenance(
    shot_count: int,
    source_file: str,
    dataset_revision: str,
    source_file_sha256: str,
) -> None:
    if isinstance(shot_count, bool) or shot_count not in SOURCE_FILES:
        raise ValueError("shot_count must be 0 or 5.")
    if (
        not isinstance(source_file, str)
        or Path(source_file).name != source_file
        or source_file != SOURCE_FILES[shot_count]
    ):
        raise ValueError("source_file must match the selected RouterBench shot variant.")
    if (
        not isinstance(dataset_revision, str)
        or not _REVISION_PATTERN.fullmatch(dataset_revision)
    ):
        raise ValueError("dataset_revision must be a 40-character commit SHA.")
    if (
        not isinstance(source_file_sha256, str)
        or not _SHA256_PATTERN.fullmatch(source_file_sha256)
    ):
        raise ValueError("source_file_sha256 must be a 64-character SHA-256 digest.")
    if (
        dataset_revision.lower() == ROUTERBENCH_REVISION.lower()
        and source_file_sha256.lower() != SOURCE_FILE_SHA256[shot_count]
    ):
        raise ValueError("source_file_sha256 does not match the pinned Hub file.")


def normalize_wide_frame(
    frame: pd.DataFrame,
    *,
    shot_count: int,
    source_file: str,
    source_file_sha256: str,
    dataset_revision: str = ROUTERBENCH_REVISION,
    task_category_map: dict[str, str] | None = None,
    performance_metric_map: dict[str, str] | None = None,
    source_cost_method: str = "routerbench_precomputed",
) -> pd.DataFrame:
    """Convert a trusted RouterBench wide table to one row per sample/model.

    Source model identifiers, prompts, responses, costs, and performance
    values are copied without mapping, rescoring, or recomputation.
    """
    _validate_provenance(
        shot_count,
        source_file,
        dataset_revision,
        source_file_sha256,
    )
    if not frame.columns.is_unique:
        raise ValueError("Input frame contains duplicate column names.")

    required = {"sample_id", "eval_name", "prompt"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"RouterBench frame is missing required columns: {missing}")

    response_columns = [
        column for column in frame.columns
        if isinstance(column, str) and column.endswith("|model_response")
    ]
    model_ids = sorted(column[: -len("|model_response")] for column in response_columns)
    if not model_ids:
        raise ValueError("No per-model '|model_response' columns were found.")

    for model_id in model_ids:
        required_model_columns = {
            model_id,
            f"{model_id}|total_cost",
            f"{model_id}|model_response",
        }
        missing_model_columns = sorted(
            required_model_columns.difference(frame.columns)
        )
        if missing_model_columns:
            raise ValueError(
                f"RouterBench model {model_id!r} is missing columns: "
                f"{missing_model_columns}"
            )

    task_category_map = task_category_map or {}
    performance_metric_map = performance_metric_map or {}
    has_oracle_column = "oracle_model_to_route_to" in frame.columns
    records: list[dict[str, Any]] = []

    for source_row in frame.to_dict(orient="records"):
        sample_id = source_row["sample_id"]
        eval_name = source_row["eval_name"]
        for model_id in model_ids:
            records.append({
                "schema_version": NORMALIZED_SCHEMA_VERSION,
                "dataset_name": ROUTERBENCH_DATASET,
                "dataset_revision": dataset_revision,
                "source_file": source_file,
                "source_file_sha256": source_file_sha256.lower(),
                "shot_count": shot_count,
                "source_sample_id": sample_id,
                "source_eval_name": eval_name,
                "task_category": task_category_map.get(eval_name),
                "source_prompt": source_row["prompt"],
                "source_model_id": model_id,
                "source_response": source_row[f"{model_id}|model_response"],
                "source_response_column": f"{model_id}|model_response",
                "source_performance_value": source_row[model_id],
                "source_performance_column": model_id,
                "source_performance_metric": performance_metric_map.get(eval_name),
                "source_cost_value": source_row[f"{model_id}|total_cost"],
                "source_cost_column": f"{model_id}|total_cost",
                "source_cost_currency": "USD",
                "source_cost_method": source_cost_method,
                "source_oracle_model_id": (
                    source_row["oracle_model_to_route_to"]
                    if has_oracle_column
                    else None
                ),
            })

    return pd.DataFrame.from_records(records, columns=NORMALIZED_COLUMNS)
