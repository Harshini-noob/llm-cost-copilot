import pandas as pd
import pytest

from eval.routerbench_import import (
    NORMALIZED_COLUMNS,
    NORMALIZED_DATA_DIR,
    ROUTERBENCH_REVISION,
    SOURCE_FILE_SHA256,
    normalize_wide_frame,
    normalized_output_path,
)


def _source_frame():
    return pd.DataFrame({
        "sample_id": ["gsm8k.test.0", "mmlu.test.1"],
        "eval_name": ["gsm8k", "mmlu"],
        "prompt": ["prompt one", "prompt two"],
        "gpt-3.5-turbo-1106": [1.0, 0.25],
        "gpt-3.5-turbo-1106|model_response": ["answer one", "answer two"],
        "gpt-3.5-turbo-1106|total_cost": [0.00001, 0.00002],
        "meta/llama-2-70b-chat": [0.5, 0.0],
        "meta/llama-2-70b-chat|model_response": ["response one", "response two"],
        "meta/llama-2-70b-chat|total_cost": [0.00003, 0.00004],
        "oracle_model_to_route_to": [
            "gpt-3.5-turbo-1106",
            "meta/llama-2-70b-chat",
        ],
    })


def test_normalizes_wide_rows_without_mapping_source_values():
    source = _source_frame()
    original = source.copy(deep=True)

    normalized = normalize_wide_frame(
        source,
        shot_count=0,
        source_file="routerbench_0shot.pkl",
        source_file_sha256=SOURCE_FILE_SHA256[0],
        task_category_map={"gsm8k": "math"},
        performance_metric_map={"gsm8k": "accuracy"},
    )

    assert len(normalized) == 4
    assert list(normalized.columns) == NORMALIZED_COLUMNS
    assert set(normalized["source_model_id"]) == {
        "gpt-3.5-turbo-1106",
        "meta/llama-2-70b-chat",
    }
    gpt_answer = normalized[
        (normalized["source_sample_id"] == "gsm8k.test.0")
        & (normalized["source_model_id"] == "gpt-3.5-turbo-1106")
    ].iloc[0]
    assert gpt_answer["source_prompt"] == "prompt one"
    assert gpt_answer["source_response"] == "answer one"
    assert gpt_answer["source_performance_value"] == 1.0
    assert gpt_answer["source_cost_value"] == 0.00001
    assert gpt_answer["task_category"] == "math"
    assert gpt_answer["source_performance_metric"] == "accuracy"
    assert gpt_answer["source_oracle_model_id"] == "gpt-3.5-turbo-1106"
    assert "model_id" not in normalized.columns
    assert "correct" not in normalized.columns
    pd.testing.assert_frame_equal(source, original)


def test_preserves_unmapped_task_names_and_shot_variant_provenance():
    normalized = normalize_wide_frame(
        _source_frame(),
        shot_count=5,
        source_file="routerbench_5shot.pkl",
        source_file_sha256=SOURCE_FILE_SHA256[5].upper(),
    )

    assert set(normalized["shot_count"]) == {5}
    assert set(normalized["source_eval_name"]) == {"gsm8k", "mmlu"}
    assert normalized["task_category"].isna().all()
    assert set(normalized["source_file_sha256"]) == {SOURCE_FILE_SHA256[5]}
    assert set(normalized["dataset_revision"]) == {ROUTERBENCH_REVISION}
    assert set(normalized["source_cost_method"]) == {"routerbench_precomputed"}


@pytest.mark.parametrize(
    ("shot_count", "source_file", "digest", "message"),
    [
        (1, "routerbench_0shot.pkl", SOURCE_FILE_SHA256[0], "shot_count"),
        (5, "routerbench_0shot.pkl", SOURCE_FILE_SHA256[0], "source_file"),
        (0, "routerbench_0shot.pkl", "not-a-hash", "SHA-256"),
    ],
)
def test_rejects_invalid_source_provenance(
    shot_count,
    source_file,
    digest,
    message,
):
    with pytest.raises(ValueError, match=message):
        normalize_wide_frame(
            _source_frame(),
            shot_count=shot_count,
            source_file=source_file,
            source_file_sha256=digest,
        )


def test_rejects_incomplete_or_ambiguous_wide_schema():
    missing = _source_frame().drop(columns=["gpt-3.5-turbo-1106|total_cost"])
    with pytest.raises(ValueError, match="missing columns"):
        normalize_wide_frame(
            missing,
            shot_count=0,
            source_file="routerbench_0shot.pkl",
            source_file_sha256=SOURCE_FILE_SHA256[0],
        )

    duplicate_columns = _source_frame()
    duplicate_columns.columns = [
        "sample_id" if index == 1 else column
        for index, column in enumerate(duplicate_columns.columns)
    ]
    with pytest.raises(ValueError, match="duplicate column"):
        normalize_wide_frame(
            duplicate_columns,
            shot_count=0,
            source_file="routerbench_0shot.pkl",
            source_file_sha256=SOURCE_FILE_SHA256[0],
        )


def test_normalized_path_stays_in_external_data_area():
    assert normalized_output_path(0) == (
        NORMALIZED_DATA_DIR / "routerbench_0shot.normalized.parquet"
    )
    assert "external" in normalized_output_path(5).parts
    with pytest.raises(ValueError, match="0 or 5"):
        normalized_output_path(1)
