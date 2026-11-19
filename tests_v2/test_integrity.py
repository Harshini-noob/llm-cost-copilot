import hashlib
import json

import pandas as pd

from eval.integrity import analyze_results, create_integrity_report


def _row(
    *,
    config="config-a",
    prompt="p1",
    model="model-a",
    run=1,
    success=True,
    status="scored",
    correct=True,
):
    return {
        "benchmark_config_id": config,
        "prompt_id": prompt,
        "model_id": model,
        "run": run,
        "execution_success": success,
        "evaluation_status": status,
        "correct": correct,
    }


def _consistency_by_config(report):
    return {
        row["configuration_group_id"]: row
        for row in report["consistency"]["rows"]
    }


def test_duplicate_report_separates_verified_groups_and_legacy_candidates():
    rows = [
        _row(run=1),
        _row(run=1, correct=False),
        _row(config=None, run=1),
        _row(config="legacy-unversioned", run=1),
    ]
    report = analyze_results(pd.DataFrame(rows))

    verified = report["duplicates"]["verified"]
    legacy = report["duplicates"]["legacy_candidates"]
    assert verified["group_count"] == 1
    assert verified["records_in_groups"] == 2
    assert verified["excess_records"] == 1
    assert verified["groups"][0]["verification"] == "verified_same_configuration"
    assert legacy["group_count"] == 1
    assert legacy["records_in_groups"] == 2
    assert legacy["excess_records"] == 1
    assert legacy["groups"][0]["verification"] == "candidate_configuration_unknown"


def test_consistency_never_combines_fingerprinted_configurations():
    rows = [
        _row(config="config-a", run=1, correct=True),
        _row(config="config-a", run=2, correct=True),
        _row(config="config-b", run=1, correct=False),
        _row(config="config-b", run=2, correct=False),
    ]
    report = analyze_results(pd.DataFrame(rows))

    classifications = {
        (row["configuration_group_id"], row["consistency_status"])
        for row in report["consistency"]["rows"]
    }
    assert classifications == {
        ("config-a", "consistent_correct"),
        ("config-b", "consistent_incorrect"),
    }
    assert all(row["scored_run_count"] == 2 for row in report["consistency"]["rows"])
    assert {
        row["benchmark_config_id"] for row in report["prompt_summaries"]
    } == {"config-a", "config-b"}


def test_legacy_rows_are_identified_and_not_scored_for_consistency():
    rows = [
        _row(config=None, run=1, correct=True),
        _row(config=None, run=2, correct=False),
    ]
    report = analyze_results(pd.DataFrame(rows))

    legacy_config = next(
        item for item in report["configurations"]
        if item["configuration_group_id"] == "legacy-unversioned"
    )
    consistency = _consistency_by_config(report)["legacy-unversioned"]
    assert legacy_config["fingerprint_available"] is False
    assert legacy_config["benchmark_config_id"] is None
    assert consistency["consistency_status"] == "insufficient_sample"
    assert consistency["insufficient_reason"] == "missing_configuration_fingerprint"
    assert consistency["label_agreement"] is None


def test_execution_failure_is_separate_from_correctness():
    rows = [
        _row(run=1, success=False, status="execution_failed", correct=None),
        _row(run=2, success=True, status="scored", correct=True),
    ]
    report = analyze_results(pd.DataFrame(rows))
    metrics = report["model_summaries"][0]["metrics"]

    assert metrics["sample_sizes"]["execution_failures"] == 1
    assert metrics["execution_success_rate"] == 0.5
    assert metrics["accuracy"] == 1.0
    assert metrics["sample_sizes"]["scored_evaluations"] == 1
    consistency = report["consistency"]["rows"][0]
    assert consistency["consistency_status"] == "insufficient_sample"
    assert consistency["scored_run_count"] == 1


def test_unscored_and_unsupported_rows_do_not_enter_accuracy_or_agreement():
    rows = [
        _row(run=1, status="scored", correct=True),
        _row(run=2, status="unsupported", correct=None),
        _row(run=3, status="unscored", correct=None),
    ]
    report = analyze_results(pd.DataFrame(rows))
    metrics = report["model_summaries"][0]["metrics"]

    assert metrics["sample_sizes"]["scored_evaluations"] == 1
    assert metrics["accuracy"] == 1.0
    assert metrics["label_agreement"] is None
    assert metrics["evaluation_coverage"] == 1.0
    assert metrics["evaluation_status_counts"] == {
        "scored": 1,
        "unscored": 1,
        "unsupported": 1,
    }
    assert metrics["insufficient_samples"]["accuracy"] is True
    assert metrics["insufficient_samples"]["label_agreement"] is True


def test_repeated_run_classifications_and_separate_metrics():
    rows = [
        _row(model="correct", run=1, correct=True),
        _row(model="correct", run=2, correct=True),
        _row(model="incorrect", run=1, correct=False),
        _row(model="incorrect", run=2, correct=False),
        _row(model="mixed", run=1, correct=True),
        _row(model="mixed", run=2, correct=False),
        _row(model="mixed", run=3, success=False, status="execution_failed", correct=None),
        _row(model="not-enough", run=1, correct=True),
    ]
    report = analyze_results(pd.DataFrame(rows))
    states = {
        row["model_id"]: row["consistency_status"]
        for row in report["consistency"]["rows"]
    }
    assert states == {
        "correct": "consistent_correct",
        "incorrect": "consistent_incorrect",
        "mixed": "inconsistent",
        "not-enough": "insufficient_sample",
    }

    mixed_metrics = next(
        row["metrics"] for row in report["model_summaries"]
        if row["model_id"] == "mixed"
    )
    assert mixed_metrics["execution_success_rate"] == 2 / 3
    assert mixed_metrics["evaluation_coverage"] == 1.0
    assert mixed_metrics["accuracy"] == 0.5
    assert mixed_metrics["label_agreement"] == 0.5
    assert mixed_metrics["sample_sizes"]["records"] == 3
    assert mixed_metrics["sample_sizes"]["label_agreement_scored_runs"] == 2


def test_report_includes_source_hash_and_does_not_modify_source(tmp_path):
    source = tmp_path / "matrix.parquet"
    output_dir = tmp_path / "reports"
    pd.DataFrame([_row()]).to_parquet(source, index=False)
    original_bytes = source.read_bytes()
    expected_hash = hashlib.sha256(original_bytes).hexdigest()

    report_path = create_integrity_report(source, output_dir)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["analysis_version"] == "1.4.0"
    assert report["source"]["parquet_sha256"] == expected_hash
    assert report["source"]["record_count"] == 1
    assert source.read_bytes() == original_bytes
    assert report_path.parent == output_dir
