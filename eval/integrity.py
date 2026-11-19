"""Read-only analysis of benchmark integrity and repeated-run consistency."""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "data" / "processed" / "matrix.parquet"
ANALYSIS_VERSION = "1.4.0"
REPORT_SCHEMA_VERSION = 1
LEGACY_CONFIG_GROUP = "legacy-unversioned"
MIN_CONSISTENCY_SAMPLE_SIZE = 2
_RECORDED_EVALUATION_STATUSES = {
    "scored",
    "unscored",
    "unsupported",
    "needs_review",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _json_value(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return str(value)


def _configuration_id(value: Any) -> str | None:
    if _is_missing(value):
        return None
    normalized = str(value).strip()
    if not normalized or normalized == LEGACY_CONFIG_GROUP:
        return None
    return normalized


def _as_boolean(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _stable_key(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"))


def _normalized_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for source_record in frame.to_dict(orient="records"):
        config_id = _configuration_id(source_record.get("benchmark_config_id"))
        status_value = source_record.get("evaluation_status")
        status = "missing" if _is_missing(status_value) else str(status_value).strip().lower()
        if not status:
            status = "missing"
        records.append({
            "configuration_group_id": config_id or LEGACY_CONFIG_GROUP,
            "benchmark_config_id": config_id,
            "fingerprint_available": config_id is not None,
            "prompt_id": _json_value(source_record.get("prompt_id")),
            "model_id": _json_value(source_record.get("model_id")),
            "run": _json_value(source_record.get("run")),
            "execution_success": _as_boolean(source_record.get("execution_success")),
            "evaluation_status": status,
            "correct": _as_boolean(source_record.get("correct")),
        })
    return records


def _record_is_scored(record: dict[str, Any]) -> bool:
    return (
        record["execution_success"] is True
        and record["evaluation_status"] == "scored"
        and record["correct"] is not None
    )


def _consistency_rows(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    excluded_incomplete_key_records = 0
    for record in records:
        if record["prompt_id"] is None or record["model_id"] is None:
            excluded_incomplete_key_records += 1
            continue
        key = (
            record["configuration_group_id"],
            _stable_key(record["prompt_id"]),
            _stable_key(record["model_id"]),
        )
        groups[key].append(record)

    rows = []
    for (config_group_id, _, _), group_records in sorted(groups.items()):
        first = group_records[0]
        scored = [record for record in group_records if _record_is_scored(record)]
        correct_count = sum(record["correct"] is True for record in scored)
        incorrect_count = sum(record["correct"] is False for record in scored)
        scored_count = len(scored)
        agreement_count = max(correct_count, incorrect_count) if scored_count else 0

        if not first["fingerprint_available"]:
            consistency_status = "insufficient_sample"
            insufficient_reason = "missing_configuration_fingerprint"
            label_agreement = None
        elif scored_count < MIN_CONSISTENCY_SAMPLE_SIZE:
            consistency_status = "insufficient_sample"
            insufficient_reason = "fewer_than_two_scored_runs"
            label_agreement = None
        elif correct_count == scored_count:
            consistency_status = "consistent_correct"
            insufficient_reason = None
            label_agreement = _rate(agreement_count, scored_count)
        elif incorrect_count == scored_count:
            consistency_status = "consistent_incorrect"
            insufficient_reason = None
            label_agreement = _rate(agreement_count, scored_count)
        else:
            consistency_status = "inconsistent"
            insufficient_reason = None
            label_agreement = _rate(agreement_count, scored_count)

        rows.append({
            "configuration_group_id": config_group_id,
            "benchmark_config_id": first["benchmark_config_id"],
            "fingerprint_available": first["fingerprint_available"],
            "prompt_id": first["prompt_id"],
            "model_id": first["model_id"],
            "run_count": len(group_records),
            "scored_run_count": scored_count,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "accuracy": _rate(correct_count, scored_count),
            "label_agreement": label_agreement,
            "consistency_status": consistency_status,
            "insufficient_reason": insufficient_reason,
        })

    return rows, excluded_incomplete_key_records


def _duplicates(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    verified: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    legacy: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    incomplete_key_records = 0

    for record in records:
        if (
            record["prompt_id"] is None
            or record["model_id"] is None
            or record["run"] is None
        ):
            incomplete_key_records += 1
            continue
        if record["fingerprint_available"]:
            key = (
                record["benchmark_config_id"],
                _stable_key(record["prompt_id"]),
                _stable_key(record["model_id"]),
                _stable_key(record["run"]),
            )
            verified[key].append(record)
        else:
            key = (
                _stable_key(record["prompt_id"]),
                _stable_key(record["model_id"]),
                _stable_key(record["run"]),
            )
            legacy[key].append(record)

    def report_groups(groups, include_config):
        duplicate_groups = []
        for group in groups.values():
            if len(group) < 2:
                continue
            first = group[0]
            row = {
                "configuration_group_id": first["configuration_group_id"],
                "benchmark_config_id": first["benchmark_config_id"],
                "fingerprint_available": first["fingerprint_available"],
                "prompt_id": first["prompt_id"],
                "model_id": first["model_id"],
                "run": first["run"],
                "record_count": len(group),
                "excess_records": len(group) - 1,
                "evaluation_statuses": sorted({
                    record["evaluation_status"] for record in group
                }),
            }
            if include_config:
                row["verification"] = "verified_same_configuration"
            else:
                row["verification"] = "candidate_configuration_unknown"
            duplicate_groups.append(row)
        duplicate_groups.sort(
            key=lambda row: (
                row["configuration_group_id"],
                str(row["prompt_id"]),
                str(row["model_id"]),
                str(row["run"]),
            )
        )
        records_in_groups = sum(row["record_count"] for row in duplicate_groups)
        excess_records = sum(row["excess_records"] for row in duplicate_groups)
        return {
            "group_count": len(duplicate_groups),
            "records_in_groups": records_in_groups,
            "excess_records": excess_records,
            "groups": duplicate_groups,
        }

    return report_groups(verified, True), report_groups(legacy, False), incomplete_key_records


def _summary_metrics(
    records: list[dict[str, Any]],
    consistency_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    record_count = len(records)
    success_count = sum(record["execution_success"] is True for record in records)
    failure_count = sum(record["execution_success"] is False for record in records)
    unknown_execution_count = record_count - success_count - failure_count
    scored = [record for record in records if _record_is_scored(record)]
    correct_count = sum(record["correct"] is True for record in scored)
    incorrect_count = sum(record["correct"] is False for record in scored)
    evaluation_covered_count = sum(
        record["execution_success"] is True
        and record["evaluation_status"] in _RECORDED_EVALUATION_STATUSES
        for record in records
    )

    agreement_groups = [
        row for row in consistency_rows
        if row["fingerprint_available"]
        and row["scored_run_count"] >= MIN_CONSISTENCY_SAMPLE_SIZE
    ]
    agreement_labels = sum(row["scored_run_count"] for row in agreement_groups)
    agreement_matches = sum(
        max(row["correct_count"], row["incorrect_count"])
        for row in agreement_groups
    )

    statuses = defaultdict(int)
    for record in records:
        statuses[record["evaluation_status"]] += 1

    return {
        "sample_sizes": {
            "records": record_count,
            "execution_successes": success_count,
            "execution_failures": failure_count,
            "execution_outcome_unknown": unknown_execution_count,
            "evaluation_covered_successes": evaluation_covered_count,
            "scored_evaluations": len(scored),
            "label_agreement_groups": len(agreement_groups),
            "label_agreement_scored_runs": agreement_labels,
        },
        "execution_success_rate": _rate(success_count, record_count),
        "evaluation_coverage": _rate(evaluation_covered_count, success_count),
        "accuracy": _rate(correct_count, len(scored)),
        "label_agreement": _rate(agreement_matches, agreement_labels),
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "evaluation_status_counts": dict(sorted(statuses.items())),
        "insufficient_samples": {
            "execution_success_rate": record_count < MIN_CONSISTENCY_SAMPLE_SIZE,
            "evaluation_coverage": success_count < MIN_CONSISTENCY_SAMPLE_SIZE,
            "accuracy": len(scored) < MIN_CONSISTENCY_SAMPLE_SIZE,
            "label_agreement": len(agreement_groups) == 0,
        },
    }


def analyze_results(
    frame: pd.DataFrame,
    *,
    source_path: str | Path | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Analyze benchmark rows without mutating the input frame."""
    records = _normalized_records(frame.copy())
    consistency_rows, incomplete_consistency_keys = _consistency_rows(records)
    verified_duplicates, legacy_candidates, incomplete_duplicate_keys = _duplicates(records)

    config_ids = sorted({
        record["configuration_group_id"] for record in records
    })
    configuration_summaries = []
    for config_group_id in config_ids:
        config_records = [
            record for record in records
            if record["configuration_group_id"] == config_group_id
        ]
        config_consistency = [
            row for row in consistency_rows
            if row["configuration_group_id"] == config_group_id
        ]
        is_fingerprinted = config_group_id != LEGACY_CONFIG_GROUP
        config_duplicate_groups = [
            row for row in verified_duplicates["groups"]
            if row["configuration_group_id"] == config_group_id
        ]
        legacy_duplicate_groups = [
            row for row in legacy_candidates["groups"]
            if row["configuration_group_id"] == config_group_id
        ]
        configuration_summaries.append({
            "configuration_group_id": config_group_id,
            "benchmark_config_id": config_group_id if is_fingerprinted else None,
            "fingerprint_available": is_fingerprinted,
            "record_count": len(config_records),
            "verified_duplicate_group_count": len(config_duplicate_groups),
            "legacy_duplicate_candidate_group_count": len(legacy_duplicate_groups),
            "metrics": _summary_metrics(config_records, config_consistency),
        })

    def make_summaries(group_field: str) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            group_value = record[group_field]
            if group_value is None:
                continue
            groups[
                (record["configuration_group_id"], _stable_key(group_value))
            ].append(record)

        summaries = []
        for (config_group_id, encoded_value), group_records in sorted(groups.items()):
            group_value = group_records[0][group_field]
            if group_field == "model_id":
                related_consistency = [
                    row for row in consistency_rows
                    if row["configuration_group_id"] == config_group_id
                    and _stable_key(row["model_id"]) == encoded_value
                ]
                summary_key = {"model_id": group_value}
            else:
                related_consistency = [
                    row for row in consistency_rows
                    if row["configuration_group_id"] == config_group_id
                    and _stable_key(row["prompt_id"]) == encoded_value
                ]
                summary_key = {"prompt_id": group_value}
            fingerprint_available = config_group_id != LEGACY_CONFIG_GROUP
            summaries.append({
                "configuration_group_id": config_group_id,
                "benchmark_config_id": (
                    config_group_id if fingerprint_available else None
                ),
                "fingerprint_available": fingerprint_available,
                **summary_key,
                "metrics": _summary_metrics(group_records, related_consistency),
            })
        return summaries

    return {
        "analysis_version": ANALYSIS_VERSION,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "parquet_path": str(Path(source_path).resolve()) if source_path else None,
            "parquet_sha256": source_sha256,
            "record_count": len(records),
        },
        "rules": {
            "minimum_scored_runs_for_consistency": MIN_CONSISTENCY_SAMPLE_SIZE,
            "accuracy_definition": "correct scored labels / all scored labels",
            "label_agreement_definition": (
                "Within each eligible prompt/model/configuration group, "
                "majority label count / scored labels; summary values are "
                "weighted across eligible groups."
            ),
            "evaluation_coverage_definition": (
                "Successful executions with a recorded evaluation status "
                "(scored, unscored, unsupported, or needs_review) / successful executions."
            ),
            "execution_success_rate_definition": (
                "Successful executions / all benchmark records."
            ),
            "legacy_consistency": (
                "Legacy rows are isolated and excluded from consistency "
                "calculations because their configuration fingerprint is unknown."
            ),
        },
        "configurations": configuration_summaries,
        "duplicates": {
            "verified": verified_duplicates,
            "legacy_candidates": legacy_candidates,
            "incomplete_key_record_count": incomplete_duplicate_keys,
        },
        "model_summaries": make_summaries("model_id"),
        "prompt_summaries": make_summaries("prompt_id"),
        "consistency": {
            "rows": consistency_rows,
            "excluded_incomplete_key_record_count": incomplete_consistency_keys,
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_integrity_report(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    output_dir: str | Path | None = None,
) -> Path:
    """Read a Parquet matrix and create a uniquely named JSON analysis report."""
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"Benchmark Parquet file not found: {source}")

    source_hash = _sha256_file(source)
    frame = pd.read_parquet(source)
    report = analyze_results(
        frame,
        source_path=source,
        source_sha256=source_hash,
    )

    destination_dir = Path(output_dir) if output_dir else source.parent
    destination_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = destination_dir / f"matrix_integrity_{timestamp}.json"
    with destination.open("x", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2, ensure_ascii=False, allow_nan=False)
        report_file.write("\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a read-only integrity report for benchmark results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help="Source benchmark Parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report directory (defaults to the source file's directory).",
    )
    args = parser.parse_args()
    report_path = create_integrity_report(args.input, args.output_dir)
    print(f"Integrity report saved to: {report_path}")


if __name__ == "__main__":
    main()
