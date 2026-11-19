"""Pinned, local-only adapters for separately stored external benchmarks.

This module accepts Parquet or JSONL files already present on disk. It has no
network or pickle-loading path; dataset downloads require separate approval.
"""

import argparse
import hashlib
import json
import math
import re
import unicodedata
import warnings
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from eval.dataset_registry import (
    TrustedDatasetEntry,
    assert_test_mode,
    lookup_trusted_entry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DATA_ROOT = PROJECT_ROOT / "data" / "external" / "benchmarks"
EXTERNAL_SCHEMA_VERSION = "1.0"
ADAPTER_VERSION = "1.0"
SPLIT_POLICY_VERSION = "1.0"

DATASET_SPECS = {
    "gsm8k": {
        "dataset_id": "openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "config": "main",
        "license_id": "MIT",
        "license_url": "https://opensource.org/license/mit/",
        "attribution": (
            "Cobbe et al., Training Verifiers to Solve Math Word Problems "
            "(2021), https://arxiv.org/abs/2110.14168"
        ),
        "citation_url": "https://arxiv.org/abs/2110.14168",
        "source_splits": {"train", "test"},
        "task_type": "math",
        "evaluation_method": "numeric_exact",
        "prompt_template_version": "gsm8k-v1",
    },
    "mmlu": {
        "dataset_id": "cais/mmlu",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "config": "all",
        "license_id": "MIT",
        "license_url": "https://opensource.org/license/mit/",
        "attribution": (
            "Hendrycks et al., Measuring Massive Multitask Language "
            "Understanding (ICLR 2021), https://arxiv.org/abs/2009.03300"
        ),
        "citation_url": "https://arxiv.org/abs/2009.03300",
        "source_splits": {"dev", "validation", "test"},
        "task_type": "knowledge",
        "evaluation_method": "multiple_choice_exact",
        "prompt_template_version": "mmlu-mc-v1",
    },
    "mbpp": {
        "dataset_id": "google-research-datasets/mbpp",
        "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        "config": "sanitized",
        "license_id": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": (
            "Austin et al., Program Synthesis with Large Language Models "
            "(2021), https://arxiv.org/abs/2108.07732"
        ),
        "citation_url": "https://arxiv.org/abs/2108.07732",
        "source_splits": {"train", "prompt", "validation", "test"},
        "task_type": "code",
        "evaluation_method": "unit_test",
        "prompt_template_version": "mbpp-code-v1",
    },
}

NORMALIZED_COLUMNS = [
    "schema_version",
    "dataset_key",
    "dataset_id",
    "dataset_revision",
    "dataset_config",
    "license_id",
    "license_url",
    "attribution",
    "citation_url",
    "adapter_version",
    "split_policy_version",
    "prompt_template_version",
    "source_split",
    "source_file_sha256",
    "source_file_format",
    "source_manifest_json",
    "benchmark_split",
    "source_id",
    "source_row_locator",
    "prompt_id",
    "source_question",
    "source_reference_answer_json",
    "source_record_json",
    "task_type",
    "prompt",
    "reference_answer",
    "evaluation_method",
    "choices",
    "choice_labels",
    "test_cases",
    "test_imports",
    "test_setup_code",
    "answer_variable",
]

_REQUIRED_BENCHMARK_SPLITS = {"train", "validation", "test"}
_ALLOWED_BENCHMARK_SPLITS = _REQUIRED_BENCHMARK_SPLITS | {"few_shot"}
_NEAR_DUPLICATE_TOKEN_JACCARD = 0.8


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Source records cannot contain non-finite numbers.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"Unsupported source value type: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def read_safe_source(path: str | Path) -> list[dict[str, Any]]:
    """Read only local Parquet/JSONL data; pickle formats are rejected."""
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        raise ValueError("Pickle source files are not supported.")
    return _parse_safe_source(source_path.read_bytes(), suffix)


def _parse_safe_source(content: bytes, suffix: str) -> list[dict[str, Any]]:
    if suffix == ".parquet":
        frame = pd.read_parquet(BytesIO(content))
        if not frame.columns.is_unique:
            raise ValueError("Source file contains duplicate column names.")
        return frame.to_dict(orient="records")
    if suffix == ".jsonl":
        rows = []
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("JSONL source must be UTF-8.") from exc
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}."
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL line {line_number} must contain an object."
                )
            rows.append(value)
        return rows
    raise ValueError("Only local .parquet and .jsonl files are supported.")


def _required_text(row: Mapping[str, Any], field: str, dataset: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{dataset} source field {field!r} must be non-empty text.")
    return value


def _benchmark_split(dataset_key: str, source_split: str, source_id: str) -> str:
    if dataset_key == "gsm8k":
        if source_split == "test":
            return "test"
        digest = hashlib.sha256(
            f"{DATASET_SPECS[dataset_key]['revision']}\0{source_id}".encode("utf-8")
        ).digest()
        return "validation" if int.from_bytes(digest[:8], "big") % 10 == 0 else "train"
    if dataset_key == "mmlu":
        return {
            "dev": "train",
            "validation": "validation",
            "test": "test",
        }[source_split]
    return {
        "train": "train",
        "prompt": "few_shot",
        "validation": "validation",
        "test": "test",
    }[source_split]


def _choice_label(value: Any) -> str:
    if not isinstance(value, (str, int, bool)) and hasattr(value, "item"):
        value = value.item()
    labels = "ABCD"
    if isinstance(value, bool):
        raise ValueError("MMLU answer label must be an index from 0 to 3 or A-D.")
    if isinstance(value, int) and 0 <= value < len(labels):
        return labels[value]
    if isinstance(value, str) and value.upper() in labels:
        return value.upper()
    raise ValueError("MMLU answer label must be an index from 0 to 3 or A-D.")


def _gsm8k_reference(answer: str) -> str:
    if answer.count("####") != 1:
        raise ValueError("GSM8K answer must contain exactly one #### delimiter.")
    reference = answer.rsplit("####", 1)[1].strip()
    if not reference:
        raise ValueError("GSM8K answer after #### must not be empty.")
    return reference


def _mmlu_prompt(subject: str, question: str, choices: list[str]) -> str:
    labels = ["A", "B", "C", "D"]
    return (
        f"Subject: {subject.replace('_', ' ')}\n"
        f"Question: {question}\n"
        + "\n".join(
            f"{label}. {choice}"
            for label, choice in zip(labels, choices)
        )
        + "\nReturn exactly one answer label (A, B, C, or D)."
    )


def _mbpp_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "Return Python code only. Do not execute or import external services."
    )


def _validate_source_manifest(
    dataset_key: str,
    source_split: str,
    path: Path,
    manifest: Mapping[str, Any],
    *,
    trusted_registry: Mapping[str, TrustedDatasetEntry] | None = None,
) -> tuple[str, TrustedDatasetEntry]:
    if not isinstance(manifest, Mapping):
        raise ValueError("A source manifest is required for every source split.")
    required = {
        "dataset_name",
        "approved_revision",
        "split",
        "format",
        "attribution",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(
            f"Source manifest is missing fields: {sorted(missing)}"
        )

    spec = DATASET_SPECS[dataset_key]
    if manifest["dataset_name"] != spec["dataset_id"]:
        raise ValueError("Source manifest dataset_name does not match the dataset.")
    if manifest["approved_revision"] != spec["revision"]:
        raise ValueError("Source manifest revision is not approved.")
    if manifest["split"] != source_split:
        raise ValueError("Source manifest split does not match the source file.")
    if manifest["attribution"] != spec["attribution"]:
        raise ValueError("Source manifest attribution does not match the dataset.")

    if trusted_registry is not None:
        assert_test_mode("Overriding trusted_registry")

    source_format = manifest["format"]
    expected_suffix = {"parquet": ".parquet", "jsonl": ".jsonl"}.get(source_format)
    if expected_suffix is None or path.suffix.lower() != expected_suffix:
        raise ValueError("Source manifest format does not match the source file.")

    trusted_entry = lookup_trusted_entry(
        dataset_name=manifest["dataset_name"],
        revision=manifest["approved_revision"],
        split=source_split,
        file_format=source_format,
        registry=trusted_registry,
        entry_id=manifest.get("registry_entry_id"),
    )

    if trusted_entry.approval_status != "approved":
        raise ValueError(
            f"Trusted registry entry '{trusted_entry.entry_id}' approval status is "
            f"'{trusted_entry.approval_status}'; ingestion is blocked."
        )

    if not trusted_entry.sha256:
        raise ValueError(
            f"Trusted registry entry '{trusted_entry.entry_id}' has no independently "
            "verified SHA-256; ingestion is blocked."
        )

    if "source_file_sha256" in manifest:
        user_hash = manifest["source_file_sha256"]
        if (
            not isinstance(user_hash, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", user_hash)
        ):
            raise ValueError(
                "Source manifest SHA-256 must be 64 hexadecimal characters."
            )
        if user_hash.lower() != trusted_entry.sha256.lower():
            raise ValueError(
                f"Source manifest SHA-256 ({user_hash}) does not match trusted "
                f"registry ({trusted_entry.sha256})."
            )

    return source_format, trusted_entry


def _read_manifest_source(
    dataset_key: str,
    source_split: str,
    path: str | Path,
    manifest: Mapping[str, Any] | None,
    *,
    trusted_registry: Mapping[str, TrustedDatasetEntry] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = Path(path)
    if source_path.suffix.lower() in {".pkl", ".pickle"}:
        raise ValueError("Pickle source files are not supported.")
    if manifest is None:
        raise ValueError("A source manifest is required for every source split.")
    source_format, trusted_entry = _validate_source_manifest(
        dataset_key,
        source_split,
        source_path,
        manifest,
        trusted_registry=trusted_registry,
    )
    content = source_path.read_bytes()
    actual_hash = hashlib.sha256(content).hexdigest()
    if actual_hash.lower() != trusted_entry.sha256.lower():
        raise ValueError(
            f"Source SHA-256 mismatch for {source_split}: "
            f"expected {trusted_entry.sha256}, received {actual_hash}."
        )
    rows = _parse_safe_source(content, source_path.suffix.lower())
    return rows, {
        "dataset_name": manifest["dataset_name"],
        "approved_revision": manifest["approved_revision"],
        "source_file_sha256": actual_hash,
        "split": manifest["split"],
        "format": source_format,
        "attribution": manifest["attribution"],
        "registry_entry_id": trusted_entry.entry_id,
        "publisher": trusted_entry.publisher,
        "source_url": trusted_entry.source_url,
        "license_approval_reference": trusted_entry.license_approval_reference,
        "approval_status": trusted_entry.approval_status,
    }


def _normalize_record(
    dataset_key: str,
    source_split: str,
    row_index: int,
    row: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset_key]
    locator = f"{source_split}:{row_index:07d}"
    choices = None
    choice_labels = None
    test_cases = None
    test_imports = None
    test_setup_code = None
    answer_variable = None

    if dataset_key == "gsm8k":
        question = _required_text(row, "question", dataset_key)
        original_answer = _required_text(row, "answer", dataset_key)
        reference = _gsm8k_reference(original_answer)
        source_id = locator
        prompt = question

    elif dataset_key == "mmlu":
        question = _required_text(row, "question", dataset_key)
        subject = _required_text(row, "subject", dataset_key)
        raw_choices = row.get("choices")
        if not isinstance(raw_choices, (list, tuple)) or len(raw_choices) != 4:
            raise ValueError("MMLU choices must contain exactly four options.")
        if any(not isinstance(choice, str) for choice in raw_choices):
            raise ValueError("MMLU choices must be strings.")
        choices = list(raw_choices)
        choice_labels = ["A", "B", "C", "D"]
        original_answer = row.get("answer")
        reference = _choice_label(original_answer)
        source_id = f"{locator}:{subject}"
        prompt = _mmlu_prompt(subject, question, choices)

    else:
        question = _required_text(row, "prompt", dataset_key)
        code = _required_text(row, "code", dataset_key)
        raw_task_id = row.get("task_id")
        if raw_task_id is None or isinstance(raw_task_id, bool):
            raise ValueError("MBPP task_id is required.")
        source_id = str(raw_task_id)
        raw_tests = row.get("test_list")
        if not isinstance(raw_tests, (list, tuple)) or not raw_tests:
            raise ValueError("MBPP test_list must be a non-empty list.")
        if any(not isinstance(test, str) or not test.strip() for test in raw_tests):
            raise ValueError("MBPP test_list entries must be non-empty strings.")
        raw_imports = row.get("test_imports", [])
        if not isinstance(raw_imports, (list, tuple)) or any(
            not isinstance(item, str) for item in raw_imports
        ):
            raise ValueError("MBPP test_imports must be a list of strings.")
        raw_setup = row.get("test_setup_code")
        if raw_setup is not None and not isinstance(raw_setup, str):
            raise ValueError("MBPP test_setup_code must be text when present.")
        reference = code
        test_cases = list(raw_tests)
        test_imports = list(raw_imports)
        test_setup_code = raw_setup
        prompt = _mbpp_prompt(question)
        original_answer = code

    benchmark_split = _benchmark_split(dataset_key, source_split, source_id)
    if benchmark_split not in _ALLOWED_BENCHMARK_SPLITS:
        raise ValueError("Adapter produced an unsupported benchmark split.")

    return {
        "schema_version": EXTERNAL_SCHEMA_VERSION,
        "dataset_key": dataset_key,
        "dataset_id": spec["dataset_id"],
        "dataset_revision": spec["revision"],
        "dataset_config": spec["config"],
        "license_id": spec["license_id"],
        "license_url": spec["license_url"],
        "attribution": spec["attribution"],
        "citation_url": spec["citation_url"],
        "adapter_version": ADAPTER_VERSION,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "prompt_template_version": spec["prompt_template_version"],
        "source_split": source_split,
        "source_file_sha256": source_manifest["source_file_sha256"],
        "source_file_format": source_manifest["format"],
        "source_manifest_json": _canonical_json(dict(source_manifest)),
        "benchmark_split": benchmark_split,
        "source_id": source_id,
        "source_row_locator": locator,
        "prompt_id": (
            f"{dataset_key}:{spec['revision']}:{spec['config']}:"
            f"{source_id}"
        ),
        "source_question": question,
        "source_reference_answer_json": _canonical_json(original_answer),
        "source_record_json": _canonical_json(dict(row)),
        "task_type": spec["task_type"],
        "prompt": prompt,
        "reference_answer": reference,
        "evaluation_method": spec["evaluation_method"],
        "choices": choices,
        "choice_labels": choice_labels,
        "test_cases": test_cases,
        "test_imports": test_imports,
        "test_setup_code": test_setup_code,
        "answer_variable": answer_variable,
    }


def normalize_source_splits(
    dataset_key: str,
    source_paths: Mapping[str, str | Path],
    source_manifests: Mapping[str, Mapping[str, Any]],
    *,
    trusted_registry: Mapping[str, TrustedDatasetEntry] | None = None,
) -> list[dict[str, Any]]:
    """Normalize source files only after each per-split manifest is verified."""
    if trusted_registry is not None:
        assert_test_mode("Overriding trusted_registry")

    spec = DATASET_SPECS.get(dataset_key)
    if spec is None:
        raise ValueError(f"Unsupported external dataset: {dataset_key}")
    if set(source_paths) != spec["source_splits"]:
        missing = sorted(spec["source_splits"].difference(source_paths))
        extra = sorted(set(source_paths).difference(spec["source_splits"]))
        raise ValueError(
            f"Source splits mismatch; missing={missing}, unsupported={extra}."
        )
    if set(source_manifests) != spec["source_splits"]:
        missing = sorted(spec["source_splits"].difference(source_manifests))
        extra = sorted(set(source_manifests).difference(spec["source_splits"]))
        raise ValueError(
            f"Source manifests mismatch; missing={missing}, unsupported={extra}."
        )

    records = []
    for source_split in sorted(source_paths):
        source_rows, source_manifest = _read_manifest_source(
            dataset_key,
            source_split,
            source_paths[source_split],
            source_manifests[source_split],
            trusted_registry=trusted_registry,
        )
        for row_index, row in enumerate(source_rows):
            if not isinstance(row, Mapping):
                raise ValueError("Every source row must be a mapping.")
            records.append(
                _normalize_record(
                    dataset_key,
                    source_split,
                    row_index,
                    row,
                    source_manifest,
                )
            )
    validate_split_integrity(records, trusted_registry=trusted_registry)
    return records


def validate_normalized_records(
    records: list[Mapping[str, Any]],
    *,
    trusted_registry: Mapping[str, TrustedDatasetEntry] | None = None,
) -> None:
    if trusted_registry is not None:
        assert_test_mode("Overriding trusted_registry")

    seen_prompt_ids = set()
    for index, record in enumerate(records):
        missing = set(NORMALIZED_COLUMNS).difference(record)
        if missing:
            raise ValueError(
                f"Normalized row {index} is missing columns: {sorted(missing)}"
            )
        dataset_key = record["dataset_key"]
        spec = DATASET_SPECS.get(dataset_key)
        if spec is None:
            raise ValueError(f"Unsupported external dataset: {dataset_key}")
        if any(
            record[field] != spec[expected]
            for field, expected in (
                ("dataset_id", "dataset_id"),
                ("dataset_revision", "revision"),
                ("dataset_config", "config"),
                ("license_id", "license_id"),
                ("license_url", "license_url"),
                ("attribution", "attribution"),
                ("citation_url", "citation_url"),
                ("prompt_template_version", "prompt_template_version"),
            )
        ):
            raise ValueError("Normalized row does not match its pinned dataset.")
        if (
            record["schema_version"] != EXTERNAL_SCHEMA_VERSION
            or record["adapter_version"] != ADAPTER_VERSION
            or record["split_policy_version"] != SPLIT_POLICY_VERSION
        ):
            raise ValueError("Normalized row has an unsupported schema version.")
        if (
            not isinstance(record["source_file_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["source_file_sha256"])
        ):
            raise ValueError("Normalized row has an invalid source-file hash.")
        try:
            source_manifest = json.loads(record["source_manifest_json"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("Normalized source manifest is invalid JSON.") from exc
        if not isinstance(source_manifest, dict):
            raise ValueError("Normalized source manifest must be a JSON object.")

        # Every normalized record must contain a valid registry_entry_id
        registry_entry_id = source_manifest.get("registry_entry_id")
        if not isinstance(registry_entry_id, str) or not registry_entry_id.strip():
            raise ValueError("Normalized record is missing a valid registry_entry_id.")

        # Every normalized record must contain expected publisher, source URL, approval reference, and approval_status="approved"
        for field in ("publisher", "source_url", "license_approval_reference", "approval_status"):
            val = source_manifest.get(field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"Normalized record manifest is missing valid {field}.")
        if source_manifest["approval_status"] != "approved":
            raise ValueError(
                f"Normalized record approval status is '{source_manifest['approval_status']}'; "
                "must be 'approved'."
            )

        # Resolve against trusted approval registry using dataset name, revision, split, format, and registry entry ID
        trusted_entry = lookup_trusted_entry(
            dataset_name=record["dataset_id"],
            revision=record["dataset_revision"],
            split=record["source_split"],
            file_format=record["source_file_format"],
            registry=trusted_registry,
            entry_id=registry_entry_id,
        )

        if trusted_entry.approval_status != "approved":
            raise ValueError(
                f"Trusted registry entry '{trusted_entry.entry_id}' approval status is "
                f"'{trusted_entry.approval_status}'; must be 'approved'."
            )
        if not trusted_entry.sha256:
            raise ValueError(
                f"Trusted registry entry '{trusted_entry.entry_id}' has no independently verified SHA-256."
            )

        # Compare normalized source hash and manifest hash against trusted registry SHA-256
        if record["source_file_sha256"].lower() != trusted_entry.sha256.lower():
            raise ValueError(
                f"Normalized record source hash ({record['source_file_sha256']}) does not match "
                f"trusted registry SHA-256 ({trusted_entry.sha256})."
            )
        if source_manifest.get("source_file_sha256", "").lower() != trusted_entry.sha256.lower():
            raise ValueError("Source manifest SHA-256 does not match trusted registry SHA-256.")

        # Reject records whose source manifest metadata does not match the trusted entry
        if source_manifest["publisher"] != trusted_entry.publisher:
            raise ValueError("Normalized record publisher does not match trusted registry.")
        if source_manifest["source_url"] != trusted_entry.source_url:
            raise ValueError("Normalized record source_url does not match trusted registry.")
        if source_manifest["license_approval_reference"] != trusted_entry.license_approval_reference:
            raise ValueError("Normalized record license_approval_reference does not match trusted registry.")
        if source_manifest["attribution"] != trusted_entry.attribution:
            raise ValueError("Normalized record attribution does not match trusted registry.")
        if source_manifest["dataset_name"] != trusted_entry.dataset_name:
            raise ValueError("Normalized record dataset_name does not match trusted registry.")
        if source_manifest["approved_revision"] != trusted_entry.approved_revision:
            raise ValueError("Normalized record approved_revision does not match trusted registry.")
        if source_manifest["split"] != trusted_entry.split:
            raise ValueError("Normalized record split does not match trusted registry.")
        if source_manifest["format"] != trusted_entry.file_format:
            raise ValueError("Normalized record format does not match trusted registry.")
        if record["source_split"] not in spec["source_splits"]:
            raise ValueError("Normalized row has an unsupported source split.")
        if record["benchmark_split"] not in _ALLOWED_BENCHMARK_SPLITS:
            raise ValueError("Normalized row has an unsupported benchmark split.")
        if record["benchmark_split"] != _benchmark_split(
            dataset_key,
            record["source_split"],
            record["source_id"],
        ):
            raise ValueError("Normalized row violates the fixed split policy.")
        if record["task_type"] != spec["task_type"]:
            raise ValueError("Normalized row has an invalid task type.")
        if record["evaluation_method"] != spec["evaluation_method"]:
            raise ValueError("Normalized row has an invalid evaluation method.")
        if not isinstance(record["source_id"], str) or not record["source_id"]:
            raise ValueError("Normalized row must retain a source_id.")
        if not isinstance(record["prompt"], str) or not record["prompt"].strip():
            raise ValueError("Normalized prompt must be non-empty text.")
        if not isinstance(record["source_question"], str):
            raise ValueError("Normalized row must retain its source question.")
        try:
            source_record = json.loads(record["source_record_json"])
            source_answer = json.loads(record["source_reference_answer_json"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("Normalized source provenance is invalid JSON.") from exc
        if not isinstance(source_record, dict):
            raise ValueError("source_record_json must encode an object.")
        if record["source_record_json"] != _canonical_json(source_record):
            raise ValueError("source_record_json must use canonical serialization.")
        if not isinstance(record["reference_answer"], str) or not record[
            "reference_answer"
        ]:
            raise ValueError("Normalized row must have a reference answer.")
        if not isinstance(record["source_row_locator"], str) or not record[
            "source_row_locator"
        ]:
            raise ValueError("Normalized row must retain a source row locator.")
        if record["source_question"] != source_record.get(
            "question" if dataset_key != "mbpp" else "prompt"
        ):
            raise ValueError("Normalized source question was changed.")
        if dataset_key == "gsm8k":
            expected_answer = source_record.get("answer")
            if (
                source_answer != expected_answer
                or record["reference_answer"] != _gsm8k_reference(expected_answer)
                or record["prompt"] != record["source_question"]
                or record["source_id"] != record["source_row_locator"]
            ):
                raise ValueError("GSM8K source answer or prompt was changed.")
        if record["dataset_key"] == "mmlu":
            expected_answer = source_record.get("answer")
            expected_choices = source_record.get("choices")
            expected_subject = source_record.get("subject")
            if (
                record["choice_labels"] != ["A", "B", "C", "D"]
                or not isinstance(record["choices"], list)
                or len(record["choices"]) != 4
                or record["choices"] != expected_choices
                or record["reference_answer"] != _choice_label(expected_answer)
                or source_answer != expected_answer
                or record["prompt"]
                != _mmlu_prompt(
                    expected_subject,
                    record["source_question"],
                    expected_choices,
                )
            ):
                raise ValueError("MMLU row changed its answer or choices.")
        if record["dataset_key"] == "mbpp":
            expected_code = source_record.get("code")
            expected_tests = source_record.get("test_list")
            expected_imports = source_record.get("test_imports", [])
            if not isinstance(record["test_cases"], list) or not record[
                "test_cases"
            ]:
                raise ValueError("MBPP row must preserve its test cases.")
            if not isinstance(record["test_imports"], list):
                raise ValueError("MBPP row must preserve its test imports.")
            if (
                source_answer != expected_code
                or record["reference_answer"] != expected_code
                or record["test_cases"] != expected_tests
                or record["test_imports"] != expected_imports
                or record["test_setup_code"] != source_record.get("test_setup_code")
                or record["prompt"] != _mbpp_prompt(record["source_question"])
                or record["source_id"] != str(source_record.get("task_id"))
            ):
                raise ValueError("MBPP source code, prompt, or tests were changed.")
        prompt_id = record["prompt_id"]
        expected_prompt_id = (
            f"{dataset_key}:{spec['revision']}:{spec['config']}:"
            f"{record['source_id']}"
        )
        if prompt_id != expected_prompt_id:
            raise ValueError("Normalized prompt_id does not match source provenance.")
        if prompt_id in seen_prompt_ids:
            raise ValueError(f"Duplicate normalized prompt_id: {prompt_id}")
        seen_prompt_ids.add(prompt_id)


def _prompt_template_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\w<>]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _prompt_tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return set(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _near_duplicate_across_splits(
    left: list[Mapping[str, Any]],
    right: list[Mapping[str, Any]],
) -> list[tuple[str, str, float]]:
    documents = [
        (record, _prompt_tokens(record["source_question"]), split)
        for split, group in (("left", left), ("right", right))
        for record in group
    ]
    documents = [document for document in documents if document[1]]
    frequencies = Counter(
        token
        for _, tokens, _ in documents
        for token in tokens
    )
    documents.sort(
        key=lambda item: (
            len(item[1]),
            str(item[0]["source_id"]),
            item[2],
        )
    )
    prefix_index: dict[str, list[int]] = {}
    matches: dict[tuple[str, str], float] = {}
    for document_index, (record, tokens, split) in enumerate(documents):
        prefix_length = (
            len(tokens)
            - math.ceil(_NEAR_DUPLICATE_TOKEN_JACCARD * len(tokens))
            + 1
        )
        candidates: set[int] = set()
        for token in tokens:
            candidates.update(prefix_index.get(token, ()))

        for candidate_index in candidates:
            candidate_record, candidate_tokens, candidate_split = documents[
                candidate_index
            ]
            if candidate_split == split:
                continue
            union = candidate_tokens | tokens
            similarity = len(candidate_tokens & tokens) / len(union)
            if similarity >= _NEAR_DUPLICATE_TOKEN_JACCARD:
                pair = (
                    str(candidate_record["source_id"]),
                    str(record["source_id"]),
                )
                matches[pair] = similarity

        ordered_tokens = sorted(
            tokens,
            key=lambda token: (frequencies[token], token),
        )
        for token in ordered_tokens[:prefix_length]:
            prefix_index.setdefault(token, []).append(document_index)
    return [
        (source_id_a, source_id_b, similarity)
        for (source_id_a, source_id_b), similarity in sorted(matches.items())
    ]


def validate_split_integrity(
    records: list[Mapping[str, Any]],
    *,
    trusted_registry: Mapping[str, TrustedDatasetEntry] | None = None,
) -> None:
    """Reject missing splits and duplicate/near-duplicate prompts across splits."""
    validate_normalized_records(records, trusted_registry=trusted_registry)
    dataset_keys = {record["dataset_key"] for record in records}
    if len(dataset_keys) != 1:
        raise ValueError("A normalized external dataset cannot mix datasets.")
    benchmark_splits = {record["benchmark_split"] for record in records}
    if not _REQUIRED_BENCHMARK_SPLITS.issubset(benchmark_splits):
        missing = sorted(_REQUIRED_BENCHMARK_SPLITS.difference(benchmark_splits))
        raise ValueError(f"Normalized data is missing benchmark splits: {missing}")

    source_ids: dict[str, str] = {}
    prompt_splits: dict[str, tuple[str, str]] = {}
    records_by_split: dict[str, list[Mapping[str, Any]]] = {}
    source_file_by_split: dict[str, tuple[str, str]] = {}
    for record in records:
        source_id = record["source_id"]
        split = record["benchmark_split"]
        records_by_split.setdefault(split, []).append(record)
        source_file = (
            record["source_file_sha256"],
            record["source_manifest_json"],
        )
        prior_source_file = source_file_by_split.get(record["source_split"])
        if prior_source_file is not None and prior_source_file != source_file:
            raise ValueError(
                f"Source split {record['source_split']!r} contains multiple "
                "source manifests."
            )
        source_file_by_split[record["source_split"]] = source_file
        prior_split = source_ids.get(source_id)
        if prior_split is not None and prior_split != split:
            raise ValueError(
                f"Source id {source_id!r} leaks across benchmark splits."
            )
        source_ids[source_id] = split

        key = _prompt_template_key(record["source_question"])
        if not key:
            raise ValueError("Source question is empty after normalization.")
        previous = prompt_splits.get(key)
        if previous is not None and previous[0] != split:
            raise ValueError(
                "Exact duplicate prompts cross "
                f"splits ({previous[0]} and {split}; "
                f"{previous[1]} and {record['source_id']})."
            )
        prompt_splits[key] = (split, source_id)

    split_names = sorted(records_by_split)
    for left_index, left_split in enumerate(split_names):
        for right_split in split_names[left_index + 1 :]:
            near_duplicates = _near_duplicate_across_splits(
                records_by_split[left_split],
                records_by_split[right_split],
            )
            for source_id_a, source_id_b, similarity in near_duplicates:
                warnings.warn(
                    "Potential near-duplicate prompts across benchmark splits "
                    f"{left_split}/{right_split}: source_ids="
                    f"{source_id_a!r},{source_id_b!r}; "
                    f"token_jaccard={similarity:.3f}. Review without removing "
                    "or modifying source records.",
                    UserWarning,
                    stacklevel=2,
                )


def normalized_split_path(dataset_key: str, benchmark_split: str) -> Path:
    spec = DATASET_SPECS.get(dataset_key)
    if spec is None:
        raise ValueError(f"Unsupported external dataset: {dataset_key}")
    if benchmark_split not in _ALLOWED_BENCHMARK_SPLITS:
        raise ValueError(f"Unsupported benchmark split: {benchmark_split}")
    return (
        EXTERNAL_DATA_ROOT
        / dataset_key
        / spec["revision"]
        / spec["config"]
        / "normalized"
        / f"{benchmark_split}.parquet"
    )


def write_normalized_splits(
    dataset_key: str,
    records: list[Mapping[str, Any]],
    *,
    root: str | Path | None = None,
) -> list[Path]:
    validate_split_integrity(records)
    root_path = Path(root) if root is not None else EXTERNAL_DATA_ROOT
    spec = DATASET_SPECS[dataset_key]
    output_dir = (
        root_path / dataset_key / spec["revision"] / spec["config"] / "normalized"
    )
    grouped = {
        split: [
            dict(record)
            for record in records
            if record["benchmark_split"] == split
        ]
        for split in sorted({record["benchmark_split"] for record in records})
    }
    outputs = [output_dir / f"{split}.parquet" for split in grouped]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite external normalized data: "
            + ", ".join(existing)
        )

    for split, rows in grouped.items():
        path = output_dir / f"{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame.from_records(rows, columns=NORMALIZED_COLUMNS).to_parquet(
            path,
            index=False,
        )
    return outputs


def normalize_local_files(
    dataset_key: str,
    split_paths: Mapping[str, str | Path],
    source_manifests: Mapping[str, Mapping[str, Any]],
    *,
    write: bool = False,
    trusted_registry: Mapping[str, TrustedDatasetEntry] | None = None,
) -> list[dict[str, Any]]:
    """Normalize caller-provided local safe files; never fetch remote data."""
    if trusted_registry is not None:
        assert_test_mode("Overriding trusted_registry")
    records = normalize_source_splits(
        dataset_key,
        split_paths,
        source_manifests,
        trusted_registry=trusted_registry,
    )
    if write:
        write_normalized_splits(dataset_key, records)
    return records


def _parse_split_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected SOURCE_SPLIT=LOCAL_PATH.")
    split, path = value.split("=", 1)
    if not split or not path:
        raise argparse.ArgumentTypeError("Expected SOURCE_SPLIT=LOCAL_PATH.")
    return split, Path(path)


def _read_manifest_files(
    parser: argparse.ArgumentParser,
    manifest_args: list[tuple[str, Path]],
) -> dict[str, dict[str, Any]]:
    manifests = {}
    for split, path in manifest_args:
        if split in manifests:
            parser.error(f"Manifest for split {split!r} was provided more than once.")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"Cannot read manifest {path}: {exc}")
        if not isinstance(manifest, dict):
            parser.error(f"Manifest {path} must contain a JSON object.")
        manifests[split] = manifest
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize pinned external benchmark files already on disk."
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_SPECS), required=True)
    parser.add_argument(
        "--split",
        action="append",
        type=_parse_split_argument,
        required=True,
        metavar="SOURCE_SPLIT=LOCAL_PATH",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=_parse_split_argument,
        required=True,
        metavar="SOURCE_SPLIT=MANIFEST_JSON",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write normalized Parquet files under data/external/benchmarks.",
    )
    args = parser.parse_args()
    split_paths = dict(args.split)
    if len(split_paths) != len(args.split):
        parser.error("Each source split may be provided only once.")
    manifests = _read_manifest_files(parser, args.manifest)
    records = normalize_local_files(
        args.dataset,
        split_paths,
        manifests,
        write=args.write,
    )
    print(f"Normalized rows: {len(records)}")
    print(f"Benchmark splits: {sorted({row['benchmark_split'] for row in records})}")
    if args.write:
        print("Saved normalized Parquet files under data/external/benchmarks.")
    else:
        print("Dry normalization only; no files were written.")


if __name__ == "__main__":
    main()
